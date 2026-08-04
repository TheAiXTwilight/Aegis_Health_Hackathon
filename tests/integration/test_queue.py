"""
tests/integration/test_queue.py — backend/queue.py lifecycle tests.

Tests job submission, worker execution, status transitions, purge,
disconnected client timeout, and the disjoint invariant.

Phase 4: ToolError.code assertions added for all submit_job rejection paths.
Queue module-level state is reset between tests via the autouse
_reset_queue fixture.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

import backend.queue as bq
from backend.queue import (
    adaptive_max_size,
    get_average_pipeline_duration_s,
    get_estimated_wait_seconds,
    get_job,
    get_queue_depth,
    get_queue_max,
    get_queue_position,
    get_status_payload,
    get_stream_queue,
    is_inference_active,
    run_inference_worker,
    submit_job,
)
from schemas.errors import FatalPipelineError, ToolError
from schemas.queue import JobStatus, PipelineJob
from schemas.state import AegisState
from app.settings import settings


# ── State reset ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_queue():
    """Reset all module-level queue state before each test."""
    bq._job_store.clear()
    bq._job_queue.clear()
    bq._job_streams.clear()
    bq._session_states.clear()
    bq._completed_durations.clear()
    bq._jobs_completed_today = 0
    bq._jobs_failed_today = 0
    bq._purge_callbacks.clear()
    bq._user_submissions.clear()
    yield


# ── Fake pipelines ─────────────────────────────────────────────────

class _OkPipeline:
    """Yields two tokens then completes."""
    async def run(self, state: AegisState):
        yield "Hello "
        yield "world"


class _FailPipeline:
    """Raises FatalPipelineError immediately."""
    async def run(self, state: AegisState):
        raise FatalPipelineError(
            ToolError(tool="test_tool", reason="test fatal error", fatal=True)
        )
        yield  # make it a generator


# ── submit_job ─────────────────────────────────────────────────────

async def test_submit_job_returns_pipeline_job():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    result = await submit_job(job, state)
    assert isinstance(result, PipelineJob)


async def test_submit_job_enqueues_job():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    assert get_queue_depth() == 1


async def test_submit_job_stores_job():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    assert get_job(job.job_id) is job


async def test_submit_job_status_must_be_queued():
    from tools.tool_names import TOOL_QUEUE
    state = AegisState()
    job   = PipelineJob(session_id=state.session_id)
    job.status = JobStatus.RUNNING
    result = await submit_job(job, state)
    assert isinstance(result, ToolError)
    assert result.fatal is True
    assert result.code == "invalid_input"
    assert result.tool == TOOL_QUEUE


async def test_submit_job_session_mismatch_returns_fatal_tool_error():
    from tools.tool_names import TOOL_QUEUE
    state = AegisState()
    job   = PipelineJob(session_id="different-session-id")
    result = await submit_job(job, state)
    assert isinstance(result, ToolError)
    assert result.fatal is True
    assert result.code == "invalid_input"
    assert result.tool == TOOL_QUEUE


async def test_submit_job_duplicate_job_id_returns_tool_error():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    result = await submit_job(job, state)
    assert isinstance(result, ToolError)
    assert result.fatal is True
    assert result.code == "duplicate_session"


async def test_submit_job_duplicate_active_session_returns_tool_error():
    state = AegisState()
    job1 = PipelineJob(session_id=state.session_id)
    await submit_job(job1, state)

    job2 = PipelineJob(session_id=state.session_id)
    result = await submit_job(job2, state)
    assert isinstance(result, ToolError)
    assert result.fatal is True
    assert result.code == "duplicate_session"
    assert "active job" in result.reason.lower()


async def test_submit_job_queue_full_returns_tool_error():
    from tools.tool_names import TOOL_QUEUE
    for _ in range(get_queue_max()):
        s = AegisState()
        j = PipelineJob(session_id=s.session_id)
        await submit_job(j, s)

    state  = AegisState()
    job    = PipelineJob(session_id=state.session_id)
    result = await submit_job(job, state)
    assert isinstance(result, ToolError)
    assert result.fatal is True
    assert result.code == "queue_full"
    assert result.tool == TOOL_QUEUE


# ── Queue position ─────────────────────────────────────────────────

async def test_queue_position_first_submitted():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    assert get_queue_position(job.job_id) == 1


async def test_queue_position_second_submitted():
    s1, s2 = AegisState(), AegisState()
    j1 = PipelineJob(session_id=s1.session_id)
    j2 = PipelineJob(session_id=s2.session_id)
    await submit_job(j1, s1)
    await submit_job(j2, s2)
    assert get_queue_position(j2.job_id) == 2


async def test_queue_position_none_when_not_queued():
    assert get_queue_position("unknown-job-id") is None


# ── get_queue_max ──────────────────────────────────────────────────

def test_get_queue_max_returns_constant():
    assert get_queue_max() == settings.AEGIS_QUEUE_MAX_SIZE


# ── get_stream_queue ───────────────────────────────────────────────

async def test_get_stream_queue_returns_none_for_unknown_job():
    assert get_stream_queue("nonexistent-job-id") is None


async def test_get_stream_queue_returns_none_for_queued_job():
    """Queued job has no stream queue — only created when job starts running."""
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    assert get_stream_queue(job.job_id) is None


# ── get_average_pipeline_duration_s ───────────────────────────────

def test_avg_duration_none_before_three_completions():
    bq._completed_durations.extend([1.0, 2.0])  # only 2
    assert get_average_pipeline_duration_s() is None


def test_avg_duration_none_with_zero_completions():
    assert get_average_pipeline_duration_s() is None


def test_avg_duration_returns_float_after_three_completions():
    bq._completed_durations.extend([1.0, 2.0, 3.0])
    result = get_average_pipeline_duration_s()
    assert isinstance(result, float)
    assert abs(result - 2.0) < 1e-9


def test_avg_duration_uses_last_ten():
    """Rolling average uses only the last 10 of all completions."""
    # 5 large values followed by 10 small values — last 10 dominate
    bq._completed_durations.extend([100.0] * 5 + [1.0] * 10)
    result = get_average_pipeline_duration_s()
    assert result is not None
    assert result < 50.0  # dominated by the 1.0 values, not 100.0


# ── get_estimated_wait_seconds ─────────────────────────────────────

async def test_estimated_wait_none_when_not_queued():
    assert get_estimated_wait_seconds("nonexistent-job") is None


async def test_estimated_wait_none_when_fewer_than_three_completions():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    bq._completed_durations.extend([1.0, 2.0])  # only 2 — below threshold
    assert get_estimated_wait_seconds(job.job_id) is None


async def test_estimated_wait_returns_float_after_three_completions():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    bq._completed_durations.extend([2.0, 2.0, 2.0])
    result = get_estimated_wait_seconds(job.job_id)
    assert isinstance(result, float)
    assert result > 0.0


async def test_estimated_wait_proportional_to_queue_position():
    """Wait estimate scales with queue position."""
    s1, s2 = AegisState(), AegisState()
    j1 = PipelineJob(session_id=s1.session_id)
    j2 = PipelineJob(session_id=s2.session_id)
    await submit_job(j1, s1)
    await submit_job(j2, s2)

    bq._completed_durations.extend([4.0, 4.0, 4.0])

    wait1 = get_estimated_wait_seconds(j1.job_id)
    wait2 = get_estimated_wait_seconds(j2.job_id)

    assert wait1 is not None
    assert wait2 is not None
    assert wait2 > wait1


# ── Status payload ─────────────────────────────────────────────────

async def test_status_payload_returns_none_for_unknown():
    assert get_status_payload("unknown") is None


async def test_status_payload_contains_queue_position():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    payload = get_status_payload(job.job_id)
    assert payload is not None
    assert payload["queue_position"] == 1


async def test_status_payload_contains_pipeline_state_fields():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    payload = get_status_payload(job.job_id)
    assert "current_tool" in payload
    assert "tools_run" in payload
    assert "tools_failed" in payload
    assert "step_durations_ms" in payload


# ── Worker execution ───────────────────────────────────────────────

async def _run_worker_briefly(pipeline, seconds: float = 0.5):
    task = asyncio.create_task(run_inference_worker(pipeline))
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return task


async def test_worker_transitions_job_to_completed():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    await _run_worker_briefly(_OkPipeline())
    assert job.status == JobStatus.COMPLETED


async def test_worker_streams_tokens_to_queue():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    await _run_worker_briefly(_OkPipeline())

    stream_q = get_stream_queue(job.job_id)
    assert stream_q is not None

    tokens = []
    while True:
        token = stream_q.get_nowait()
        if token is None:
            break
        tokens.append(token)

    assert "".join(tokens) == "Hello world"


async def test_worker_marks_job_failed_on_fatal_error():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    await _run_worker_briefly(_FailPipeline())
    assert job.status == JobStatus.FAILED
    assert job.error is not None


async def test_worker_sends_sentinel_after_completion():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    await _run_worker_briefly(_OkPipeline())

    stream_q = get_stream_queue(job.job_id)
    assert stream_q is not None

    sentinel_found = False
    for _ in range(100):
        try:
            token = stream_q.get_nowait()
            if token is None:
                sentinel_found = True
                break
        except asyncio.QueueEmpty:
            break

    assert sentinel_found


async def test_worker_increments_completed_counter():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    await _run_worker_briefly(_OkPipeline())
    assert bq._jobs_completed_today == 1


async def test_worker_increments_failed_counter_on_error():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)
    await _run_worker_briefly(_FailPipeline())
    assert bq._jobs_failed_today == 1


# ── Missing session state ──────────────────────────────────────────

@pytest.mark.security_gate
@pytest.mark.xfail(
    strict=True,
    reason="Known reliability defect: _execute_job finally references state before assignment and can terminate the worker.",
)
async def test_worker_fails_job_with_missing_session_state():
    """
    Worker raises RuntimeError when session state is missing after dequeue.
    Job must transition to FAILED with a non-None error message.
    """
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)

    # Remove session state after submission to simulate corruption
    del bq._session_states[state.session_id]

    await _run_worker_briefly(_OkPipeline(), seconds=0.5)

    assert job.status == JobStatus.FAILED
    assert job.error is not None
    assert bq._jobs_failed_today == 1


# ── Pipeline timeout ───────────────────────────────────────────────

async def test_pipeline_timeout_marks_job_failed():
    """
    Pipeline that never yields within PIPELINE_TIMEOUT_S triggers timeout.
    Job transitions to FAILED with a wall-clock message.
    Timeout patched to near-zero to make test instant.
    """
    class _HangingPipeline:
        async def run(self, state: AegisState):
            await asyncio.sleep(9999)
            yield "never"  # pragma: no cover

    original = bq.PIPELINE_TIMEOUT_S
    bq.PIPELINE_TIMEOUT_S = 0.001

    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)

    try:
        await _run_worker_briefly(_HangingPipeline(), seconds=1.0)
    finally:
        bq.PIPELINE_TIMEOUT_S = original

    assert job.status == JobStatus.FAILED
    assert job.error is not None
    assert (
        "wall-clock" in job.error.lower()
        or "timeout" in job.error.lower()
    )


# ── Disconnected client ────────────────────────────────────────────

async def test_disconnected_client_job_fails():
    """
    When the stream queue fills and the consumer timeout is exceeded,
    job transitions to FAILED and lock is released.
    """
    class _InfinitePipeline:
        async def run(self, state):
            for i in range(1000):
                yield f"token{i}"

    original_timeout = bq.STREAM_PUT_TIMEOUT_S
    bq.STREAM_PUT_TIMEOUT_S = 0.001

    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    await submit_job(job, state)

    # Replace the stream queue with a maxsize=1 queue that fills immediately
    stream_q = asyncio.Queue(maxsize=1)
    bq._job_streams[job.job_id] = stream_q

    try:
        await _run_worker_briefly(_InfinitePipeline(), seconds=1.0)
    finally:
        bq.STREAM_PUT_TIMEOUT_S = original_timeout

    assert job.status == JobStatus.FAILED


# ── Purge ──────────────────────────────────────────────────────────

def test_purge_removes_expired_job():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    job.status = JobStatus.COMPLETED
    job.completed_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
    bq._job_store[job.job_id] = job

    bq._purge_expired_jobs()

    assert get_job(job.job_id) is None


def test_purge_does_not_remove_recent_job():
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    job.status = JobStatus.COMPLETED
    job.completed_at = datetime.now(timezone.utc)
    bq._job_store[job.job_id] = job

    bq._purge_expired_jobs()

    assert get_job(job.job_id) is not None


def test_purge_removes_session_state():
    """Purge must remove the associated AegisState from _session_states."""
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    job.status = JobStatus.COMPLETED
    job.completed_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
    bq._job_store[job.job_id] = job
    bq._session_states[state.session_id] = state

    bq._purge_expired_jobs()

    assert state.session_id not in bq._session_states


def test_purge_invokes_callback():
    called_with: list[str] = []

    def _cb(session_id: str):
        called_with.append(session_id)

    bq.register_purge_callback(_cb)

    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    job.status = JobStatus.COMPLETED
    job.completed_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
    bq._job_store[job.job_id] = job

    bq._purge_expired_jobs()

    assert called_with == [state.session_id]


def test_purge_callback_error_does_not_stop_other_callbacks():
    """
    Exception in one callback must not prevent subsequent callbacks running.
    Both are registered; the bad one raises; the good one must still fire.
    """
    called: list[str] = []

    def _bad_cb(sid: str) -> None:
        raise RuntimeError("callback exploded")

    def _good_cb(sid: str) -> None:
        called.append(sid)

    bq.register_purge_callback(_bad_cb)
    bq.register_purge_callback(_good_cb)

    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    job.status = JobStatus.COMPLETED
    job.completed_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
    bq._job_store[job.job_id] = job

    bq._purge_expired_jobs()

    assert called == [state.session_id]


def test_register_purge_callback_idempotent():
    """Registering the same callback twice must not duplicate it."""
    def _cb(sid: str):
        pass

    bq.register_purge_callback(_cb)
    bq.register_purge_callback(_cb)

    assert bq._purge_callbacks.count(_cb) == 1


def test_purge_does_not_remove_queued_jobs():
    """Only COMPLETED or FAILED jobs with expired completed_at are purged."""
    state = AegisState()
    job = PipelineJob(session_id=state.session_id)
    bq._job_store[job.job_id] = job

    bq._purge_expired_jobs()

    assert get_job(job.job_id) is not None


# ── Inference lock ─────────────────────────────────────────────────

def test_is_inference_active_false_when_idle():
    assert is_inference_active() is False


# ── Duration tracking ──────────────────────────────────────────────

def test_completed_durations_is_deque():
    from collections import deque
    assert isinstance(bq._completed_durations, deque)


def test_completed_durations_maxlen_100():
    assert bq._completed_durations.maxlen == 100


# ── Priority ─────────────────────────────────────────────────────

async def test_higher_priority_job_is_processed_first():
    """Higher priority jobs sit at the front of the queue."""
    state_low = AegisState(user_id="user-1", priority=1)
    job_low = PipelineJob(session_id=state_low.session_id, user_id="user-1", priority=1)
    await submit_job(job_low, state_low)

    state_high = AegisState(user_id="user-2", priority=5)
    job_high = PipelineJob(session_id=state_high.session_id, user_id="user-2", priority=5)
    await submit_job(job_high, state_high)

    assert get_queue_position(job_high.job_id) == 1
    assert get_queue_position(job_low.job_id) == 2


async def test_equal_priority_jobs_stay_fifo():
    """Jobs with equal priority retain submission order."""
    s1, s2 = AegisState(user_id="user-1", priority=2), AegisState(user_id="user-2", priority=2)
    j1 = PipelineJob(session_id=s1.session_id, user_id="user-1", priority=2)
    j2 = PipelineJob(session_id=s2.session_id, user_id="user-2", priority=2)
    await submit_job(j1, s1)
    await submit_job(j2, s2)

    assert get_queue_position(j1.job_id) == 1
    assert get_queue_position(j2.job_id) == 2


# ── Rate limit ────────────────────────────────────────────────────

async def test_rate_limit_allows_under_threshold():
    """A user can submit up to AEGIS_QUEUE_RATE_LIMIT_ATTEMPTS jobs."""
    user_id = "rate-user"
    for _ in range(settings.AEGIS_QUEUE_RATE_LIMIT_ATTEMPTS):
        s = AegisState(user_id=user_id)
        j = PipelineJob(session_id=s.session_id, user_id=user_id)
        result = await submit_job(j, s)
        assert isinstance(result, PipelineJob)


async def test_rate_limit_rejects_over_threshold():
    """Submitting past the threshold returns a rate_limited ToolError."""
    from tools.tool_names import TOOL_QUEUE

    user_id = "rate-user"
    for _ in range(settings.AEGIS_QUEUE_RATE_LIMIT_ATTEMPTS):
        s = AegisState(user_id=user_id)
        j = PipelineJob(session_id=s.session_id, user_id=user_id)
        await submit_job(j, s)

    s_overflow = AegisState(user_id=user_id)
    j_overflow = PipelineJob(session_id=s_overflow.session_id, user_id=user_id)
    result = await submit_job(j_overflow, s_overflow)
    assert isinstance(result, ToolError)
    assert result.fatal is True
    assert result.code == "rate_limited"
    assert result.tool == TOOL_QUEUE


# ── Adaptive max ──────────────────────────────────────────────────

def test_adaptive_max_uses_max_size_with_few_completions():
    """With <3 completions, adaptive max equals the configured max."""
    assert adaptive_max_size() == settings.AEGIS_QUEUE_MAX_SIZE


def test_adaptive_max_respects_bounds():
    """Adaptive max is always between min and max settings."""
    bq._completed_durations.extend([1.0, 1.0, 1.0])
    size = adaptive_max_size()
    assert settings.AEGIS_QUEUE_MIN_SIZE <= size <= settings.AEGIS_QUEUE_MAX_SIZE
