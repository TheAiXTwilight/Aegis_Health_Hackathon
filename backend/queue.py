from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Protocol

from loguru import logger

from schemas.errors import FatalPipelineError, ToolError
from schemas.queue import JobStatus, PipelineJob
from schemas.state import AegisState


class PipelineRunner(Protocol):
    """
    Minimal protocol expected by the queue worker.

    AegisPipeline will satisfy this by exposing:

        def run(self, state: AegisState) -> AsyncIterator[str]

    The pipeline mutates AegisState in place and yields report tokens.
    """

    def run(self, state: AegisState) -> AsyncIterator[str]:
        ...


# ── Constants ────────────────────────────────────────────────────
# NOTE: All state below is module-level.
# This works correctly only with a single Uvicorn worker
# (--workers 1), which is mandatory per the spec.
# Running --workers 2 would give each worker independent state
# and silently break queue, job store, and stream isolation.

MAX_QUEUE_SIZE = 10
PIPELINE_TIMEOUT_S = 180
JOB_RETENTION_SECONDS = 3600

STREAM_QUEUE_MAXSIZE = 256
STREAM_PUT_TIMEOUT_S = 30.0


# ── In-memory state ──────────────────────────────────────────────

_job_store: dict[str, PipelineJob] = {}
_job_queue: deque[str] = deque()
_job_streams: dict[str, asyncio.Queue[str | None]] = {}
_session_states: dict[str, AegisState] = {}

# asyncio.Lock() created at module import time.
# Safe in Python 3.10+ — no running event loop required for Lock creation.
_inference_lock: asyncio.Lock = asyncio.Lock()

_completed_durations: deque[float] = deque(maxlen=100)

_jobs_completed_today: int = 0
_jobs_failed_today: int = 0

_purge_callbacks: list[Callable[[str], None]] = []


# ── Purge callback registration ──────────────────────────────────

def register_purge_callback(callback: Callable[[str], None]) -> None:
    """
    Register a function to be called when a job is purged.

    The callback receives the purged job's session_id. Used by
    backend/main.py to clean up uploaded files when a job's
    retention window expires.

    Idempotent — registering the same callable twice is a no-op.
    This protects against duplicate registration if the FastAPI
    lifespan is re-entered (e.g. across test sessions).

    Exceptions raised inside callbacks are logged but do not stop
    other callbacks from running.
    """
    if callback not in _purge_callbacks:
        _purge_callbacks.append(callback)


# ── Public queue inspection helpers ──────────────────────────────

def get_job(job_id: str) -> PipelineJob | None:
    """Return a stored job, or None if unknown/purged."""
    return _job_store.get(job_id)


def get_stream_queue(job_id: str) -> asyncio.Queue[str | None] | None:
    """
    Return the active stream queue for a running/completed job.

    Returns None when:
        - job is still queued
        - job_id is unknown
        - job was purged after retention
    """
    return _job_streams.get(job_id)


def get_queue_position(job_id: str) -> int | None:
    """
    Return current 1-based position, or None if not queued.

    Computed dynamically — never stored. O(n) over MAX_QUEUE_SIZE=10.
    """
    try:
        return list(_job_queue).index(job_id) + 1
    except ValueError:
        return None


def get_estimated_wait_seconds(job_id: str) -> float | None:
    """
    Return estimated wait in seconds, or None if not queued or if fewer
    than 3 completed durations exist.

    Uses rolling average of the last 10 completed pipeline durations.
    """
    position = get_queue_position(job_id)
    if position is None or len(_completed_durations) < 3:
        return None

    recent = list(_completed_durations)[-10:]
    avg = sum(recent) / len(recent)
    return position * avg


def get_average_pipeline_duration_s() -> float | None:
    """
    Return rolling average duration over last 10 completions.

    Returns None until at least 3 completions exist.
    """
    if len(_completed_durations) < 3:
        return None

    recent = list(_completed_durations)[-10:]
    return sum(recent) / len(recent)


def get_queue_depth() -> int:
    """Number of jobs currently waiting in FIFO queue."""
    return len(_job_queue)


def get_queue_max() -> int:
    """Maximum number of waiting jobs allowed."""
    return MAX_QUEUE_SIZE


def is_inference_active() -> bool:
    """True when the single inference worker is executing a job."""
    return _inference_lock.locked()


def get_jobs_completed_today() -> int:
    """In-memory completion counter. Resets on process restart."""
    return _jobs_completed_today


def get_jobs_failed_today() -> int:
    """In-memory failure counter. Resets on process restart."""
    return _jobs_failed_today


def get_status_payload(job_id: str) -> dict[str, Any] | None:
    """
    API-friendly job status payload.

    Adds dynamic fields beyond PipelineJob's stored state:

        queue_position          — 1-based FIFO position, None if not queued
        estimated_wait_seconds  — rolling avg * position, None if < 3 completions
        current_tool            — tool name currently executing, None if not running
        tools_run               — tools that completed successfully so far
        tools_failed            — tools that produced a ToolError so far
        step_durations_ms       — wall-clock ms per completed tool

    The pipeline state fields (current_tool, tools_run, tools_failed,
    step_durations_ms) are read directly from AegisState so the frontend
    sidebar gets live progress without any stream parsing.
    """
    job = get_job(job_id)
    if job is None:
        return None

    payload = job.model_dump(mode="json")
    payload["queue_position"]         = get_queue_position(job_id)
    payload["estimated_wait_seconds"] = get_estimated_wait_seconds(job_id)

    state = _session_states.get(job.session_id)
    if state is not None:
        payload["current_tool"]      = state.current_tool
        payload["tools_run"]         = list(state.tools_run)
        payload["tools_failed"]      = list(state.tools_failed)
        payload["step_durations_ms"] = dict(state.step_durations_ms)
    else:
        payload["current_tool"]      = None
        payload["tools_run"]         = []
        payload["tools_failed"]      = []
        payload["step_durations_ms"] = {}

    return payload


# ── Submission ───────────────────────────────────────────────────

async def submit_job(
    job: PipelineJob,
    state: AegisState,
) -> PipelineJob | ToolError:
    """
    Enqueue a job and register its AegisState.

    Runs after upload/input validation and before pipeline execution.

    Returns:
        PipelineJob on success
        ToolError(fatal=True) when rejected before queue entry
    """
    if job.status != JobStatus.QUEUED:
        return ToolError(
            tool="queue",
            reason=(
                f"Only queued jobs can be submitted. "
                f"Got status={job.status.value!r}."
            ),
            fatal=True,
        )

    if job.session_id != state.session_id:
        return ToolError(
            tool="queue",
            reason=(
                f"Job/session mismatch: job.session_id={job.session_id!r}, "
                f"state.session_id={state.session_id!r}."
            ),
            fatal=True,
        )

    if job.job_id in _job_store:
        return ToolError(
            tool="queue",
            reason=f"Duplicate job_id: {job.job_id}",
            fatal=True,
        )

    active_same_session = any(
        existing.session_id == job.session_id
        and existing.status in (JobStatus.QUEUED, JobStatus.RUNNING)
        for existing in _job_store.values()
    )
    if active_same_session:
        return ToolError(
            tool="queue",
            reason=f"Session already has an active job: {job.session_id}",
            fatal=True,
        )

    if len(_job_queue) >= MAX_QUEUE_SIZE:
        return ToolError(
            tool="queue",
            reason=f"Queue full ({MAX_QUEUE_SIZE} jobs). Try again shortly.",
            fatal=True,
        )

    _job_store[job.job_id] = job
    _session_states[job.session_id] = state
    _job_queue.append(job.job_id)

    logger.info(
        "Job queued",
        job_id=job.job_id,
        session_id=job.session_id,
        queue_depth=len(_job_queue),
    )

    return job


# ── Worker ───────────────────────────────────────────────────────

async def run_inference_worker(pipeline: PipelineRunner) -> None:
    """
    Single worker — runs for application lifetime.

    Processes queued jobs FIFO. Only one pipeline runs at a time due to
    _inference_lock. The pipeline dependency is injected by backend.main,
    avoiding global pipeline state and circular imports.
    """
    while True:
        _purge_expired_jobs()

        if not _job_queue:
            await asyncio.sleep(0.1)
            continue

        job_id = _job_queue.popleft()
        job = _job_store.get(job_id)
        if job is None:
            logger.warning("Queued job missing from store", job_id=job_id)
            continue

        async with _inference_lock:
            await _execute_job(job, pipeline)


async def _execute_job(job: PipelineJob, pipeline: PipelineRunner) -> None:
    """
    Execute one pipeline job under the inference lock.

    Called exclusively by run_inference_worker() while _inference_lock is held.

    Sentinel guarantee:
        stream_q sentinel (None) is attempted in finally exactly once.
        put_nowait() ensures cleanup cannot block.

    Disconnected client protection:
        each stream_q.put(token) is wrapped with asyncio.wait_for(...).
        If the consumer does not read within STREAM_PUT_TIMEOUT_S, job is
        marked FAILED and the lock is released.

    Pipeline wall-clock limit:
        asyncio.timeout(PIPELINE_TIMEOUT_S) wraps the whole streaming loop.
        Exceeded → asyncio.TimeoutError → job marked FAILED.

    Fatal pipeline error:
        FatalPipelineError carries the originating ToolError. Job is marked
        FAILED with the ToolError's reason as job.error.
    """
    global _jobs_completed_today, _jobs_failed_today

    stream_q: asyncio.Queue[str | None] = asyncio.Queue(
        maxsize=STREAM_QUEUE_MAXSIZE
    )
    _job_streams[job.job_id] = stream_q

    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)

    logger.info(
        "Job started",
        job_id=job.job_id,
        session_id=job.session_id,
    )

    try:
        if job.session_id not in _session_states:
            raise RuntimeError(
                f"Missing AegisState for session_id={job.session_id!r}"
            )

        state = _session_states[job.session_id]

        async with asyncio.timeout(PIPELINE_TIMEOUT_S):
            async for token in pipeline.run(state):
                try:
                    await asyncio.wait_for(
                        stream_q.put(token),
                        timeout=STREAM_PUT_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.now(timezone.utc)
                    job.error = (
                        f"Stream consumer did not read within "
                        f"{STREAM_PUT_TIMEOUT_S}s. "
                        "Client may have disconnected or become too slow."
                    )
                    _jobs_failed_today += 1

                    logger.warning(
                        "Job failed: stream consumer timeout",
                        job_id=job.job_id,
                        session_id=job.session_id,
                        timeout_s=STREAM_PUT_TIMEOUT_S,
                    )
                    return

        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)

        if job.started_at is not None:
            duration = (job.completed_at - job.started_at).total_seconds()
            _completed_durations.append(duration)

        _jobs_completed_today += 1

        logger.info(
            "Job completed",
            job_id=job.job_id,
            session_id=job.session_id,
            duration_s=(
                (job.completed_at - job.started_at).total_seconds()
                if job.started_at is not None
                else None
            ),
        )

    except asyncio.TimeoutError:
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.error = f"Pipeline exceeded {PIPELINE_TIMEOUT_S}s wall-clock limit."
        _jobs_failed_today += 1

        logger.warning(
            "Job failed: pipeline timeout",
            job_id=job.job_id,
            session_id=job.session_id,
            timeout_s=PIPELINE_TIMEOUT_S,
        )

    except FatalPipelineError as e:
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.error = e.tool_error.reason
        _jobs_failed_today += 1

        logger.warning(
            "Job failed: fatal pipeline error",
            job_id=job.job_id,
            session_id=job.session_id,
            tool=e.tool_error.tool,
            reason=e.tool_error.reason,
        )

    except Exception as e:
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.error = str(e)
        _jobs_failed_today += 1

        logger.exception(
            "Job failed: unhandled pipeline error",
            job_id=job.job_id,
            session_id=job.session_id,
        )

    finally:
        try:
            stream_q.put_nowait(None)
        except asyncio.QueueFull:
            pass


# ── Cleanup ──────────────────────────────────────────────────────

def _purge_expired_jobs() -> None:
    """
    Purge completed/failed jobs after JOB_RETENTION_SECONDS.

    Removes associated stream queues and session states, and invokes
    every registered purge callback with the job's session_id.
    """
    now = datetime.now(timezone.utc)

    expired = [
        jid
        for jid, job in _job_store.items()
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
        and job.completed_at is not None
        and (now - job.completed_at).total_seconds() > JOB_RETENTION_SECONDS
    ]

    for jid in expired:
        job = _job_store.pop(jid)
        _job_streams.pop(jid, None)
        _session_states.pop(job.session_id, None)

        for callback in _purge_callbacks:
            try:
                callback(job.session_id)
            except Exception:
                logger.exception(
                    "Purge callback failed",
                    job_id=jid,
                    session_id=job.session_id,
                    callback=getattr(callback, "__name__", repr(callback)),
                )

        logger.info(
            "Purged expired job",
            job_id=jid,
            session_id=job.session_id,
        )