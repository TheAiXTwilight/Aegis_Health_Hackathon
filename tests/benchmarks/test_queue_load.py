"""
tests/benchmarks/test_queue_load.py — Queue load and throughput.

Verifies MAX_QUEUE_SIZE enforcement under concurrent submission and
that the worker can drain a full queue in reasonable wall-clock time.

Marked benchmark — excluded from the default run when performance
infrastructure is not available:

    pytest -m "not benchmark"
"""

from __future__ import annotations

import asyncio
import time

import pytest

import backend.queue as bq
from backend.queue import (
    get_queue_depth,
    get_queue_max,
    run_inference_worker,
    submit_job,
)
from schemas.errors import ToolError
from schemas.queue import JobStatus, PipelineJob
from schemas.state import AegisState


@pytest.fixture(autouse=True)
def _reset_queue():
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


class _FastPipeline:
    """Pipeline that completes in one iteration — minimal overhead."""
    async def run(self, state: AegisState):
        yield "ok"


async def _run_worker_briefly(pipeline, seconds: float):
    task = asyncio.create_task(run_inference_worker(pipeline))
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── MAX_QUEUE_SIZE enforcement ─────────────────────────────────────

@pytest.mark.benchmark
async def test_queue_rejects_at_max_size():
    """Submitting capacity + 1 jobs — last one must be rejected."""
    capacity = get_queue_max()
    for _ in range(capacity):
        s = AegisState()
        j = PipelineJob(session_id=s.session_id)
        result = await submit_job(j, s)
        assert isinstance(result, PipelineJob)

    assert get_queue_depth() == capacity

    s_overflow = AegisState()
    j_overflow = PipelineJob(session_id=s_overflow.session_id)
    result = await submit_job(j_overflow, s_overflow)
    assert isinstance(result, ToolError)
    assert "queue full" in result.reason.lower()


@pytest.mark.benchmark
async def test_queue_accepts_exactly_max_size():
    """Exactly capacity jobs must all be accepted."""
    capacity = get_queue_max()
    results = []
    for _ in range(capacity):
        s = AegisState()
        j = PipelineJob(session_id=s.session_id)
        result = await submit_job(j, s)
        results.append(result)

    assert all(isinstance(r, PipelineJob) for r in results)
    assert get_queue_depth() == capacity


# ── Worker drains queue ────────────────────────────────────────────

@pytest.mark.benchmark
async def test_worker_drains_full_queue():
    """
    Worker must process all capacity jobs to COMPLETED.
    Wall-clock budget: capacity * 0.5s per job.
    """
    capacity = get_queue_max()
    jobs = []
    for _ in range(capacity):
        s = AegisState()
        j = PipelineJob(session_id=s.session_id)
        await submit_job(j, s)
        jobs.append(j)

    assert get_queue_depth() == capacity

    task = asyncio.create_task(run_inference_worker(_FastPipeline()))
    start = time.perf_counter()
    budget_s = capacity * 0.5

    while time.perf_counter() - start < budget_s:
        all_done = all(
            j.status in (JobStatus.COMPLETED, JobStatus.FAILED)
            for j in jobs
        )
        if all_done:
            break
        await asyncio.sleep(0.05)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    completed = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)
    assert completed == capacity, (
        f"Only {completed}/{capacity} jobs completed within budget"
    )


@pytest.mark.benchmark
async def test_worker_processes_jobs_fifo_by_started_at():
    """Jobs with equal priority are processed in submission order (verified via started_at)."""
    jobs = []
    for _ in range(3):
        s = AegisState()
        j = PipelineJob(session_id=s.session_id)
        await submit_job(j, s)
        jobs.append(j)

    await _run_worker_briefly(_FastPipeline(), seconds=1.0)

    completed_jobs = [j for j in jobs if j.started_at is not None]
    if len(completed_jobs) >= 2:
        started_ats = [j.started_at for j in completed_jobs]
        assert started_ats == sorted(started_ats), (
            "Jobs with equal priority were not processed in FIFO order by started_at"
        )


@pytest.mark.benchmark
async def test_completed_durations_recorded_after_drain():
    """After draining, _completed_durations must have entries."""
    for _ in range(3):
        s = AegisState()
        j = PipelineJob(session_id=s.session_id)
        await submit_job(j, s)

    await _run_worker_briefly(_FastPipeline(), seconds=0.5)

    assert len(bq._completed_durations) > 0
    assert all(isinstance(d, float) for d in bq._completed_durations)