from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Protocol

from loguru import logger

from app.db.session import SessionLocal
from app.db.models import User, PipelineJobRow, HealthRecord
from schemas.errors import FatalPipelineError, ToolError
from schemas.queue import JobStatus, PipelineJob
from schemas.state import AegisState
from tools.tool_names import TOOL_QUEUE


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
_job_queue: list[str] = []  # priority-ordered, highest priority first
_job_streams: dict[str, asyncio.Queue[str | None]] = {}
_session_states: dict[str, AegisState] = {}

# asyncio.Lock() created at module import time.
# Safe in Python 3.10+ — no running event loop required for Lock creation.
_inference_lock: asyncio.Lock = asyncio.Lock()

_completed_durations: deque[float] = deque(maxlen=100)

_jobs_completed_today: int = 0
_jobs_failed_today: int = 0

_purge_callbacks: list[Callable[[str], None]] = []

# Per-user submission rate limit state
_user_submissions: dict[str, deque[float]] = {}

from app.settings import settings


# ── Purge callback registration ──────────────────────────────────

def register_purge_callback(callback: Callable[[str], None]) -> None:
    """
    Register a function to be called when a job is purged.

    The callback receives the purged job's session_id. Used by
    backend/main.py to clean up uploaded files when a job's
    retention window expires.

    Lifecycle invariant:
        MUST be called only during application startup (lifespan handler),
        before the inference worker task is created. The worker calls
        _purge_expired_jobs() on every iteration of its loop — registering
        a callback after the worker has started is not thread-safe and
        produces undefined behaviour.

        Concretely: register_purge_callback() is called from the FastAPI
        lifespan context manager. asyncio.create_task(run_inference_worker())
        is called immediately after. This ordering is enforced by lifespan
        and must not be changed.

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


def remove_completed_job(job_id: str) -> PipelineJob | None:
    """
    Remove a completed/failed job from retained in-memory state.

    Active jobs are never removed here. Report deletion calls this only after
    its user-scoped database transaction succeeds.
    """
    job = _job_store.get(job_id)
    if job is None or job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        return None

    _job_store.pop(job_id, None)
    _job_streams.pop(job_id, None)
    _session_states.pop(job.session_id, None)

    try:
        _job_queue.remove(job_id)
    except ValueError:
        pass

    for callback in _purge_callbacks:
        try:
            callback(job.session_id)
        except Exception:
            logger.exception(
                "Report deletion cleanup callback failed",
                job_id=job_id,
                session_id=job.session_id,
                callback=getattr(callback, "__name__", repr(callback)),
            )

    logger.info("Removed deleted report from retained job state", job_id=job_id)
    return job


def get_state_by_job_id(job_id: str) -> AegisState | None:
    """
    Return the AegisState associated with a job_id, or None if unknown/purged.

    Used by /queue/result/{job_id} to expose the final structured pipeline
    result to the frontend after streaming completes.
    """
    job = _job_store.get(job_id)
    if job is None:
        return None

    return _session_states.get(job.session_id)


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

    Computed dynamically — never stored. O(n) over the adaptive queue max.
    """
    try:
        return _job_queue.index(job_id) + 1
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
    avg    = sum(recent) / len(recent)
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
    """
    Maximum number of waiting jobs allowed.

    Adaptive: shrinks when recent pipeline runs are slow, expands when
    they are fast, bounded by AEGIS_QUEUE_MIN_SIZE and AEGIS_QUEUE_MAX_SIZE.
    """
    return adaptive_max_size()


def adaptive_max_size() -> int:
    """
    Compute an adaptive queue capacity from recent pipeline duration.

    Goal: keep the total queued wait near AEGIS_QUEUE_TARGET_WAIT_S.
    """
    if len(_completed_durations) < 3:
        return settings.AEGIS_QUEUE_MAX_SIZE

    avg = sum(_completed_durations) / len(_completed_durations)
    if avg <= 0:
        return settings.AEGIS_QUEUE_MAX_SIZE

    capacity = int(settings.AEGIS_QUEUE_TARGET_WAIT_S / avg)
    return max(
        settings.AEGIS_QUEUE_MIN_SIZE,
        min(settings.AEGIS_QUEUE_MAX_SIZE, capacity),
    )


def check_rate_limit(user_id: str | None) -> ToolError | None:
    """
    Enforce per-user submission rate limit.

    Returns None when allowed, or a fatal ToolError when the user has
    exceeded AEGIS_QUEUE_RATE_LIMIT_ATTEMPTS within
    AEGIS_QUEUE_RATE_LIMIT_WINDOW_S seconds.
    """
    if not user_id:
        return None

    now = time.monotonic()
    window = settings.AEGIS_QUEUE_RATE_LIMIT_WINDOW_S
    attempts = settings.AEGIS_QUEUE_RATE_LIMIT_ATTEMPTS
    submissions = _user_submissions.setdefault(user_id, deque())

    while submissions and now - submissions[0] > window:
        submissions.popleft()

    if len(submissions) >= attempts:
        retry_after = max(1, int(window - (now - submissions[0])))
        return ToolError(
            tool=TOOL_QUEUE,
            code="rate_limited",
            reason=f"Too many submissions. Please wait {retry_after}s before retrying.",
            fatal=True,
        )

    submissions.append(now)
    return None


def _priority_sort_key(job_id: str) -> tuple[int, datetime]:
    """
    Sort key for queued jobs: higher priority first, then older first.
    """
    job = _job_store.get(job_id)
    if job is None:
        return (-1, datetime.min.replace(tzinfo=timezone.utc))
    return (-job.priority, job.submitted_at)


def _insert_by_priority(job_id: str) -> None:
    """Insert a job id into _job_queue while preserving priority order."""
    _job_queue.append(job_id)
    _job_queue.sort(key=_priority_sort_key)


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

    Error codes:
        invalid_input     — malformed job state or session mismatch
        duplicate_session — job_id already exists or session already active
        queue_full        — queue at adaptive capacity
        rate_limited      — user exceeded submission rate limit
    """
    if job.status != JobStatus.QUEUED:
        return ToolError(
            tool=TOOL_QUEUE,
            code="invalid_input",
            reason=(
                f"Only queued jobs can be submitted. "
                f"Got status={job.status.value!r}."
            ),
            fatal=True,
        )

    if job.session_id != state.session_id:
        return ToolError(
            tool=TOOL_QUEUE,
            code="invalid_input",
            reason=(
                f"Job/session mismatch: job.session_id={job.session_id!r}, "
                f"state.session_id={state.session_id!r}."
            ),
            fatal=True,
        )

    if job.job_id in _job_store:
        return ToolError(
            tool=TOOL_QUEUE,
            code="duplicate_session",
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
            tool=TOOL_QUEUE,
            code="duplicate_session",
            reason=f"Session already has an active job: {job.session_id}",
            fatal=True,
        )

    max_size = adaptive_max_size()
    if len(_job_queue) >= max_size:
        return ToolError(
            tool=TOOL_QUEUE,
            code="queue_full",
            reason=f"Queue full ({max_size} jobs). Try again shortly.",
            fatal=True,
        )

    rate_limit_err = check_rate_limit(job.user_id)
    if rate_limit_err is not None:
        return rate_limit_err

    _job_store[job.job_id]         = job
    _session_states[job.session_id] = state
    _insert_by_priority(job.job_id)

    if job.user_id and job.user_id != "anonymous":
        try:
            with SessionLocal() as db:
                if db.query(User).filter_by(id=job.user_id).first():
                    row = PipelineJobRow(
                        job_id=job.job_id,
                        user_id=job.user_id,
                        session_id=job.session_id,
                        status=job.status.value,
                        priority=job.priority,
                        submitted_at=job.submitted_at,
                    )
                    db.add(row)
                    db.commit()
        except Exception as e:
            logger.warning("queue · failed to persist PipelineJobRow", error=str(e), job_id=job.job_id)

    logger.info(
        "Job queued",
        job_id=job.job_id,
        session_id=job.session_id,
        user_id=job.user_id,
        priority=job.priority,
        queue_depth=len(_job_queue),
    )

    return job


def _save_checkpoint(job_id: str, state: AegisState) -> None:
    try:
        from pathlib import Path
        path = Path("/tmp/aegis_checkpoint") / f"{job_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(state.model_dump_json())
    except Exception as e:
        logger.warning("queue · failed to write checkpoint", job_id=job_id, error=str(e))


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

        job_id = _job_queue.pop(0)
        job    = _job_store.get(job_id)
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

    job.status     = JobStatus.RUNNING
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
        _save_checkpoint(job.job_id, state)

        from backend.cache import result_cache, compute_cache_key
        cache_key = compute_cache_key(state)
        cached_result = result_cache.get(cache_key)
        if cached_result:
            logger.info("queue · result cache hit, bypassing inference", job_id=job.job_id)
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            _jobs_completed_today += 1

            cached_result["patient"] = {
                "name": getattr(state, "patient_name", None),
                "dob": getattr(state, "patient_dob", None),
                "sex": getattr(state, "patient_sex", None),
                "blood_group": getattr(state, "patient_blood_group", None),
                "weight_kg": getattr(state, "patient_weight_kg", None),
                "height_cm": getattr(state, "patient_height_cm", None),
                "allergies": getattr(state, "patient_allergies", None),
                "medical_conditions": list(getattr(state, "patient_medical_conditions", []) or []),
            }
            cached_result["job"] = job.model_dump(mode="json")
            if "report" in cached_result and isinstance(cached_result["report"], dict):
                cached_result["report"]["cached_result"] = True
                cached_result["report"]["confidence"] = 1.0

            from schemas.report import TriageReport
            state.report = TriageReport.model_validate(cached_result["report"]) if "report" in cached_result else None
            state.pipeline_complete = True
            setattr(state, "_cached_result_dict", cached_result)
            setattr(state, "clinical_picture", cached_result.get("clinical_picture", {}) or {})

            text = cached_result.get("report", {}).get("text") or "Cached triage report."
            try:
                await stream_q.put(text)
            except Exception:
                pass
            return

        async with asyncio.timeout(PIPELINE_TIMEOUT_S):
            async for token in pipeline.run(state):
                _save_checkpoint(job.job_id, state)
                try:
                    await asyncio.wait_for(
                        stream_q.put(token),
                        timeout=STREAM_PUT_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    job.status       = JobStatus.FAILED
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

        job.status       = JobStatus.COMPLETED
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
        job.status       = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.error        = f"Pipeline exceeded {PIPELINE_TIMEOUT_S}s wall-clock limit."
        _jobs_failed_today += 1

        logger.warning(
            "Job failed: pipeline timeout",
            job_id=job.job_id,
            session_id=job.session_id,
            timeout_s=PIPELINE_TIMEOUT_S,
        )

    except FatalPipelineError as e:
        job.status       = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.error        = e.tool_error.reason
        _jobs_failed_today += 1

        logger.warning(
            "Job failed: fatal pipeline error",
            job_id=job.job_id,
            session_id=job.session_id,
            tool=e.tool_error.tool,
            reason=e.tool_error.reason,
        )

    except Exception as e:
        job.status       = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.error        = str(e)
        _jobs_failed_today += 1

        logger.exception(
            "Job failed: unhandled pipeline error",
            job_id=job.job_id,
            session_id=job.session_id,
        )

    finally:
        if job.user_id and job.user_id != "anonymous":
            try:
                with SessionLocal() as db:
                    row = db.query(PipelineJobRow).filter_by(job_id=job.job_id).first()
                    if row:
                        row.status = job.status.value
                        row.started_at = job.started_at
                        row.completed_at = job.completed_at
                        row.error = job.error
                        db.commit()
                    
                    if job.status == JobStatus.COMPLETED and state and getattr(state, "report", None):
                        user_row = db.query(User).filter_by(id=job.user_id).first()
                        if user_row and not db.query(HealthRecord).filter_by(job_id=job.job_id).first():
                            report_dict = state.report.model_dump(mode="json")
                            _cached = getattr(state, "_cached_result_dict", None)
                            if _cached:
                                # Cache-hit path: reuse the rehydrated cached result
                                # (already carries patient/job/report from the hit
                                # branch above) instead of rebuilding from state,
                                # since the pipeline never ran this time and most
                                # state.*_result fields were never populated.
                                result_dict = _cached
                                result_dict["report"] = report_dict
                            else:
                                result_dict = {
                                    "job": job.model_dump(mode="json"),
                                    "patient": {
                                        "name": getattr(state, "patient_name", None),
                                        "dob": getattr(state, "patient_dob", None),
                                        "sex": getattr(state, "patient_sex", None),
                                        "blood_group": getattr(state, "patient_blood_group", None),
                                        "weight_kg": getattr(state, "patient_weight_kg", None),
                                        "height_cm": getattr(state, "patient_height_cm", None),
                                        "allergies": getattr(state, "patient_allergies", None),
                                        "medical_conditions": list(getattr(state, "patient_medical_conditions", []) or []),
                                    },
                                    "submitted": {
                                        "symptoms_text": getattr(state, "submitted_symptoms_text", None) or state.raw_symptoms_text,
                                        "medications": list(state.medications_raw),
                                        "xray_findings": list(state.xray_findings_raw),
                                        "xray_free_text": state.xray_free_text_raw,
                                        "lab_pdf_uploaded": bool(getattr(state, "lab_pdf_path", None)),
                                        "xray_image_uploaded": bool(getattr(state, "xray_image_path", None)),
                                        "audio_uploaded": bool(getattr(state, "audio_file_path", None)),
                                    },
                                    "report": report_dict,
                                    "execution_plan": state.execution_plan.model_dump(mode="json") if getattr(state, "execution_plan", None) else None,
                                    "rule_validator_result": state.rule_validator_result.model_dump(mode="json") if getattr(state, "rule_validator_result", None) else None,
                                    "severity_result": state.severity_result.model_dump(mode="json") if getattr(state, "severity_result", None) else None,
                                    "voice_result": state.voice_result.model_dump(mode="json") if getattr(state, "voice_result", None) else None,
                                    "symptom_result": state.symptom_result.model_dump(mode="json") if getattr(state, "symptom_result", None) else None,
                                    "lab_result": state.lab_result.model_dump(mode="json") if getattr(state, "lab_result", None) else None,
                                    "xray_result": state.xray_result.model_dump(mode="json") if getattr(state, "xray_result", None) else None,
                                    "drug_result": state.drug_result.model_dump(mode="json") if getattr(state, "drug_result", None) else None,
                                    "rag_result": state.rag_result.model_dump(mode="json") if getattr(state, "rag_result", None) else None,
                                    "clinical_picture": getattr(state, "clinical_picture", {}) or {},
                                    "pipeline": {
                                        "current_tool": state.current_tool,
                                        "tools_run": list(state.tools_run),
                                        "tools_failed": list(state.tools_failed),
                                        "step_durations_ms": dict(state.step_durations_ms),
                                        "pipeline_complete": state.pipeline_complete,
                                        "pipeline_start_ms": state.pipeline_start_ms,
                                        "pipeline_end_ms": state.pipeline_end_ms,
                                    },
                                }
                            sev = getattr(state.report, "severity", None) or (getattr(state.severity_result, "severity", None) if getattr(state, "severity_result", None) else "LOW")
                            conf = getattr(state.report, "confidence", None) or 0.0
                            val_stat = getattr(state.report, "validation_status", None)
                            if not val_stat and getattr(state, "rule_validator_result", None):
                                st = getattr(state.rule_validator_result, "status", None)
                                val_stat = getattr(st, "value", str(st)) if st else None

                            from backend.cache import result_cache, compute_cache_key
                            record_cache_key = compute_cache_key(state)

                            record = HealthRecord(
                                user_id=job.user_id,
                                job_id=job.job_id,
                                severity=sev,
                                confidence=conf,
                                validation_status=val_stat,
                                symptoms_text=getattr(state, "submitted_symptoms_text", None) or state.raw_symptoms_text,
                                medications_json=json.dumps(list(state.medications_raw)),
                                xray_findings_json=json.dumps(list(state.xray_findings_raw)),
                                report_json=json.dumps(report_dict),
                                result_json=json.dumps(result_dict),
                                cache_key=record_cache_key,
                            )
                            db.add(record)
                            db.commit()
                            result_cache.set(record_cache_key, result_dict)
                            logger.info("queue · persisted HealthRecord & set cache", job_id=job.job_id, user_id=job.user_id)
            except Exception as e:
                logger.warning("queue · failed to persist job completion / HealthRecord", error=str(e), job_id=job.job_id)

        _save_checkpoint(job.job_id, state)
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

    Exceptions raised inside callbacks are logged but do not stop
    other callbacks from running.
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