"""
backend/main.py — FastAPI application entrypoint.

Wires together:
    - AegisPipeline (real pipeline, dependency-injected into worker)
    - Inference worker (background task started in lifespan)
    - Upload validation (backend.uploads)
    - Queue submission (backend.queue)
    - Submit / status endpoints
    - Health and streaming routers (mounted from sibling modules)

Upload storage:
    Files are persisted to /tmp/aegis_uploads/{session_id}/ and removed when
    the job is purged via register_purge_callback().

ToolError code vocabulary (Phase 4):
    invalid_input     — malformed or type-invalid caller input
    missing_input     — at least one required modality absent
    internal_error    — server-side failure (filesystem, etc.)
    duplicate_session — active job already exists for this session
    queue_full        — FIFO queue at MAX_QUEUE_SIZE capacity

Upload size enforcement (M3):
    _save_upload() enforces the per-kind byte limit while streaming the upload
    to disk. UploadTooLargeError is raised immediately when the limit is exceeded —
    no further bytes are read or written, and the partial file is removed before
    the exception propagates. This prevents a client from exhausting disk or memory
    by uploading a file that only gets rejected after a full write.

    UploadTooLargeError is a private exception scoped to this module. It is
    self-documenting and cannot be accidentally conflated with unrelated ValueError
    instances from other code paths.

    UPLOAD_CHUNK_SIZE controls the streaming granularity. 1 MB chunks balance
    memory pressure against syscall overhead. Tune if profiling reveals a bottleneck
    on Jetson.
"""
from __future__ import annotations

import os

# Must run before chromadb, onnxruntime, or faiss are imported anywhere
# in the app (directly or transitively — chromadb itself depends on
# onnxruntime). On macOS in particular, these libraries can each
# statically link their own copy of the OpenMP runtime; loading more
# than one copy into the same process aborts with "OMP: Error #15:
# Initializing libomp.dylib, but found libomp.dylib already
# initialized." This is a well-known conflict with a standard
# (if officially "unsupported") workaround — see
# http://openmp.llvm.org/ for the upstream explanation. Setting it
# here, before any other import, ensures it takes effect regardless
# of which entrypoint (uvicorn CLI, gunicorn, tests, a script) starts
# the process, rather than relying on it being set in the shell.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import asyncio
import contextlib
import json
import shutil
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from agents.pipeline import AegisPipeline
from backend.chat import router as chat_router
from backend.dashboard import build_report_measurement_groups, router as dashboard_router
from backend.health import router as health_router
from backend.model_registry import admin_router
from backend.pdf_export import router as pdf_router
from backend.records import register_report_state_remover, router as records_router
from backend.queue import (
    get_job,
    remove_completed_job,
    get_state_by_job_id,
    get_status_payload,
    register_purge_callback,
    run_inference_worker,
    submit_job,
)
from backend.streaming import router as streaming_router
from backend.uploads import (
    MAX_AUDIO_BYTES,
    MAX_PDF_BYTES,
    MAX_XRAY_BYTES,
    validate_audio,
    validate_lab_pdf,
    validate_medications,
    validate_xray_image,
)
from backend.vitals import router as vitals_router

from schemas.errors import ToolError
from schemas.queue import PipelineJob
from schemas.state import AegisState
from tools.tool_names import TOOL_INPUT_VALIDATION, TOOL_QUEUE

from app.settings import settings
from app.db.session import init_db, SessionLocal
from app.db.models import User, PipelineJobRow, HealthRecord
from app.db.seed import seed_demo_users
from app.auth import get_optional_user, get_current_user
from backend.exports import router as export_router
from backend.account import router as account_router
from backend.auth import router as auth_router
from backend.tts import router as tts_router

# Piper TTS synthesizer — imported so the lifespan handler can run the
# idle-eviction monitor task. Warmup itself is triggered on demand from
# backend/queue.py (when a report job starts running) and from
# backend/tts.py (when the frontend polls /tts/status/{job_id}), so we
# do NOT preload the voice at server startup — that would waste ~180MB
# on servers where nobody clicks Voice TTS.
from tools import tts_synthesizer
from backend.ollama_manager import start_ollama, stop_ollama

UPLOAD_ROOT = Path("/tmp/aegis_uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
# backend/main.py -> backend/ -> repo root -> frontend/dist
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
register_report_state_remover(remove_completed_job)

# Streaming chunk size for upload writes.
# 1 MB balances memory pressure against syscall overhead.
# Adjust if Jetson profiling reveals a bottleneck.
UPLOAD_CHUNK_SIZE = 1024 * 1024 # 1 MB

# ── HTTP status code mapping (M2) ────────────────────────────────
# Module-level constant — built once at import time, not per call.
# Maps queue-layer ToolError codes to HTTP status codes.
# Unknown codes default to 500 — an unrecognised code indicates a
# programming error, not a client error.
QUEUE_ERROR_HTTP_STATUS: dict[str | None, int] = {
    "queue_full": 503,
    "duplicate_session": 409,
    "invalid_input": 400,
    "rate_limited": 429,
}


# ── Upload size enforcement (M3) ──────────────────────────────────
class UploadTooLargeError(Exception):
    """
    Raised by _save_upload() when a streaming upload exceeds the configured byte limit.
    Private to this module. Self-documenting and impossible to accidentally conflate
    with unrelated ValueError instances from other code paths.

    Attributes:
        kind    — upload kind label ("audio", "lab", "xray")
        written — bytes written before the limit was hit
        limit   — configured limit in bytes
    """
    def __init__(self, kind: str, written: int, limit: int) -> None:
        self.kind = kind
        self.written = written
        self.limit = limit
        super().__init__(
            f"{kind} upload exceeded size limit: "
            f"{written} bytes written (max {limit} bytes)"
        )


# ── ToolError helpers ─────────────────────────────────────────────
def _input_error(code: str, reason: str) -> ToolError:
    """
    Build a fatal ToolError attributed to TOOL_INPUT_VALIDATION.
    Centralises the repeated pattern:
        ToolError(tool=TOOL_INPUT_VALIDATION, code=..., reason=..., fatal=True)
    Used for all pre-queue validation failures in submit(). Does not cover
    queue-layer errors (those use TOOL_QUEUE via submit_job).
    """
    return ToolError(
        tool=TOOL_INPUT_VALIDATION,
        code=code,
        reason=reason,
        fatal=True,
    )


# ── Upload helpers ───────────────────────────────────────────────
async def _save_upload(
    file: UploadFile, session_dir: Path, kind: str, max_bytes: int,
) -> Path:
    """
    Save an UploadFile to disk under session_dir. Returns the saved path.

    Enforces max_bytes while streaming — raises UploadTooLargeError and removes
    the partial file immediately if the limit is exceeded. Uses UPLOAD_CHUNK_SIZE
    async reads to avoid blocking the event loop.

    OSError during unlink of the partial file is suppressed — the UploadTooLargeError
    is always re-raised regardless of cleanup success.

    Raises:
        UploadTooLargeError — upload exceeded max_bytes during streaming
        OSError             — filesystem write failure
    """
    suffix = Path(file.filename or kind).suffix or ""
    target = session_dir / f"{kind}{suffix}"
    written = 0
    try:
        async with aiofiles.open(target, "wb") as out:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise UploadTooLargeError(kind, written, max_bytes)
                await out.write(chunk)
    except UploadTooLargeError:
        with contextlib.suppress(OSError):
            target.unlink()
        raise
    return target


def cleanup_session_uploads(session_id: str) -> None:
    """
    Remove all uploaded files for a session. Called by the queue's purge
    mechanism when a job's retention window expires.
    """
    session_dir = UPLOAD_ROOT / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
    logger.info("Cleaned up uploads", session_id=session_id)


def _cleanup_on_rejection(session_dir: Path) -> None:
    """Remove a session directory when submission is rejected."""
    shutil.rmtree(session_dir, ignore_errors=True)


def _tool_error_response(err: ToolError, status_code: int) -> JSONResponse:
    """Convert a ToolError into a JSON response with the chosen status code."""
    return JSONResponse(
        status_code=status_code,
        content=err.model_dump(mode="json"),
    )


# ── TTS idle-eviction monitor ─────────────────────────────────────
async def _tts_idle_monitor() -> None:
    """
    Background task that unloads the Piper voice model after
    AEGIS_TTS_IDLE_EVICT_SECS of no TTS activity, reclaiming ~180MB.

    Runs for the application lifetime, started/stopped by the lifespan
    handler. Checks every AEGIS_TTS_IDLE_CHECK_SECS. The eviction
    itself is cheap (dropping a Python reference so the GC can reclaim
    the ONNX session); reload on next TTS request is transparent —
    the request just pays the ~500ms cold-load cost once.

    A setting of AEGIS_TTS_IDLE_EVICT_SECS <= 0 disables eviction
    entirely (voice stays loaded until process exit once first used).
    """
    evict_after = settings.AEGIS_TTS_IDLE_EVICT_SECS
    check_interval = settings.AEGIS_TTS_IDLE_CHECK_SECS

    if evict_after <= 0:
        logger.info("TTS idle-eviction disabled (AEGIS_TTS_IDLE_EVICT_SECS <= 0)")
        return

    logger.info(
        "TTS idle-eviction monitor started",
        evict_after_secs=evict_after,
        check_interval_secs=check_interval,
    )

    while True:
        try:
            await asyncio.sleep(check_interval)
            if not tts_synthesizer.is_loaded():
                continue
            idle_secs = tts_synthesizer.seconds_since_last_activity()
            if idle_secs > evict_after:
                # Run eviction in a thread so the lock acquisition (which
                # could briefly wait behind an in-flight synthesis) never
                # blocks the event loop.
                await asyncio.to_thread(tts_synthesizer.evict_voice)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Never let a monitor error kill the task. Log and continue.
            logger.exception("TTS idle monitor iteration failed (continuing)")


# ── Lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start the inference worker and register cleanup callbacks on startup.
    Cancel the worker cleanly on shutdown.

    register_purge_callback is idempotent — re-entering lifespan in tests
    does not accumulate duplicate registrations.

    Also starts the TTS idle-eviction monitor (see _tts_idle_monitor).
    Note we do NOT preload the Piper voice at boot — warmup happens
    on-demand when a report job starts running (backend/queue.py) or
    when the frontend polls /tts/status (backend/tts.py). This keeps
    ~180MB free on servers where nobody clicks Voice TTS.
    """
    logger.info("Starting Aegis Health", db_url=settings.AEGIS_DB_URL, seed_demo=settings.AEGIS_SEED_DEMO_USERS)

    # Ensure Ollama is running before anything that might call it.
    # No more separate `ollama serve` terminal required.
    await start_ollama()

    init_db()
    if settings.AEGIS_SEED_DEMO_USERS:
        from app.db.session import SessionLocal
        db = SessionLocal()
        try:
            users = seed_demo_users(db)
            logger.info("Demo users ready", count=len(users), emails=[u.email for u in users])
        finally:
            db.close()
    pipeline = AegisPipeline()
    register_purge_callback(cleanup_session_uploads)
    worker_task = asyncio.create_task(run_inference_worker(pipeline))
    logger.info("Inference worker started")

    # Start the TTS idle-eviction monitor. Runs for the app lifetime,
    # cancelled cleanly in the finally block below.
    tts_monitor_task = asyncio.create_task(_tts_idle_monitor())

    try:
        yield
    finally:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
        logger.info("Inference worker stopped")

        tts_monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tts_monitor_task
        logger.info("TTS idle-eviction monitor stopped")

        # Only stops Ollama if this process is the one that started it.
        await stop_ollama()


# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Aegis Health",
    version="1.0",
    lifespan=lifespan,
)


# ── HTTP status code mapping ──────────────────────────────────────
def _status_code_for_queue_error(err: ToolError) -> int:
    """
    Map queue-layer ToolError codes to HTTP status codes.
    Uses the module-level QUEUE_ERROR_HTTP_STATUS constant.
    Unknown codes default to 500 — an unrecognised code indicates a
    programming error, not a client error.
    """
    code = err.code
    if code not in QUEUE_ERROR_HTTP_STATUS:
        logger.error(
            "Unrecognised queue ToolError code — defaulting to 500",
            code=code,
            reason=err.reason,
            tool=err.tool,
        )
        return 500
    return QUEUE_ERROR_HTTP_STATUS[code]


def _dump_pipeline_value(value: Any) -> Any:
    """
    Convert pipeline result objects into JSON-safe data.
    Handles:
        None      -> None
        ToolError -> {"error": ...}
        Pydantic  -> model_dump(mode="json")
        primitive -> unchanged
    """
    if value is None:
        return None
    if isinstance(value, ToolError):
        return {"error": value.model_dump(mode="json")}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _derive_priority(symptoms_text: str | None, priority: int | None) -> int:
    """
    Determine job priority.

    If the caller explicitly supplies a priority in [1, 5], use it.
    Otherwise infer from symptom keywords: critical phrases get the
    highest priority so urgent cases are processed first.
    """
    if priority is not None and 1 <= priority <= 5:
        return priority

    text = (symptoms_text or "").lower()
    critical = [
        "critical", "severe", "heart attack", "chest pain",
        "can't breathe", "cannot breathe", "unconscious", "stroke",
        "anaphylaxis", "allergic reaction", "suicide",
    ]
    high = [
        "high", "intense", "bleeding", "fall", "fracture",
        "poison", "burn", "seizure", "fainting",
    ]
    moderate = [
        "moderate", "fever", "cough", "pain", "vomiting",
        "diarrhea", "dizziness", "nausea",
    ]

    if any(word in text for word in critical):
        return 5
    if any(word in text for word in high):
        return 4
    if any(word in text for word in moderate):
        return 3
    return 1


# ── Endpoints: queue submit / status ─────────────────────────────
@app.post("/queue/submit")
async def submit(
    request: Request,
    user: User | None = Depends(get_optional_user),
    patient_name: str | None = Form(None),
    patient_dob: str | None = Form(None),
    patient_sex: str | None = Form(None),
    patient_blood_group: str | None = Form(None),
    patient_weight_kg: float | None = Form(None),
    patient_height_cm: float | None = Form(None),
    patient_allergies: str | None = Form(None),
    patient_medical_conditions: str | None = Form(None),
    symptoms_text: str | None = Form(None),
    medications: str | None = Form(None),
    xray_findings: str | None = Form(None),
    xray_free_text: str | None = Form(None),
    priority: int | None = Form(None),
    lab_pdf: list[UploadFile] = File(default=[]),
    xray_image: list[UploadFile] = File(default=[]),
    audio: UploadFile | None = File(None),
) -> JSONResponse:
    """
    Validate uploads → build AegisState → enqueue a PipelineJob.

    Form contract:
        symptoms_text — optional plain text symptoms
        medications   — JSON-encoded list[str], default "[]"
        xray_findings — JSON-encoded list[str], default "[]"
        xray_free_text— optional clinician free-text X-ray note
        priority      — optional explicit 1-5 priority; inferred from symptoms if omitted
        audio         — optional audio file (WAV preferred)
        lab_pdf       — optional list of lab report PDFs
        xray_image    — optional list of X-ray images

    Status codes:
        200 — accepted, returns PipelineJob JSON
        400 — invalid input (size, duration, count, malformed JSON)
        409 — duplicate job or active session
        429 — user/IP submission rate limit exceeded
        500 — server-side failure (filesystem error saving upload)
        503 — queue full
    """
    # ── Parse JSON form fields ────────────────────────────────────
    try:
        medical_conditions_list = json.loads(patient_medical_conditions) if patient_medical_conditions else []
        if not isinstance(medical_conditions_list, list) or not all(
            isinstance(condition, str) for condition in medical_conditions_list
        ):
            raise ValueError("patient_medical_conditions must be a JSON list of strings")
    except (json.JSONDecodeError, ValueError) as e:
        return _tool_error_response(
            _input_error("invalid_input", f"Invalid medical conditions field: {e}"),
            status_code=400,
        )

    try:
        medications_list = json.loads(medications) if medications else []
        if not isinstance(medications_list, list) or not all(
            isinstance(m, str) for m in medications_list
        ):
            raise ValueError("medications must be a JSON list of strings")
    except (json.JSONDecodeError, ValueError) as e:
        return _tool_error_response(
            _input_error("invalid_input", f"Invalid medications field: {e}"),
            status_code=400,
        )

    try:
        xray_findings_list = json.loads(xray_findings) if xray_findings else []
        if not isinstance(xray_findings_list, list) or not all(
            isinstance(f, str) for f in xray_findings_list
        ):
            raise ValueError("xray_findings must be a JSON list of strings")
    except (json.JSONDecodeError, ValueError) as e:
        return _tool_error_response(
            _input_error("invalid_input", f"Invalid xray_findings field: {e}"),
            status_code=400,
        )

    if patient_weight_kg is not None and not 10 <= patient_weight_kg <= 500:
        return _tool_error_response(
            _input_error("invalid_input", "Weight must be between 10 and 500 kg."),
            status_code=400,
        )
    if patient_height_cm is not None and not 50 <= patient_height_cm <= 250:
        return _tool_error_response(
            _input_error("invalid_input", "Height must be between 50 and 250 cm."),
            status_code=400,
        )

    if not any([
        symptoms_text,
        audio,
        lab_pdf,
        xray_image,
        medications_list,
        xray_findings_list,
        xray_free_text,
    ]):
        return _tool_error_response(
            _input_error("missing_input", "At least one input must be provided."),
            status_code=400,
        )

    user_id = user.id if user else (request.client.host if request.client else "anonymous")
    derived_priority = _derive_priority(symptoms_text, priority)

    state = AegisState(
        user_id=user_id,
        priority=derived_priority,
        patient_name=patient_name,
        patient_dob=patient_dob,
        patient_sex=patient_sex,
        patient_blood_group=patient_blood_group,
        patient_weight_kg=patient_weight_kg,
        patient_height_cm=patient_height_cm,
        patient_allergies=patient_allergies,
        patient_medical_conditions=medical_conditions_list,
        raw_symptoms_text=symptoms_text,
        submitted_symptoms_text=symptoms_text,
        medications_raw=medications_list,
        xray_findings_raw=xray_findings_list,
        xray_free_text_raw=xray_free_text,
    )

    session_dir = UPLOAD_ROOT / state.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # ── Save uploads (size enforced during streaming) ─────────────
    try:
        if audio is not None:
            state.audio_file_path = str(
                await _save_upload(audio, session_dir, "audio", MAX_AUDIO_BYTES)
            )
        
        # Save multiple Lab Report PDFs
        lab_paths: list[str] = []
        for i, file_obj in enumerate(lab_pdf, start=1):
            if file_obj.filename:
                saved = await _save_upload(file_obj, session_dir, f"lab_{i}", MAX_PDF_BYTES)
                lab_paths.append(str(saved))
        if lab_paths:
            state.lab_pdf_path = lab_paths

        # Save multiple X-Ray Images
        xray_paths: list[str] = []
        for i, file_obj in enumerate(xray_image, start=1):
            if file_obj.filename:
                saved = await _save_upload(file_obj, session_dir, f"xray_{i}", MAX_XRAY_BYTES)
                xray_paths.append(str(saved))
        if xray_paths:
            state.xray_image_path = xray_paths

    except UploadTooLargeError as e:
        _cleanup_on_rejection(session_dir)
        return _tool_error_response(
            _input_error("invalid_input", str(e)),
            status_code=400,
        )
    except OSError as e:
        _cleanup_on_rejection(session_dir)
        return _tool_error_response(
            _input_error("internal_error", f"Failed to save uploaded file: {e}"),
            status_code=500,
        )

    # ── Post-write validators ─────────────────────────────────────
    # Size is enforced during streaming above. These validators handle
    # constraints that cannot be checked during the write:
    # validate_medications — entry count on deserialized JSON list
    # validate_audio       — WAV duration and format (requires header parse)
    # validate_lab_pdf     — readability guard (existence confirmed by save)
    # validate_xray_image  — readability guard (existence confirmed by save)
    if state.medications_raw:
        err = validate_medications(state.medications_raw)
        if err is not None:
            _cleanup_on_rejection(session_dir)
            return _tool_error_response(err, status_code=400)

    file_validators: list[tuple[Callable[[Any], ToolError | None], Any]] = [
        (validate_audio, getattr(state, "audio_file_path", None)),
        (validate_lab_pdf, getattr(state, "lab_pdf_path", None)),
        (validate_xray_image, getattr(state, "xray_image_path", None)),
    ]

    for validator, path in file_validators:
        if not path:
            continue
        err = validator(path)
        if err is not None:
            _cleanup_on_rejection(session_dir)
            return _tool_error_response(err, status_code=400)

    job = PipelineJob(
        session_id=state.session_id,
        user_id=state.user_id,
        priority=state.priority,
    )
    result = await submit_job(job, state)

    if isinstance(result, ToolError):
        _cleanup_on_rejection(session_dir)
        status_code = _status_code_for_queue_error(result)
        return _tool_error_response(result, status_code=status_code)

    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )


@app.post("/queue/recover/{job_id}")
async def recover_job(job_id: str, user: User = Depends(get_current_user)) -> dict[str, Any]:
    """Recover a job from checkpoint."""
    path = Path("/tmp/aegis_checkpoint") / f"{job_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Checkpoint not found for this job_id")
    try:
        data = json.loads(path.read_text())
        state = AegisState.model_validate(data)
        if state.user_id != user.id and user.role != "admin":
            raise HTTPException(status_code=403, detail="Not authorized to recover this job")

        job = PipelineJob(session_id=state.session_id, user_id=state.user_id, priority=state.priority)
        job.job_id = job_id
        from backend.queue import _job_store, _session_states, _insert_by_priority
        _job_store[job_id] = job
        _session_states[state.session_id] = state
        _insert_by_priority(job_id)
        return {"job_id": job_id, "recovered": True, "status": "queued"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to recover checkpoint: {e}")


@app.get("/queue/status/{job_id}")
async def status(job_id: str) -> dict[str, Any]:
    """Return live status payload for a job. 404 if unknown or purged."""
    payload = get_status_payload(job_id)
    if payload is None:
        with SessionLocal() as db:
            row = db.query(PipelineJobRow).filter_by(job_id=job_id).first()
            if row:
                return {
                    "job_id": row.job_id,
                    "session_id": row.session_id,
                    "user_id": row.user_id,
                    "status": row.status,
                    "priority": row.priority,
                    "submitted_at": row.submitted_at.isoformat() if row.submitted_at else None,
                    "started_at": row.started_at.isoformat() if row.started_at else None,
                    "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                    "error": row.error,
                    "queue_position": None,
                    "estimated_wait_seconds": None,
                    "current_tool": None,
                    "tools_run": [],
                    "tools_failed": [],
                    "step_durations_ms": {},
                }
        raise HTTPException(
            status_code=404,
            detail=f"Unknown job_id: {job_id}",
        )
    return payload


@app.get("/queue/result/{job_id}")
async def result(job_id: str) -> dict[str, Any]:
    """
    Return final structured pipeline result for a job.
    This is used by the frontend after streaming completes to build:
        - detailed PDF report
        - patient metadata section
        - severity/confidence cards
        - key insights dashboard
        - validation warning/override banner
    """
    job = get_job(job_id)
    state = get_state_by_job_id(job_id)
    if job is None or state is None:
        with SessionLocal() as db:
            rec = db.query(HealthRecord).filter_by(job_id=job_id).first()
            if rec and rec.result_json:
                try:
                    persisted_result = json.loads(rec.result_json)
                    persisted_result["measurement_groups"] = build_report_measurement_groups(
                        persisted_result
                    )
                    return persisted_result
                except Exception:
                    pass
        raise HTTPException(
            status_code=404,
            detail=f"Unknown job_id or pipeline state: {job_id}",
        )
    result_payload = {
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
            "symptoms_text": (
                getattr(state, "submitted_symptoms_text", None) or state.raw_symptoms_text
            ),
            "medications": list(state.medications_raw),
            "xray_findings": list(state.xray_findings_raw),
            "xray_free_text": state.xray_free_text_raw,
            "lab_pdf_uploaded": bool(getattr(state, "lab_pdf_path", None)),
            "xray_image_uploaded": bool(getattr(state, "xray_image_path", None)),
            "audio_uploaded": bool(getattr(state, "audio_file_path", None)),
        },
        "report": _dump_pipeline_value(state.report),
        "execution_plan": _dump_pipeline_value(state.execution_plan),
        "rule_validator_result": _dump_pipeline_value(state.rule_validator_result),
        "severity_result": _dump_pipeline_value(state.severity_result),
        "voice_result": _dump_pipeline_value(state.voice_result),
        "symptom_result": _dump_pipeline_value(state.symptom_result),
        "lab_result": _dump_pipeline_value(state.lab_result),
        "xray_result": _dump_pipeline_value(state.xray_result),
        "drug_result": _dump_pipeline_value(state.drug_result),
        "rag_result": _dump_pipeline_value(state.rag_result),
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
    result_payload["measurement_groups"] = build_report_measurement_groups(result_payload)
    return result_payload


@app.get("/queue/heatmap/{session_id}/{filename}")
async def get_heatmap_file(session_id: str, filename: str) -> FileResponse:
    """Serve generated Grad-CAM heatmap PNG artifact by filename."""
    file_path = UPLOAD_ROOT / session_id / filename
    if not file_path.is_file() or file_path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="Heatmap artifact not found.")
    return FileResponse(path=file_path, media_type="image/png")


@app.get("/queue/heatmap/{job_id}")
async def get_heatmap_by_job(job_id: str) -> FileResponse:
    """Serve generated Grad-CAM heatmap PNG artifact by job_id."""
    state = get_state_by_job_id(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No pipeline state found for job_id: {job_id}")
    
    heatmap_path = None
    if state.xray_result and hasattr(state.xray_result, "heatmap_path") and state.xray_result.heatmap_path:
        heatmap_path = Path(state.xray_result.heatmap_path)
    else:
        session_dir = UPLOAD_ROOT / state.session_id
        if session_dir.is_dir():
            for p in session_dir.glob("*_heatmap.png"):
                heatmap_path = p
                break

    if not heatmap_path or not heatmap_path.is_file():
        raise HTTPException(status_code=404, detail="Heatmap artifact not found for this job.")
    
    return FileResponse(path=heatmap_path, media_type="image/png")


# ── Router mounts ────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(health_router)
app.include_router(streaming_router)
app.include_router(pdf_router)
app.include_router(export_router)
app.include_router(account_router)
app.include_router(admin_router)
app.include_router(vitals_router)
app.include_router(chat_router)
app.include_router(dashboard_router)
app.include_router(records_router)
app.include_router(tts_router)

from backend.security import install_security
install_security(app)

# ── Frontend static file serving ──────────────────────────────────
if FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="frontend-assets",
    )

    _API_PREFIXES = (
        "queue", "auth", "health", "stream", "pdf", "export",
        "account", "admin", "vitals", "chat", "api",
        "records", "tts", "docs", "openapi.json", "redoc",
    )

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str) -> FileResponse:
        if full_path.split("/", 1)[0] in _API_PREFIXES:
            raise HTTPException(status_code=404, detail="Not found")

        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)

        return FileResponse(FRONTEND_DIST / "index.html")
else:
    logger.warning(
        "Frontend dist not found — skipping static file mount",
        expected_path=str(FRONTEND_DIST),
    )