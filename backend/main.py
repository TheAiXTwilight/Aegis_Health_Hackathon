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
    Files are persisted to /tmp/aegis_uploads/{session_id}/ and removed
    when the job is purged via register_purge_callback().
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from agents.pipeline import AegisPipeline
from backend.health import router as health_router
from backend.queue import (
    get_status_payload,
    register_purge_callback,
    run_inference_worker,
    submit_job,
)
from backend.streaming import router as streaming_router
from backend.uploads import (
    validate_audio,
    validate_lab_pdf,
    validate_medications,
    validate_xray_image,
)
from schemas.errors import ToolError
from schemas.queue import PipelineJob
from schemas.state import AegisState


UPLOAD_ROOT = Path("/tmp/aegis_uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


# ── Upload helpers ───────────────────────────────────────────────

async def _save_upload(file: UploadFile, session_dir: Path, kind: str) -> Path:
    """
    Save an UploadFile to disk under session_dir. Returns the saved path.
    Uses async chunked writes to avoid blocking the event loop on large files.
    """
    suffix = Path(file.filename or kind).suffix or ""
    target = session_dir / f"{kind}{suffix}"

    async with aiofiles.open(target, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            await out.write(chunk)

    return target


def cleanup_session_uploads(session_id: str) -> None:
    """
    Remove all uploaded files for a session. Called by the queue's
    purge mechanism when a job's retention window expires.
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


# ── Lifespan ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start the inference worker and register cleanup callbacks on startup.
    Cancel the worker cleanly on shutdown.

    register_purge_callback is idempotent — re-entering lifespan in tests
    does not accumulate duplicate registrations.
    """
    pipeline = AegisPipeline()
    register_purge_callback(cleanup_session_uploads)

    worker_task = asyncio.create_task(run_inference_worker(pipeline))
    logger.info("Inference worker started")

    try:
        yield
    finally:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
        logger.info("Inference worker stopped")


# ── App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Aegis Health",
    version="1.0",
    lifespan=lifespan,
)


# ── Endpoints: queue submit / status ─────────────────────────────

@app.post("/queue/submit")
async def submit(
    symptoms_text: str | None = Form(None),
    medications: str = Form("[]"),
    xray_findings: str = Form("[]"),
    xray_free_text: str | None = Form(None),
    audio: UploadFile | None = File(None),
    lab_pdf: UploadFile | None = File(None),
    xray_image: UploadFile | None = File(None),
) -> JSONResponse:
    """
    Validate uploads → build AegisState → enqueue a PipelineJob.

    Form contract:
        symptoms_text   — optional plain text symptoms
        medications     — JSON-encoded list[str], default "[]"
        xray_findings   — JSON-encoded list[str], default "[]"
        xray_free_text  — optional clinician free-text X-ray note
        audio           — optional audio file (WAV preferred)
        lab_pdf         — optional lab report PDF
        xray_image      — optional X-ray image

    Status codes:
        200 — accepted, returns PipelineJob JSON
        400 — invalid input (size, duration, count, malformed JSON)
        409 — duplicate job or active session
        503 — queue full
    """
    # Parse JSON-encoded form fields.
    try:
        medications_list = json.loads(medications)
        if not isinstance(medications_list, list) or not all(
            isinstance(m, str) for m in medications_list
        ):
            raise ValueError("medications must be a JSON list of strings")
    except (json.JSONDecodeError, ValueError) as e:
        return _tool_error_response(
            ToolError(
                tool="input_validation",
                reason=f"Invalid medications field: {e}",
                fatal=True,
            ),
            status_code=400,
        )

    try:
        xray_findings_list = json.loads(xray_findings)
        if not isinstance(xray_findings_list, list) or not all(
            isinstance(f, str) for f in xray_findings_list
        ):
            raise ValueError("xray_findings must be a JSON list of strings")
    except (json.JSONDecodeError, ValueError) as e:
        return _tool_error_response(
            ToolError(
                tool="input_validation",
                reason=f"Invalid xray_findings field: {e}",
                fatal=True,
            ),
            status_code=400,
        )

    # At least one input modality must be provided.
    if not any([
        symptoms_text, audio, lab_pdf, xray_image,
        medications_list, xray_findings_list, xray_free_text,
    ]):
        return _tool_error_response(
            ToolError(
                tool="input_validation",
                reason="At least one input must be provided.",
                fatal=True,
            ),
            status_code=400,
        )

    # Pre-build state to get a server-generated session_id.
    state = AegisState(
        raw_symptoms_text=symptoms_text,
        medications_raw=medications_list,
        xray_findings_raw=xray_findings_list,
        xray_free_text_raw=xray_free_text,
    )

    # Per-session upload directory.
    session_dir = UPLOAD_ROOT / state.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files to disk.
    try:
        if audio is not None:
            state.audio_file_path = str(
                await _save_upload(audio, session_dir, "audio")
            )
        if lab_pdf is not None:
            state.lab_pdf_path = str(
                await _save_upload(lab_pdf, session_dir, "lab")
            )
        if xray_image is not None:
            state.xray_image_path = str(
                await _save_upload(xray_image, session_dir, "xray")
            )
    except OSError as e:
        _cleanup_on_rejection(session_dir)
        return _tool_error_response(
            ToolError(
                tool="input_validation",
                reason=f"Failed to save uploaded file: {e}",
                fatal=True,
            ),
            status_code=400,
        )

    # Validate uploads.
    validators_with_inputs = [
        (validate_medications, state.medications_raw),
        (validate_audio, state.audio_file_path),
        (validate_lab_pdf, state.lab_pdf_path),
        (validate_xray_image, state.xray_image_path),
    ]
    for validator, input_value in validators_with_inputs:
        if input_value is None or input_value == []:
            continue
        err = validator(input_value)
        if err is not None:
            _cleanup_on_rejection(session_dir)
            return _tool_error_response(err, status_code=400)

    # Build and submit job.
    # No try/except ValidationError here. Both objects are already
    # constructed and valid. submit_job() only inspects them.
    job = PipelineJob(session_id=state.session_id)
    result = await submit_job(job, state)

    if isinstance(result, ToolError):
        _cleanup_on_rejection(session_dir)
        status_code = _status_code_for_queue_error(result)
        return _tool_error_response(result, status_code=status_code)

    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )


def _status_code_for_queue_error(err: ToolError) -> int:
    """
    Map queue-layer ToolError reasons to HTTP status codes.

    TEMPORARY: substring matching is fragile. Phase 5 will replace this
    with a structured code field on ToolError. Tracked in open items.
    """
    reason = err.reason.lower()
    if "queue full" in reason:
        return 503
    if "duplicate" in reason or "already has an active job" in reason:
        return 409
    return 400


@app.get("/queue/status/{job_id}")
async def status(job_id: str) -> dict[str, Any]:
    """Return live status payload for a job. 404 if unknown or purged."""
    payload = get_status_payload(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return payload


# ── Router mounts ────────────────────────────────────────────────

app.include_router(health_router)
app.include_router(streaming_router)