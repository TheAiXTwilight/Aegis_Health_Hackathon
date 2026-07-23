"""
backend/streaming.py — GET /queue/stream/{job_id}

Bridges the internal asyncio.Queue (filled by the inference worker) to
an HTTP streaming response that the Streamlit frontend consumes via
st.write_stream.

Protocol decisions:
    - Raw chunked transfer encoding (no SSE framing, no JSON wrapping)
    - Plain string tokens, yielded exactly as the pipeline produces them
    - The None sentinel placed by backend.queue terminates the stream

Status codes:
    200  Stream begins immediately.
    404  No such job_id (unknown, never submitted, or purged).
    425  Job exists but is still QUEUED — stream not yet available.
"""
from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from backend.queue import get_job, get_stream_queue
from schemas.queue import JobStatus


router = APIRouter()


@router.get("/queue/stream/{job_id}")
async def stream(job_id: str) -> StreamingResponse:
    """
    Stream pipeline tokens for a running or completed job.

    The worker in backend/queue.py creates an asyncio.Queue when a job
    transitions to RUNNING, fills it with tokens, and enqueues None when
    the pipeline finishes (success or failure). This endpoint drains that
    queue and yields each token to the HTTP client until the sentinel.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown job_id: {job_id}",
        )

    if job.status == JobStatus.QUEUED:
        raise HTTPException(
            status_code=425,
            detail=(
                f"Job {job_id} is still queued and has no stream yet. "
                "Poll /queue/status until status transitions to running."
            ),
        )

    stream_q = get_stream_queue(job_id)
    if stream_q is None:
        raise HTTPException(
            status_code=404,
            detail=f"Stream for job_id={job_id} is no longer available.",
        )

    async def token_stream() -> AsyncIterator[str]:
        """
        Drain the worker's asyncio.Queue until the None sentinel.

        Yields raw str. StreamingResponse handles UTF-8 encoding and
        chunked transfer encoding using the declared media_type.
        """
        while True:
            token = await stream_q.get()
            if token is None:
                return
            yield token

    logger.info(
        "Stream started",
        job_id=job_id,
        session_id=job.session_id,
        job_status=job.status.value,
    )

    return StreamingResponse(
        token_stream(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
