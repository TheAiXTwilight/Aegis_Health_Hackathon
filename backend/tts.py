"""
backend/tts.py — Local text-to-speech endpoint (Piper, fully offline).

    POST /tts/speak          — authenticated user submits report text
                                (optionally with job_id), gets back WAV
                                audio bytes synthesized locally.
    POST /tts/speak/stream   — same, but streams WAV chunks as each
                                report segment finishes synthesis so
                                playback starts after ~100-300 ms
                                instead of waiting for the full report.
    GET  /tts/status/{job_id} — check whether background synthesis for
                                a job has finished, without triggering
                                a new synthesis. Also fires a
                                fire-and-forget Piper warmup so the
                                voice model is loaded by the time the
                                user clicks Voice TTS.

Replaces the frontend's previous use of window.speechSynthesis, which
depends on OS-installed voices and silently produces no audio on many
machines/browsers (see tools/tts_synthesizer.py docstring). No request
ever leaves this server — synthesis runs in-process via Piper.

Background synthesis (backend/tts_cache.py) may already have prepared
audio for a job_id by the time /speak is called, if the pipeline
kicked off synthesis in parallel with report generation (see
backend/queue.py). When that's the case, /speak returns the cached
audio immediately instead of re-synthesizing from scratch.

/speak/stream bypasses the cache entirely — streaming is only useful
when synthesis has not finished yet, so there is nothing to cache-hit.
It runs synthesis in a background thread, pushing each PCM segment to
the client via chunked transfer as soon as it is ready.
"""

from __future__ import annotations

import asyncio
import queue
import threading

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from loguru import logger

from app.auth import get_current_user
from app.db.models import User
from app.settings import settings
from backend import tts_cache
from schemas.tts import TextToSpeechRequest
from tools import tts_synthesizer
from tools.tts_synthesizer import (
    TTSError,
    synthesize_speech,
    synthesize_speech_stream,
)

router = APIRouter(prefix="/tts", tags=["tts"])


_REASON_TO_STATUS = {
    "model_missing": 503,   # voice model not installed — server misconfig
    "text_too_long": 413,
    "synthesis_failed": 500,
}

# How long /speak is willing to wait for an in-progress background
# synthesis to finish before falling back to synthesizing itself.
# Keeps /speak's latency bounded even if the background job is stuck.
_CACHE_WAIT_TIMEOUT_S = 8.0
_CACHE_POLL_INTERVAL_S = 0.15

# Sentinels pushed onto the streaming queue by the producer thread.
# _STREAM_DONE  — synthesis finished, no more chunks coming.
# _STREAM_ERROR — synthesis failed; the next item on the queue is the
#                 exception instance.
_STREAM_DONE = object()
_STREAM_ERROR = object()


@router.post("/speak")
async def speak(
    body: TextToSpeechRequest,
    current_user: User = Depends(get_current_user),
) -> Response:
    """
    Synthesize `body.text` to speech locally and return WAV audio.

    If `body.job_id` is provided and background synthesis for that job
    is already complete (see backend/tts_cache.py), the cached audio
    is returned immediately with no new synthesis. If background
    synthesis is still in progress, this waits briefly for it rather
    than duplicating the work. Otherwise it synthesizes fresh.

    Returns:
        200 — audio/wav bytes.
        413 — text exceeds AEGIS_TTS_MAX_CHARS.
        503 — Piper voice model not installed on this server.
        500 — synthesis failed for another reason.
    """
    if body.job_id:
        entry = tts_cache.get_cached(body.job_id)

        if entry is not None and entry.status == "pending":
            # Background synthesis is already running — wait for it
            # instead of starting a redundant second synthesis.
            waited = 0.0
            while entry.status == "pending" and waited < _CACHE_WAIT_TIMEOUT_S:
                await asyncio.sleep(_CACHE_POLL_INTERVAL_S)
                waited += _CACHE_POLL_INTERVAL_S
                entry = tts_cache.get_cached(body.job_id)

        if entry is not None and entry.status == "ready" and entry.audio_bytes:
            return Response(content=entry.audio_bytes, media_type="audio/wav")

        if entry is not None and entry.status == "failed":
            status_code = _REASON_TO_STATUS.get(entry.error_reason, 500)
            raise HTTPException(
                status_code=status_code,
                detail=entry.error_reason or "Background synthesis failed.",
            )
        # entry is None or still pending after timeout — synthesize fresh.

    try:
        # synthesize_speech() is synchronous and CPU-bound (Piper neural
        # inference). asyncio.to_thread() moves it off the event loop so
        # it does not freeze other requests during synthesis.
        wav_bytes = await asyncio.to_thread(
            synthesize_speech,
            body.text,
            max_chars=settings.AEGIS_TTS_MAX_CHARS,
        )
    except TTSError as exc:
        status_code = _REASON_TO_STATUS.get(exc.reason, 500)
        logger.warning(
            "tts · synthesis error",
            user_id=current_user.id,
            reason=exc.reason,
            detail=str(exc),
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    return Response(content=wav_bytes, media_type="audio/wav")


@router.post("/speak/stream")
async def speak_stream(
    body: TextToSpeechRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Streaming variant of /speak.

    Synthesizes `body.text` segment-by-segment and pushes WAV bytes to
    the client via chunked transfer encoding as each segment finishes,
    so the browser can start playing after ~100-300 ms instead of
    waiting for the entire report to be synthesized.

    Always synthesizes fresh — does not check the background cache,
    because streaming is only useful before synthesis is complete.

    Response format: valid WAV stream — 44-byte header with data-size
    0xFFFFFFFF (streaming sentinel, per WAV spec) followed by raw
    int16-LE PCM chunks as they are produced.

    Architecture — thread + queue bridge:
        synthesize_speech_stream() is a synchronous generator.
        asyncio.to_thread() runs a callable to completion and returns
        one result — it cannot incrementally yield from a thread.
        Instead we spin a plain threading.Thread that puts chunks onto
        a queue.Queue, and an async generator on the event loop drains
        that queue by BLOCKING on queue.get() inside an executor. This
        replaces the earlier busy-polling implementation (get_nowait +
        asyncio.sleep(0.01)) which spun at 100 Hz per request while
        Piper was synthesizing and pegged every CPU core.

    Returns:
        200 — audio/wav chunked stream.
        413 — text exceeds AEGIS_TTS_MAX_CHARS.
        503 — Piper voice model not installed on this server.
        500 — synthesis failed for another reason.
    """
    # Run validation + voice loading synchronously in a thread BEFORE
    # opening the StreamingResponse. Once we return 200 the status code
    # is already sent — we cannot switch to 4xx/5xx mid-stream.
    # _prime_and_start_producer() calls synthesize_speech_stream() up
    # to (and including) its first yield (the WAV header), which
    # triggers _validate_text() and _load_voice() — both run before
    # any yield.
    try:
        chunk_queue = await asyncio.to_thread(
            _prime_and_start_producer,
            body.text,
            settings.AEGIS_TTS_MAX_CHARS,
        )
    except TTSError as exc:
        status_code = _REASON_TO_STATUS.get(exc.reason, 500)
        logger.warning(
            "tts · stream validation error",
            user_id=current_user.id,
            reason=exc.reason,
            detail=str(exc),
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    async def async_chunk_generator():
        # Bridge the blocking queue.get() to asyncio via
        # loop.run_in_executor(). This blocks a worker thread (cheap —
        # near-zero CPU while waiting) instead of busy-polling the
        # event loop at 100 Hz, which was the root cause of the CPU
        # spike in the earlier implementation: get_nowait() +
        # asyncio.sleep(0.01) spun continuously across every active
        # request while Piper took 200-500 ms per segment, pegging
        # every core for the whole synthesis duration.
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, chunk_queue.get)

            if item is _STREAM_DONE:
                break

            if item is _STREAM_ERROR:
                # Retrieve the exception the producer put after the sentinel.
                exc_obj = await loop.run_in_executor(None, chunk_queue.get)
                logger.error(
                    "tts · stream synthesis error mid-stream",
                    user_id=current_user.id,
                    error=str(exc_obj),
                )
                # Headers are already sent — close the stream cleanly.
                # The browser will play whatever audio arrived before
                # the error.
                break

            yield item

    return StreamingResponse(
        async_chunk_generator(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _prime_and_start_producer(text: str, max_chars: int) -> "queue.Queue":
    """
    Create the synthesize_speech_stream() generator, advance it to its
    first yield (the WAV header — which triggers all validation and
    voice loading), put that header onto the queue, then hand off the
    remaining iteration to a background daemon thread.

    Returns the queue immediately so the caller (speak_stream) can
    start draining it via the async generator.

    This function is called via asyncio.to_thread() so validation and
    voice loading run on a worker thread rather than blocking the loop.

    Raises TTSError if validation or voice loading fails — before the
    queue is created, so the caller can still return a proper 4xx/5xx.
    """
    # Create the generator. _validate_text() and _load_voice() run
    # synchronously inside synthesize_speech_stream() before the first
    # yield, so any TTSError surfaces here.
    gen = synthesize_speech_stream(text, max_chars=max_chars)

    # Advance to the first yield (WAV header). Raises TTSError on
    # validation failure or StopIteration if the generator is empty
    # (should never happen given the not-segments guard inside).
    try:
        first_chunk = next(gen)
    except StopIteration:
        raise TTSError(
            "Stream generator produced no output.", reason="synthesis_failed"
        )

    # Everything validated — set up the queue and hand off to the
    # producer thread for the remaining chunks.
    chunk_queue: queue.Queue = queue.Queue(maxsize=64)
    chunk_queue.put(first_chunk)  # WAV header, ready immediately

    def _producer() -> None:
        try:
            for chunk in gen:
                chunk_queue.put(chunk)  # blocks on full queue (backpressure)
            chunk_queue.put(_STREAM_DONE)
        except TTSError as exc:
            chunk_queue.put(_STREAM_ERROR)
            chunk_queue.put(exc)
        except Exception as exc:  # pragma: no cover — defensive
            chunk_queue.put(_STREAM_ERROR)
            chunk_queue.put(exc)

    t = threading.Thread(target=_producer, daemon=True)
    t.start()

    return chunk_queue


@router.get("/status/{job_id}")
async def tts_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Check whether background synthesis for job_id has finished, without
    triggering any new synthesis. Used by the frontend to know whether
    clicking "Voice TTS" will play instantly or need to wait.

    Side effect: fire-and-forget Piper voice warmup. Any status check
    is a strong signal that the user has opened a report page and is
    likely to click Voice TTS soon — so we start loading the voice
    now (in a thread, non-blocking) if it is not already loaded. This
    is idempotent and thread-safe (see tts_synthesizer.warmup), and
    also updates the idle-eviction "last activity" timestamp so a
    voice loaded specifically for this user doesn't get evicted
    before they click.

    Returns: {"status": "pending" | "ready" | "failed" | "not_found"}
    """
    # DISABLED on Jetson board: this endpoint is polled automatically
    # by the frontend just from having a report page open, so firing
    # Piper warmup here means simply viewing a report — not clicking
    # Voice TTS — was loading/keeping Piper warm in the background,
    # competing with Ollama/report generation for CPU on this shared,
    # memory-constrained device. Warmup now only happens via the
    # actual on-click synthesis path (POST /tts/speak).
    # tts_synthesizer.touch_activity()
    # asyncio.create_task(asyncio.to_thread(tts_synthesizer.warmup))

    entry = tts_cache.get_cached(job_id)
    if entry is None:
        return {"status": "not_found"}
    return {"status": entry.status}