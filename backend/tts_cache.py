"""
backend/tts_cache.py — Background TTS pre-synthesis, keyed by job_id.

Purpose:
    Let voice synthesis start the moment a report finishes generating,
    running in parallel with whatever the user does next (reading the
    report, etc.), instead of only starting after the user clicks the
    "Voice TTS" button.

Design constraints this module exists to satisfy:
    - Piper synthesis is synchronous and CPU-bound (see
      tools/tts_synthesizer.py). It must never run directly on the
      asyncio event loop — that blocks every other request on this
      single-worker server for the full duration of synthesis.
    - It must not fire unconditionally for every job regardless of
      whether anyone will ever listen to it, or CPU usage scales with
      report volume for no benefit. Callers should still gate this
      behind some signal of intent (e.g. only for jobs whose owner has
      voice readout enabled / has used it before), but the mechanism
      itself is available to call as soon as a report is ready.
    - Cache entries are in-memory only, per-process, and intentionally
      bounded/evictable — this is a latency optimization, not a
      durable store. A cache miss just means normal synthesis-on-click
      still works exactly as before.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from tools.tts_synthesizer import TTSError, synthesize_speech

# ── In-memory cache ─────────────────────────────────────────────────
# job_id -> _TTSCacheEntry. Bounded by _MAX_ENTRIES with simple FIFO
# eviction, since this is a short-lived latency win, not a store of
# record — the report text (and therefore the ability to re-synthesize
# on demand) always lives elsewhere.

_MAX_ENTRIES = 200


@dataclass
class _TTSCacheEntry:
    status: str  # "pending" | "ready" | "failed"
    audio_bytes: Optional[bytes] = None
    error_reason: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)


_cache: dict[str, _TTSCacheEntry] = {}
_cache_order: list[str] = []  # insertion order, for FIFO eviction
_inflight: dict[str, asyncio.Task] = {}


def _evict_if_needed() -> None:
    while len(_cache_order) > _MAX_ENTRIES:
        oldest = _cache_order.pop(0)
        _cache.pop(oldest, None)


def get_cached(job_id: str) -> _TTSCacheEntry | None:
    """Return the cache entry for job_id, or None if nothing has been prepared."""
    return _cache.get(job_id)


def start_background_synthesis(job_id: str, report_text: str) -> None:
    """
    Kick off Piper synthesis for report_text in a background thread,
    without blocking the caller or the event loop.

    Safe to call multiple times for the same job_id — a synthesis
    already in flight or already cached is not repeated.

    This only schedules work; it does not await it. Call get_cached()
    later (e.g. from the /tts/speak endpoint) to check whether the
    result is ready yet.
    """
    if job_id in _cache or job_id in _inflight:
        return

    if not report_text or not report_text.strip():
        return

    _cache[job_id] = _TTSCacheEntry(status="pending")
    _cache_order.append(job_id)
    _evict_if_needed()

    async def _run() -> None:
        try:
            # asyncio.to_thread moves the blocking, CPU-bound Piper
            # call onto a worker thread so it does not stall the event
            # loop — this is the same fix applied to the on-click path
            # in backend/tts.py, applied here for the background path.
            audio_bytes = await asyncio.to_thread(synthesize_speech, report_text)
            entry = _cache.get(job_id)
            if entry is not None:
                entry.status = "ready"
                entry.audio_bytes = audio_bytes
            logger.info("tts_cache · background synthesis ready", job_id=job_id)
        except TTSError as exc:
            entry = _cache.get(job_id)
            if entry is not None:
                entry.status = "failed"
                entry.error_reason = exc.reason
            logger.warning(
                "tts_cache · background synthesis failed",
                job_id=job_id,
                reason=exc.reason,
            )
        except Exception as exc:  # pragma: no cover - defensive
            entry = _cache.get(job_id)
            if entry is not None:
                entry.status = "failed"
                entry.error_reason = "synthesis_failed"
            logger.exception(
                "tts_cache · unexpected error during background synthesis",
                job_id=job_id,
            )
        finally:
            _inflight.pop(job_id, None)

    task = asyncio.create_task(_run())
    _inflight[job_id] = task