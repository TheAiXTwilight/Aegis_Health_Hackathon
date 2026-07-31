"""
tools/tts_synthesizer.py — Thin client + lifecycle manager for the Piper TTS
worker process (tools/piper_server.py).

WHY A SEPARATE PROCESS
    Piper is the ONLY pipeline component that used to run inside the backend
    process. Its ONNX inference fans out across cores and its model/arena
    memory bloated the backend, crashing it on memory-constrained Jetson
    boards. Every other tool (symptom extraction, report generation, ...)
    already offloads to the separate Ollama process, so the backend stays
    light. This module now does the same for TTS: Piper runs in its own
    worker process, and this module is just a client.

LIFECYCLE (the "hybrid")
    - warmup() / ensure_worker() spawn the worker and load the voice. Called
      at server startup (prewarm) so the first read-out is instant.
    - evict_voice() KILLS the worker → 100% of its RAM returns to the OS
      (no ONNX-arena leak). Called by the idle monitor after
      AEGIS_TTS_IDLE_EVICT_SECS and on shutdown.
    - The next synthesis after eviction respawns the worker (~1s once).

PUBLIC API IS UNCHANGED from the old in-process version (warmup,
synthesize_speech, synthesize_speech_stream, is_loaded, evict_voice,
seconds_since_last_activity, touch_activity, TTSError,
clean_text_for_speech), so backend/tts.py, backend/queue.py and
backend/main.py need NO edits.
"""
from __future__ import annotations

import os
import re
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import httpx
from loguru import logger

from app.settings import settings


# ── ONNX/OpenMP thread cap ────────────────────────────────────────
# Also enforced at the top of piper_server.py (the worker is where the work
# happens). Kept here too so any in-process ONNX (RAG embedder) is capped.
_ONNX_THREADS = os.environ.get("AEGIS_PIPER_THREADS", "2")


def _apply_onnx_thread_cap() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", _ONNX_THREADS)
    os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", _ONNX_THREADS)
    os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", _ONNX_THREADS)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", _ONNX_THREADS)


_apply_onnx_thread_cap()


class TTSError(Exception):
    """Raised for any TTS failure. `reason` is machine-readable."""

    def __init__(self, message: str, reason: str = "synthesis_failed"):
        super().__init__(message)
        self.reason = reason


# ════════════════════════════════════════════════════════════════════
# PURE TEXT / WAV HELPERS — shared with piper_server.py (imported there).
# These contain no voice state; safe to import in either process.
# ════════════════════════════════════════════════════════════════════

_MARKDOWN_HEADER_RE = re.compile(r"#{1,6}\s?")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_BULLET_RE = re.compile(r"[-*•]")

_RAW_HTML_BLOCK_RE = re.compile(
    r"<!--RAW_HTML_START-->([\s\S]*?)<!--RAW_HTML_END-->"
)
_HTML_BLOCK_TAG_RE = re.compile(
    r"</?(?:div|p|li|ul|ol|br|tr|td|th|table|section|span)\b[^>]*>",
    re.IGNORECASE,
)
_HTML_ANY_TAG_RE = re.compile(r"<[^>]+>")
_HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")
_HTML_ENTITY_MAP = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&#39;": "'", "&nbsp;": " ",
}


def _html_to_speech_text(html: str) -> str:
    text = _HTML_COMMENT_RE.sub(" ", html)
    text = _HTML_BLOCK_TAG_RE.sub(" ", text)
    text = _HTML_ANY_TAG_RE.sub(" ", text)
    for entity, char in _HTML_ENTITY_MAP.items():
        text = text.replace(entity, char)
    return re.sub(r"[ \t]+", " ", text)


def clean_text_for_speech(text: str) -> str:
    cleaned = _RAW_HTML_BLOCK_RE.sub(
        lambda m: _html_to_speech_text(m.group(1)), text
    )
    cleaned = _MARKDOWN_HEADER_RE.sub("", cleaned)
    cleaned = _MARKDOWN_BOLD_RE.sub("", cleaned)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _MARKDOWN_BULLET_RE.sub("", cleaned)
    cleaned = _HTML_ANY_TAG_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ── Segment-based synthesis (pause-aware) ─────────────────────────
_LINE_PAUSE_SECS: float = 0.35
_SECTION_PAUSE_SECS: float = 0.75
_HEADING_PAUSE_SECS: float = 0.90
_MAX_HEADING_WORDS = 6


def _is_heading_line(line: str) -> bool:
    words = line.split()
    return (
        1 <= len(words) <= _MAX_HEADING_WORDS
        and bool(words[0][:1].isupper())
    )


def _segment_report_text(cleaned: str) -> list[tuple[str, float]]:
    lines = cleaned.split("\n")
    segments: list[tuple[str, float]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if segments:
                seg_text, _old = segments[-1]
                segments[-1] = (seg_text, _SECTION_PAUSE_SECS)
            continue
        if not line.endswith((".", "!", "?", ":", ";")):
            line = line + "."
        pause = _HEADING_PAUSE_SECS if _is_heading_line(line.rstrip(".!?:;")) else _LINE_PAUSE_SECS
        segments.append((line, pause))
    return segments


def _validate_text(text: str, max_chars: int) -> str:
    if not text or not text.strip():
        raise TTSError("No text provided.", reason="synthesis_failed")
    cleaned = clean_text_for_speech(text)
    if not cleaned:
        raise TTSError("Text was empty after cleanup.", reason="synthesis_failed")
    if len(cleaned) > max_chars:
        raise TTSError(
            f"Text is {len(cleaned)} characters, exceeds limit of {max_chars}.",
            reason="text_too_long",
        )
    return cleaned


# ── WAV streaming header (unknown length) ─────────────────────────
_WAV_CHANNELS = 1
_WAV_SAMPLE_WIDTH = 2


def _make_wav_header(sample_rate: int) -> bytes:
    byte_rate = sample_rate * _WAV_CHANNELS * _WAV_SAMPLE_WIDTH
    block_align = _WAV_CHANNELS * _WAV_SAMPLE_WIDTH
    bits_per_sample = _WAV_SAMPLE_WIDTH * 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE",
        b"fmt ", 16, 1, _WAV_CHANNELS, sample_rate, byte_rate, block_align,
        bits_per_sample, b"data", 0xFFFFFFFF,
    )


# ════════════════════════════════════════════════════════════════════
# WORKER PROCESS LIFECYCLE
# ════════════════════════════════════════════════════════════════════

_WORKER_LOCK = threading.Lock()
_worker_proc: subprocess.Popen | None = None
_last_activity_ts: float = 0.0

# tiny health cache so frequent /tts/status polls don't hammer the worker
_health_cache_ts: float = 0.0
_health_cache_val: bool = False
_HEALTH_CACHE_TTL = 0.5


def _worker_port() -> int:
    return int(os.environ.get("AEGIS_PIPER_PORT", "9880"))


def _worker_url(path: str = "/health") -> str:
    return f"http://127.0.0.1:{_worker_port()}{path}"


def _repo_root() -> str:
    # tools/tts_synthesizer.py -> parent.parent == repo root
    return str(Path(__file__).resolve().parent.parent)


def _touch_activity() -> None:
    global _last_activity_ts
    _last_activity_ts = time.monotonic()


def touch_activity() -> None:
    """Public: record TTS intent (e.g. a status poll) to keep the worker hot."""
    _touch_activity()


def _health(force: bool = False) -> bool:
    global _health_cache_ts, _health_cache_val
    now = time.monotonic()
    if not force and (now - _health_cache_ts) < _HEALTH_CACHE_TTL:
        return _health_cache_val
    try:
        r = httpx.get(_worker_url("/health"), timeout=1.0)
        ok = r.status_code == 200 and r.json().get("ok") is True
    except Exception:
        ok = False
    _health_cache_ts = now
    _health_cache_val = ok
    return ok


def ensure_worker(timeout: float = 90.0) -> None:
    """
    Make sure the Piper worker process is up and the voice is loaded.
    Spawns it (once) if needed. Thread-safe; idempotent.
    Raises TTSError(reason="model_missing") if it can't become ready.
    """
    global _worker_proc
    with _WORKER_LOCK:
        if _worker_proc is not None and _worker_proc.poll() is None and _health(force=True):
            _touch_activity()
            return

        env = os.environ.copy()
        cmd = [
            sys.executable, "-m", "tools.piper_server",
            "--host", "127.0.0.1", "--port", str(_worker_port()),
        ]
        log_path = os.environ.get("AEGIS_PIPER_LOG", "/tmp/aegis_piper_worker.log")
        try:
            log_f = open(log_path, "ab")
        except OSError:
            log_f = subprocess.DEVNULL  # type: ignore

        logger.info("tts_synthesizer · spawning piper worker", port=_worker_port())
        _worker_proc = subprocess.Popen(
            cmd,
            cwd=_repo_root(),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=(sys.platform != "win32"),
        )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _worker_proc.poll() is not None:
                _worker_proc = None
                raise TTSError(
                    "piper worker exited during startup "
                    "(check /tmp/aegis_piper_worker.log)",
                    reason="model_missing",
                )
            if _health(force=True):
                _touch_activity()
                logger.info("tts_synthesizer · piper worker ready")
                return
            time.sleep(0.3)
        _worker_proc = None
        raise TTSError(
            "piper worker did not become ready in time "
            "(check /tmp/aegis_piper_worker.log)",
            reason="model_missing",
        )


def is_loaded() -> bool:
    """True when the worker is up AND the voice is loaded."""
    return _health()


def warmup() -> None:
    """Pre-load the worker (spawn + voice load). Safe/idempotent; never raises."""
    try:
        ensure_worker(timeout=90.0)
    except Exception as exc:
        logger.warning(f"tts_synthesizer · warmup failed (will retry lazily): {exc}")


def evict_voice() -> bool:
    """
    Kill the worker process to reclaim its RAM. Returns True if a worker
    was actually stopped. Safe to call when nothing is running.
    """
    global _worker_proc, _health_cache_val
    with _WORKER_LOCK:
        proc = _worker_proc
        _worker_proc = None
        _health_cache_val = False
        if proc is None or proc.poll() is not None:
            return False
        try:
            if sys.platform != "win32":
                try:
                    os.killpg(os.getpgid(proc.pid), 15)
                except ProcessLookupError:
                    pass
            else:
                proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        logger.info("tts_synthesizer · piper worker killed — RAM reclaimed")
        return True


def seconds_since_last_activity() -> float:
    if _last_activity_ts == 0.0:
        return float("inf")
    return time.monotonic() - _last_activity_ts


# ════════════════════════════════════════════════════════════════════
# SYNTHESIS — delegated to the worker over local HTTP
# ════════════════════════════════════════════════════════════════════

def synthesize_speech(text: str, max_chars: int | None = None) -> bytes:
    if max_chars is None:
        max_chars = settings.AEGIS_TTS_MAX_CHARS
    ensure_worker(timeout=90.0)
    _touch_activity()
    try:
        r = httpx.post(
            _worker_url("/synthesize"),
            json={"text": text, "max_chars": max_chars},
            timeout=180.0,
        )
        if r.status_code != 200:
            raise TTSError(f"worker error {r.status_code}: {r.text[:200]}")
        _touch_activity()
        return r.content
    except TTSError:
        raise
    except Exception as exc:
        raise TTSError(f"speech synthesis failed: {exc}") from exc


def synthesize_speech_stream(
    text: str, max_chars: int | None = None
) -> Iterator[bytes]:
    if max_chars is None:
        max_chars = settings.AEGIS_TTS_MAX_CHARS
    ensure_worker(timeout=90.0)
    _touch_activity()
    try:
        with httpx.stream(
            "POST",
            _worker_url("/synthesize?stream=1"),
            json={"text": text, "max_chars": max_chars},
            timeout=180.0,
        ) as r:
            if r.status_code != 200:
                raise TTSError(f"worker error {r.status_code}")
            for chunk in r.iter_bytes():
                if chunk:
                    _touch_activity()
                    yield chunk
    except TTSError:
        raise
    except Exception as exc:
        raise TTSError(f"speech synthesis failed: {exc}") from exc
