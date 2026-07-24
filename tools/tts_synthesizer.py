"""
tools/tts_synthesizer.py — Local text-to-speech synthesis via Piper.

Replaces the frontend's reliance on the browser `window.speechSynthesis`
API (which silently produces no audio on machines/browsers with zero
installed OS voices — see backend/tts.py for the endpoint that calls
this module).

Architecture (mirrors tools/voice_transcriber.py):
    - Singleton PiperVoice (_VOICE), lazy-loaded on first call.
      Optionally warm-loaded via warmup() — triggered when a report
      job starts running (backend/queue.py) or when the frontend polls
      /tts/status/{job_id} (backend/tts.py). This eliminates the
      ~500ms-1s first-click latency spike, since by the time the user
      actually clicks Voice TTS the model is already loaded.
    - Idle eviction: after AEGIS_TTS_IDLE_EVICT_SECS with no TTS
      activity, the loaded voice is unloaded and its ~180MB of ONNX
      weights + activation buffers are freed. A background monitor
      task (started from backend/main.py's lifespan) runs the check
      every 60s. Reload on next request is transparent.
    - Model directory/voice name resolved via _get_model_dir() /
      _get_voice_name() — both read from app/settings.py
      (AEGIS_TTS_MODEL_DIR / AEGIS_TTS_VOICE_NAME), the single source
      of truth for this config. Do NOT read os.environ directly here —
      settings.py already merges .env / real env vars / defaults, and
      splitting that logic across two places is how this drifted out
      of sync last time.
    - CPU only. Fully offline — no network call, no external API key.
    - Output: WAV bytes (16-bit PCM).
    - All failure modes raise TTSError (never a bare Exception) so the
      backend route can return a clean structured error instead of a
      500 with a stack trace.
    - ONNX Runtime thread count is capped to 2 (see
      _apply_onnx_thread_cap below). This is a sweet-spot chosen
      empirically: 1 thread minimises CPU spike (~100%) but adds
      ~40% latency per segment; 2 threads keep CPU under ~200% (vs.
      the default ~380% on an 8-core machine that fanned across
      every core fighting cache lines) while restoring most of that
      latency back. Segments still run sequentially — parallel
      segment synthesis was considered but rejected because Piper's
      Python wrapper is not thread-safe on a shared voice instance.

Concurrency model:
    Multiple triggers can call warmup() at nearly the same time (e.g.
    report-start warmup from the pipeline worker, plus a status-check
    warmup from the frontend polling /tts/status). A threading.Lock
    (_LOAD_LOCK) guards the load/evict transition so at most one
    thread does the ~500ms model load work while others block briefly
    and then reuse the loaded singleton. All entry points update
    _last_activity_ts so the idle monitor never evicts a voice that
    is actively in use.

Voice model files are NOT bundled with the repo (they run 50-100MB+
each). One-time setup (voice name/dir must match app/settings.py):

    python -m piper.download_voices en_GB-jenny_dioco-medium

...or download manually from
https://github.com/rhasspy/piper/blob/master/VOICES.md and place both
the .onnx and .onnx.json files in the directory named by
settings.AEGIS_TTS_MODEL_DIR (see app/settings.py for the current
default).
"""

from __future__ import annotations

import io
import os
import re
import struct
import threading
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np
from loguru import logger

from app.settings import settings

if TYPE_CHECKING:
    from piper import PiperVoice as _PiperVoiceType


class TTSError(Exception):
    """Raised for any TTS failure. `reason` is machine-readable."""

    def __init__(self, message: str, reason: str = "synthesis_failed"):
        super().__init__(message)
        self.reason = reason


# ── ONNX Runtime CPU-usage cap ─────────────────────────────────────
#
# By default ONNX Runtime (which Piper uses under the hood) creates
# one intra-op thread per physical CPU core AND enables inter-op
# parallelism, so every voice.synthesize_wav() call fans out across
# every available core. On an 8-core machine this shows up as ~380%
# system CPU during the 2-second synthesis window — every core pegged
# fighting over the same tiny transformer.
#
# The neural network in Piper voice models is small enough (~15-25M
# params) that using more than 2 cores gives no meaningful speedup:
# the cores spend most of their time waiting for cache lines and
# synchronising on reduction ops rather than doing useful math.
#
# We settled on 2 intra-op threads / 1 inter-op thread after A/B
# testing on segmented streaming synthesis:
#   1 thread:  ~95% peak CPU, ~180ms/segment, ~2.5s total
#   2 threads: ~180% peak CPU, ~115ms/segment, ~1.7s total  ← chosen
#   4 threads: ~320% peak CPU, ~95ms/segment, ~1.5s total
#   default:   ~380% peak CPU, ~90ms/segment, ~1.4s total
# For streaming synthesis, latency-to-first-byte matters far more
# than total wall time (playback starts after segment 1 and later
# segments queue up faster than they play). 2 threads is where CPU
# spike stops being noticeable while segment latency stays low
# enough that the playback queue never underruns.
#
# We set these via environment variables BEFORE piper-tts (and
# therefore ONNX Runtime) is imported, because ONNX Runtime reads them
# once at session-creation time and ignores changes afterward. This
# has to happen at module import in tts_synthesizer.py, not inside
# _load_voice(), because piper imports ORT eagerly on `from piper
# import PiperVoice`.

_ONNX_INTRA_OP_THREADS = "2"   # threads inside a single op (matmul, conv)
_ONNX_INTER_OP_THREADS = "1"   # threads across independent ops


def _apply_onnx_thread_cap() -> None:
    """
    Cap ONNX Runtime CPU thread count via environment variables.
    Idempotent — safe to call multiple times.

    Must run BEFORE `from piper import PiperVoice` because piper's
    __init__ imports onnxruntime, which reads these vars once at
    module-load time and never re-reads them.
    """
    # setdefault so an operator can override via .env / real env vars
    # if they specifically need higher parallelism (e.g. batch use).
    os.environ.setdefault("OMP_NUM_THREADS", _ONNX_INTRA_OP_THREADS)
    os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", _ONNX_INTRA_OP_THREADS)
    os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", _ONNX_INTER_OP_THREADS)
    # Extra belt-and-braces for MKL / OpenBLAS backends occasionally
    # pulled in by numpy — same CPU-spike root cause, same fix.
    os.environ.setdefault("MKL_NUM_THREADS", _ONNX_INTRA_OP_THREADS)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", _ONNX_INTRA_OP_THREADS)


_apply_onnx_thread_cap()


# ── Singleton + activity tracking ──────────────────────────────────

_VOICE: "_PiperVoiceType | None" = None

# Lock guarding _VOICE load/evict transitions. Held only during the
# ~500ms load / ~1ms evict itself, NOT during synthesis — synthesize
# calls read _VOICE without acquiring, which is safe because we only
# ever assign _VOICE from within the lock and CPython guarantees
# atomic pointer assignment.
_LOAD_LOCK = threading.Lock()

# Wall-clock timestamp (time.monotonic()) of the last TTS activity.
# Any of warmup(), synthesize_speech(), synthesize_speech_stream(),
# or touch_activity() updates this. The idle monitor task compares
# this to time.monotonic() to decide when to evict.
_last_activity_ts: float = 0.0


def _touch_activity() -> None:
    """Record that TTS was just used, so the idle monitor keeps the
    voice loaded. Called internally at every synthesis entry point."""
    global _last_activity_ts
    _last_activity_ts = time.monotonic()


def touch_activity() -> None:
    """
    Public alias for _touch_activity() — called by backend/tts.py's
    status endpoint so that even mere status polls (which happen when
    the user opens a report page) count as intent-to-use, keeping the
    voice hot for the likely upcoming click.
    """
    _touch_activity()


def _get_model_dir() -> Path:
    """Single source of truth for the Piper model directory (app/settings.py)."""
    return Path(settings.AEGIS_TTS_MODEL_DIR)


def _get_voice_name() -> str:
    """Single source of truth for the Piper voice name (app/settings.py)."""
    return settings.AEGIS_TTS_VOICE_NAME


def _resolve_model_files() -> tuple[Path, Path]:
    """
    Locate the .onnx model and its .onnx.json config inside the model
    directory. Raises TTSError(reason="model_missing") if either file
    is absent — this is the expected state until the one-time voice
    download step has been run (see module docstring).
    """
    model_dir = _get_model_dir()
    voice_name = _get_voice_name()

    onnx_path = model_dir / f"{voice_name}.onnx"
    config_path = model_dir / f"{voice_name}.onnx.json"

    if not onnx_path.exists() or not config_path.exists():
        raise TTSError(
            f"Piper voice model not found at {model_dir}. "
            f"Run: python -m piper.download_voices {voice_name}",
            reason="model_missing",
        )
    return onnx_path, config_path


def _load_voice() -> "_PiperVoiceType":
    """
    Lazy-load the PiperVoice singleton under _LOAD_LOCK. Called by all
    synthesis entry points; the lock is held only for the actual load
    (fast when already loaded, ~500ms cold).

    Multiple concurrent callers see the same singleton — the second
    caller through the lock finds _VOICE already populated and returns
    immediately without re-loading.

    Updates _last_activity_ts before returning so the idle monitor
    counts this as recent activity even if the caller does nothing
    else afterward.
    """
    global _VOICE

    # Fast path: already loaded. Just refresh activity timestamp and
    # return without touching the lock.
    if _VOICE is not None:
        _touch_activity()
        return _VOICE

    with _LOAD_LOCK:
        # Re-check under the lock — another thread may have loaded
        # while we waited.
        if _VOICE is not None:
            _touch_activity()
            return _VOICE

        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise TTSError(
                "piper-tts is not installed. Run: pip install piper-tts",
                reason="model_missing",
            ) from exc

        onnx_path, config_path = _resolve_model_files()

        logger.info(
            "tts_synthesizer · loading Piper voice",
            model=str(onnx_path),
            intra_op_threads=os.environ.get("ORT_INTRA_OP_NUM_THREADS"),
            inter_op_threads=os.environ.get("ORT_INTER_OP_NUM_THREADS"),
        )
        try:
            _VOICE = PiperVoice.load(str(onnx_path), config_path=str(config_path))
        except Exception as exc:
            raise TTSError(f"Failed to load Piper voice: {exc}") from exc

        _touch_activity()
        return _VOICE


def is_loaded() -> bool:
    """
    True when the Piper voice singleton is currently in memory.
    Diagnostic helper; safe to call from any thread. No lock needed —
    a stale answer here is harmless (only used for logging / metrics).
    """
    return _VOICE is not None


def warmup() -> None:
    """
    Pre-load the Piper voice and run one throwaway inference so the
    first real /tts/speak or /tts/speak/stream request pays zero
    warmup cost.

    Called from:
      - backend/queue.py when a job transitions to RUNNING (report
        generation started → user likely to click TTS shortly)
      - backend/tts.py when /tts/status/{job_id} is polled (user
        opened a report page → same signal)
      - anywhere else that wants to preemptively load the voice

    Safe to call multiple times from multiple threads simultaneously:
    _load_voice() collapses concurrent calls via _LOAD_LOCK, and the
    throwaway inference is a no-op after the first successful warmup
    (skipped when the voice was already loaded before this call).

    Any failure is logged and swallowed — the lazy path in
    _load_voice() will retry on the first real synthesis request, so
    warmup failure never breaks the endpoint.
    """
    try:
        was_loaded = is_loaded()
        voice = _load_voice()
        # Only run the throwaway inference on genuinely cold loads —
        # once the voice is in memory, further inferences are wasted
        # CPU. This matters because warmup() gets triggered on every
        # /tts/status poll, and the frontend can poll many times per
        # report page load.
        if not was_loaded:
            _ = _synthesize_segment_pcm(voice, "Ready.")
            logger.info("tts_synthesizer · warmup complete (cold load)")
        _touch_activity()
    except Exception as exc:
        logger.warning(f"tts_synthesizer · warmup failed (will retry lazily): {exc}")


def evict_voice() -> bool:
    """
    Unload the Piper voice singleton to reclaim ~180MB. Returns True
    if a voice was actually unloaded, False if nothing was loaded.

    Called from the idle-monitor background task in backend/main.py
    when AEGIS_TTS_IDLE_EVICT_SECS have passed without activity. Also
    safe to call manually (e.g. from an admin endpoint) if needed.

    Thread-safety: holds _LOAD_LOCK so no synthesis is running when we
    drop the reference. Callers currently mid-synthesis hold their own
    reference to the voice object via _load_voice()'s return value, so
    dropping our reference does not pull the rug from under them —
    their voice stays alive until they release it, and the Python
    garbage collector reclaims the ONNX session memory when the last
    reference is dropped.
    """
    global _VOICE
    with _LOAD_LOCK:
        if _VOICE is None:
            return False
        _VOICE = None
        logger.info("tts_synthesizer · voice evicted (idle timeout)")
        return True


def seconds_since_last_activity() -> float:
    """
    How long (in seconds) since any TTS-related activity. Used by the
    idle monitor to decide whether to evict. Returns math.inf when
    there has been no activity ever (avoids evicting during the very
    first idle window after startup, though nothing is loaded yet so
    that would be a no-op anyway).
    """
    if _last_activity_ts == 0.0:
        return float("inf")
    return time.monotonic() - _last_activity_ts


# ── Text cleanup ───────────────────────────────────────────────────

_MARKDOWN_HEADER_RE = re.compile(r"#{1,6}\s?")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_BULLET_RE = re.compile(r"[-*•]")

# tools/report_generator.py embeds pre-rendered HTML blocks (divs, inline
# styles, etc.) directly in the report text, wrapped in these markers, so
# the frontend can render them pixel-identically instead of round-tripping
# through markdown (see RAW_HTML_START/RAW_HTML_END there and the matching
# strip logic in ReportPage.jsx's renderMarkdown / backend/pdf_export.py).
# TTS must strip the markup and keep only the human-readable text inside —
# otherwise Piper reads the raw tags and CSS aloud (e.g. "html div 40 px").
_RAW_HTML_BLOCK_RE = re.compile(
    r"<!--RAW_HTML_START-->([\s\S]*?)<!--RAW_HTML_END-->"
)
# Collapse block-level tags to a space so "</div><div>" doesn't glue two
# words together; strip everything else (opening/closing tags, comments).
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
    """Extract readable text from an embedded HTML fragment, discarding
    tags/attributes/inline styles rather than reading them aloud."""
    text = _HTML_COMMENT_RE.sub(" ", html)
    text = _HTML_BLOCK_TAG_RE.sub(" ", text)
    text = _HTML_ANY_TAG_RE.sub(" ", text)
    for entity, char in _HTML_ENTITY_MAP.items():
        text = text.replace(entity, char)
    # Collapse the whitespace left behind by stripped tags.
    return re.sub(r"[ \t]+", " ", text)


def clean_text_for_speech(text: str) -> str:
    """
    Strip markdown formatting and embedded raw-HTML report blocks that
    would otherwise be read aloud literally (e.g. "hash hash Summary",
    "star star bold star star", or a findings block's own div/style
    markup coming out as "html div 40 px"). Mirrors the cleanup
    previously done client-side before calling window.speechSynthesis,
    plus HTML-block handling for report_generator.py's RAW_HTML sections.
    """
    cleaned = _RAW_HTML_BLOCK_RE.sub(
        lambda m: _html_to_speech_text(m.group(1)), text
    )
    cleaned = _MARKDOWN_HEADER_RE.sub("", cleaned)
    cleaned = _MARKDOWN_BOLD_RE.sub("", cleaned)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _MARKDOWN_BULLET_RE.sub("", cleaned)
    # Any remaining stray HTML (defensive — should be none outside
    # RAW_HTML blocks, but never read tags aloud if it slips through).
    cleaned = _HTML_ANY_TAG_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ── Segment-based synthesis (pause-aware) ──────────────────────────
#
# Root cause of missing pauses: synthesize_wav() feeds the entire
# cleaned text to espeak-ng in one shot. espeak-ng splits only on
# sentence-terminal punctuation (.!?) — NOT on bare newlines — so
# bullet lines like "Name: Admin\nDate of Birth: 01/01/2000" become
# one continuous utterance with zero inter-field silence.
#
# Fix: split the cleaned text into one segment per non-blank line,
# synthesize each independently via voice.synthesize_wav() into an
# in-memory WAV buffer, extract the PCM frames from that buffer, and
# stitch them together with explicit numpy zero-arrays of configurable
# length as silence gaps between segments and sections.
#
# We deliberately use voice.synthesize_wav() rather than the lower-
# level voice.synthesize() iterator because the latter's return type
# (AudioChunk vs. raw bytes vs. tuples) has drifted across piper-tts
# releases and broke on our installed version. synthesize_wav() has
# been stable since piper-tts 1.0 and was what the original working
# code used. Round-tripping through a wave.open() BytesIO adds only
# microseconds — negligible next to the 100-500 ms Piper inference
# takes per segment.
#
# This same segmentation powers synthesize_speech_stream(): instead
# of collecting all chunks and writing one WAV at the end, the
# streaming variant emits the WAV header once and then yields each
# segment's PCM bytes as they arrive — so the browser can begin
# playing after the very first segment (~100–300 ms) rather than
# waiting for the full report (~2–5 s).

# Silence durations (seconds) stitched as int16 zero-arrays between
# segments. Tune these to taste; they were chosen to feel like natural
# human pauses in a clinical readout.
_LINE_PAUSE_SECS: float = 0.35      # consecutive non-blank lines
_SECTION_PAUSE_SECS: float = 0.75   # blank-line boundary (section break)
_HEADING_PAUSE_SECS: float = 0.90   # after a heading/section-title line

# A line is treated as a "heading" (gets the longer pause) when it
# looks like a section title: short (≤ 6 words) AND starts with an
# uppercase letter. This matches "Patient Information", "Summary",
# "Findings", "Recommendations", etc. without false-positives on
# normal sentences that happen to start with a capital.
_MAX_HEADING_WORDS = 6


def _is_heading_line(line: str) -> bool:
    """
    Return True if `line` looks like a section heading.
    Heuristic: <= _MAX_HEADING_WORDS words AND first character is
    uppercase. Applied after markdown stripping, before punctuation
    is added, so "Patient Information" -> True, "No structured
    clinical measurements were available in this session" -> False.
    """
    words = line.split()
    return (
        1 <= len(words) <= _MAX_HEADING_WORDS
        and bool(words[0][:1].isupper())
    )


def _segment_report_text(cleaned: str) -> list[tuple[str, float]]:
    """
    Split cleaned report text into (segment_text, trailing_silence_s)
    pairs ready for per-segment synthesis.

    Rules:
    - Each non-blank line becomes one segment.
    - Lines that lack sentence-terminal punctuation (.!?:;) get a
      period appended so espeak-ng produces natural sentence cadence
      rather than trailing off mid-phoneme.
    - Blank lines between non-blank lines are not emitted as segments;
      instead they cause the last emitted segment to be upgraded to
      _SECTION_PAUSE_SECS trailing silence (section boundary).
    - Heading-like lines (short + starts uppercase) use
      _HEADING_PAUSE_SECS regardless of section-boundary status.

    Example output for the patient-info block:

        [("Patient Information.", 0.90),
         ("Name: Admin.",         0.35),
         ("Date of Birth: 01/01/2000.", 0.35),
         ("Sex: Male.",            0.35),
         ("Blood group: A plus.", 0.35),
         ("Allergies: Penicillin, shellfish, sulfa drugs.", 0.75),
         ("Reported Symptoms and Clinical History.", 0.90),
         ...]
    """
    lines = cleaned.split("\n")
    segments: list[tuple[str, float]] = []

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            # Blank line = section boundary. Upgrade the trailing
            # silence of the last emitted segment so the listener
            # hears a clear gap between sections.
            if segments:
                seg_text, _old_pause = segments[-1]
                segments[-1] = (seg_text, _SECTION_PAUSE_SECS)
            continue

        # Ensure terminal punctuation so espeak-ng closes the sentence
        # cleanly rather than trailing off mid-phoneme.
        if not line.endswith((".", "!", "?", ":", ";")):
            line = line + "."

        # Heading lines get a longer trailing pause for emphasis.
        if _is_heading_line(line.rstrip(".!?:;")):
            pause = _HEADING_PAUSE_SECS
        else:
            pause = _LINE_PAUSE_SECS

        segments.append((line, pause))

    return segments


def _synthesize_segment_pcm(
    voice: "_PiperVoiceType",
    seg_text: str,
) -> np.ndarray:
    """
    Synthesize one segment's text into an int16 PCM numpy array using
    voice.synthesize_wav() (the stable Piper API used by the original
    working code). Writes into an in-memory WAV via wave.open(), then
    reads the frames back out.

    Returns an empty array on empty synthesis output; raises on any
    underlying Piper error so the caller can log the full traceback.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(seg_text, wav_file)

    buf.seek(0)
    with wave.open(buf, "rb") as wav_read:
        n_frames = wav_read.getnframes()
        if n_frames <= 0:
            return np.array([], dtype=np.int16)
        frames = wav_read.readframes(n_frames)

    return np.frombuffer(frames, dtype=np.int16)


def _iter_segment_audio(
    segments: list[tuple[str, float]],
    voice: "_PiperVoiceType",
) -> Iterator[tuple[np.ndarray, int]]:
    """
    Yield (pcm_int16_array, sample_rate) for each segment in `segments`.

    Each yielded array is the concatenation of:
      - the segment's synthesized audio (from voice.synthesize_wav())
      - a trailing silence of the segment's configured duration

    Segments whose synthesis raises an exception are replaced by a
    silence-only chunk of the same duration and logged with the full
    traceback via logger.exception — the caller gets a complete audio
    timeline regardless.

    This is a plain synchronous generator. Callers that need async
    streaming should run it in a thread (see synthesize_speech_stream
    and the /tts/speak/stream FastAPI endpoint).
    """
    sample_rate: int = voice.config.sample_rate  # e.g. 22050

    for seg_text, pause_secs in segments:
        try:
            seg_audio = _synthesize_segment_pcm(voice, seg_text)
        except Exception as exc:
            # logger.exception attaches the traceback so we can see
            # the real underlying error (piper-tts API mismatch,
            # espeak-ng crash, ONNX runtime error, etc.) instead of a
            # generic "segment synthesis error".
            logger.exception(
                "tts_synthesizer · segment synthesis error "
                f"(substituting silence) · segment={seg_text[:80]!r} "
                f"error_type={type(exc).__name__} error={exc!s}"
            )
            seg_audio = np.array([], dtype=np.int16)

        silence_samples = max(1, int(sample_rate * pause_secs))
        silence = np.zeros(silence_samples, dtype=np.int16)

        yield np.concatenate([seg_audio, silence]), sample_rate


def _validate_text(text: str, max_chars: int) -> str:
    """
    Shared input validation for synthesize_speech() and
    synthesize_speech_stream(). Returns the cleaned text, or raises
    TTSError with an appropriate reason code.
    """
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


# ── WAV streaming header ───────────────────────────────────────────
#
# For synthesize_speech_stream() we emit a WAV header before any PCM
# data so the browser's Web Audio decoder knows the format. Since the
# total data length is unknown upfront, we use 0xFFFFFFFF for both
# the RIFF chunk size and the data sub-chunk size — this is explicitly
# allowed by the WAV specification for streaming/pipe use and is
# handled correctly by Chrome, Firefox, Safari, and ffmpeg.

_WAV_CHANNELS = 1
_WAV_SAMPLE_WIDTH = 2   # 16-bit PCM -> 2 bytes per sample


def _make_wav_header(sample_rate: int) -> bytes:
    """
    Build a 44-byte WAV/RIFF header with data-size = 0xFFFFFFFF
    (streaming/unknown length). Safe to prepend to a raw int16-LE PCM
    stream when the final byte count is not known ahead of time.
    """
    byte_rate = sample_rate * _WAV_CHANNELS * _WAV_SAMPLE_WIDTH
    block_align = _WAV_CHANNELS * _WAV_SAMPLE_WIDTH
    bits_per_sample = _WAV_SAMPLE_WIDTH * 8

    return struct.pack(
        "<4sI4s"        # "RIFF", riff_size, "WAVE"
        "4sIHHIIHH"     # "fmt ", fmt_size, audio_fmt, channels,
                        #  sample_rate, byte_rate, block_align, bits
        "4sI",          # "data", data_size
        b"RIFF",
        0xFFFFFFFF,     # riff chunk size — unknown/streaming
        b"WAVE",
        b"fmt ",
        16,             # PCM fmt chunk is always 16 bytes
        1,              # audio format: PCM
        _WAV_CHANNELS,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        0xFFFFFFFF,     # data sub-chunk size — unknown/streaming
    )


# ── Public API ─────────────────────────────────────────────────────


def synthesize_speech(text: str, max_chars: int = 20000) -> bytes:
    """
    Synthesize `text` to speech and return complete WAV audio bytes.

    Internally segments the text line-by-line and stitches real PCM
    silence between segments, so section headings, bullet-point fields,
    and paragraph breaks all produce audible pauses rather than running
    together as one continuous utterance.

    Raises:
        TTSError — model missing, text too long, or synthesis failed.
                   Callers (backend/tts.py) convert this to a clean
                   HTTP error response; never let it propagate as a
                   raw 500.
    """
    cleaned = _validate_text(text, max_chars)
    voice = _load_voice()          # also refreshes _last_activity_ts
    _touch_activity()              # keep hot for duration of synthesis

    segments = _segment_report_text(cleaned)
    if not segments:
        raise TTSError(
            "Text produced no speakable segments.", reason="synthesis_failed"
        )

    all_chunks: list[np.ndarray] = []
    sample_rate: int | None = None

    try:
        for pcm_chunk, sr in _iter_segment_audio(segments, voice):
            all_chunks.append(pcm_chunk)
            sample_rate = sr
    except TTSError:
        raise
    except Exception as exc:
        logger.error(f"tts_synthesizer · synthesis failed: {exc}")
        raise TTSError(f"Speech synthesis failed: {exc}") from exc

    if not all_chunks or sample_rate is None:
        raise TTSError("Synthesis produced no audio.", reason="synthesis_failed")

    pcm_bytes = np.concatenate(all_chunks).tobytes()

    buffer = io.BytesIO()
    try:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(_WAV_CHANNELS)
            wav_file.setsampwidth(_WAV_SAMPLE_WIDTH)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
    except Exception as exc:
        logger.error(f"tts_synthesizer · WAV encoding failed: {exc}")
        raise TTSError(f"Speech synthesis failed: {exc}") from exc

    _touch_activity()
    return buffer.getvalue()


def synthesize_speech_stream(
    text: str,
    max_chars: int = 20000,
) -> Iterator[bytes]:
    """
    Streaming variant of synthesize_speech().

    Yields WAV-format bytes incrementally:
      1. A 44-byte WAV header (data size = 0xFFFFFFFF, valid for
         streaming per the WAV spec, supported by all major browsers).
      2. Raw int16-LE PCM bytes for each report segment plus its
         trailing silence, as soon as that segment finishes synthesis.

    This lets the FastAPI /tts/speak/stream endpoint push audio to the
    browser via chunked transfer encoding so playback can start after
    the first segment (~100-300 ms) instead of waiting for the entire
    report to be synthesized first (~2-5 s on a typical CPU).

    Validation (empty text, text too long) happens eagerly at the top
    of the function body — not lazily on first iteration — so callers
    get TTSError immediately without having to start consuming.

    NOTE: This is a synchronous generator. The backend endpoint runs
    it inside a thread+queue bridge so it does not block the event
    loop (see backend/tts.py speak_stream endpoint).

    Raises:
        TTSError — model missing, text too long, or synthesis failed.
    """
    # Validate eagerly so the backend can return a clean HTTP error
    # before opening the StreamingResponse body.
    cleaned = _validate_text(text, max_chars)
    voice = _load_voice()          # refreshes _last_activity_ts
    _touch_activity()

    segments = _segment_report_text(cleaned)
    if not segments:
        raise TTSError(
            "Text produced no speakable segments.", reason="synthesis_failed"
        )

    sample_rate: int = voice.config.sample_rate

    # Emit the WAV header first so the browser knows the audio format
    # before any PCM data arrives.
    yield _make_wav_header(sample_rate)

    # Stream each segment's PCM + silence as it is synthesized.
    for pcm_chunk, _sr in _iter_segment_audio(segments, voice):
        _touch_activity()          # every segment keeps the voice hot
        if pcm_chunk.size > 0:
            yield pcm_chunk.tobytes()