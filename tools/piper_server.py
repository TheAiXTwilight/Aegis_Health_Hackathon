"""
tools/piper_server.py — Standalone Piper TTS worker process.

Spawned and managed by tools/tts_synthesizer.py (the client). Keeping Piper
in its own process means:
  - the FastAPI backend never imports onnxruntime/piper -> no in-process
    CPU/RAM spike, no crash risk to the backend;
  - killing this process returns 100% of its RAM to the OS (no ONNX-arena
    leak that plagues in-process unload);
  - ONNX threads are capped here (env set below, before any ONNX import).

Protocol (local HTTP on 127.0.0.1):
    GET  /health                  -> {"ok": <voice loaded?>}
    POST /synthesize              -> full WAV bytes  (body: {"text","max_chars"})
    POST /synthesize?stream=1     -> streaming WAV (44B header + PCM, chunked)

A pidfile is written on start; on startup any stale worker holding it is
killed first, so backend restarts never leak orphan workers.

PIPER_STUB=1 swaps the real voice for a silence-generating stub so the whole
server (load + segment + WAV + HTTP) can be exercised without piper-tts or
the model installed — used by the sandbox test.

Run: python3 -m tools.piper_server --host 127.0.0.1 --port 9880
"""
from __future__ import annotations

import argparse
import io
import json
import os
import socket
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ── Cap ONNX/OpenMP threads BEFORE importing anything that pulls onnxruntime ──
# PiperVoice import triggers onnxruntime; without this cap ORT grabs one thread
# per core. Must run before the piper/numpy imports below.
_THREADS = os.environ.get("AEGIS_PIPER_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", _THREADS)
os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _THREADS)

import numpy as np
from loguru import logger

# Pure text/WAV helpers live in the client module (no voice state there).
from tools.tts_synthesizer import (
    TTSError,
    _WAV_CHANNELS,
    _WAV_SAMPLE_WIDTH,
    _make_wav_header,
    _segment_report_text,
    _validate_text,
)
from app.settings import settings


# ── Voice loading ─────────────────────────────────────────────────
_VOICE = None
_SYNTH_LOCK = threading.Lock()


class _StubVoice:
    """Silence-generating stand-in used when PIPER_STUB=1 (tests)."""

    class _Cfg:
        sample_rate = 22050

    config = _Cfg()

    def synthesize_wav(self, text, wav_file):
        n = int(self.config.sample_rate * 0.15)  # ~0.15s silence per segment
        with wave.open(wav_file, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.config.sample_rate)
            w.writeframes(b"\x00\x00" * n)


def _resolve_model_files() -> tuple[Path, Path]:
    model_dir = Path(settings.AEGIS_TTS_MODEL_DIR)
    voice_name = settings.AEGIS_TTS_VOICE_NAME
    onnx = model_dir / f"{voice_name}.onnx"
    cfg = model_dir / f"{voice_name}.onnx.json"
    if not onnx.exists() or not cfg.exists():
        raise TTSError(
            f"Piper voice model not found at {model_dir}. "
            f"Run: python -m piper.download_voices {voice_name}",
            reason="model_missing",
        )
    return onnx, cfg


def _load_voice():
    global _VOICE
    if _VOICE is not None:
        return _VOICE
    if os.environ.get("PIPER_STUB") == "1":
        logger.warning("piper_server · using STUB voice (PIPER_STUB=1)")
        _VOICE = _StubVoice()
        return _VOICE
    try:
        from piper import PiperVoice
    except ImportError as exc:
        raise TTSError(
            "piper-tts is not installed. Run: pip install piper-tts",
            reason="model_missing",
        ) from exc
    onnx, cfg = _resolve_model_files()
    logger.info("piper_server · loading Piper voice", model=str(onnx))
    _VOICE = PiperVoice.load(str(onnx), config_path=str(cfg))
    logger.info("piper_server · voice loaded")
    return _VOICE


# ── Per-segment synthesis (voice-dependent) ───────────────────────
def _synthesize_segment_pcm(voice, seg_text: str) -> np.ndarray:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(seg_text, wav_file)
    buf.seek(0)
    with wave.open(buf, "rb") as r:
        n = r.getnframes()
        if n <= 0:
            return np.array([], dtype=np.int16)
        return np.frombuffer(r.readframes(n), dtype=np.int16)


def _iter_segment_audio(segments, voice):
    sample_rate = voice.config.sample_rate
    for seg_text, pause_secs in segments:
        try:
            seg_audio = _synthesize_segment_pcm(voice, seg_text)
        except Exception as exc:
            logger.exception(
                f"piper_server · segment failed; substituting silence · {seg_text[:80]!r}"
            )
            seg_audio = np.array([], dtype=np.int16)
        silence = np.zeros(max(1, int(sample_rate * pause_secs)), dtype=np.int16)
        yield np.concatenate([seg_audio, silence]), sample_rate


def _build_full_wav(text: str, max_chars: int) -> bytes:
    cleaned = _validate_text(text, max_chars)
    voice = _load_voice()
    segments = _segment_report_text(cleaned)
    if not segments:
        raise TTSError("Text produced no speakable segments.", reason="synthesis_failed")
    chunks, sr = [], None
    with _SYNTH_LOCK:
        for pcm, sr in _iter_segment_audio(segments, voice):
            chunks.append(pcm)
    if not chunks or sr is None:
        raise TTSError("Synthesis produced no audio.", reason="synthesis_failed")
    pcm_bytes = np.concatenate(chunks).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(_WAV_CHANNELS)
        w.setsampwidth(_WAV_SAMPLE_WIDTH)
        w.setframerate(sr)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


# ── HTTP handler ──────────────────────────────────────────────────
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence default stderr access log
        pass

    def _send_bytes(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code, obj):
        self._send_bytes(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"ok": _VOICE is not None})
        else:
            self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/synthesize"):
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode() or "{}")
            text = payload.get("text", "")
            max_chars = int(payload.get("max_chars") or settings.AEGIS_TTS_MAX_CHARS)
        except Exception:
            self._send_json(400, {"ok": False, "error": "bad json body"})
            return

        stream = "stream=1" in self.path
        try:
            if not stream:
                wav = _build_full_wav(text, max_chars)
                self._send_bytes(200, wav, "audio/wav")
            else:
                self._stream_wav(text, max_chars)
        except TTSError as e:
            self._send_json(500, {"ok": False, "error": str(e), "reason": e.reason})
        except Exception as e:
            logger.exception("piper_server · synth error")
            self._send_json(500, {"ok": False, "error": str(e)})

    def _stream_wav(self, text, max_chars):
        cleaned = _validate_text(text, max_chars)
        voice = _load_voice()
        segments = _segment_report_text(cleaned)
        if not segments:
            raise TTSError("Text produced no speakable segments.", reason="synthesis_failed")
        sr = voice.config.sample_rate

        # Disable Nagle so small PCM chunks are pushed out immediately instead
        # of waiting ~200ms for an ACK — keeps time-to-first-audio low.
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(b: bytes):
            self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n")

        try:
            chunk(_make_wav_header(sr))
            self.wfile.flush()  # push the WAV header now so playback can init
            with _SYNTH_LOCK:
                for pcm, _sr in _iter_segment_audio(segments, voice):
                    if pcm.size > 0:
                        chunk(pcm.tobytes())
                        self.wfile.flush()  # stream each segment as soon as it's ready
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            logger.exception("piper_server · stream error")
            try:
                self.wfile.write(b"0\r\n\r\n")
            except Exception:
                pass


# ── pidfile / stale-worker cleanup ────────────────────────────────
def _pidfile() -> Path:
    return Path(os.environ.get("AEGIS_PIPER_PIDFILE", "/tmp/aegis_piper_worker.pid"))


def _kill_stale() -> None:
    pf = _pidfile()
    if not pf.exists():
        return
    try:
        pid = int(pf.read_text().strip())
    except Exception:
        pf.unlink(missing_ok=True)
        return
    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False
    if alive:
        try:
            os.kill(pid, 15)  # SIGTERM the stale worker
            logger.info("piper_server · killed stale worker", pid=pid)
        except OSError:
            pass
    pf.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--port", type=int, default=int(os.environ.get("AEGIS_PIPER_PORT", "9880"))
    )
    args = ap.parse_args()

    _kill_stale()

    # Eager load so /health reflects true readiness and the first request is instant.
    try:
        _load_voice()
    except Exception as exc:
        logger.error("piper_server · voice load failed at startup: {}", exc)

    _pidfile().write_text(str(os.getpid()))
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    logger.info(
        "piper_server · listening",
        host=args.host, port=args.port,
        stub=(os.environ.get("PIPER_STUB") == "1"),
        voice_loaded=(_VOICE is not None),
    )
    try:
        server.serve_forever()
    finally:
        _pidfile().unlink(missing_ok=True)


if __name__ == "__main__":
    main()
