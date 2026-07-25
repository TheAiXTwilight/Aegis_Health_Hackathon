"""
backend/ollama_manager.py — auto-start/stop the Ollama server as a
subprocess of the FastAPI process, so a separate `ollama serve` terminal
is no longer needed.

Behaviour:
    - On FastAPI startup, checks whether something is already answering
      at AEGIS_OLLAMA_BASE_URL/api/tags.
        - If yes: assumes an external `ollama serve` (or another instance)
          is already running and does nothing. We never kill a server we
          did not start ourselves.
        - If no: spawns `ollama serve` as a child process, and polls
          /api/tags until it responds (or a timeout is hit).
    - On FastAPI shutdown, if (and only if) this process started the
      Ollama server, it is terminated (SIGTERM, then SIGKILL if it
      doesn't exit in time).

This intentionally mirrors the existing lifespan pattern used for the
inference worker / TTS idle monitor in backend/main.py — a single
`await start_ollama()` / `await stop_ollama()` pair bracketing the
`yield`.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from typing import Optional

import httpx
from loguru import logger

from app.settings import settings

# Populated only if *this* process spawned ollama serve. None means
# either Ollama was already running, or the binary could not be found —
# either way we must not try to stop anything on shutdown.
_ollama_process: Optional[subprocess.Popen] = None

# How long to wait for `ollama serve` to come up and start answering
# /api/tags before giving up (it still keeps running in the background;
# we just stop blocking startup).
_STARTUP_TIMEOUT_SECS = 30.0
_POLL_INTERVAL_SECS = 0.5
_SHUTDOWN_GRACE_SECS = 5.0


def _ollama_base_url() -> str:
    return settings.AEGIS_OLLAMA_BASE_URL.rstrip("/")


async def _is_ollama_up() -> bool:
    """Return True if something is already answering /api/tags."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{_ollama_base_url()}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


async def start_ollama() -> None:
    """
    Ensure an Ollama server is running before the app starts serving
    requests. Idempotent and safe to call even if Ollama is already up
    (started by us in a previous call, or externally).
    """
    global _ollama_process

    if await _is_ollama_up():
        logger.info("Ollama already reachable at {url} — not spawning a new instance",
                     url=_ollama_base_url())
        return

    ollama_bin = shutil.which("ollama")
    if ollama_bin is None:
        logger.warning(
            "`ollama` binary not found on PATH — skipping auto-start. "
            "Install Ollama (https://ollama.com) or start it manually with "
            "`ollama serve` if LLM-backed features are needed."
        )
        return

    logger.info("Starting `ollama serve` as a managed subprocess")
    try:
        _ollama_process = subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            # New session so Ctrl+C / signals to the parent uvicorn
            # process don't race the child before we get a chance to
            # terminate it ourselves in stop_ollama().
            start_new_session=(sys.platform != "win32"),
        )
    except Exception:
        logger.exception("Failed to spawn `ollama serve`")
        _ollama_process = None
        return

    # Poll until it's answering, or until we time out (server keeps
    # trying to boot in the background either way).
    elapsed = 0.0
    while elapsed < _STARTUP_TIMEOUT_SECS:
        if await _is_ollama_up():
            logger.info("Ollama server is up ({url}), pid={pid}",
                        url=_ollama_base_url(), pid=_ollama_process.pid)
            return
        if _ollama_process.poll() is not None:
            logger.error(
                "`ollama serve` exited early (code={code}) during startup",
                code=_ollama_process.returncode,
            )
            _ollama_process = None
            return
        await asyncio.sleep(_POLL_INTERVAL_SECS)
        elapsed += _POLL_INTERVAL_SECS

    logger.warning(
        "Timed out waiting for Ollama to become reachable after {secs}s "
        "— continuing startup anyway; it may still be loading",
        secs=_STARTUP_TIMEOUT_SECS,
    )


async def stop_ollama() -> None:
    """
    Stop the Ollama server, but only if this process is the one that
    started it. Never touches an externally-managed instance.
    """
    global _ollama_process

    if _ollama_process is None:
        return

    if _ollama_process.poll() is not None:
        # Already exited on its own.
        _ollama_process = None
        return

    logger.info("Stopping managed `ollama serve` (pid={pid})", pid=_ollama_process.pid)
    _ollama_process.terminate()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_ollama_process.wait), timeout=_SHUTDOWN_GRACE_SECS
        )
    except asyncio.TimeoutError:
        logger.warning("`ollama serve` did not exit in time — killing")
        _ollama_process.kill()
        with contextlib_suppress():
            _ollama_process.wait(timeout=_SHUTDOWN_GRACE_SECS)

    _ollama_process = None
    logger.info("Ollama server stopped")


def contextlib_suppress():
    import contextlib
    return contextlib.suppress(Exception)