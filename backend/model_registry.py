"""
backend/model_registry.py — Centralised model loading, prewarm, and memory guard.

Provides:
    ModelRegistry — singleton that manages model lifecycle on Jetson

Responsibilities:
    - Memory guard: prevents OOM by checking available RAM before heavy tools
    - Ollama prewarm: runs a single-token generate to pull model into GPU/RAM
    - RAG prewarm: runs a minimal embedding call so first user isn't cold
    - One-heavy-path-at-a-time enforcement

Uses httpx for Ollama calls (matching the existing backend/health.py pattern).
All methods are async-safe and non-blocking where possible.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path

import httpx
from loguru import logger

from app.settings import settings


# ── Memory helpers ──────────────────────────────────────────────
def _parse_meminfo_kb() -> dict[str, int]:
    """Parse /proc/meminfo into a dict of {key: value_in_kB}."""
    try:
        text = Path("/proc/meminfo").read_text()
    except (FileNotFoundError, PermissionError):
        return {}
    result: dict[str, int] = {}
    for line in text.splitlines():
        if ":" in line:
            key, rest = line.split(":", 1)
            parts = rest.strip().split()
            if parts:
                try:
                    result[key.strip()] = int(parts[0])
                except ValueError:
                    pass
    return result


def memory_available_mb() -> int | None:
    """Return available system memory in MB, or None if unreadable."""
    mem = _parse_meminfo_kb()
    if "MemAvailable" in mem:
        return mem["MemAvailable"] // 1024

    # Fallback: MemFree + Buffers + Cached (older kernels)
    free = mem.get("MemFree", 0)
    buffers = mem.get("Buffers", 0)
    cached = mem.get("Cached", 0)
    total = free + buffers + cached
    return total // 1024 if total > 0 else None


# ── Model Registry ──────────────────────────────────────────────
class ModelRegistry:
    """
    Central registry for AI model lifecycle.

    Enforces:
        - Memory headroom checks before heavy tools
        - One inference path at a time
        - Prewarming of Ollama + RAG

    Singleton pattern — import `model_registry` from this module.
    """

    def __init__(self, memory_floor_mb: int = 900) -> None:
        self.memory_floor_mb: int = memory_floor_mb
        self._ollama_model: str = settings.AEGIS_OLLAMA_MODEL
        self._ollama_url: str = settings.AEGIS_OLLAMA_BASE_URL.rstrip("/")
        self._prewarmed: bool = False
        self._active_lock: asyncio.Lock = asyncio.Lock()

    # ── Memory guard ─────────────────────────────────────────
    def assert_memory_headroom(self, component: str) -> None:
        """
        Check available memory before a heavy tool starts.
        Raises RuntimeError if below the floor — caller should
        handle by delaying, retrying, or falling back.
        """
        available = memory_available_mb()
        if available is None:
            # Can't read memory — skip guard (e.g. macOS dev)
            logger.debug("Memory check skipped — /proc/meminfo not available")
            return

        if available < self.memory_floor_mb:
            raise RuntimeError(
                f"memory_pressure: {component} delayed — "
                f"only {available} MB available (floor {self.memory_floor_mb} MB)"
            )

        logger.debug(
            "Memory headroom OK",
            component=component,
            available_mb=available,
            floor_mb=self.memory_floor_mb,
        )

    # ── Inference guard ──────────────────────────────────────
    async def acquire(self, component: str) -> None:
        """
        Acquire the single-inference lock.
        Blocks until the current inference path is free.
        Must be released with release().
        """
        await self._active_lock.acquire()
        logger.info("Inference lock acquired", component=component)

    def release(self, component: str) -> None:
        """Release the single-inference lock."""
        try:
            self._active_lock.release()
            logger.info("Inference lock released", component=component)
        except RuntimeError:
            logger.warning("Inference lock released when not held", component=component)

    @property
    def is_active(self) -> bool:
        """True if an inference path is currently running."""
        return self._active_lock.locked()

    # ── Prewarm ──────────────────────────────────────────────
    @property
    def is_prewarmed(self) -> bool:
        return self._prewarmed

    async def prewarm(self) -> dict[str, bool]:
        """
        Prewarm Ollama and RAG embedding paths.

        Returns a dict of {component: success} so callers
        can report which prewarm steps succeeded.

        Idempotent — calling twice is safe.
        """
        self.assert_memory_headroom("prewarm")
        results: dict[str, bool] = {}

        # 1. Ollama prewarm — single-token generate
        results["ollama"] = await self._ollama_prewarm()

        # 2. RAG embedding prewarm
        results["rag_embed"] = await self._rag_embed("prewarm health check")

        if all(results.values()):
            self._prewarmed = True

        logger.info("Prewarm complete", results=results)
        return results

    async def _ollama_prewarm(self) -> bool:
        """
        Send a minimal generate request to Ollama so the model
        is loaded into GPU/RAM before the first user request.
        Uses num_predict=1 to minimise token generation time.
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self._ollama_url}/api/generate",
                    json={
                        "model": self._ollama_model,
                        "prompt": "health check",
                        "stream": False,
                        "options": {
                            "num_predict": 1,
                            "num_ctx": settings.OLLAMA_NUM_CTX,
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                logger.info(
                    "Ollama prewarmed",
                    model=self._ollama_model,
                    eval_count=data.get("eval_count", 0),
                )
                return True
        except Exception as exc:
            logger.warning("Ollama prewarm failed — model may be cold on first request",
                          error=str(exc))
            return False

    async def _rag_embed(self, text: str) -> bool:
        """
        Run a minimal embedding call so the RAG pipeline
        is not cold on the first user request.

        Calls tools.medical_rag_search._embed() directly — the same
        function the runtime query path uses (see that module's
        docstring: it deliberately keeps its own embedding copy
        rather than importing rag/embed.py, so runtime has zero
        import-time dependency on the build package). Calling it here
        for prewarm also has the side effect of loading and caching
        the ONNX session at startup instead of on the first request.

        _embed() is synchronous (plain numpy/onnxruntime, no I/O),
        so it's run directly rather than awaited.
        """
        try:
            from tools.medical_rag_search import _embed  # deferred import

            vector = _embed(text)
            if vector is None:
                logger.warning(
                    "RAG embed prewarm returned None — ONNX model or "
                    "tokenizer likely unavailable"
                )
                return False
            return True
        except Exception as exc:
            logger.warning("RAG embed prewarm failed", error=str(exc))
            return False

    # ── Generation helper ────────────────────────────────────
    async def ollama_generate(
        self,
        prompt: str,
        *,
        num_predict: int | None = None,
        num_ctx: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
        timeout: float = 120.0,
    ) -> dict | httpx.Response:
        """
        Send a generate request to Ollama.

        By default uses settings for num_ctx, num_predict, temperature.
        Pass explicit values to override for specific call sites.

        Returns the parsed JSON response for non-streaming,
        or the httpx Response for streaming (caller reads .aiter_lines()).
        """
        options: dict = {}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        else:
            options["num_ctx"] = settings.OLLAMA_NUM_CTX
        if num_predict is not None:
            options["num_predict"] = num_predict
        else:
            options["num_predict"] = settings.OLLAMA_NUM_PREDICT
        if temperature is not None:
            options["temperature"] = temperature
        else:
            options["temperature"] = settings.OLLAMA_TEMPERATURE

        client = httpx.AsyncClient(timeout=timeout)
        resp = await client.post(
            f"{self._ollama_url}/api/generate",
            json={
                "model": self._ollama_model,
                "prompt": prompt,
                "stream": stream,
                "options": options,
            },
        )
        if not stream:
            resp.raise_for_status()
            return resp.json()
        return resp


    # ── Model check ──────────────────────────────────────────
    async def is_model_available(self) -> bool:
        """
        Check whether the configured Ollama model is present.

        Uses the same /api/tags endpoint as backend/health.py.
        Cached in the caller — this does raw HTTP each time.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._ollama_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                models = data.get("models", [])
                base = self._ollama_model.split(":")[0]
                for m in models:
                    name = m.get("name", "")
                    if name.split(":")[0] == base:
                        return True
                return False
        except Exception:
            return False


# ── Prewarm endpoint router ─────────────────────────────────────
from fastapi import APIRouter, HTTPException

admin_router = APIRouter(prefix="/admin", tags=["admin"])


@admin_router.post("/prewarm")
async def prewarm_endpoint():
    """
    Trigger model prewarm manually.

    Used by deployment scripts and demo setup.
    Returns which components succeeded.
    """
    try:
        results = await model_registry.prewarm()
        return {
            "prewarmed": model_registry.is_prewarmed,
            "results": results,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@admin_router.post("/demo-reset")
async def demo_reset_endpoint():
    """Reset demo accounts, purge queues, uploads, checkpoints, and LRU cache."""
    try:
        from backend.cache import result_cache
        from backend.queue import _job_store, _session_states, _job_streams, _job_queue
        from app.db.session import SessionLocal
        from app.db.seed import seed_demo_users

        result_cache.clear()
        _job_store.clear()
        _session_states.clear()
        _job_streams.clear()
        _job_queue.clear()

        shutil.rmtree("/tmp/aegis_uploads", ignore_errors=True)
        Path("/tmp/aegis_uploads").mkdir(parents=True, exist_ok=True)
        shutil.rmtree("/tmp/aegis_checkpoint", ignore_errors=True)
        Path("/tmp/aegis_checkpoint").mkdir(parents=True, exist_ok=True)

        with SessionLocal() as db:
            seed_demo_users(db)

        return {"reset": True, "message": "Demo environment successfully reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset demo: {e}")


# ── Singleton ───────────────────────────────────────────────────
model_registry = ModelRegistry(memory_floor_mb=settings.OLLAMA_MEMORY_FLOOR_MB)
