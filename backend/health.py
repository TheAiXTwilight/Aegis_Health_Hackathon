"""
backend/health.py — /health endpoint.

Returns lightweight system status with a STABLE response shape.

Response shape is stable across releases. Fields whose backing
infrastructure does not yet exist return honest placeholder values:

    False — boolean subsystem reports "not loaded / not ready"
    null  — numeric measurement not yet available

GPU detection via nvidia-smi / Jetson device nodes — torch never imported.
Memory via nvidia-smi GPU memory, falling back to /proc/meminfo system RAM.

Slow probes (Ollama model_loaded, RAG rag_index_ready) remain placeholders
until Week 2 infrastructure lands. When they do, add cached re-probes here
per the spec's "re-probed at most once per minute" rule — do NOT add them
before their backing infra exists.

Non-blocking — never acquires the inference lock.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any

from fastapi import APIRouter
from loguru import logger

from backend.queue import (
    get_average_pipeline_duration_s,
    get_jobs_completed_today,
    get_jobs_failed_today,
    get_queue_depth,
    get_queue_max,
    is_inference_active,
)


router = APIRouter()


# ── Hardware probes ───────────────────────────────────────────────
# Ported from files2/health.py. Jetson-aware: checks device nodes
# before falling back to nvidia-smi and tegrastats.
# torch is never imported — avoids loading a large framework just
# for a health check.

def _gpu_available() -> bool:
    """
    Detect GPU presence without importing torch.

    Check order:
        1. Jetson device nodes  (/dev/nvhost-gpu, /dev/nvidiactl)
        2. nvidia-smi query
        3. tegrastats (Jetson fallback)

    Returns False on any failure — health endpoint must never raise.
    """
    # Jetson exposes these device nodes when the GPU is active.
    if os.path.exists("/dev/nvhost-gpu") or os.path.exists("/dev/nvidiactl"):
        return True

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["tegrastats", "--interval", "1", "--stop"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass

    return False


def _memory_mb() -> tuple[int | None, int | None]:
    """
    Return (used_mb, total_mb).

    Probe order:
        1. nvidia-smi GPU memory  (preferred on Jetson — reflects VRAM used
           by Ollama, which is the memory we actually care about)
        2. /proc/meminfo system RAM  (fallback — reflects total process memory)

    Returns (None, None) if both probes fail — callers receive null in the
    health response rather than a misleading value.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) == 2:
                return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        pass

    try:
        with open("/proc/meminfo") as f:
            lines = {
                line.split(":")[0]: int(line.split()[1])
                for line in f
                if ":" in line
            }
        total_kb = lines.get("MemTotal", 0)
        avail_kb = lines.get("MemAvailable", 0)
        if total_kb > 0:
            total_mb = total_kb // 1024
            used_mb  = (total_kb - avail_kb) // 1024
            return used_mb, total_mb
    except Exception:
        pass

    logger.warning("health · both memory probes failed · returning null")
    return None, None


# ── Router ────────────────────────────────────────────────────────

@router.get("/health")
async def health() -> dict[str, Any]:
    """
    Return current system status.

    Implemented (real values, available now):
        system_status                — "ok" until system-level checks land
        inference_active             — True iff worker holds the lock
        gpu_available                — Jetson device node / nvidia-smi / tegrastats
        memory_used_mb               — nvidia-smi VRAM or /proc/meminfo used
        memory_total_mb              — nvidia-smi VRAM total or /proc/meminfo total
        queue_depth                  — current FIFO depth
        queue_max                    — MAX_QUEUE_SIZE
        average_pipeline_duration_s  — rolling avg of last 10, null if < 3
        jobs_completed_today         — in-memory counter, resets on restart
        jobs_failed_today            — in-memory counter, resets on restart

    Placeholder (honest False/null until backing infra lands):
        model_loaded    — False    (Ollama warmup probe, Week 2)
        rag_index_ready — False    (ChromaDB/FAISS probe, Week 2)
    """
    used_mb, total_mb = _memory_mb()

    return {
        "system_status":               "ok",
        "inference_active":            is_inference_active(),
        "model_loaded":                False,          # Week 2: real Ollama probe
        "gpu_available":               _gpu_available(),
        "memory_used_mb":              used_mb,
        "memory_total_mb":             total_mb,
        "rag_index_ready":             False,          # Week 2: ChromaDB/FAISS probe
        "queue_depth":                 get_queue_depth(),
        "queue_max":                   get_queue_max(),
        "average_pipeline_duration_s": get_average_pipeline_duration_s(),
        "jobs_completed_today":        get_jobs_completed_today(),
        "jobs_failed_today":           get_jobs_failed_today(),
    }