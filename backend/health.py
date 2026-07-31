"""
backend/health.py — /health endpoint.

Returns lightweight system status with a STABLE response shape.

Response shape is stable across releases. Fields whose backing
infrastructure does not yet exist return honest placeholder values:

    False — boolean subsystem reports "not loaded / not ready"
    null  — numeric measurement not yet available

GPU detection via nvidia-smi / Jetson device nodes — torch never imported.
CPU/RAM readings are APPLICATION-scoped (this backend process only, via
/proc/self/*) — system-wide CPU/RAM probes were removed; this endpoint
no longer reports whole-machine or whole-GPU utilization.

model_loaded probe:
    HTTP GET to OLLAMA_BASE_URL/api/tags — checks whether aegis-llama
    is present in the Ollama model registry. Cached for 60 seconds.
    Model name matched by splitting on ":" and comparing the base name
    exactly to _MODEL_TAG — prevents false positives from similarly-named
    variants such as aegis-llama-old or aegis-llama-test.
    Returns False on any failure — health endpoint must never raise.

rag_index_ready probe (Phase 3):
    Calls tools.medical_rag_search.is_index_ready().
    Returns True when ChromaDB collection exists and contains documents,
    OR when FAISS index file is loaded and non-empty.
    Returns False on any failure — health endpoint must never raise.
    Cached for _PROBE_TTL_S (60 seconds).

Both slow probes are re-probed at most once per minute per the spec.
Non-blocking — never acquires the inference lock.
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Any

import httpx
from fastapi import APIRouter
from fastapi.responses import Response
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

OLLAMA_BASE_URL = os.getenv("AEGIS_OLLAMA_BASE_URL", "http://172.17.0.1:11434")
_MODEL_TAG      = os.getenv("AEGIS_OLLAMA_MODEL", "llama3.2:1b")
_PROBE_TTL_S    = 60.0


# ── TTL cache state ───────────────────────────────────────────────
# Two independent caches — model and RAG probed separately.
# Initialised to 0.0 so the first request always triggers a real
# probe (time.monotonic() - 0.0 always exceeds _PROBE_TTL_S).

_model_last_probe: float = 0.0
_model_cached:     bool  = False

_rag_last_probe:   float = 0.0
_rag_cached:       bool  = False

_proc_cpu_last_total: float | None = None  # this process's own jiffies
_proc_cpu_last_sys:   float | None = None  # whole-system jiffies at same sample
_CLK_TCK = 100  # standard Linux default (getconf CLK_TCK); used to convert
                 # /proc/self/stat jiffy counts into seconds


# ── Application (this process) probes ─────────────────────────────
# This is the ONLY CPU/RAM reading exposed by this endpoint — scoped to
# the current backend process only, via /proc/self/* (no psutil
# dependency, Linux/Jetson-first with a macOS dev fallback). Reports
# "how much is *this app* using", not whole-machine/whole-GPU load.
#
# Caveat: aegis-llama runs inside Ollama's own separate process, not
# this one — these probes intentionally do NOT include Ollama's RAM/CPU
# usage. If "the application" should be read as "everything AegisHealth
# runs, including the model", these numbers will look artificially low
# during inference. Flagged as an open question in the handoff rather
# than guessed at, since the two answers are legitimately different
# metrics tracking different things.

def _app_cpu_percent() -> float | None:
    """
    Return CPU utilisation percentage for THIS backend process only,
    as a percentage of one core (can exceed 100% on multi-threaded work,
    matching `top`/`ps` convention — NOT normalized to total cores).

    Probe order:
        1. Linux/Jetson: /proc/self/stat (process jiffies) vs
           /proc/stat (system jiffies used only as the denominator for
           this process's share — no system-wide value is exposed).
        2. macOS dev fallback: `ps -o %cpu= -p <pid>`.

    First Linux call returns 0.0 (needs two samples for a delta).
    """
    global _proc_cpu_last_total, _proc_cpu_last_sys

    try:
        with open("/proc/self/stat", "r", encoding="utf-8") as f:
            # Fields are space-separated; comm (field 2) can itself
            # contain spaces/parens, so split after the last ')'.
            stat_line = f.read()
        after_comm = stat_line.rsplit(")", 1)[-1].split()
        # After splitting off "pid (comm)", remaining fields start at
        # original field 3 (state) = after_comm[0]. utime=14, stime=15
        # in the full /proc/pid/stat numbering → index 11, 12 here.
        utime = float(after_comm[11])
        stime = float(after_comm[12])
        proc_total = utime + stime

        with open("/proc/stat", "r", encoding="utf-8") as f:
            first = f.readline().strip().split()
        sys_total = sum(float(v) for v in first[1:]) if first and first[0] == "cpu" else None

        if sys_total is None:
            raise ValueError("no system cpu line")

        if _proc_cpu_last_total is None or _proc_cpu_last_sys is None:
            _proc_cpu_last_total = proc_total
            _proc_cpu_last_sys = sys_total
            return 0.0

        proc_delta = proc_total - _proc_cpu_last_total
        sys_delta = sys_total - _proc_cpu_last_sys

        _proc_cpu_last_total = proc_total
        _proc_cpu_last_sys = sys_total

        if sys_delta <= 0:
            return 0.0

        num_cores = os.cpu_count() or 1
        used_percent = (proc_delta / sys_delta) * 100.0 * num_cores
        return round(max(0.0, used_percent), 1)
    except Exception:
        pass

    # macOS development fallback.
    try:
        result = subprocess.run(
            ["ps", "-o", "%cpu=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            return round(float(result.stdout.strip()), 1)
    except Exception:
        pass

    return None


def _app_memory_mb() -> int | None:
    """
    Return resident memory (RSS) used by THIS backend process, in MB.

    Probe order:
        1. Linux/Jetson: VmRSS from /proc/self/status.
        2. macOS dev fallback: `ps -o rss= -p <pid>` (KB → MB).

    Returns None if both probes fail.
    """
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb // 1024
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip()) // 1024
    except Exception:
        pass

    return None


# ── Hardware probes ───────────────────────────────────────────────

def _gpu_available() -> bool:
    """
    Detect GPU presence without importing torch.

    Check order:
        1. Jetson device nodes  (/dev/nvhost-gpu, /dev/nvidiactl)
        2. nvidia-smi query
        3. tegrastats (Jetson fallback)

    Returns False on any failure — health endpoint must never raise.
    """
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


# ── Slow probes with TTL cache ────────────────────────────────────

def _probe_model_loaded() -> bool:
    """
    Check whether aegis-llama is present in the Ollama model registry.

    HTTP GET to OLLAMA_BASE_URL/api/tags. Matches by splitting the model
    name on ":" and comparing the base name exactly to _MODEL_TAG.
    Prevents false positives from aegis-llama-old, aegis-llama-test, etc.

    Cached for _PROBE_TTL_S. Returns False on any failure.
    """
    global _model_last_probe, _model_cached

    now = time.monotonic()
    if now - _model_last_probe < _PROBE_TTL_S:
        return _model_cached

    result = False
    try:
        response = httpx.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=3.0,
        )
        if response.status_code == 200:
            data   = response.json()
            models = data.get("models", [])
            result = any(
                m.get("name", "").split(":")[0] == _MODEL_TAG
                for m in models
            )
    except Exception:
        logger.warning("health · model_loaded probe failed · returning False")

    _model_last_probe = now
    _model_cached     = result
    return result


def _probe_rag_index_ready() -> bool:
    """
    Check whether the RAG index is ready (Phase 3 — real probe).

    Delegates to tools.medical_rag_search.is_index_ready(), which
    checks ChromaDB collection count > 0 (primary) or FAISS ntotal > 0
    (fallback). Import is deferred so a missing dependency does not
    crash the health endpoint — returns False gracefully on any error.

    Cached for _PROBE_TTL_S. Returns False on any failure.
    """
    global _rag_last_probe, _rag_cached

    now = time.monotonic()
    if now - _rag_last_probe < _PROBE_TTL_S:
        return _rag_cached

    result = False
    try:
        from tools.medical_rag_search import is_index_ready  # deferred import
        result = is_index_ready()
    except Exception as exc:
        logger.warning(
            "health · rag_index_ready probe failed · returning False",
            error=str(exc),
        )

    _rag_last_probe = now
    _rag_cached     = result
    return result


# ── Health endpoint ───────────────────────────────────────────────

@router.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status":           "ok",
        "model_loaded":     _probe_model_loaded(),
        "rag_index_ready":  _probe_rag_index_ready(),
        "gpu_available":    _gpu_available(),
        # Application-scoped (this backend process only, via /proc/self/*).
        # Does not include Ollama's own separate process (aegis-llama
        # runs outside this process). System-wide cpu_percent /
        # memory_used_mb / memory_total_mb were removed — this endpoint
        # now reports only what AegisHealth's own process is using.
        "app_cpu_percent":    _app_cpu_percent(),
        "app_memory_used_mb": _app_memory_mb(),
        "inference_active": is_inference_active(),
        "queue_depth":      get_queue_depth(),
        "queue_max":        get_queue_max(),
        "jobs_completed":   get_jobs_completed_today(),
        "jobs_failed":      get_jobs_failed_today(),
        "avg_duration_s":   get_average_pipeline_duration_s(),
    }


@router.get("/readyz")
def readyz() -> dict[str, Any]:
    """Readiness endpoint for deployment checks."""
    try:
        from backend.model_registry import model_registry
        prewarmed = model_registry.is_prewarmed
    except Exception:
        prewarmed = False

    return {
        "status": "ready",
        "backend_online": True,
        "prewarmed": prewarmed,
        "model_loaded": _probe_model_loaded(),
        "rag_index_ready": _probe_rag_index_ready(),
    }


@router.get("/metrics")
def metrics() -> Response:
    """Prometheus text format metrics."""
    from backend.cache import result_cache

    depth = get_queue_depth()
    max_size = get_queue_max()
    active = 1 if is_inference_active() else 0
    avg_dur = get_average_pipeline_duration_s() or 0.0
    comp = get_jobs_completed_today()
    failed = get_jobs_failed_today()
    hits = result_cache.hits
    misses = result_cache.misses

    lines = [
        "# HELP aegis_queue_depth Current queue depth.",
        "# TYPE aegis_queue_depth gauge",
        f"aegis_queue_depth {depth}",
        "# HELP aegis_queue_max Maximum adaptive queue size.",
        "# TYPE aegis_queue_max gauge",
        f"aegis_queue_max {max_size}",
        "# HELP aegis_inference_active Whether inference worker is currently active (1 or 0).",
        "# TYPE aegis_inference_active gauge",
        f"aegis_inference_active {active}",
        f"aegis_jobs_in_flight {active}",
        "# HELP aegis_avg_duration_seconds Average completed job duration in seconds.",
        "# TYPE aegis_avg_duration_seconds gauge",
        f"aegis_avg_duration_seconds {avg_dur:.2f}",
        "# HELP aegis_jobs_completed_today Total completed jobs today.",
        "# TYPE aegis_jobs_completed_today counter",
        f"aegis_jobs_completed_today {comp}",
        "# HELP aegis_jobs_failed_today Total failed jobs today.",
        "# TYPE aegis_jobs_failed_today counter",
        f"aegis_jobs_failed_today {failed}",
        "# HELP aegis_pipeline_cache_hits_total Total result cache hits.",
        "# TYPE aegis_pipeline_cache_hits_total counter",
        f"aegis_pipeline_cache_hits_total {hits}",
        "# HELP aegis_pipeline_cache_misses_total Total result cache misses.",
        "# TYPE aegis_pipeline_cache_misses_total counter",
        f"aegis_pipeline_cache_misses_total {misses}",
        "# HELP aegis_app_cpu_percent This process's own CPU utilization percent (of one core; not normalized to total cores).",
        "# TYPE aegis_app_cpu_percent gauge",
        f"aegis_app_cpu_percent {_app_cpu_percent() or 0.0}",
        "# HELP aegis_app_memory_used_mb This process's own resident memory (RSS) in MB.",
        "# TYPE aegis_app_memory_used_mb gauge",
        f"aegis_app_memory_used_mb {_app_memory_mb() or 0}",
        "# HELP aegis_auth_failures_total Total authentication failures.",
        "# TYPE aegis_auth_failures_total counter",
        "aegis_auth_failures_total 0",
        "# HELP aegis_rule_validator_total RuleValidator safety agreement counts.",
        "# TYPE aegis_rule_validator_total counter",
        'aegis_rule_validator_total{status="agreement"} 0',
        'aegis_rule_validator_total{status="warning"} 0',
        'aegis_rule_validator_total{status="override"} 0',
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")