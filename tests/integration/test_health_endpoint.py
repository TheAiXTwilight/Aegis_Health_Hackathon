"""
tests/integration/test_health_endpoint.py — GET /health (baseline).

Tests the health endpoint public API contract using FastAPI's TestClient.

Test strategy:
    Baseline tests only — assert on the actual response shape returned
    when probes operate without intervention. No probe mocking, no
    TTL cache manipulation. Every assertion is backed by observed
    behaviour from the smoke test.

    Probe-propagation tests (verifying that successful probe values
    propagate to the response) are deferred until backend/health.py
    is inspected for cache behaviour. The two slow probes
    (_probe_model_loaded, _probe_rag_index_ready) use a 60-second TTL
    cache that would defeat naive patching.

Verified response shape (from smoke test):
    {
        "status":           "ok",
        "model_loaded":     bool,
        "rag_index_ready":  bool,
        "gpu_available":    bool,
        "app_cpu_percent":    float | null,
        "app_memory_used_mb": int | null,
        "inference_active": bool,
        "queue_depth":      int,
        "queue_max":        int,
        "jobs_completed":   int,
        "jobs_failed":      int,
        "avg_duration_s":   float | null,
    }
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import backend.queue as bq
from app.settings import settings
from backend.main import app


# ── Per-test queue state reset ────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_queue_state():
    """Reset global queue counters so baseline health tests start fresh."""
    bq._job_store.clear()
    bq._job_queue.clear()
    bq._session_states.clear()
    bq._user_submissions.clear()
    bq._completed_durations.clear()
    bq._jobs_completed_today = 0
    bq._jobs_failed_today = 0
    yield
    bq._job_store.clear()
    bq._job_queue.clear()
    bq._session_states.clear()
    bq._user_submissions.clear()
    bq._completed_durations.clear()
    bq._jobs_completed_today = 0
    bq._jobs_failed_today = 0


# ── Module-scoped client ──────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    TestClient for the FastAPI app.

    Module-scoped — TestClient as context manager triggers the lifespan
    handlers, which start the inference worker. Module scope avoids
    repeated worker startup/teardown.
    """
    with TestClient(app) as c:
        yield c


# ── Expected response shape ───────────────────────────────────────

EXPECTED_KEYS = {
    "status",
    "model_loaded",
    "rag_index_ready",
    "gpu_available",
    "app_cpu_percent",
    "app_memory_used_mb",
    "inference_active",
    "queue_depth",
    "queue_max",
    "jobs_completed",
    "jobs_failed",
    "avg_duration_s",
}


# ── Status and content type ───────────────────────────────────────

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_json(client):
    response = client.get("/health")
    assert response.headers["content-type"].startswith("application/json")


# ── Response shape ────────────────────────────────────────────────

def test_health_response_contains_exactly_expected_keys(client):
    """The /health response must contain the documented set of keys."""
    data = client.get("/health").json()
    assert set(data.keys()) == EXPECTED_KEYS


def test_health_status_is_ok(client):
    """status field is always 'ok' until system-level checks land."""
    data = client.get("/health").json()
    assert data["status"] == "ok"


# ── Field types ───────────────────────────────────────────────────

def test_health_boolean_fields_are_bool(client):
    data = client.get("/health").json()
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["rag_index_ready"], bool)
    assert isinstance(data["gpu_available"], bool)
    assert isinstance(data["inference_active"], bool)


def test_health_queue_fields_are_int(client):
    data = client.get("/health").json()
    assert isinstance(data["queue_depth"], int)
    assert isinstance(data["queue_max"], int)
    assert isinstance(data["jobs_completed"], int)
    assert isinstance(data["jobs_failed"], int)


def test_health_app_resource_fields_are_correct_types(client):
    """
    app_cpu_percent is float when the probe succeeds, None on the very
    first delta-sampled call in a fresh process, or on probe failure.
    app_memory_used_mb is int when the probe succeeds, None on failure.
    Both types are valid responses per the endpoint contract.
    """
    data = client.get("/health").json()
    assert isinstance(data["app_cpu_percent"], (int, float, type(None)))
    assert isinstance(data["app_memory_used_mb"], (int, type(None)))


# ── Spec-locked values ────────────────────────────────────────────

def test_health_queue_max_is_within_configured_bounds(client):
    """queue_max is adaptive but stays within configured min/max."""
    queue_max = client.get("/health").json()["queue_max"]
    assert settings.AEGIS_QUEUE_MIN_SIZE <= queue_max <= settings.AEGIS_QUEUE_MAX_SIZE


# ── Baseline state (no jobs run yet) ──────────────────────────────

def test_health_inference_active_false_when_idle(client):
    """No job running → inference_active=False."""
    data = client.get("/health").json()
    assert data["inference_active"] is False


def test_health_queue_depth_zero_when_empty(client):
    """No jobs queued → queue_depth=0."""
    data = client.get("/health").json()
    assert data["queue_depth"] == 0


def test_health_avg_duration_none_on_fresh_run(client):
    """avg_duration_s requires 3+ completions. None on fresh run."""
    data = client.get("/health").json()
    assert data["avg_duration_s"] is None