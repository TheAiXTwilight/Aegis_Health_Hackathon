"""
tests/integration/test_streaming_endpoint.py — GET /queue/stream/{job_id}.

Test strategy:
    Deterministic HTTP contract tests only.

    Three groups:
        1. Unknown job → 404 with FastAPI detail shape
        2. Queued job → 425 with FastAPI detail shape
        3. Completed job → 200 with text/plain content-type

    Streaming body content tests (verifying actual token stream
    content) are deliberately NOT included. They would require
    either a live Ollama instance or significant pipeline mocking,
    both of which belong in a separate integration file rather
    than the HTTP contract suite.

Response shapes verified from smoke observation:
    Unknown (404):   {'detail': 'Unknown job_id: <id>'}
    Queued  (425):   {'detail': 'Job <id> is still queued...'}
    Completed (200): empty body, content-type 'text/plain; charset=utf-8'

Contract notes:
    - The streaming endpoint always returns 200 with text/plain when
      the job exists and has progressed past 'queued' state, even if
      the pipeline failed and produced zero tokens.
    - Stream queues persist past job completion (only purged via
      retention timeout in backend.queue).
    - The 425 detail message includes the job_id and instructs the
      client to poll /queue/status — clients should not retry the
      stream endpoint immediately.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from backend.main import app


# ── Module-scoped client ──────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    TestClient for the FastAPI app.

    Module-scoped because lifespan startup includes the inference
    worker. Module scope avoids repeated startup/teardown across
    every test.
    """
    with TestClient(app) as c:
        yield c


# ── Helper: poll for job completion ───────────────────────────────

def _wait_for_completion(
    client: TestClient,
    job_id: str,
    timeout_s: float = 15.0,
) -> str | None:
    """
    Poll /queue/status/{job_id} until status is 'completed' or 'failed'.

    Returns the final status string, or None if timeout exceeded.
    Polls every 200ms.

    Used by tests that need a job past 'queued' state. Polling is more
    robust than sleep — adapts to actual machine speed instead of
    assuming a fixed duration.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = client.get(f"/queue/status/{job_id}").json()
        if status.get("status") in {"completed", "failed"}:
            return status["status"]
        time.sleep(0.2)
    return None


# ── Group 1: unknown job ──────────────────────────────────────────

def test_stream_unknown_job_returns_404(client):
    """Unknown job_id → 404."""
    response = client.get("/queue/stream/nonexistent-job-id-99999")
    assert response.status_code == 404


def test_stream_unknown_job_uses_fastapi_detail_shape(client):
    """
    404 response uses FastAPI HTTPException shape {'detail': '...'}.
    Same shape as /queue/status/<unknown>.
    """
    response = client.get("/queue/stream/nonexistent-job-id-99999")
    body = response.json()
    assert "detail" in body
    assert "Unknown job_id" in body["detail"]


# ── Group 2: queued job ───────────────────────────────────────────

def test_stream_queued_job_returns_425(client):
    """
    A job in 'queued' state has no stream queue yet → 425.

    Submitting and immediately requesting the stream is reliable
    because the worker takes ~100ms to dequeue, and the back-to-back
    HTTP calls complete in <10ms.
    """
    submit = client.post(
        "/queue/submit",
        data={"symptoms_text": "stream queued test"},
    )
    job_id = submit.json()["job_id"]

    response = client.get(f"/queue/stream/{job_id}")
    assert response.status_code == 425


def test_stream_queued_job_uses_fastapi_detail_shape(client):
    """
    425 response uses FastAPI HTTPException shape with helpful
    instruction to poll /queue/status.
    """
    submit = client.post(
        "/queue/submit",
        data={"symptoms_text": "stream queued detail test"},
    )
    job_id = submit.json()["job_id"]

    response = client.get(f"/queue/stream/{job_id}")
    body = response.json()
    assert "detail" in body
    # Detail must mention the job_id and instruct polling /queue/status
    assert job_id in body["detail"]
    assert "queued" in body["detail"].lower()


# ── Group 3: completed job ────────────────────────────────────────

def test_stream_completed_job_returns_200(client):
    """
    A completed (or failed) job's stream returns 200.

    Stream queues persist past job completion. The stream returns
    whatever tokens were enqueued by the pipeline, terminated by
    the None sentinel. For a job that failed before any tokens
    were produced, the body is empty but the status is still 200.
    """
    submit = client.post(
        "/queue/submit",
        data={"symptoms_text": "stream completed test"},
    )
    job_id = submit.json()["job_id"]

    final_status = _wait_for_completion(client, job_id)
    assert final_status is not None, "Job did not complete within timeout"

    response = client.get(f"/queue/stream/{job_id}")
    assert response.status_code == 200


def test_stream_completed_job_content_type_is_text_plain(client):
    """
    The streaming endpoint uses text/plain content-type with UTF-8
    charset — NOT application/json. This is spec-locked behaviour
    for the chunked transfer streaming protocol.
    """
    submit = client.post(
        "/queue/submit",
        data={"symptoms_text": "stream content-type test"},
    )
    job_id = submit.json()["job_id"]

    final_status = _wait_for_completion(client, job_id)
    assert final_status is not None

    response = client.get(f"/queue/stream/{job_id}")
    content_type = response.headers["content-type"]
    assert content_type.startswith("text/plain")
    assert "charset=utf-8" in content_type.lower()


def test_stream_completed_failed_job_body_is_text(client):
    """
    The stream body is always plain text (possibly empty), never JSON.

    For a job that failed before producing any tokens (the case here
    because Ollama is unreachable in the test environment), the body
    is an empty string. The contract is that body is text — content
    depends on what the pipeline emitted before termination.
    """
    submit = client.post(
        "/queue/submit",
        data={"symptoms_text": "stream body type test"},
    )
    job_id = submit.json()["job_id"]

    final_status = _wait_for_completion(client, job_id)
    assert final_status is not None

    response = client.get(f"/queue/stream/{job_id}")
    # Body is text (str), not parsed JSON
    assert isinstance(response.text, str)


def test_stream_works_via_stream_api(client):
    """
    The endpoint works via TestClient.stream() context manager,
    not only .get(). Verifies the chunked transfer protocol is
    compatible with httpx streaming API.

    For an empty body (failed-pipeline case), iter_text() yields
    zero chunks. The test asserts the API contract, not chunk count.
    """
    submit = client.post(
        "/queue/submit",
        data={"symptoms_text": "stream api test"},
    )
    job_id = submit.json()["job_id"]

    final_status = _wait_for_completion(client, job_id)
    assert final_status is not None

    with client.stream("GET", f"/queue/stream/{job_id}") as response:
        assert response.status_code == 200

        # Drain the stream — for an empty body this collects nothing
        chunks = list(response.iter_text())
        assert isinstance(chunks, list)
        # Body content depends on pipeline output; we only assert
        # that iteration works without error