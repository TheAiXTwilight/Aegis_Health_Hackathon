"""Deterministic tests for /queue/stream/{job_id}.

The old version started the full application lifespan and waited for Ollama/Piper
and the real pipeline. These tests construct queue state directly and test the
HTTP streaming contract without GPU, models, sleeps, or network services.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.queue as queue_module
from backend.streaming import router as streaming_router
from schemas.queue import JobStatus, PipelineJob
from schemas.state import AegisState


@pytest.fixture(autouse=True)
def reset_queue_state():
    queue_module._job_store.clear()
    queue_module._job_queue.clear()
    queue_module._job_streams.clear()
    queue_module._session_states.clear()
    queue_module._completed_durations.clear()
    queue_module._user_submissions.clear()
    yield
    queue_module._job_store.clear()
    queue_module._job_queue.clear()
    queue_module._job_streams.clear()
    queue_module._session_states.clear()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(streaming_router)
    return TestClient(app)


def register_job(*, status: JobStatus, tokens: list[str] | None = None) -> PipelineJob:
    state = AegisState()
    job = PipelineJob(session_id=state.session_id, user_id="alice-id")
    job.status = status
    queue_module._job_store[job.job_id] = job
    queue_module._session_states[state.session_id] = state

    if tokens is not None:
        stream: asyncio.Queue[str | None] = asyncio.Queue()
        for token in tokens:
            stream.put_nowait(token)
        stream.put_nowait(None)
        queue_module._job_streams[job.job_id] = stream
    return job


def test_unknown_job_returns_404_with_fastapi_error_shape(client):
    response = client.get("/queue/stream/not-a-real-job")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_queued_job_returns_425_until_stream_exists(client):
    job = register_job(status=JobStatus.QUEUED)
    response = client.get(f"/queue/stream/{job.job_id}")
    assert response.status_code == 425
    assert "still queued" in response.json()["detail"].lower()


@pytest.mark.parametrize("status", [JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.FAILED])
def test_available_stream_returns_plain_text_tokens_in_order(client, status):
    job = register_job(status=status, tokens=["first ", "second", " third"])

    response = client.get(f"/queue/stream/{job.job_id}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; charset=utf-8")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.text == "first second third"


def test_job_with_no_stream_queue_returns_404_after_stream_is_unavailable(client):
    job = register_job(status=JobStatus.COMPLETED)
    response = client.get(f"/queue/stream/{job.job_id}")
    assert response.status_code == 404
    assert "no longer available" in response.json()["detail"].lower()


@pytest.mark.security_gate
@pytest.mark.xfail(strict=True, reason="Known P0: streaming route has no current-user/owner authorization.")
def test_stream_requires_authenticated_owner(client):
    job = register_job(status=JobStatus.COMPLETED, tokens=["private triage report"])
    response = client.get(f"/queue/stream/{job.job_id}")
    assert response.status_code == 401
