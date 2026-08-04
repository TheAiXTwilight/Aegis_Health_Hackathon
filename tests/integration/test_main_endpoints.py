"""
tests/integration/test_main_endpoints.py — POST /queue/submit and GET /queue/status.

Test strategy:
    Deterministic, fast, race-free tests only.

    Groups:
        1. Submission validation — rejects malformed/empty/oversized input
        2. Successful submit response — verifies the response contract
        3. Unknown status endpoint — verifies 404 + FastAPI detail shape
        4. Status payload shape for known job
        5. Queue full / duplicate session HTTP codes (via state injection)
        6. _status_code_for_queue_error pure function coverage

    Worker timing tests (queued/running/failed state observations) are
    deliberately NOT included here. They belong in a separate file
    that explicitly tests asynchronous pipeline behaviour.

Phase 4: ToolError.code assertions where applicable.
         _status_code_for_queue_error no longer uses substring matching.
         503 tested via direct queue state injection (HTTP layer).
         409 tested via pure function call (_status_code_for_queue_error).
         asyncio.get_event_loop() never used — sync tests stay sync.
"""

from __future__ import annotations

import json
from uuid import UUID

import pytest
import backend.queue as bq
from fastapi.testclient import TestClient

from backend.main import app, _status_code_for_queue_error
from schemas.errors import ToolError
from schemas.queue import PipelineJob
from schemas.state import AegisState


# ── Module-scoped client ──────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """TestClient without entering the production lifespan.

    These tests exercise synchronous input/status endpoint contracts. Starting
    the real application lifespan would also start the global inference worker,
    allowing it to race with queue-unit tests that intentionally manipulate
    queue internals. Worker behaviour is covered separately with injected
    fake pipelines.
    """
    return TestClient(app)


# ── Per-test queue state reset ────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_queue_state():
    """
    Reset queue module state before each test that manipulates it.

    Required for tests that plant jobs directly into queue internals
    (503 path). Safe to run even when no state was planted.
    """
    yield
    bq._job_store.clear()
    bq._job_queue.clear()
    bq._session_states.clear()
    bq._user_submissions.clear()


# ── Expected response shapes ──────────────────────────────────────

SUBMIT_KEYS = {
    "job_id",
    "session_id",
    "status",
    "submitted_at",
    "started_at",
    "completed_at",
    "error",
    "schema_version",
}


# ── Group 1: submission validation ────────────────────────────────

def test_submit_rejects_empty_form(client):
    """No inputs of any kind → 400."""
    response = client.post("/queue/submit", data={})
    assert response.status_code == 400


def test_submit_empty_form_returns_tool_error_shape(client):
    """400 response uses ToolError shape with 'reason' and 'fatal' keys."""
    response = client.post("/queue/submit", data={})
    body = response.json()
    assert "reason" in body
    assert "fatal" in body
    assert body["fatal"] is True


def test_submit_empty_form_code_is_missing_input(client):
    """Empty submission uses missing_input code."""
    response = client.post("/queue/submit", data={})
    body = response.json()
    assert body.get("code") == "missing_input"


def test_submit_rejects_only_empty_lists(client):
    """medications=[] and xray_findings=[] alone is not enough."""
    response = client.post(
        "/queue/submit",
        data={"medications": "[]", "xray_findings": "[]"},
    )
    assert response.status_code == 400


def test_submit_rejects_malformed_medications_json(client):
    response = client.post(
        "/queue/submit",
        data={"medications": "not json"},
    )
    assert response.status_code == 400
    body = response.json()
    assert body.get("code") == "invalid_input"
    assert "medications" in body["reason"].lower()


def test_submit_rejects_medications_not_a_list(client):
    """JSON parses but is not a list → 400."""
    response = client.post(
        "/queue/submit",
        data={"medications": '{"key": "value"}'},
    )
    assert response.status_code == 400
    assert response.json().get("code") == "invalid_input"


def test_submit_rejects_medications_with_non_string_items(client):
    """JSON list with non-string items → 400."""
    response = client.post(
        "/queue/submit",
        data={"medications": "[1, 2, 3]"},
    )
    assert response.status_code == 400
    assert response.json().get("code") == "invalid_input"


def test_submit_rejects_malformed_xray_findings_json(client):
    response = client.post(
        "/queue/submit",
        data={
            "symptoms_text": "headache",
            "xray_findings": "not json",
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body.get("code") == "invalid_input"
    assert "xray_findings" in body["reason"].lower()


def test_submit_rejects_too_many_medications(client):
    """MAX_MEDICATIONS = 50. 51 items rejected."""
    response = client.post(
        "/queue/submit",
        data={
            "symptoms_text": "test",
            "medications": json.dumps([f"drug{i}" for i in range(51)]),
        },
    )
    assert response.status_code == 400


def test_submit_rejects_oversized_audio_during_stream(client):
    """
    UploadTooLargeError raised mid-stream returns 400 with invalid_input code.
    Size is enforced during streaming — no full write before rejection.
    """
    from backend.uploads import MAX_AUDIO_BYTES

    oversized = b"x" * (MAX_AUDIO_BYTES + 1)

    response = client.post(
        "/queue/submit",
        files={"audio": ("big.wav", oversized, "audio/wav")},
        data={"symptoms_text": "headache"},
    )
    assert response.status_code == 400
    body = response.json()
    assert body.get("code") == "invalid_input"
    assert body["fatal"] is True



# ── Group 2: successful submit response ───────────────────────────

def test_submit_text_only_returns_200(client):
    """Minimal valid input: symptoms_text only."""
    response = client.post(
        "/queue/submit",
        data={"symptoms_text": "mild headache"},
    )
    assert response.status_code == 200


def test_submit_response_contains_required_keys(client):
    """
    Subset comparison (<=) allows additive evolution of the response.
    SUBMIT_KEYS documents the required floor.
    """
    response = client.post(
        "/queue/submit",
        data={"symptoms_text": "mild headache"},
    )
    body = response.json()
    assert SUBMIT_KEYS <= body.keys()


def test_submit_response_status_is_queued(client):
    """A fresh submission starts in 'queued' status."""
    response = client.post(
        "/queue/submit",
        data={"symptoms_text": "mild headache"},
    )
    assert response.json()["status"] == "queued"


def test_submit_response_identifiers_are_valid_uuids(client):
    """
    job_id and session_id are server-generated UUIDs.
    UUID() raises ValueError on any malformed identifier.
    """
    response = client.post(
        "/queue/submit",
        data={"symptoms_text": "mild headache"},
    )
    body = response.json()
    UUID(body["job_id"])
    UUID(body["session_id"])


def test_submit_response_started_and_completed_are_null(client):
    """At submit time, the job has not started or completed."""
    response = client.post(
        "/queue/submit",
        data={"symptoms_text": "mild headache"},
    )
    body = response.json()
    assert body["started_at"] is None
    assert body["completed_at"] is None
    assert body["error"] is None


def test_submit_response_schema_version_is_one_dot_one(client):
    """
    schema_version is exactly '1.1' — increments only on breaking changes.
    """
    response = client.post(
        "/queue/submit",
        data={"symptoms_text": "mild headache"},
    )
    assert response.json()["schema_version"] == "1.1"


def test_submit_generates_unique_session_ids(client):
    """Each submit generates a fresh session_id server-side."""
    r1 = client.post("/queue/submit", data={"symptoms_text": "first"})
    r2 = client.post("/queue/submit", data={"symptoms_text": "second"})
    assert r1.json()["session_id"] != r2.json()["session_id"]


def test_submit_accepts_exactly_50_medications(client):
    """MAX_MEDICATIONS = 50 is the inclusive upper bound."""
    response = client.post(
        "/queue/submit",
        data={
            "symptoms_text": "test",
            "medications": json.dumps([f"drug{i}" for i in range(50)]),
        },
    )
    assert response.status_code == 200


def test_submit_accepts_xray_findings_list(client):
    """The endpoint accepts a JSON list of clinician X-ray findings."""
    response = client.post(
        "/queue/submit",
        data={
            "symptoms_text": "test",
            "xray_findings": json.dumps(["Cardiomegaly"]),
        },
    )
    assert response.status_code == 200


def test_submit_accepts_xray_free_text(client):
    """xray_free_text form field is accepted alongside other inputs."""
    response = client.post(
        "/queue/submit",
        data={
            "symptoms_text": "test",
            "xray_free_text": "mild interstitial markings noted",
        },
    )
    assert response.status_code == 200


def test_submit_empty_form_tool_is_input_validation(client):
    """tool field uses TOOL_INPUT_VALIDATION constant, not a bare string."""
    from tools.tool_names import TOOL_INPUT_VALIDATION
    response = client.post("/queue/submit", data={})
    assert response.json()["tool"] == TOOL_INPUT_VALIDATION


def test_submit_queue_full_tool_is_queue(client):
    """tool field uses TOOL_QUEUE for queue-layer errors."""
    from tools.tool_names import TOOL_QUEUE

    for _ in range(bq.get_queue_max()):
        s = AegisState()
        j = PipelineJob(session_id=s.session_id)
        bq._job_store[j.job_id]         = j
        bq._session_states[s.session_id] = s
        bq._job_queue.append(j.job_id)

    response = client.post(
        "/queue/submit",
        data={"symptoms_text": "overflow"},
    )
    assert response.status_code == 503
    assert response.json()["tool"] == TOOL_QUEUE


# ── Group 3: unknown status ───────────────────────────────────────

def test_status_unknown_job_returns_404(client):
    """Unknown job_id → 404."""
    response = client.get("/queue/status/nonexistent-job-id-12345")
    assert response.status_code == 404


def test_status_unknown_job_uses_fastapi_detail_shape(client):
    """
    FastAPI HTTPException returns {'detail': '...'} — NOT the ToolError shape.
    """
    response = client.get("/queue/status/nonexistent-job-id-12345")
    body = response.json()
    assert "detail" in body
    assert "Unknown job_id" in body["detail"]


# ── Group 4: status payload shape for known job ───────────────────

def test_status_known_job_contains_required_fields(client):
    """
    After submit, status payload must contain all documented fields
    including dynamic pipeline-state fields.
    """
    submit = client.post(
        "/queue/submit",
        data={"symptoms_text": "headache"},
    )
    job_id = submit.json()["job_id"]

    payload = client.get(f"/queue/status/{job_id}").json()

    required = {
        "job_id",
        "session_id",
        "status",
        "submitted_at",
        "queue_position",
        "estimated_wait_seconds",
        "current_tool",
        "tools_run",
        "tools_failed",
        "step_durations_ms",
    }
    assert required <= payload.keys()


def test_status_known_job_id_matches(client):
    """Returned payload job_id matches the submitted job_id."""
    submit = client.post(
        "/queue/submit",
        data={"symptoms_text": "headache"},
    )
    job_id = submit.json()["job_id"]
    payload = client.get(f"/queue/status/{job_id}").json()
    assert payload["job_id"] == job_id


def test_status_tools_run_is_list(client):
    submit = client.post(
        "/queue/submit",
        data={"symptoms_text": "headache"},
    )
    job_id = submit.json()["job_id"]
    payload = client.get(f"/queue/status/{job_id}").json()
    assert isinstance(payload["tools_run"], list)
    assert isinstance(payload["tools_failed"], list)


def test_status_step_durations_is_dict(client):
    submit = client.post(
        "/queue/submit",
        data={"symptoms_text": "headache"},
    )
    job_id = submit.json()["job_id"]
    payload = client.get(f"/queue/status/{job_id}").json()
    assert isinstance(payload["step_durations_ms"], dict)


def test_status_queued_job_has_integer_queue_position(client):
    """
    Immediately after submit the job is likely still queued.
    If it is, queue_position must be a positive integer.
    """
    submit = client.post(
        "/queue/submit",
        data={"symptoms_text": "headache"},
    )
    job_id = submit.json()["job_id"]
    payload = client.get(f"/queue/status/{job_id}").json()

    if payload["status"] == "queued":
        assert isinstance(payload["queue_position"], int)
        assert payload["queue_position"] >= 1


# ── Group 5: 503 via HTTP (queue state injection) ─────────────────

def test_submit_returns_503_when_queue_full(client):
    """
    When the queue is at capacity, the next submit returns 503.

    Jobs are planted directly into queue internals to avoid timing
    dependency on the worker. The HTTP endpoint still goes through
    the real submit_job path and hits the queue_full guard.
    """

    for _ in range(bq.get_queue_max()):
        s = AegisState()
        j = PipelineJob(session_id=s.session_id)
        bq._job_store[j.job_id] = j
        bq._session_states[s.session_id] = s
        bq._job_queue.append(j.job_id)

    response = client.post(
        "/queue/submit",
        data={"symptoms_text": "overflow submission"},
    )
    assert response.status_code == 503
    body = response.json()
    assert body.get("code") == "queue_full"
    assert body["fatal"] is True


# ── Group 6: _status_code_for_queue_error pure function ───────────
#
# _status_code_for_queue_error is a pure synchronous function.
# It maps ToolError.code → HTTP status int.
# No async, no HTTP client, no event loop needed.
# Tested directly here so all branches are covered without
# the architectural constraint that the endpoint generates
# a new session_id for every submit call.

def test_status_code_queue_full_returns_503():
    err = ToolError(
        tool="queue",
        code="queue_full",
        reason="Queue full (10 jobs). Try again shortly.",
        fatal=True,
    )
    assert _status_code_for_queue_error(err) == 503


def test_status_code_duplicate_session_returns_409():
    err = ToolError(
        tool="queue",
        code="duplicate_session",
        reason="Session already has an active job: abc-123",
        fatal=True,
    )
    assert _status_code_for_queue_error(err) == 409


def test_status_code_invalid_input_returns_400():
    err = ToolError(
        tool="queue",
        code="invalid_input",
        reason="Only queued jobs can be submitted.",
        fatal=True,
    )
    assert _status_code_for_queue_error(err) == 400


def test_status_code_unknown_code_returns_500():
    """
    Unrecognised codes default to 500, not 400.
    This catches future programming errors where a new queue error
    code is introduced but not added to the mapping.
    """
    err = ToolError(
        tool="queue",
        code="some_future_unrecognised_code",
        reason="Something went wrong.",
        fatal=True,
    )
    assert _status_code_for_queue_error(err) == 500


def test_status_code_none_code_returns_500():
    """
    ToolError with code=None (pre-migration ToolError) defaults to 500.
    Ensures old un-coded errors are not silently treated as client errors.
    """
    err = ToolError(
        tool="queue",
        code=None,
        reason="Legacy error without structured code.",
        fatal=True,
    )
    assert _status_code_for_queue_error(err) == 500
