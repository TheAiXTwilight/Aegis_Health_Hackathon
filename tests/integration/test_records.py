"""Updated user-scoped report history and deletion tests.

Key update: the implemented dashboard endpoint is /api/dashboard, not the old
/dashboard route used by the original test file.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth import get_current_user
from app.db.models import HealthRecord, PipelineJobRow
from tests.integration._support import (
    install_db_override,
    make_session_factory,
    mutable_user_override,
    seed_user,
)


@pytest.fixture
def session_factory():
    return make_session_factory()


@pytest.fixture
def client(session_factory):
    from backend.dashboard import router as dashboard_router
    from backend.records import router as records_router

    seed_user(session_factory, user_id="alice-id", email="alice@example.test", display_name="Alice")
    seed_user(
        session_factory,
        user_id="bob-id",
        email="bob@example.test",
        display_name="Bob",
        security_answer="Rex",
    )

    now = datetime.now(timezone.utc)
    with session_factory() as db:
        for offset, owner_id, job_id, severity in [
            (3, "alice-id", "alice-job-1", "LOW"),
            (2, "alice-id", "alice-job-2", "MODERATE"),
            (1, "alice-id", "alice-job-3", "HIGH"),
            (1, "bob-id", "bob-job-1", "LOW"),
        ]:
            report = {"text": f"Report text for {job_id}", "severity": severity}
            db.add(
                HealthRecord(
                    id=f"record-{job_id}",
                    user_id=owner_id,
                    job_id=job_id,
                    severity=severity,
                    confidence=0.8,
                    report_json=json.dumps(report),
                    result_json=json.dumps({"report": report}),
                    created_at=now - timedelta(days=offset),
                )
            )
            db.add(
                PipelineJobRow(
                    job_id=job_id,
                    user_id=owner_id,
                    session_id=f"session-{job_id}",
                    status="completed",
                )
            )
        db.commit()

    app = FastAPI()
    app.include_router(records_router)
    app.include_router(dashboard_router)
    install_db_override(app, session_factory)
    selected = {"value": "alice-id"}
    mutable_user_override(app, session_factory, selected, get_current_user)
    test_client = TestClient(app)
    test_client.selected_user = selected
    return test_client


def test_history_contains_only_current_users_reports_and_uses_descending_order(client):
    response = client.get("/records")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert [record["job_id"] for record in body["records"]] == [
        "alice-job-3", "alice-job-2", "alice-job-1"
    ]
    assert all("alice" in record["report_text"] for record in body["records"])


def test_get_single_record_returns_owner_data(client):
    response = client.get("/records/record-alice-job-2")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == "alice-job-2"
    assert body["report_text"] == "Report text for alice-job-2"
    assert body["result_data"]["measurement_groups"] == {
        "critical": [],
        "under_observation": [],
        "normal": [],
        "counts": {"critical": 0, "under_observation": 0, "normal": 0},
    }


def test_get_single_record_hides_non_owner_record(client):
    response = client.get("/records/record-bob-job-1")
    assert response.status_code == 404


def test_new_user_starts_with_zero_reports(client, session_factory):
    seed_user(
        session_factory,
        user_id="new-id",
        email="new@example.test",
        display_name="New",
        security_answer="Nova",
    )
    client.selected_user["value"] = "new-id"
    response = client.get("/records")
    assert response.status_code == 200
    assert response.json() == {"records": [], "total": 0}


def test_delete_removes_owned_report_and_updates_api_dashboard_count(client):
    initial_dashboard = client.get("/api/dashboard")
    assert initial_dashboard.status_code == 200
    assert initial_dashboard.json()["total_records"] == 3
    assert len(initial_dashboard.json()["recent_records"]) == 3

    deleted = client.delete("/queue/alice-job-2")
    assert deleted.status_code == 204

    history = client.get("/records").json()
    assert history["total"] == 2
    assert {record["job_id"] for record in history["records"]} == {
        "alice-job-1", "alice-job-3"
    }
    assert client.get("/records/record-alice-job-2").status_code == 404

    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["total_records"] == 2
    assert len(dashboard.json()["recent_records"]) == 2


def test_delete_cannot_remove_another_users_report(client):
    response = client.delete("/queue/bob-job-1")
    assert response.status_code == 404

    client.selected_user["value"] = "bob-id"
    assert client.get("/records").json()["total"] == 1
