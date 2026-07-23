"""User-scoped report history, deletion, and dashboard-count tests."""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth import get_current_user
from app.db import models  # noqa: F401
from app.db.models import HealthRecord, PipelineJobRow, User
from app.db.session import Base, get_db
from backend.dashboard import router as dashboard_router
from backend.records import router as records_router


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@pytest.fixture
def seeded(session_factory):
    with session_factory() as db:
        alice = User(
            id="alice-id",
            email="alice@example.com",
            display_name="Alice",
            password_hash="unused",
        )
        bob = User(
            id="bob-id",
            email="bob@example.com",
            display_name="Bob",
            password_hash="unused",
        )
        db.add_all([alice, bob])
        db.flush()

        for owner, job_id, severity in (
            (alice, "alice-job-1", "LOW"),
            (alice, "alice-job-2", "MEDIUM"),
            (alice, "alice-job-3", "HIGH"),
            (bob, "bob-job-1", "LOW"),
        ):
            report = {"text": f"Report text for {job_id}", "severity": severity}
            db.add(
                HealthRecord(
                    user_id=owner.id,
                    job_id=job_id,
                    severity=severity,
                    confidence=0.8,
                    report_json=json.dumps(report),
                    result_json=json.dumps({"report": report}),
                )
            )
            db.add(
                PipelineJobRow(
                    job_id=job_id,
                    user_id=owner.id,
                    session_id=f"session-{job_id}",
                    status="completed",
                )
            )
        db.commit()

    return {"alice_id": "alice-id", "bob_id": "bob-id"}


@pytest.fixture
def client(session_factory, seeded):
    app = FastAPI()
    app.include_router(records_router)
    app.include_router(dashboard_router)
    current_user_id = {"value": seeded["alice_id"]}

    def override_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    def override_user():
        with session_factory() as db:
            return db.get(User, current_user_id["value"])

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = override_user
    test_client = TestClient(app)
    test_client.current_user_id = current_user_id
    return test_client


def test_history_contains_only_current_users_reports(client):
    response = client.get("/records")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert {record["job_id"] for record in body["records"]} == {
        "alice-job-1",
        "alice-job-2",
        "alice-job-3",
    }
    assert all(record["report_text"].startswith("Report text for alice") for record in body["records"])


def test_new_user_starts_with_zero_reports(client, session_factory):
    with session_factory() as db:
        db.add(
            User(
                id="new-user-id",
                email="new@example.com",
                display_name="New User",
                password_hash="unused",
            )
        )
        db.commit()

    client.current_user_id["value"] = "new-user-id"
    response = client.get("/records")
    assert response.status_code == 200
    assert response.json() == {"records": [], "total": 0}


def test_delete_removes_owned_report_and_updates_dashboard_count(client):
    initial_dashboard = client.get("/dashboard").json()
    assert initial_dashboard["total_records"] == 3
    assert len(initial_dashboard["recent_records"]) == 3

    delete_response = client.delete("/queue/alice-job-2")
    assert delete_response.status_code == 204

    history = client.get("/records").json()
    assert history["total"] == 2
    assert {record["job_id"] for record in history["records"]} == {
        "alice-job-1",
        "alice-job-3",
    }

    dashboard = client.get("/dashboard").json()
    assert dashboard["total_records"] == 2
    assert len(dashboard["recent_records"]) == 2


def test_delete_cannot_remove_another_users_report(client):
    response = client.delete("/queue/bob-job-1")
    assert response.status_code == 404

    client.current_user_id["value"] = "bob-id"
    assert client.get("/records").json()["total"] == 1
