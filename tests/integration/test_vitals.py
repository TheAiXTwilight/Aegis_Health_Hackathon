"""Real endpoint tests for /vitals/checkin and /vitals/trends.

Replaces the earlier placeholder-only vitals tests with an in-memory DB,
actual router, user switching, persistence checks, and baseline assertions.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth import get_current_user
from app.db.models import VitalSnapshot
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
    from backend.vitals import router as vitals_router

    seed_user(
        session_factory,
        user_id="alice-id",
        email="alice@example.test",
        display_name="Alice",
    )
    seed_user(
        session_factory,
        user_id="bob-id",
        email="bob@example.test",
        display_name="Bob",
        security_answer="Rex",
    )

    app = FastAPI()
    app.include_router(vitals_router)
    install_db_override(app, session_factory)
    selected = {"value": "alice-id"}
    mutable_user_override(app, session_factory, selected, get_current_user)

    test_client = TestClient(app)
    test_client.selected_user = selected
    return test_client


def test_checkin_requires_at_least_one_vital(client, session_factory):
    response = client.post("/vitals/checkin", json={"notes": "no measurements"})
    assert response.status_code == 400

    with session_factory() as db:
        assert db.query(VitalSnapshot).count() == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"systolic_bp": 49},
        {"systolic_bp": 301},
        {"diastolic_bp": 19},
        {"diastolic_bp": 201},
        {"heart_rate": 19},
        {"heart_rate": 301},
        {"spo2": 49.9},
        {"spo2": 100.1},
        {"temperature_c": 29.9},
        {"temperature_c": 45.1},
        {"glucose_mg_dl": 19.9},
        {"glucose_mg_dl": 600.1},
        {"weight_kg": 9.9},
        {"weight_kg": 500.1},
        {"heart_rate": 72, "notes": "x" * 501},
    ],
)
def test_checkin_enforces_schema_boundaries(client, payload):
    response = client.post("/vitals/checkin", json=payload)
    assert response.status_code == 422


def test_checkin_persists_only_current_users_snapshot(client, session_factory):
    response = client.post(
        "/vitals/checkin",
        json={
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "heart_rate": 72,
            "spo2": 98.0,
            "temperature_c": 36.6,
            "glucose_mg_dl": 95.0,
            "weight_kg": 62.0,
            "notes": "Synthetic baseline check-in",
        },
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Vitals saved"

    with session_factory() as db:
        row = db.query(VitalSnapshot).one()
        assert row.user_id == "alice-id"
        assert row.heart_rate == 72
        assert row.notes == "Synthetic baseline check-in"


def test_trends_are_chronological_and_user_scoped(client, session_factory):
    now = datetime.now(timezone.utc)
    with session_factory() as db:
        db.add_all(
            [
                VitalSnapshot(
                    id="a-1", user_id="alice-id", heart_rate=60,
                    created_at=now - timedelta(days=3),
                ),
                VitalSnapshot(
                    id="a-2", user_id="alice-id", heart_rate=70,
                    created_at=now - timedelta(days=2),
                ),
                VitalSnapshot(
                    id="a-3", user_id="alice-id", heart_rate=80,
                    created_at=now - timedelta(days=1),
                ),
                VitalSnapshot(
                    id="b-1", user_id="bob-id", heart_rate=150,
                    created_at=now - timedelta(days=1),
                ),
            ]
        )
        db.commit()

    alice = client.get("/vitals/trends")
    assert alice.status_code == 200
    body = alice.json()
    assert body["sample_count"] == 3
    assert [item["id"] for item in body["timeline"]] == ["a-1", "a-2", "a-3"]
    heart_rate = next(item for item in body["baselines"] if item["vital"] == "heart rate")
    assert heart_rate["current"] == 80
    assert heart_rate["sample_size"] == 3
    assert heart_rate["insufficient_history"] is False
    assert heart_rate["z_score"] is not None

    client.selected_user["value"] = "bob-id"
    bob = client.get("/vitals/trends")
    assert bob.status_code == 200
    assert bob.json()["sample_count"] == 1
    assert [item["id"] for item in bob.json()["timeline"]] == ["b-1"]


def test_trends_mark_insufficient_history_for_field_with_less_than_three_values(client, session_factory):
    with session_factory() as db:
        db.add_all(
            [
                VitalSnapshot(id="a-1", user_id="alice-id", heart_rate=70, spo2=98.0),
                VitalSnapshot(id="a-2", user_id="alice-id", heart_rate=72),
                VitalSnapshot(id="a-3", user_id="alice-id", heart_rate=74),
            ]
        )
        db.commit()

    response = client.get("/vitals/trends")
    assert response.status_code == 200
    spo2 = next(item for item in response.json()["baselines"] if item["vital"] == "SpO2")
    assert spo2["sample_size"] == 1
    assert spo2["insufficient_history"] is True
    assert spo2["z_score"] is None


def test_vitals_router_rejects_unauthenticated_request():
    from backend.vitals import router as vitals_router

    app = FastAPI()
    app.include_router(vitals_router)
    client = TestClient(app)
    response = client.post("/vitals/checkin", json={"heart_rate": 72})
    assert response.status_code == 401
