"""Authenticated health-profile persistence and isolation tests."""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth import get_current_user
from app.db import models  # noqa: F401
from app.db.models import User, UserProfile
from app.db.session import Base, get_db
from backend.account import router as account_router


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
def client(session_factory):
    with session_factory() as db:
        db.add_all([
            User(id="user-a", email="a@example.com", display_name="User A", password_hash="unused"),
            User(id="user-b", email="b@example.com", display_name="User B", password_hash="unused"),
        ])
        db.commit()

    app = FastAPI()
    app.include_router(account_router)
    active_user = {"id": "user-a"}

    def override_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    def override_user(db=Depends(get_db)):
        return db.get(User, active_user["id"])

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = override_user
    test_client = TestClient(app)
    test_client.active_user = active_user
    test_client.session_factory = session_factory
    return test_client


PROFILE = {
    "full_name": "Asha Patient",
    "date_of_birth": "12/03/1990",
    "sex": "Female",
    "blood_group": "B+",
    "weight_kg": 62.5,
    "height_cm": 165,
    "allergies": ["Penicillin"],
    "medical_conditions": ["Asthma"],
    "current_medications": ["Salbutamol inhaler"],
}


def test_new_user_profile_is_incomplete(client):
    response = client.get("/account/profile")
    assert response.status_code == 200
    body = response.json()
    assert body["profile_complete"] is False
    assert body["full_name"] == "User A"
    assert body["weight_kg"] is None


def test_save_and_load_profile(client):
    saved = client.put("/account/profile", json=PROFILE)
    assert saved.status_code == 200
    assert saved.json()["profile_complete"] is True
    assert saved.json()["allergies"] == ["Penicillin"]

    loaded = client.get("/account/profile")
    assert loaded.status_code == 200
    assert loaded.json()["date_of_birth"] == "12/03/1990"
    assert loaded.json()["current_medications"] == ["Salbutamol inhaler"]


def test_profile_updates_user_display_name(client):
    client.put("/account/profile", json=PROFILE)
    with client.session_factory() as db:
        assert db.get(User, "user-a").display_name == "Asha Patient"
        assert db.get(UserProfile, "user-a") is not None


def test_profiles_are_user_scoped(client):
    client.put("/account/profile", json=PROFILE)
    client.active_user["id"] = "user-b"

    other = client.get("/account/profile").json()
    assert other["profile_complete"] is False
    assert other["full_name"] == "User B"
    assert other["allergies"] == []


def test_invalid_dob_rejected(client):
    response = client.put("/account/profile", json={**PROFILE, "date_of_birth": "1990-03-12"})
    assert response.status_code == 422


def test_weight_is_optional(client):
    response = client.put("/account/profile", json={**PROFILE, "weight_kg": None})
    assert response.status_code == 200
    assert response.json()["weight_kg"] is None


def test_invalid_weight_rejected(client):
    response = client.put("/account/profile", json={**PROFILE, "weight_kg": 0})
    assert response.status_code == 422
