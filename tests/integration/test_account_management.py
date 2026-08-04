"""New integration tests for account-management routes.

The module uses real auth + password hashing, but an isolated in-memory DB.
Known P0 privacy/deletion requirements are represented as strict xfails until
application code is corrected.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth import verify_password
from app.db.models import User
from tests.integration._support import (
    install_db_override,
    login,
    make_session_factory,
    registration_payload,
)


@pytest.fixture
def db_session_factory():
    return make_session_factory()


@pytest.fixture
def client(db_session_factory, monkeypatch):
    from app.settings import settings
    monkeypatch.setattr(settings, "COOKIE_SECURE", False)

    from backend.account import router as account_router
    from backend.auth import router as auth_router

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(account_router)
    install_db_override(app, db_session_factory)
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_auth_state():
    from backend import auth as auth_module

    auth_module._login_attempts.clear()
    auth_module._password_reset_tokens.clear()
    yield
    auth_module._login_attempts.clear()
    auth_module._password_reset_tokens.clear()


def register(client: TestClient, **overrides):
    return client.post("/auth/register", json=registration_payload(**overrides))


def register_and_login(client: TestClient, **overrides):
    response = register(client, **overrides)
    assert response.status_code == 201, response.text
    credentials = login(
        client,
        email=overrides.get("email", "alice@example.com").lower(),
        password=overrides.get("password", "Password123!"),
    )
    return credentials


class TestChangeEmail:
    def test_change_email_requires_correct_password_and_normalizes_new_address(self, client, db_session_factory):
        register_and_login(client)

        wrong = client.put(
            "/account/email",
            json={"new_email": "new@example.com", "password": "wrong-password"},
        )
        assert wrong.status_code == 401

        changed = client.put(
            "/account/email",
            json={"new_email": "New@Example.COM", "password": "Password123!"},
        )
        assert changed.status_code == 200
        assert changed.json()["email"] == "new@example.com"

        with db_session_factory() as db:
            assert db.query(User).filter_by(email="new@example.com").count() == 1

    def test_change_email_rejects_same_and_duplicate_address(self, client):
        register_and_login(client)
        second = TestClient(client.app)
        register_and_login(
            second,
            email="bob@example.com",
            username="bob",
            security_answer="Rex",
        )

        same = client.put(
            "/account/email",
            json={"new_email": "alice@example.com", "password": "Password123!"},
        )
        assert same.status_code == 400

        duplicate = client.put(
            "/account/email",
            json={"new_email": "bob@example.com", "password": "Password123!"},
        )
        assert duplicate.status_code == 409


class TestSecurityQuestion:
    def test_authenticated_user_can_set_valid_security_question(self, client, db_session_factory):
        register_and_login(client)

        response = client.put(
            "/account/security-question",
            json={"security_question": "favorite_food", "security_answer": "Mango"},
        )
        assert response.status_code == 200
        assert response.json()["security_question"] == "What is your favorite food?"

        with db_session_factory() as db:
            user = db.query(User).filter_by(email="alice@example.com").one()
            assert user.security_answer_hash is not None
            assert verify_password("mango", user.security_answer_hash)

    def test_invalid_security_question_is_rejected(self, client):
        register_and_login(client)
        response = client.put(
            "/account/security-question",
            json={"security_question": "invented-question", "security_answer": "answer"},
        )
        assert response.status_code == 400


class TestAccountDeletion:
    def test_delete_account_removes_authenticated_user_when_correct_password_supplied(self, client, db_session_factory):
        register_and_login(client)

        response = client.request(
            "DELETE",
            "/account",
            json={"password": "Password123!"},
        )
        assert response.status_code == 200
        assert response.json() == {"deleted": True}

        with db_session_factory() as db:
            assert db.query(User).filter_by(email="alice@example.com").count() == 0
        assert client.get("/auth/me").status_code == 401


@pytest.mark.security_gate
@pytest.mark.xfail(strict=True, reason="Known P0: account deletion currently accepts missing/wrong password.")
def test_delete_account_rejects_missing_or_wrong_password_and_preserves_data(client, db_session_factory):
    register_and_login(client)

    missing = client.request("DELETE", "/account", json={})
    assert missing.status_code in {401, 403}

    wrong = client.request("DELETE", "/account", json={"password": "not-the-password"})
    assert wrong.status_code in {401, 403}

    with db_session_factory() as db:
        assert db.query(User).filter_by(email="alice@example.com").count() == 1


@pytest.mark.security_gate
@pytest.mark.xfail(strict=True, reason="Known P0: security answer remains in plaintext column after update.")
def test_security_answer_is_hash_only_at_rest(client, db_session_factory):
    register_and_login(client)
    assert client.put(
        "/account/security-question",
        json={"security_question": "favorite_food", "security_answer": "Mango"},
    ).status_code == 200

    with db_session_factory() as db:
        user = db.query(User).filter_by(email="alice@example.com").one()
        assert user.security_answer is None
        assert verify_password("mango", user.security_answer_hash)
