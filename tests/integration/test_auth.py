"""Refreshed integration coverage for the current authentication contract.

This replaces the older test_auth.py helper that registered users without the
now-required security question/answer fields. Tests use an isolated SQLite DB;
they never touch the project's local database.

Security-gate tests are strict xfails because the reviewed implementation still
returns plaintext security answers and enumerates recovery accounts. Remove the
xfail decorators when the application fixes land.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth import verify_password
from app.db.models import RefreshToken, User
from app.db.session import get_db
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
def app(db_session_factory, monkeypatch) -> FastAPI:
    from app.settings import settings
    monkeypatch.setattr(settings, "COOKIE_SECURE", False)

    from backend.auth import router as auth_router

    test_app = FastAPI()
    test_app.include_router(auth_router)
    install_db_override(test_app, db_session_factory)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_auth_module_state():
    """Avoid leaking in-memory limiter/reset state between tests."""
    from backend import auth as auth_module

    auth_module._login_attempts.clear()
    auth_module._password_reset_tokens.clear()
    yield
    auth_module._login_attempts.clear()
    auth_module._password_reset_tokens.clear()


def register(client: TestClient, **overrides):
    return client.post("/auth/register", json=registration_payload(**overrides))


class TestRegistration:
    def test_registers_complete_security_question_payload(self, client, db_session_factory):
        response = register(client, email="Alice@Example.COM")

        assert response.status_code == 201, response.text
        body = response.json()
        assert body["email"] == "alice@example.com"
        assert body["display_name"] == "Alice Example"
        assert body["profile_complete"] is False
        assert body["security_question"] == "What is your favorite pet's name?"
        assert "password_hash" not in body
        assert "password" not in body

        with db_session_factory() as db:
            user = db.query(User).filter_by(email="alice@example.com").one()
            assert verify_password("Password123!", user.password_hash)
            assert user.security_answer_hash is not None
            assert verify_password("milo", user.security_answer_hash)

    @pytest.mark.parametrize(
        "payload",
        [
            {"security_question": None},
            {"security_question": ""},
            {"security_question": "not-a-real-question"},
            {"security_answer": None},
            {"security_answer": "   "},
        ],
    )
    def test_rejects_missing_blank_or_invalid_security_recovery_input(self, client, payload):
        response = register(client, **payload)
        assert response.status_code == 422

    def test_normalizes_email_and_rejects_case_insensitive_duplicate(self, client):
        assert register(client, email="Alice@Example.COM").status_code == 201
        duplicate = register(
            client,
            email="ALICE@example.com",
            username="another-user",
        )
        assert duplicate.status_code == 409
        assert "already exists" in duplicate.json()["detail"].lower()

    def test_rejects_duplicate_username(self, client):
        assert register(client, username="alice").status_code == 201
        duplicate = register(
            client,
            email="second@example.com",
            username="alice",
        )
        assert duplicate.status_code == 409

    @pytest.mark.parametrize(
        "field,value",
        [
            ("password", "x" * 7),
            ("password", "x" * 129),
            ("display_name", ""),
            ("display_name", "x" * 121),
            ("phone", "x" * 21),
            ("security_answer", "x" * 101),
        ],
    )
    def test_schema_boundaries_reject_invalid_registration_values(self, client, field, value):
        response = register(client, **{field: value})
        assert response.status_code == 422


class TestLoginAndTokens:
    def test_successful_login_sets_http_only_access_and_refresh_cookies(self, client):
        assert register(client).status_code == 201

        response = client.post(
            "/auth/login",
            json={"email": "alice@example.com", "password": "Password123!"},
        )

        assert response.status_code == 200
        assert response.json()["token_type"] == "bearer"
        assert response.json()["access_token"]
        headers = response.headers.get_list("set-cookie")
        assert any("aegis_access=" in h and "HttpOnly" in h and "Path=/" in h for h in headers)
        assert any("aegis_refresh=" in h and "HttpOnly" in h and "Path=/auth" in h for h in headers)

    def test_unknown_user_and_wrong_password_share_public_response(self, client):
        assert register(client).status_code == 201

        wrong_password = client.post(
            "/auth/login",
            json={"email": "alice@example.com", "password": "WrongPassword!"},
        )
        unknown_user = client.post(
            "/auth/login",
            json={"email": "nobody@example.com", "password": "WrongPassword!"},
        )

        assert wrong_password.status_code == unknown_user.status_code == 401
        assert wrong_password.json() == unknown_user.json()
        assert "access_token" not in wrong_password.json()

    def test_login_is_rate_limited_after_configured_threshold(self, client):
        assert register(client).status_code == 201
        for _ in range(5):
            assert client.post(
                "/auth/login",
                json={"email": "alice@example.com", "password": "wrong-password"},
            ).status_code == 401

        throttled = client.post(
            "/auth/login",
            json={"email": "alice@example.com", "password": "wrong-password"},
        )
        assert throttled.status_code == 429
        assert int(throttled.headers["Retry-After"]) >= 1

    def test_me_accepts_bearer_header_and_http_only_cookie(self, client):
        assert register(client).status_code == 201
        token = login(client)["access_token"]

        by_header = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert by_header.status_code == 200
        assert by_header.json()["email"] == "alice@example.com"

        # TestClient preserves the access cookie set by login().
        by_cookie = client.get("/auth/me")
        assert by_cookie.status_code == 200

    def test_invalid_bearer_token_is_rejected(self, client):
        response = client.get(
            "/auth/me",
            headers={"Authorization": "Bearer not.a.valid.jwt"},
        )
        assert response.status_code == 401

    def test_refresh_rotates_and_replay_of_old_token_is_rejected(self, client):
        assert register(client).status_code == 201
        login_response = client.post(
            "/auth/login",
            json={"email": "alice@example.com", "password": "Password123!"},
        )
        old_refresh = login_response.cookies.get("aegis_refresh")
        assert old_refresh

        client.cookies.set("aegis_refresh", old_refresh)
        refreshed = client.post("/auth/refresh")
        assert refreshed.status_code == 200
        assert refreshed.json()["access_token"]
        assert refreshed.cookies.get("aegis_refresh") != old_refresh

        client.cookies.set("aegis_refresh", old_refresh)
        assert client.post("/auth/refresh").status_code == 401

    def test_expired_refresh_token_is_rejected(self, client, db_session_factory):
        assert register(client).status_code == 201
        login_response = client.post(
            "/auth/login",
            json={"email": "alice@example.com", "password": "Password123!"},
        )
        raw_refresh = login_response.cookies.get("aegis_refresh")
        assert raw_refresh

        with db_session_factory() as db:
            row = db.query(RefreshToken).one()
            row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            db.commit()

        client.cookies.set("aegis_refresh", raw_refresh)
        assert client.post("/auth/refresh").status_code == 401

    def test_logout_revokes_refresh_and_clears_authenticated_cookie_session(self, client):
        assert register(client).status_code == 201
        login_response = client.post(
            "/auth/login",
            json={"email": "alice@example.com", "password": "Password123!"},
        )
        refresh = login_response.cookies.get("aegis_refresh")

        logout_response = client.post("/auth/logout")
        assert logout_response.status_code == 200
        assert logout_response.json() == {"logged_out": True}
        assert client.get("/auth/me").status_code == 401

        client.cookies.set("aegis_refresh", refresh)
        assert client.post("/auth/refresh").status_code == 401


class TestPasswordReset:
    def test_reset_token_is_single_use_and_revokes_existing_refresh_sessions(self, client):
        assert register(client).status_code == 201
        login_response = client.post(
            "/auth/login",
            json={"email": "alice@example.com", "password": "Password123!"},
        )
        old_refresh = login_response.cookies.get("aegis_refresh")

        verified = client.post(
            "/auth/verify-security-answer",
            json={"email": "alice@example.com", "security_answer": "  MILO  "},
        )
        assert verified.status_code == 200
        assert verified.json()["verified"] is True
        token = parse_qs(urlparse(verified.json()["reset_link"]).query)["token"][0]

        reset = client.post(
            "/auth/reset-password",
            json={"token": token, "new_password": "NewPassword123!"},
        )
        assert reset.status_code == 200

        assert client.post(
            "/auth/login",
            json={"email": "alice@example.com", "password": "Password123!"},
        ).status_code == 401
        assert client.post(
            "/auth/login",
            json={"email": "alice@example.com", "password": "NewPassword123!"},
        ).status_code == 200
        assert client.post(
            "/auth/reset-password",
            json={"token": token, "new_password": "AnotherPassword123!"},
        ).status_code == 400

        client.cookies.set("aegis_refresh", old_refresh)
        assert client.post("/auth/refresh").status_code == 401


@pytest.mark.security_gate
@pytest.mark.xfail(strict=True, reason="Known P0: UserOut currently exposes security_answer.")
def test_user_responses_never_expose_security_answer(client):
    response = register(client)
    assert response.status_code == 201
    assert "security_answer" not in response.json()
    assert "security_answer_hash" not in response.json()

    me = client.get("/auth/me")
    assert me.status_code == 200
    assert "security_answer" not in me.json()


@pytest.mark.security_gate
@pytest.mark.xfail(strict=True, reason="Known P0: forgot-password currently enumerates account existence/question.")
def test_forgot_password_has_same_public_response_for_known_and_unknown_email(client):
    assert register(client).status_code == 201
    known = client.post("/auth/forgot-password", json={"email": "alice@example.com"})
    unknown = client.post("/auth/forgot-password", json={"email": "unknown@example.com"})

    assert known.status_code == unknown.status_code == 200
    assert known.json() == unknown.json()
    assert "account_exists" not in known.json()
    assert "security_question" not in known.json()
