"""
tests/integration/test_auth.py — Tests for /auth/register, /auth/login,
/auth/refresh, /auth/logout, /auth/me.

Uses an isolated in-memory SQLite database per test (via a FastAPI
dependency override on get_db) so these tests never touch the real
aegis.db file and can run in full isolation/parallel-safety.
"""
from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base
from app.db import models  # noqa: F401 — ensure models register on Base


# ── Isolated test DB + app fixtures ────────────────────────────
@pytest.fixture
def db_session_factory():
    # StaticPool + check_same_thread=False: a single shared in-memory
    # SQLite connection reused across the whole test, so tables created
    # by create_all() are visible to every session opened from the
    # sessionmaker (a plain ":memory:" engine gives each new connection
    # its own empty database otherwise).
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@pytest.fixture
def app(db_session_factory) -> FastAPI:
    from backend.auth import router as auth_router

    test_app = FastAPI()
    test_app.include_router(auth_router)

    def override_get_db():
        db = db_session_factory()
        try:
            yield db
        finally:
            db.close()

    from app.db.session import get_db

    test_app.dependency_overrides[get_db] = override_get_db
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_login_rate_limiter():
    """
    backend.auth keeps the login rate limiter as module-level state
    (matching the codebase's single-worker invariant). Reset it before
    each test so failed-login tests don't trip the limiter for
    unrelated tests running in the same session.
    """
    from backend.auth import _login_attempts

    _login_attempts.clear()
    yield
    _login_attempts.clear()


def _register(client: TestClient, email="alice@example.com", password="password123", display_name="Alice"):
    return client.post(
        "/auth/register",
        json={"email": email, "password": password, "display_name": display_name},
    )


# ── Register ────────────────────────────────────────────────────
class TestRegister:
    def test_register_creates_user(self, client):
        resp = _register(client)
        assert resp.status_code == 201
        body = resp.json()
        assert body["email"] == "alice@example.com"
        assert body["display_name"] == "Alice"
        assert body["profile_complete"] is False
        assert "password" not in body
        assert "password_hash" not in body

    def test_register_duplicate_email_rejected(self, client):
        _register(client)
        resp = _register(client)
        assert resp.status_code == 409

    def test_register_short_password_rejected(self, client):
        resp = client.post(
            "/auth/register",
            json={"email": "bob@example.com", "password": "short", "display_name": "Bob"},
        )
        assert resp.status_code == 422

    def test_register_email_lowercased(self, client):
        resp = _register(client, email="Alice@Example.com")
        assert resp.status_code == 201
        assert resp.json()["email"] == "alice@example.com"


# ── Login ───────────────────────────────────────────────────────
class TestLogin:
    def test_login_success_returns_access_token(self, client):
        _register(client)
        resp = client.post("/auth/login", json={"email": "alice@example.com", "password": "password123"})
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    def test_login_sets_http_only_auth_cookies(self, client):
        _register(client)
        resp = client.post("/auth/login", json={"email": "alice@example.com", "password": "password123"})
        assert "aegis_refresh" in resp.cookies
        assert "aegis_access" in resp.cookies

        set_cookie_headers = resp.headers.get_list("set-cookie")
        assert any("aegis_refresh=" in value and "HttpOnly" in value for value in set_cookie_headers)
        assert any("aegis_access=" in value and "HttpOnly" in value for value in set_cookie_headers)

    def test_login_wrong_password_rejected(self, client):
        _register(client)
        resp = client.post("/auth/login", json={"email": "alice@example.com", "password": "wrongpass"})
        assert resp.status_code == 401

    def test_login_unknown_email_rejected(self, client):
        resp = client.post("/auth/login", json={"email": "ghost@example.com", "password": "whatever1"})
        assert resp.status_code == 401

    def test_login_timing_close_for_unknown_vs_wrong_password(self, client):
        """
        Timing-safe login: response time for an unknown email should be
        close to the response time for a known email + wrong password.
        This is a coarse smoke check, not a precise timing-attack proof.
        """
        _register(client)

        start = time.perf_counter()
        client.post("/auth/login", json={"email": "ghost@example.com", "password": "whatever1"})
        unknown_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        client.post("/auth/login", json={"email": "alice@example.com", "password": "wrongpass"})
        known_elapsed = time.perf_counter() - start

        # Both should be bounded near the enforced floor; allow generous slack
        # for CI/sandbox jitter while still catching gross timing leaks.
        assert abs(unknown_elapsed - known_elapsed) < 0.5

    def test_login_rate_limited_after_threshold(self, client):
        _register(client)
        for _ in range(5):
            client.post("/auth/login", json={"email": "alice@example.com", "password": "wrongpass"})
        resp = client.post("/auth/login", json={"email": "alice@example.com", "password": "wrongpass"})
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers


# ── Refresh ─────────────────────────────────────────────────────
class TestRefresh:
    def test_refresh_without_cookie_rejected(self, client):
        resp = client.post("/auth/refresh")
        assert resp.status_code == 401

    def test_refresh_rotates_token(self, client):
        _register(client)
        login_resp = client.post("/auth/login", json={"email": "alice@example.com", "password": "password123"})
        refresh_cookie = login_resp.cookies.get("aegis_refresh")
        assert refresh_cookie

        client.cookies.set("aegis_refresh", refresh_cookie)
        refresh_resp = client.post("/auth/refresh")
        assert refresh_resp.status_code == 200
        assert "access_token" in refresh_resp.json()
        assert "aegis_access" in refresh_resp.cookies

    def test_refresh_replay_rejected(self, client):
        """A refresh token that has already been used (rotated) must be rejected on reuse."""
        _register(client)
        login_resp = client.post("/auth/login", json={"email": "alice@example.com", "password": "password123"})
        old_refresh = login_resp.cookies.get("aegis_refresh")

        client.cookies.set("aegis_refresh", old_refresh)
        first_refresh = client.post("/auth/refresh")
        assert first_refresh.status_code == 200

        # Replay the same (now-revoked) refresh cookie again.
        client.cookies.set("aegis_refresh", old_refresh)
        replay_resp = client.post("/auth/refresh")
        assert replay_resp.status_code == 401


# ── Logout ──────────────────────────────────────────────────────
class TestLogout:
    def test_logout_clears_cookie_and_revokes_token(self, client):
        _register(client)
        login_resp = client.post("/auth/login", json={"email": "alice@example.com", "password": "password123"})
        refresh_cookie = login_resp.cookies.get("aegis_refresh")

        client.cookies.set("aegis_refresh", refresh_cookie)
        logout_resp = client.post("/auth/logout")
        assert logout_resp.status_code == 200
        assert logout_resp.json()["logged_out"] is True
        assert client.get("/auth/me").status_code == 401

        # The revoked token must not work for refresh anymore.
        client.cookies.set("aegis_refresh", refresh_cookie)
        refresh_resp = client.post("/auth/refresh")
        assert refresh_resp.status_code == 401


# ── Me ──────────────────────────────────────────────────────────
class TestMe:
    def test_me_without_token_rejected(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_me_with_valid_token(self, client):
        _register(client)
        login_resp = client.post("/auth/login", json={"email": "alice@example.com", "password": "password123"})
        token = login_resp.json()["access_token"]

        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["email"] == "alice@example.com"

    def test_me_with_http_only_access_cookie(self, client):
        _register(client)
        client.post("/auth/login", json={"email": "alice@example.com", "password": "password123"})

        # TestClient retains the login cookies, exactly as the browser does.
        resp = client.get("/auth/me")
        assert resp.status_code == 200
        assert resp.json()["email"] == "alice@example.com"

    def test_me_with_invalid_token_rejected(self, client):
        resp = client.get("/auth/me", headers={"Authorization": "Bearer garbage.token.here"})
        assert resp.status_code == 401


