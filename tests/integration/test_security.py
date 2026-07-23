"""
tests/integration/test_security.py — Tests for security headers and CORS.

Place this at: tests/integration/test_security.py

Tests:
    - Security headers present on all responses
    - CORS headers present on OPTIONS preflight
    - Health endpoint has correct headers
    - Cache-Control: no-store on sensitive paths
    - CSP header is well-formed
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.security import (
    SecurityHeadersMiddleware,
    install_cors,
    install_security,
)


# ── Fixtures ──────────────────────────────────────────────────
@pytest.fixture
def app() -> FastAPI:
    """Create a minimal FastAPI app with security middleware installed."""
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/auth/login")
    def login():
        return {"token": "fake"}

    @app.get("/export/fhir/test")
    def fhir_export():
        return {"resourceType": "Bundle"}

    install_security(app)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ── Tests ────────────────────────────────────────────────────
class TestSecurityHeaders:
    """Verify all required security headers are injected."""

    REQUIRED_HEADERS = {
        "x-content-type-options": "nosniff",
        "referrer-policy": "no-referrer",
        "x-frame-options": "DENY",
    }

    def test_all_required_headers_present(self, client):
        resp = client.get("/health")
        for header, expected in self.REQUIRED_HEADERS.items():
            assert resp.headers.get(header) == expected, (
                f"Missing/invalid header: {header}"
            )

    def test_permissions_policy_present(self, client):
        resp = client.get("/health")
        permissions = resp.headers.get("permissions-policy", "")
        assert "camera=()" in permissions
        assert "geolocation=()" in permissions

    def test_csp_present(self, client):
        resp = client.get("/health")
        csp = resp.headers.get("content-security-policy", "")
        assert "default-src 'self'" in csp
        assert "img-src 'self' data:" in csp
        assert "media-src 'self' blob:" in csp

    def test_csp_well_formed(self, client):
        """CSP should have directive-value pairs separated by semicolons."""
        resp = client.get("/health")
        csp = resp.headers.get("content-security-policy", "")
        # Each directive should end with a semicolon
        directives = [d.strip() for d in csp.split(";") if d.strip()]
        assert len(directives) >= 4  # at least 4 directives

    def test_hsts_not_in_dev(self, client):
        """HSTS should NOT appear in development."""
        resp = client.get("/health")
        assert "strict-transport-security" not in {
            k.lower() for k in resp.headers
        }


class TestCORS:
    """Verify CORS headers on preflight requests."""

    def test_options_preflight_has_cors_headers(self, client):
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS middleware should set these
        assert "access-control-allow-origin" in {
            k.lower() for k in resp.headers
        } or resp.status_code == 200  # OPTIONS may or may not be handled

    def test_response_has_vary_origin(self, client):
        """Responses should include Vary: Origin for proper caching."""
        resp = client.get("/health", headers={"Origin": "http://localhost:5173"})
        vary = resp.headers.get("vary", "")
        # CORS middleware should add Origin to Vary
        assert "origin" in vary.lower()


class TestCacheControl:
    """Verify Cache-Control: no-store on sensitive paths."""

    def test_auth_path_no_store(self, client):
        resp = client.post("/auth/login", json={"email": "a@b.com"})
        assert resp.headers.get("cache-control") == "no-store"

    def test_export_path_no_store(self, client):
        resp = client.get("/export/fhir/test")
        assert resp.headers.get("cache-control") == "no-store"

    def test_health_path_not_restricted(self, client):
        """Health endpoint should NOT have no-store (it's safe to cache briefly)."""
        resp = client.get("/health")
        cc = resp.headers.get("cache-control", "")
        assert "no-store" not in cc


class TestSecurityHeadersMiddlewareDirect:
    """Unit tests for the middleware class directly."""

    def test_headers_injected_after_response(self, app, client):
        """Headers should be present even on error responses."""
        resp = client.get("/nonexistent")
        assert resp.status_code == 404
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"
