"""
backend/security.py — Security middleware and CORS configuration.

Provides:
    SecurityHeadersMiddleware  — injects security headers on every response
    install_cors(app)          — configure CORS from settings
    install_security(app)      — attach middleware to a FastAPI app

CSRF protection:
    Cookie-bearing mutation endpoints (POST/PUT/DELETE that use refresh tokens)
    are protected via SameSite + origin checking. The refresh cookie is scoped
    to /auth path only and uses SameSite=Lax, which prevents cross-site form
    submissions from including the cookie.

References:
    https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP
    https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.settings import settings


# ── Security Headers Middleware ──────────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Inject security headers on every response.

    Headers applied:
        X-Content-Type-Options  → nosniff
        Referrer-Policy         → no-referrer
        X-Frame-Options         → DENY
        Permissions-Policy      → restrict camera, geolocation, payment
        Content-Security-Policy → allow self + data: images + blob: media
        Strict-Transport-Security → only in production (HTTPS)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)

        # ── Standard security headers ────────────────────────────
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = (
            "camera=(), "
            "geolocation=(), "
            "microphone=(), "
            "payment=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data:; "
            "media-src 'self' blob:; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'"
        )

        # ── HSTS (production only) ───────────────────────────────
        if settings.AEGIS_ENV == "production":
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )

        # ── Cache control for sensitive endpoints ────────────────
        # Auth and account endpoints should never be cached
        path = request.url.path
        if any(
            path.startswith(p)
            for p in ("/auth/", "/account", "/export/")
        ):
            response.headers["Cache-Control"] = "no-store"

        return response


# ── CORS Installation ────────────────────────────────────────────
def install_cors(app: FastAPI) -> None:
    """
    Configure CORS for the FastAPI app.

    Development: allows localhost origins (Vite dev server).
    Production:  locked to the specific origin from settings.

    Because this is a local-edge app, origins are narrow by default.
    Credentials are allowed so httpOnly refresh cookies work
    with cross-origin fetch from the frontend dev server.
    """
    origins = settings.AEGIS_CORS_ORIGINS

    # ── CSRF-safe: only allow methods that make sense ────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "Accept",
        ],
        max_age=600,  # 10-minute preflight cache
    )


# ── High-level installer ─────────────────────────────────────────
def install_security(app: FastAPI) -> None:
    """
    Apply all security middleware at once.

    Call this once during app creation, after all routers are mounted.
    Middleware is applied in reverse order — the first added runs last.
    We add CORS first, then security headers, so headers are applied
    after the CORS middleware has processed the response.
    """
    install_cors(app)
    app.add_middleware(SecurityHeadersMiddleware)
