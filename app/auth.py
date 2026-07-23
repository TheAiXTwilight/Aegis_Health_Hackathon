"""
app/auth.py — Authentication core: JWT access tokens, bcrypt hashing,
refresh-token helpers, and FastAPI auth dependencies.

Replaces the earlier STUB module. Keeps the same import path and the
same public function signatures (get_current_user, get_optional_user)
so backend/account.py, backend/chat.py, backend/dashboard.py,
backend/exports.py, backend/pdf_export.py, and backend/vitals.py
require zero changes.

Design notes:
    - Access tokens are short-lived JWTs (HS256), never persisted server-side;
      browsers receive them in an httpOnly cookie.
    - Refresh tokens are opaque random strings; only their SHA-256 hash is
      stored in the `refresh_tokens` table (see backend/auth.py for the
      issuing/rotation flow used by /auth/login and /auth/refresh).
    - Login is timing-safe: a bcrypt verify always runs (against a dummy
      hash when the email is unknown) so response timing does not leak
      whether an account exists.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import bcrypt
import jwt
from fastapi import Cookie, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import User
from app.db.session import get_db
from app.settings import settings

# ── Password hashing (bcrypt, no passlib dependency) ───────────────
# A fixed dummy hash used to keep login timing constant when the
# supplied email does not match any user. Computed once at import time.
_DUMMY_PASSWORD_HASH = bcrypt.hashpw(b"aegis-dummy-password", bcrypt.gensalt())


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt. Returns a UTF-8 string."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # Malformed hash — never crash the login path.
        return False


def verify_password_timing_safe(password: str, stored_hash: str | None) -> bool:
    """
    Verify a password, always performing a bcrypt comparison even when
    the user does not exist. This prevents user enumeration via response
    timing differences between "unknown email" and "wrong password".
    """
    if not stored_hash:
        bcrypt.checkpw(password.encode("utf-8"), _DUMMY_PASSWORD_HASH)
        return False
    return verify_password(password, stored_hash)


# ── JWT access tokens ────────────────────────────────────────────
def create_access_token(user: User) -> str:
    """Create a short-lived JWT access token for the given user."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.id,
        "email": user.email,
        "role": user.role,
        "jti": str(uuid4()),
        "iat": int(now.timestamp()),
        "exp": int(
            (now + timedelta(minutes=settings.AEGIS_ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()
        ),
    }
    return jwt.encode(payload, settings.AEGIS_SECRET_KEY, algorithm=settings.AEGIS_JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and verify a JWT access token. Raises jwt.PyJWTError on failure."""
    return jwt.decode(
        token,
        settings.AEGIS_SECRET_KEY,
        algorithms=[settings.AEGIS_JWT_ALGORITHM],
    )


# ── FastAPI dependencies ─────────────────────────────────────────
def _extract_access_token(
    authorization: str | None,
    access_cookie: str | None,
) -> str | None:
    """
    Resolve an access token from the Bearer header or the httpOnly cookie.

    The header takes precedence for API clients. The cookie lets ordinary
    same-origin browser requests authenticate without exposing JWTs to
    JavaScript (including existing direct fetches and image requests).
    """
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        if token:
            return token
    return access_cookie or None


async def get_current_user(
    authorization: str | None = Header(default=None),
    access_cookie: str | None = Cookie(
        default=None,
        alias=settings.AEGIS_ACCESS_COOKIE_NAME,
    ),
    db: Session = Depends(get_db),
) -> User:
    """
    Resolve the authenticated user from a Bearer token or the httpOnly
    access-token cookie. Raises 401 when the token is missing, invalid,
    expired, or the user is inactive/deleted.
    """
    token = _extract_access_token(authorization, access_cookie)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing access token")

    try:
        payload = decode_access_token(token)
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token")

    user = db.get(User, payload.get("sub"))
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User inactive or not found")
    return user


async def get_optional_user(
    authorization: str | None = Header(default=None),
    access_cookie: str | None = Cookie(
        default=None,
        alias=settings.AEGIS_ACCESS_COOKIE_NAME,
    ),
    db: Session = Depends(get_db),
) -> User | None:
    """
    Same resolution as get_current_user, but returns None instead of
    raising when no/invalid token is supplied. Used by endpoints that
    behave differently for authenticated vs anonymous callers.
    """
    token = _extract_access_token(authorization, access_cookie)
    if not token:
        return None
    try:
        payload = decode_access_token(token)
    except jwt.PyJWTError:
        return None
    user = db.get(User, payload.get("sub"))
    if not user or not user.is_active:
        return None
    return user


