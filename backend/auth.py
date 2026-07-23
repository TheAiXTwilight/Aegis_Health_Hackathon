"""
backend/auth.py — Auth router with security-question-based password reset.

Flow:
  1. POST /auth/forgot-password — enter email → returns security question
  2. POST /auth/verify-security-answer — answer question → returns reset link
  3. POST /auth/reset-password — set new password with token

Phone number field:
  - RegisterIn now accepts an optional `phone` field.

Security question:
  - RegisterIn requires `security_question` and `security_answer`.
  - Answer is hashed (bcrypt) and stored in `security_answer_hash`.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from secrets import token_urlsafe

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from loguru import logger

from app.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password_timing_safe,
)
from app.db.models import AuditLog, RefreshToken, User
from app.db.session import get_db
from app.settings import settings

router = APIRouter(prefix="/auth", tags=["auth"])

# ── Valid security questions ────────────────────────────────────────
SECURITY_QUESTIONS = {
    "pet_name": "What is your favorite pet's name?",
    "favorite_food": "What is your favorite food?",
    "birth_city": "What city were you born in?",
    "favorite_movie": "What is your favorite movie?",
}

_LOGIN_MIN_ELAPSED_S = 0.15


# ── Password reset token store (in-memory, single-worker) ──────────
# Keys are hashed tokens (SHA-256), values store user_id and expiry.
# For a multi-worker deployment this should be moved to the database.
_password_reset_tokens: dict[str, dict] = {}
_RESET_TOKEN_EXPIRE_HOURS = 1


# ── Request / response models ─────────────────────────────────────
class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=120)
    username: str | None = Field(default=None, max_length=60)
    phone: str | None = Field(default=None, max_length=20)
    security_question: str | None = Field(default=None, max_length=200)
    security_answer: str | None = Field(default=None, max_length=100)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class ForgotPasswordIn(BaseModel):
    email: EmailStr


class VerifySecurityAnswerIn(BaseModel):
    email: EmailStr
    security_answer: str = Field(min_length=1, max_length=100)


class ResetPasswordIn(BaseModel):
    token: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=128)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: str
    username: str | None
    display_name: str
    role: str
    phone: str | None
    security_question: str | None
    security_answer: str | None
    created_at: datetime
    profile_complete: bool


def _user_out(user: User) -> UserOut:
    return UserOut(
        id=user.id,
        email=user.email,
        username=user.username,
        display_name=user.display_name,
        role=user.role,
        phone=getattr(user, "phone", None),
        security_question=getattr(user, "security_question", None),
        security_answer=getattr(user, "security_answer", None),
        created_at=user.created_at,
        profile_complete=user.profile is not None,
    )


# ── In-memory login rate limiter (per IP, sliding window) ──────────
_login_attempts: dict[str, deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _check_login_rate_limit(ip: str) -> None:
    now = time.monotonic()
    window = settings.AEGIS_LOGIN_RATE_LIMIT_WINDOW_S
    attempts = _login_attempts[ip]

    while attempts and now - attempts[0] > window:
        attempts.popleft()

    if len(attempts) >= settings.AEGIS_LOGIN_RATE_LIMIT_ATTEMPTS:
        retry_after = max(1, int(window - (now - attempts[0])))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please wait before retrying.",
            headers={"Retry-After": str(retry_after)},
        )


def _record_login_attempt(ip: str) -> None:
    _login_attempts[ip].append(time.monotonic())


# ── Hashing / audit helpers ─────────────────────────────────────
def _hash_token(raw: str) -> str:
    return sha256(raw.encode("utf-8")).hexdigest()


def _hash_ip(ip: str) -> str:
    return sha256(ip.encode("utf-8")).hexdigest()


def _as_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _write_audit_log(
    db: Session,
    *,
    user_id: str | None,
    action: str,
    request: Request | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
) -> None:
    try:
        ip_hash = _hash_ip(_client_ip(request)) if request is not None else None
        ua_hash = (
            sha256(request.headers.get("user-agent", "").encode("utf-8")).hexdigest()
            if request is not None
            else None
        )
        db.add(
            AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_hash=ip_hash,
                user_agent_hash=ua_hash,
            )
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("Failed to write audit log", action=action, user_id=user_id)


def _issue_access_token(user: User, response: Response) -> str:
    token = create_access_token(user)
    response.set_cookie(
        key=settings.AEGIS_ACCESS_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite="lax",
        max_age=settings.AEGIS_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    return token


def _issue_refresh_token(db: Session, user: User, response: Response) -> None:
    raw = token_urlsafe(32)
    row = RefreshToken(
        user_id=user.id,
        token_hash=_hash_token(raw),
        expires_at=datetime.now(timezone.utc)
        + timedelta(hours=settings.AEGIS_REFRESH_TOKEN_EXPIRE_HOURS),
    )
    db.add(row)
    db.commit()
    response.set_cookie(
        key=settings.AEGIS_REFRESH_COOKIE_NAME,
        value=raw,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite="lax",
        max_age=settings.AEGIS_REFRESH_TOKEN_EXPIRE_HOURS * 3600,
        path="/auth",
    )


# ── Endpoints ────────────────────────────────────────────────────
@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterIn, request: Request, db: Session = Depends(get_db)) -> UserOut:
    """Create a new local account. Email must be unique."""
    # Validate security question & answer
    if not payload.security_question or not payload.security_question.strip():
        raise HTTPException(status_code=422, detail="Please select a security question.")
    if payload.security_question not in SECURITY_QUESTIONS:
        raise HTTPException(status_code=422, detail="Invalid security question selected.")
    if not payload.security_answer or not payload.security_answer.strip():
        raise HTTPException(status_code=422, detail="Please answer the security question.")

    email = payload.email.lower()
    existing = db.query(User).filter(User.email == email).first()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists")

    if payload.username:
        username_taken = db.query(User).filter(User.username == payload.username).first()
        if username_taken is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")

    try:
        user = User(
            email=email,
            username=payload.username,
            display_name=payload.display_name,
            password_hash=hash_password(payload.password),
            phone=payload.phone,
        security_question=SECURITY_QUESTIONS[payload.security_question],
        security_answer_hash=hash_password(payload.security_answer.strip().lower()) if payload.security_answer else None,
        security_answer=payload.security_answer.strip() if payload.security_answer else None,
            role="user",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    except Exception as db_err:
        db.rollback()
        logger.error("Failed to create user", error=str(db_err))
        # If security_question column doesn't exist, try without it
        try:
            user = User(
                email=email,
                username=payload.username,
                display_name=payload.display_name,
                password_hash=hash_password(payload.password),
                phone=payload.phone,
                role="user",
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.warning("Created user without security question — run migration: python -m migrations.add_security_questions")
        except Exception as db_err2:
            db.rollback()
            logger.error("Failed to create user even without security question", error=str(db_err2))
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_err2)}")

    _write_audit_log(db, user_id=user.id, action="register", request=request, resource_type="account", resource_id=user.id)
    logger.info("User registered", user_id=user.id, email=user.email)
    return _user_out(user)


@router.post("/login", response_model=TokenOut)
def login(payload: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)) -> TokenOut:
    ip = _client_ip(request)
    _check_login_rate_limit(ip)
    _record_login_attempt(ip)

    started = time.perf_counter()
    email = payload.email.lower()
    user = db.query(User).filter(User.email == email, User.is_active.is_(True)).first()
    password_ok = verify_password_timing_safe(payload.password, user.password_hash if user else None)

    elapsed = time.perf_counter() - started
    if elapsed < _LOGIN_MIN_ELAPSED_S:
        time.sleep(_LOGIN_MIN_ELAPSED_S - elapsed)

    if not user or not password_ok:
        _write_audit_log(db, user_id=user.id if user else None, action="login_failed", request=request, resource_type="account")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    user.last_login_at = datetime.now(timezone.utc)
    db.commit()

    _issue_refresh_token(db, user, response)
    access_token = _issue_access_token(user, response)
    _write_audit_log(db, user_id=user.id, action="login_success", request=request, resource_type="account", resource_id=user.id)

    return TokenOut(access_token=access_token)


@router.post("/forgot-password")
def forgot_password(payload: ForgotPasswordIn, request: Request, db: Session = Depends(get_db)) -> dict:
    """
    Step 1: Enter email. Returns the security question if account exists.
    """
    email = payload.email.lower().strip()
    user = db.query(User).filter(User.email == email, User.is_active.is_(True)).first()

    if not user:
        return {
            "detail": "No account found with this email address.",
            "account_exists": False,
        }

    if not user.security_question:
        return {
            "detail": "Security question not set for this account. Please contact support.",
            "account_exists": True,
            "security_question": None,
        }

    return {
        "detail": "Please answer your security question to reset your password.",
        "account_exists": True,
        "security_question": user.security_question,
    }


@router.post("/verify-security-answer")
def verify_security_answer(payload: VerifySecurityAnswerIn, request: Request, db: Session = Depends(get_db)) -> dict:
    """
    Step 2: Verify the security answer. If correct, return the reset link.
    """
    email = payload.email.lower().strip()
    user = db.query(User).filter(User.email == email, User.is_active.is_(True)).first()

    if not user:
        return {
            "detail": "No account found with this email address.",
            "account_exists": False,
        }

    if not user.security_answer_hash:
        return {
            "detail": "Security question not set for this account.",
            "account_exists": True,
            "verified": False,
        }

    # Verify answer (case-insensitive, trimmed)
    answer_ok = verify_password_timing_safe(
        payload.security_answer.strip().lower(),
        user.security_answer_hash,
    )

    if not answer_ok:
        _write_audit_log(
            db,
            user_id=user.id,
            action="security_answer_failed",
            request=request,
            resource_type="account",
            resource_id=user.id,
        )
        return {
            "detail": "Incorrect answer. Please try again.",
            "account_exists": True,
            "verified": False,
        }

    # Answer correct — generate reset token
    raw_token = token_urlsafe(48)
    token_hash = _hash_token(raw_token)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=_RESET_TOKEN_EXPIRE_HOURS)

    _password_reset_tokens[token_hash] = {
        "user_id": user.id,
        "email": user.email,
        "expires_at": expires_at,
    }

    # Clean up expired tokens
    now = datetime.now(timezone.utc)
    expired_keys = [k for k, v in _password_reset_tokens.items() if v["expires_at"] < now]
    for k in expired_keys:
        del _password_reset_tokens[k]

    _write_audit_log(
        db,
        user_id=user.id,
        action="forgot_password_requested",
        request=request,
        resource_type="account",
        resource_id=user.id,
    )

    reset_link = f"{settings.AEGIS_PUBLIC_URL}/reset-password?token={raw_token}"

    logger.info("Password reset token generated (security answer verified)", user_id=user.id, email=user.email)

    return {
        "detail": "Answer verified! Click below to reset your password.",
        "account_exists": True,
        "verified": True,
        "reset_link": reset_link,
    }


@router.post("/reset-password")
def reset_password(payload: ResetPasswordIn, request: Request, db: Session = Depends(get_db)) -> dict:
    """
    Reset password using a valid reset token.

    Validates the token, updates the user's password, and invalidates
    the token so it cannot be reused.
    """
    token_hash = _hash_token(payload.token)
    token_data = _password_reset_tokens.get(token_hash)

    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token.",
        )

    now = datetime.now(timezone.utc)
    if token_data["expires_at"] < now:
        del _password_reset_tokens[token_hash]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token has expired. Please request a new one.",
        )

    user = db.query(User).filter(User.id == token_data["user_id"], User.is_active.is_(True)).first()
    if not user:
        del _password_reset_tokens[token_hash]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account not found.",
        )

    # Update the password
    user.password_hash = hash_password(payload.new_password)
    db.commit()

    # Invalidate the token
    del _password_reset_tokens[token_hash]

    # Revoke all refresh tokens for this user (force re-login)
    db.query(RefreshToken).filter(RefreshToken.user_id == user.id).update({"revoked_at": now})
    db.commit()

    _write_audit_log(
        db,
        user_id=user.id,
        action="password_reset_completed",
        request=request,
        resource_type="account",
        resource_id=user.id,
    )

    logger.info("Password reset completed", user_id=user.id, email=user.email)

    return {"detail": "Password has been reset successfully. Please sign in with your new password."}


@router.post("/refresh", response_model=TokenOut)
def refresh(request: Request, response: Response, db: Session = Depends(get_db)) -> TokenOut:
    raw = request.cookies.get(settings.AEGIS_REFRESH_COOKIE_NAME)
    if not raw:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token")

    token_hash = _hash_token(raw)
    row = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
    now = datetime.now(timezone.utc)

    if not row or row.revoked_at is not None or _as_aware_utc(row.expires_at) <= now:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    row.revoked_at = now
    user = db.get(User, row.user_id)
    if not user or not user.is_active:
        db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User unavailable")

    db.commit()
    _issue_refresh_token(db, user, response)
    access_token = _issue_access_token(user, response)

    return TokenOut(access_token=access_token)


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)) -> dict:
    raw = request.cookies.get(settings.AEGIS_REFRESH_COOKIE_NAME)
    user_id: str | None = None
    if raw:
        token_hash = _hash_token(raw)
        row = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
        if row and row.revoked_at is None:
            row.revoked_at = datetime.now(timezone.utc)
            user_id = row.user_id
            db.commit()

    response.delete_cookie(settings.AEGIS_REFRESH_COOKIE_NAME, path="/auth")
    response.delete_cookie(settings.AEGIS_ACCESS_COOKIE_NAME, path="/")
    _write_audit_log(db, user_id=user_id, action="logout", request=request, resource_type="account")
    return {"logged_out": True}


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> UserOut:
    return _user_out(user)


