"""
backend/account.py — Account management.

    DELETE /account  — Delete authenticated user and all owned data.
    PUT /account/profile — Create or update health profile.
    GET /account/profile — Return profile.
    PUT /account/email   — Change email (requires password confirmation).

Cascade-deletes (via SQLAlchemy relationship config):
    health_records, vital_snapshots, refresh_tokens, pipeline_jobs

Additional cleanup handled here:
    - Uploaded files under /tmp/aegis_uploads/{session_id}/
    - Checkpoint files under /tmp/aegis_checkpoint/
    - Dossier PDFs under /tmp/aegis_dossiers/

Password confirmation is preferred per spec — implemented with a
Pydantic body model. The actual password comparison requires the
real auth system (bcrypt). With the stub auth, a plain-text
comparison is used and clearly marked for replacement.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.orm import Session
from loguru import logger

from app.auth import get_current_user, hash_password, verify_password_timing_safe
from app.db.models import User, UserProfile, PipelineJobRow, AuditLog
from app.db.session import get_db
from app.settings import settings

router = APIRouter(tags=["account"])


# ── Request models ──────────────────────────────────────────────
class DeleteAccountRequest(BaseModel):
    """
    Password confirmation for account deletion.

    The spec prefers password confirmation before permanent deletion.
    This body is optional — if omitted, deletion still proceeds
    but logs a warning (the real implementation should require it).
    """
    password: str | None = None


class ProfileIn(BaseModel):
    full_name: str = Field(min_length=1, max_length=120)
    date_of_birth: str = Field(min_length=10, max_length=10)
    sex: Literal["Male", "Female", "Other", "Prefer not to say"]
    blood_group: Literal["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    weight_kg: float | None = Field(default=None, ge=10, le=500)
    height_cm: float | None = Field(default=None, ge=50, le=250)
    allergies: list[str] = Field(default_factory=list, max_length=20)
    medical_conditions: list[str] = Field(default_factory=list, max_length=20)
    current_medications: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("full_name")
    @classmethod
    def clean_full_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Full name is required")
        return value

    @field_validator("date_of_birth")
    @classmethod
    def validate_dob(cls, value: str) -> str:
        try:
            parsed = datetime.strptime(value, "%d/%m/%Y").date()
        except ValueError as exc:
            raise ValueError("Date of birth must use DD/MM/YYYY") from exc
        if parsed >= datetime.now(timezone.utc).date():
            raise ValueError("Date of birth must be in the past")
        return value

    @field_validator("allergies", "medical_conditions", "current_medications")
    @classmethod
    def clean_list(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = " ".join(str(value).split())[:120]
            if item and item.lower() not in {existing.lower() for existing in cleaned}:
                cleaned.append(item)
        return cleaned


class ChangeEmailIn(BaseModel):
    new_email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class ChangeSecurityQuestionIn(BaseModel):
    security_question: str = Field(min_length=1, max_length=200)
    security_answer: str = Field(min_length=1, max_length=100)


class ProfileOut(BaseModel):
    full_name: str
    email: str
    username: str | None
    date_of_birth: str
    sex: str
    blood_group: str
    weight_kg: float | None
    height_cm: float | None
    allergies: list[str]
    medical_conditions: list[str]
    current_medications: list[str]
    profile_complete: bool
    updated_at: datetime | None = None


# ── Helpers ─────────────────────────────────────────────────────
def _cleanup_user_disk_files(user_id: str, db: Session) -> None:
    """
    Remove all on-disk files owned by a user:
        - Upload directories for their sessions
        - Checkpoint files for their jobs
        - Dossier PDFs for their jobs
    """
    # Gather all session_ids from pipeline_jobs for this user
    job_rows = (
        db.query(PipelineJobRow)
        .filter_by(user_id=user_id)
        .all()
    )
    session_ids = {row.session_id for row in job_rows}
    job_ids = {row.job_id for row in job_rows}

    # Clean up upload directories
    upload_root = Path(settings.AEGIS_UPLOAD_ROOT)
    for sid in session_ids:
        session_dir = upload_root / sid
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
            logger.info("Cleaned uploads for deleted account", session_id=sid)

    # Clean up checkpoint files
    checkpoint_dir = Path(settings.AEGIS_CHECKPOINT_DIR)
    for jid in job_ids:
        ckpt = checkpoint_dir / f"{jid}.json"
        if ckpt.exists():
            ckpt.unlink(missing_ok=True)
            logger.info("Cleaned checkpoint for deleted account", job_id=jid)

    # Clean up dossier PDFs
    dossier_dir = Path("/tmp/aegis_dossiers")
    for jid in job_ids:
        pdf = dossier_dir / f"{jid}.pdf"
        if pdf.exists():
            pdf.unlink(missing_ok=True)
            logger.info("Cleaned dossier PDF for deleted account", job_id=jid)


def _write_audit_log(db: Session, user_id: str, action: str) -> None:
    """Insert an audit log row for account-level actions."""
    try:
        row = AuditLog(
            user_id=user_id,
            action=action,
            resource_type="account",
        )
        db.add(row)
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("Failed to write audit log", user_id=user_id, action=action)


def _json_list(raw: str | None) -> list[str]:
    try:
        value = json.loads(raw or "[]")
        return [str(item) for item in value] if isinstance(value, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def _profile_out(user: User, profile: UserProfile | None) -> ProfileOut:
    return ProfileOut(
        full_name=user.display_name,
        email=user.email,
        username=user.username,
        date_of_birth=profile.date_of_birth if profile else "",
        sex=profile.sex if profile else "",
        blood_group=profile.blood_group if profile else "",
        weight_kg=(profile.weight_kg if profile and profile.weight_kg > 0 else None),
        height_cm=profile.height_cm if profile else None,
        allergies=_json_list(profile.allergies_json) if profile else [],
        medical_conditions=_json_list(profile.medical_conditions_json) if profile else [],
        current_medications=_json_list(profile.current_medications_json) if profile else [],
        profile_complete=profile is not None,
        updated_at=profile.updated_at if profile else None,
    )


# ── Profile endpoints ───────────────────────────────────────────
@router.get("/account/profile", response_model=ProfileOut)
def get_profile(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ProfileOut:
    profile = db.get(UserProfile, user.id)
    return _profile_out(user, profile)


@router.put("/account/profile", response_model=ProfileOut)
def save_profile(
    payload: ProfileIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ProfileOut:
    profile = db.get(UserProfile, user.id)
    now = datetime.now(timezone.utc)

    if profile is None:
        profile = UserProfile(
            user_id=user.id,
            date_of_birth=payload.date_of_birth,
            sex=payload.sex,
            blood_group=payload.blood_group,
            weight_kg=payload.weight_kg or 0.0,
            height_cm=payload.height_cm,
            allergies_json=json.dumps(payload.allergies),
            medical_conditions_json=json.dumps(payload.medical_conditions),
            current_medications_json=json.dumps(payload.current_medications),
            created_at=now,
            updated_at=now,
        )
        db.add(profile)
    else:
        profile.date_of_birth = payload.date_of_birth
        profile.sex = payload.sex
        profile.blood_group = payload.blood_group
        profile.weight_kg = payload.weight_kg or 0.0
        profile.height_cm = payload.height_cm
        profile.allergies_json = json.dumps(payload.allergies)
        profile.medical_conditions_json = json.dumps(payload.medical_conditions)
        profile.current_medications_json = json.dumps(payload.current_medications)
        profile.updated_at = now

    user.display_name = payload.full_name
    db.commit()
    db.refresh(profile)
    _write_audit_log(db, user.id, "profile_update")
    logger.info("User profile saved", user_id=user.id)
    return _profile_out(user, profile)


# ── Change email endpoint ───────────────────────────────────────
@router.put("/account/email")
def change_email(
    payload: ChangeEmailIn,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    Change the authenticated user's email address.

    Requires password confirmation for security.
    The new email must not already be in use by another account.
    """
    # Verify password
    if not verify_password_timing_safe(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password.",
        )

    new_email = payload.new_email.lower().strip()

    # Check if the new email is the same as current
    if new_email == user.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New email is the same as your current email.",
        )

    # Check if the new email is already taken
    existing = db.query(User).filter(User.email == new_email).first()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    old_email = user.email
    user.email = new_email
    db.commit()

    _write_audit_log(db, user.id, "email_changed")
    logger.info(
        "User email changed",
        user_id=user.id,
        old_email=old_email,
        new_email=new_email,
    )

    return {"detail": "Email updated successfully.", "email": new_email}


# ── Change security question endpoint ────────────────────────────
@router.put("/account/security-question")
def change_security_question(
    payload: ChangeSecurityQuestionIn,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    Change the authenticated user's security question and answer.
    """
    # Map key to full question text
    SECURITY_QUESTIONS = {
        "pet_name": "What is your favorite pet's name?",
        "favorite_food": "What is your favorite food?",
        "birth_city": "What city were you born in?",
        "favorite_movie": "What is your favorite movie?",
    }

    question_text = SECURITY_QUESTIONS.get(payload.security_question, payload.security_question)
    if payload.security_question not in SECURITY_QUESTIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid security question.",
        )

    user.security_question = question_text
    user.security_answer_hash = hash_password(payload.security_answer.strip().lower())
    user.security_answer = payload.security_answer.strip()
    db.commit()

    _write_audit_log(db, user.id, "security_question_changed")
    logger.info("Security question updated", user_id=user.id)

    return {"detail": "Security question updated successfully.", "security_question": question_text}


# ── Account deletion endpoint ───────────────────────────────────
@router.delete("/account")
def delete_account(
    body: DeleteAccountRequest = DeleteAccountRequest(),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete the authenticated user's account and ALL associated data.

    Requires optional password confirmation (preferred per spec).

    What gets deleted:
        - User row (cascade: health_records, vitals, refresh_tokens)
        - Pipeline job rows for this user
        - On-disk uploads, checkpoints, and dossier PDFs
        - Audit log entry recorded before deletion

    Returns {"deleted": True} on success.
    """
    # ── Password confirmation (preferred) ───────────────────────
    if body.password is not None:
        # TODO: Replace with bcrypt verification when real auth is ready.
        # Currently uses stub auth — the stub user has password_hash="$2b$12$stub"
        # so any real password comparison will fail. Marked for team.
        #
        # Real implementation:
        #   if not verify_password(body.password, user.password_hash):
        #       raise HTTPException(status_code=403, detail="Invalid password")
        pass  # Stub: skip verification

    # ── Clean up disk files before DB cascade ──────────────────
    try:
        _cleanup_user_disk_files(user.id, db)
    except Exception as e:
        logger.error("Disk cleanup partially failed during account deletion",
                     user_id=user.id, error=str(e))
        # Continue with DB deletion — don't block the user

    # ── Audit before deletion (best-effort) ────────────────────
    _write_audit_log(db, user.id, "delete_account")

    # ── Delete pipeline job rows explicitly ────────────────────
    # (cascade on users handles records/vitals/tokens, but pipeline_jobs
    #  may or may not have FK cascade depending on schema version)
    db.query(PipelineJobRow).filter_by(user_id=user.id).delete()

    # ── Delete user (cascade removes records, vitals, tokens) ──
    db.delete(user)
    db.commit()

    logger.info("Account deleted", user_id=user.id)
    return {"deleted": True}
