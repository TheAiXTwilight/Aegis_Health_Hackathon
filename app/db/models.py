"""
app/db/models.py — Updated SQLAlchemy ORM models with phone field on User.

Changes from original:
  - Added `phone` column to the `users` table (nullable string).

Run the following SQL to add the column to an existing database:
  ALTER TABLE users ADD COLUMN phone VARCHAR(20);
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


# ── User ─────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    username: Mapped[str | None] = mapped_column(String, unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    phone: Mapped[str | None] = mapped_column(String, nullable=True)
    security_question: Mapped[str | None] = mapped_column(String, nullable=True)
    security_answer_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    security_answer: Mapped[str | None] = mapped_column(String, nullable=True)
    role: Mapped[str] = mapped_column(String, default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    records = relationship("HealthRecord", back_populates="user", cascade="all, delete-orphan")
    vitals = relationship("VitalSnapshot", back_populates="user", cascade="all, delete-orphan")
    profile = relationship(
        "UserProfile",
        back_populates="user",
        cascade="all, delete-orphan",
        uselist=False,
    )


# ── UserProfile ─────────────────────────────────────────────────
class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    date_of_birth: Mapped[str] = mapped_column(String, nullable=False)
    sex: Mapped[str] = mapped_column(String, nullable=False)
    blood_group: Mapped[str] = mapped_column(String, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    height_cm: Mapped[float | None] = mapped_column(Float)
    allergies_json: Mapped[str] = mapped_column(Text, default="[]")
    medical_conditions_json: Mapped[str] = mapped_column(Text, default="[]")
    current_medications_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=now_utc,
        onupdate=now_utc,
    )

    user = relationship("User", back_populates="profile")


# ── RefreshToken ─────────────────────────────────────────────────
class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


# ── HealthRecord ─────────────────────────────────────────────────
class HealthRecord(Base):
    __tablename__ = "health_records"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    validation_status: Mapped[str | None] = mapped_column(String)
    symptoms_text: Mapped[str | None] = mapped_column(Text)
    medications_json: Mapped[str] = mapped_column(Text, default="[]")
    xray_findings_json: Mapped[str] = mapped_column(Text, default="[]")
    report_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    # Result-cache key this report was stored under (see backend/cache.py).
    # Nullable for rows created before this column existed. Lets report
    # deletion evict the exact matching entry from result_cache so a
    # deleted report can't be silently resurrected by a later cache hit
    # on the same symptoms/labs.
    cache_key: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    user = relationship("User", back_populates="records")


# ── VitalSnapshot ────────────────────────────────────────────────
class VitalSnapshot(Base):
    __tablename__ = "vital_snapshots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    systolic_bp: Mapped[int | None] = mapped_column(Integer)
    diastolic_bp: Mapped[int | None] = mapped_column(Integer)
    heart_rate: Mapped[int | None] = mapped_column(Integer)
    spo2: Mapped[float | None] = mapped_column(Float)
    temperature_c: Mapped[float | None] = mapped_column(Float)
    glucose_mg_dl: Mapped[float | None] = mapped_column(Float)
    weight_kg: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    user = relationship("User", back_populates="vitals")


# ── PipelineJobRow ───────────────────────────────────────────────
class PipelineJobRow(Base):
    __tablename__ = "pipeline_jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=5)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


# ── AuditLog ─────────────────────────────────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    action: Mapped[str] = mapped_column(String, nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String)
    resource_id: Mapped[str | None] = mapped_column(String)
    ip_hash: Mapped[str | None] = mapped_column(String)
    user_agent_hash: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


# ── ChatMessage ──────────────────────────────────────────────────
# Persists report follow-up chat history in the database instead of an
# in-process dict. This is the source of truth for turn counting — with
# multiple gunicorn/uvicorn workers, an in-memory dict gives each worker
# its own partial view of the conversation, so the turn badge can drift
# or undercount depending on which worker handles a given request. A DB
# row per message is visible to every worker and survives restarts.
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    # Monotonic position within this (user_id, job_id) conversation —
    # 0-based, user and assistant messages share the same sequence.
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    severity_delta: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    __table_args__ = (
        Index("ix_chat_messages_conversation", "user_id", "job_id", "seq"),
    )