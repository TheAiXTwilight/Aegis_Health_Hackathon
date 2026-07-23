from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """
    Pipeline job lifecycle states.

    Transitions:
        QUEUED    → RUNNING    (worker dequeues)
        RUNNING   → COMPLETED  (pipeline finishes cleanly)
        RUNNING   → FAILED     (timeout, exception, disconnected client)

    No other transitions are valid.
    Transitions enforced by backend.queue, not by this model.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineJob(BaseModel):
    """
    One submitted pipeline run.

    Lives in backend/queue.py's _job_store for JOB_RETENTION_SECONDS
    after reaching a terminal state, then purged.

    queue_position is intentionally NOT stored here. It is computed
    dynamically via get_queue_position(job_id) because stored values
    go stale as jobs ahead complete.

    Lifecycle invariants (documented only — not validated at runtime):
        submitted_at <= started_at        when started_at is set
        submitted_at <= completed_at      when completed_at is set
        started_at   <= completed_at      when both are set
    """

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str | None = None
    priority: int = Field(default=1, ge=1, le=5)
    status: JobStatus = JobStatus.QUEUED
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    schema_version: str = "1.1"