"""Authenticated report-history and report-deletion endpoints."""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.db.models import AuditLog, HealthRecord, PipelineJobRow, User
from app.db.session import get_db
from app.settings import settings
from backend.dashboard import build_report_measurement_groups, _utc_iso

router = APIRouter(tags=["records"])

# Registered by backend.main after backend.queue is imported. Keeping this as
# a callback avoids coupling the records router to the inference tool package.
_report_state_remover: Callable[[str], Any] | None = None


def register_report_state_remover(remover: Callable[[str], Any]) -> None:
    global _report_state_remover
    _report_state_remover = remover


def _json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _record_payload(record: HealthRecord) -> dict[str, Any]:
    report_data = _json_object(record.report_json)
    result_data = _json_object(record.result_json)
    result_report = result_data.get("report")
    if not isinstance(result_report, dict):
        result_report = {}
    result_data["measurement_groups"] = build_report_measurement_groups(result_data)

    return {
        "id": record.id,
        "job_id": record.job_id,
        "severity": record.severity,
        "confidence": record.confidence,
        "validation_status": record.validation_status,
        "symptoms_text": record.symptoms_text,
        "created_at": _utc_iso(record.created_at),
        "status": "completed",
        "report_text": report_data.get("text") or result_report.get("text") or "",
        "report_data": report_data,
        "result_data": result_data,
    }


def _cleanup_persisted_job_files(job_id: str, session_id: str | None) -> None:
    if session_id:
        shutil.rmtree(Path(settings.AEGIS_UPLOAD_ROOT) / session_id, ignore_errors=True)

    (Path(settings.AEGIS_CHECKPOINT_DIR) / f"{job_id}.json").unlink(missing_ok=True)
    (Path("/tmp/aegis_dossiers") / f"{job_id}.pdf").unlink(missing_ok=True)


@router.get("/records")
def list_records(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return only the authenticated user's persisted reports."""
    records = (
        db.query(HealthRecord)
        .filter(HealthRecord.user_id == user.id)
        .order_by(HealthRecord.created_at.desc())
        .limit(25)
        .all()
    )
    total = db.query(HealthRecord).filter(HealthRecord.user_id == user.id).count()
    return {
        "records": [_record_payload(record) for record in records],
        "total": total,
    }


@router.get("/records/{record_id}")
def get_record(
    record_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return one report when it belongs to the authenticated user."""
    record = (
        db.query(HealthRecord)
        .filter(
            HealthRecord.id == record_id,
            HealthRecord.user_id == user.id,
        )
        .first()
    )
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return _record_payload(record)


@router.delete("/queue/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_report(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """Permanently delete one completed report owned by the current user."""
    record = (
        db.query(HealthRecord)
        .filter(
            HealthRecord.job_id == job_id,
            HealthRecord.user_id == user.id,
        )
        .first()
    )
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    job_row = (
        db.query(PipelineJobRow)
        .filter(
            PipelineJobRow.job_id == job_id,
            PipelineJobRow.user_id == user.id,
        )
        .first()
    )
    session_id = job_row.session_id if job_row else None
    record_cache_key = record.cache_key

    db.delete(record)
    if job_row is not None:
        db.delete(job_row)
    db.add(
        AuditLog(
            user_id=user.id,
            action="report_delete",
            resource_type="health_record",
            resource_id=job_id,
        )
    )
    db.commit()

    # Evict the result-cache entry this report was served from (if any),
    # so a later submission with the same symptoms/medications/xray/lab
    # combination can't silently resurrect the just-deleted report via a
    # cache hit. Rows persisted before the cache_key column existed have
    # no key here — nothing to evict for those, which just means the old
    # stale-cache-hit behavior persists for pre-migration reports only.
    from backend.cache import result_cache
    result_cache.delete(record_cache_key)

    # Remove retained in-memory state and files only after the DB transaction
    # succeeds. The helper also invokes the registered upload cleanup callback.
    removed_job = (
        _report_state_remover(job_id)
        if _report_state_remover is not None
        else None
    )
    if removed_job is None:
        _cleanup_persisted_job_files(job_id, session_id)
    else:
        (Path(settings.AEGIS_CHECKPOINT_DIR) / f"{job_id}.json").unlink(missing_ok=True)
        (Path("/tmp/aegis_dossiers") / f"{job_id}.pdf").unlink(missing_ok=True)

    return Response(status_code=status.HTTP_204_NO_CONTENT)