"""
backend/exports.py — Data export endpoints.

    GET  /export/fhir/{record_id}  — FHIR R4 Bundle (single record)
    GET  /export/pdf/{job_id}      — PDF clinical dossier (stub)
    GET  /export/zip               — Full user data ZIP

All endpoints require authentication.
All record access is scoped to the authenticated user.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.db.models import User, HealthRecord, VitalSnapshot, AuditLog
from app.db.session import get_db
from backend.fhir_bundle import build_fhir_bundle
from backend.pdf_export import export_pdf as _pdf_handler
from backend.zip_builder import build_user_zip

router = APIRouter(prefix="/export", tags=["export"])


# ── FHIR R4 Export ──────────────────────────────────────────────
@router.get("/fhir/{record_or_job_id}")
def fhir_export(
    record_or_job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Download one user-owned HealthRecord as a FHIR R4 JSON Bundle.

    The identifier may be either the HealthRecord ID or its job ID so the
    Report page can export both persisted history rows and the active report.
    """
    record = (
        db.query(HealthRecord)
        .filter(
            HealthRecord.user_id == user.id,
            or_(
                HealthRecord.id == record_or_job_id,
                HealthRecord.job_id == record_or_job_id,
            ),
        )
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    bundle = build_fhir_bundle(
        user_id=user.id,
        display_name=user.display_name,
        record_id=record.id,
        job_id=record.job_id,
        severity=record.severity,
        confidence=record.confidence,
        validation_status=record.validation_status,
        symptoms_text=record.symptoms_text,
        medications_json=record.medications_json,
        xray_findings_json=record.xray_findings_json,
        report_json=record.report_json,
        result_json=record.result_json,
        created_at=record.created_at,
    )

    return JSONResponse(
        content=bundle,
        media_type="application/fhir+json",
        headers={
            "Content-Disposition": (
                f'attachment; filename="aegis_fhir_{record.job_id[:8]}.fhir.json"'
            ),
            "Cache-Control": "no-store",
        },
    )


# ── ZIP Data Export ─────────────────────────────────────────────
@router.get("/zip")
def export_zip(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Export all user-owned data as an in-memory ZIP file.

    ZIP contains:
        profile.json         — user account info
        health_records.json  — all triage records
        vital_snapshots.json — all vitals check-ins
        fhir/                — one FHIR R4 Bundle per record
        dossiers/            — PDF dossiers (if generated on disk)
        audit_metadata.json  — export timestamp and counts
    """
    records = (
        db.query(HealthRecord)
        .filter_by(user_id=user.id)
        .order_by(HealthRecord.created_at.desc())
        .all()
    )

    vitals = (
        db.query(VitalSnapshot)
        .filter_by(user_id=user.id)
        .order_by(VitalSnapshot.created_at.desc())
        .all()
    )

    audit_rows = (
        db.query(AuditLog)
        .filter_by(user_id=user.id)
        .order_by(AuditLog.created_at.desc())
        .limit(200)
        .all()
    )

    zip_buf = build_user_zip(
        user=user,
        records=records,
        vitals=vitals,
        audit_rows=audit_rows,
    )

    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="aegis_export_{user.id}.zip"'
            ),
        },
    )


