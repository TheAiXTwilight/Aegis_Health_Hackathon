"""
backend/zip_builder.py — In-memory ZIP builder for user data export.

Builds a ZIP containing:
    profile.json          — user profile (name, email, role, created_at)
    health_records.json   — all health records as JSON array
    vital_snapshots.json  — all vital check-ins as JSON array
    fhir/                 — one FHIR R4 Bundle per health record
    audit_metadata.json   — export timestamp, record count, schema version

PDF dossiers are included when the file exists on disk — skipped silently
when not generated yet (the PDF Clinical Dossier feature is a separate task).
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from backend.fhir_bundle import build_fhir_bundle

if TYPE_CHECKING:
    from app.db.models import User, HealthRecord, VitalSnapshot, AuditLog


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _model_to_dict(obj, exclude: set[str] | None = None) -> dict:
    """Convert a SQLAlchemy model row to a JSON-safe dict."""
    skip = exclude or set()
    result: dict = {}
    for col in obj.__table__.columns:
        if col.name in skip:
            continue
        val = getattr(obj, col.name)
        if isinstance(val, datetime):
            val = val.isoformat()
        result[col.name] = val
    return result


def build_user_zip(
    *,
    user: "User",
    records: list["HealthRecord"],
    vitals: list["VitalSnapshot"],
    audit_rows: list["AuditLog"] | None = None,
) -> io.BytesIO:
    """
    Build an in-memory ZIP file containing all exportable data
    owned by the authenticated user.

    Returns a BytesIO buffer positioned at the start.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:

        # ── 1. profile.json ────────────────────────────────────
        profile = {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
        }
        zf.writestr("profile.json", json.dumps(profile, indent=2, default=str))

        # ── 2. health_records.json ─────────────────────────────
        records_data = [
            _model_to_dict(r, exclude={"report_json", "result_json"})
            for r in records
        ]
        zf.writestr("health_records.json", json.dumps(records_data, indent=2, default=str))

        # ── 3. vital_snapshots.json ────────────────────────────
        vitals_data = [_model_to_dict(v) for v in vitals]
        zf.writestr("vital_snapshots.json", json.dumps(vitals_data, indent=2, default=str))

        # ── 4. FHIR bundles (one per record) ───────────────────
        for record in records:
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
            safe_id = record.id.replace("/", "_").replace("\\", "_")
            zf.writestr(
                f"fhir/bundle_{safe_id}.json",
                json.dumps(bundle, indent=2, default=str),
            )

        # ── 5. PDF dossiers (when generated on disk) ───────────
        pdf_dir = Path("/tmp/aegis_dossiers")
        for record in records:
            pdf_path = pdf_dir / f"{record.job_id}.pdf"
            if pdf_path.exists():
                zf.write(pdf_path, arcname=f"dossiers/{record.job_id}.pdf")

        # ── 6. audit_metadata.json ─────────────────────────────
        audit_data = {
            "exported_at": _now_iso(),
            "exported_by": user.display_name,
            "record_count": len(records),
            "vital_snapshot_count": len(vitals),
            "schema_version": "1.0.0",
            "audit_log_entries": (
                [_model_to_dict(a) for a in audit_rows]
                if audit_rows else []
            ),
        }
        zf.writestr("audit_metadata.json", json.dumps(audit_data, indent=2, default=str))

    buf.seek(0)
    return buf
