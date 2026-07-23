"""
backend/vitals.py — Daily vitals check-in and trends endpoints.

    POST /vitals/checkin   — Save a vitals snapshot for authenticated user
    GET  /vitals/trends     — Return historical vitals with baseline z-scores
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.db.models import User, VitalSnapshot
from app.db.session import get_db
from backend.baseline import baseline_summary

router = APIRouter(prefix="/vitals", tags=["vitals"])


# ── Request model ───────────────────────────────────────────────
class VitalsCheckIn(BaseModel):
    systolic_bp: int | None = Field(None, ge=50, le=300)
    diastolic_bp: int | None = Field(None, ge=20, le=200)
    heart_rate: int | None = Field(None, ge=20, le=300)
    spo2: float | None = Field(None, ge=50.0, le=100.0)
    temperature_c: float | None = Field(None, ge=30.0, le=45.0)
    glucose_mg_dl: float | None = Field(None, ge=20.0, le=600.0)
    weight_kg: float | None = Field(None, ge=10.0, le=500.0)
    notes: str | None = Field(None, max_length=500)


_VITAL_FIELDS = (
    ("systolic_bp", "systolic BP"),
    ("diastolic_bp", "diastolic BP"),
    ("heart_rate", "heart rate"),
    ("spo2", "SpO2"),
    ("temperature_c", "temperature"),
    ("glucose_mg_dl", "glucose"),
    ("weight_kg", "weight"),
)


@router.post("/checkin")
def vitals_checkin(
    body: VitalsCheckIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save a daily vitals snapshot. At least one vital required."""
    values = {
        "systolic_bp": body.systolic_bp,
        "diastolic_bp": body.diastolic_bp,
        "heart_rate": body.heart_rate,
        "spo2": body.spo2,
        "temperature_c": body.temperature_c,
        "glucose_mg_dl": body.glucose_mg_dl,
        "weight_kg": body.weight_kg,
    }
    if all(v is None for v in values.values()):
        raise HTTPException(status_code=400, detail="At least one vital must be provided")

    snapshot = VitalSnapshot(
        user_id=user.id,
        systolic_bp=body.systolic_bp,
        diastolic_bp=body.diastolic_bp,
        heart_rate=body.heart_rate,
        spo2=body.spo2,
        temperature_c=body.temperature_c,
        glucose_mg_dl=body.glucose_mg_dl,
        weight_kg=body.weight_kg,
        notes=body.notes,
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)

    return {
        "id": snapshot.id,
        "created_at": snapshot.created_at.isoformat(),
        "message": "Vitals saved",
    }


@router.get("/trends")
def vitals_trends(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return historical vitals with baseline z-scores per vital."""
    snapshots = (
        db.query(VitalSnapshot)
        .filter_by(user_id=user.id)
        .order_by(VitalSnapshot.created_at.asc())
        .all()
    )

    timeline = []
    for s in snapshots:
        timeline.append({
            "id": s.id,
            "created_at": s.created_at.isoformat(),
            "systolic_bp": s.systolic_bp,
            "diastolic_bp": s.diastolic_bp,
            "heart_rate": s.heart_rate,
            "spo2": s.spo2,
            "temperature_c": s.temperature_c,
            "glucose_mg_dl": s.glucose_mg_dl,
            "weight_kg": s.weight_kg,
            "notes": s.notes,
        })

    if not snapshots:
        return {"baselines": [], "timeline": [], "sample_count": 0}

    latest = snapshots[-1]
    baselines = []

    for attr, display_name in _VITAL_FIELDS:
        history = [getattr(s, attr) for s in snapshots if getattr(s, attr) is not None]
        current = getattr(latest, attr)

        if current is None or len(history) < 3:
            baselines.append({
                "vital": display_name,
                "current": current,
                "mean": (sum(history) / len(history)) if history else None,
                "std": None,
                "z_score": None,
                "sample_size": len(history),
                "interpretation": None,
                "insufficient_history": True,
            })
        else:
            baselines.append(baseline_summary(
                current=current, history=history, vital_name=display_name,
            ))

    return {
        "baselines": baselines,
        "timeline": timeline,
        "sample_count": len(snapshots),
    }
