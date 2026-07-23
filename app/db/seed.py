"""
app/db/seed.py — Demo users and sample history for Aegis Health.

Provides:
    seed_demo_users(db)      — idempotent seed of Priya, Arjun, and Judge.
    clear_demo_users(db)   — remove demo users and their owned data.

Intended use:
    - Standalone CLI: `python scripts/seed_demo.py`
    - Auto-seed on startup when AEGIS_SEED_DEMO_USERS=true.

Demo credentials:
    priya@aegis.health   / demo1234
    arjun@aegis.health   / demo1234
    judge@aegis.health   / demo1234
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from loguru import logger
from sqlalchemy.orm import Session

from app.auth import hash_password
from app.db.models import HealthRecord, User, VitalSnapshot


DEMO_USERS = [
    {
        "email": "priya@aegis.health",
        "username": "priya",
        "display_name": "Priya Sharma",
        "password": "demo1234",
        "role": "user",
    },
    {
        "email": "arjun@aegis.health",
        "username": "arjun",
        "display_name": "Arjun Patel",
        "password": "demo1234",
        "role": "user",
    },
    {
        "email": "judge@aegis.health",
        "username": "judge",
        "display_name": "Dr. Ananya Rao",
        "password": "demo1234",
        "role": "judge",
    },
]


DEMO_HISTORY = {
    "priya@aegis.health": [
        {
            "severity": "moderate",
            "confidence": 0.78,
            "symptoms_text": "Fever, dry cough, and mild fatigue for 3 days.",
            "medications": ["Paracetamol", "Vitamin C"],
            "xray_findings": [],
            "report": {
                "summary": "Likely viral upper respiratory infection. Hydration and rest advised.",
                "recommendations": ["Rest", "Hydration", "Monitor fever"],
            },
        },
        {
            "severity": "low",
            "confidence": 0.85,
            "symptoms_text": "Seasonal allergies with nasal congestion and sneezing.",
            "medications": ["Cetirizine"],
            "xray_findings": [],
            "report": {
                "summary": "Allergic rhinitis. Antihistamine management appropriate.",
                "recommendations": ["Continue antihistamine", "Avoid known triggers"],
            },
        },
    ],
    "arjun@aegis.health": [
        {
            "severity": "high",
            "confidence": 0.82,
            "symptoms_text": "Chest tightness, shortness of breath, and palpitations.",
            "medications": ["Aspirin"],
            "xray_findings": ["Normal lung fields"],
            "report": {
                "summary": "Cardiopulmonary symptoms require urgent clinical evaluation.",
                "recommendations": ["Seek emergency care", "ECG and troponins"],
            },
        },
        {
            "severity": "moderate",
            "confidence": 0.74,
            "symptoms_text": "Recurrent tension headache with neck stiffness.",
            "medications": ["Ibuprofen"],
            "xray_findings": [],
            "report": {
                "summary": "Tension-type headache. Review ergonomics and stress levels.",
                "recommendations": ["Physiotherapy", "Stress management"],
            },
        },
    ],
}


DEMO_VITALS = {
    "priya@aegis.health": [
        {"systolic_bp": 118, "diastolic_bp": 76, "heart_rate": 72, "spo2": 98.0, "temperature_c": 37.1, "weight_kg": 58.0},
        {"systolic_bp": 120, "diastolic_bp": 78, "heart_rate": 75, "spo2": 97.5, "temperature_c": 37.8, "weight_kg": 57.8},
        {"systolic_bp": 116, "diastolic_bp": 74, "heart_rate": 70, "spo2": 98.2, "temperature_c": 36.9, "weight_kg": 57.8},
    ],
    "arjun@aegis.health": [
        {"systolic_bp": 134, "diastolic_bp": 86, "heart_rate": 88, "spo2": 96.0, "temperature_c": 37.0, "weight_kg": 74.5},
        {"systolic_bp": 128, "diastolic_bp": 82, "heart_rate": 82, "spo2": 97.0, "temperature_c": 36.8, "weight_kg": 74.2},
        {"systolic_bp": 130, "diastolic_bp": 84, "heart_rate": 85, "spo2": 96.5, "temperature_c": 37.1, "weight_kg": 74.0},
    ],
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_job_id() -> str:
    return f"demo-{uuid4()}"


def seed_demo_users(db: Session) -> list[User]:
    """
    Idempotently create demo users and their sample history.

    Returns the list of created/updated users. Existing users with the same
    email have their password hash reset to the demo password so the demo
    credentials always work.
    """
    created_users: list[User] = []

    for spec in DEMO_USERS:
        user = db.query(User).filter(User.email == spec["email"]).first()
        if user is not None:
            # Ensure demo credentials always work, even if the database was
            # previously seeded with a different password hash.
            user.password_hash = hash_password(spec["password"])
            user.is_active = True
            logger.info("Reset demo user password", email=user.email, username=user.username)
        else:
            # If the preferred username is already taken by another account,
            # skip the username so the demo user can still be created.
            username = spec["username"]
            if username and db.query(User).filter(User.username == username).first():
                logger.warning(
                    "Demo username already taken; creating demo user without username",
                    requested_username=username,
                    email=spec["email"],
                )
                username = None

            user = User(
                email=spec["email"],
                username=username,
                display_name=spec["display_name"],
                password_hash=hash_password(spec["password"]),
                role=spec["role"],
                is_active=True,
            )
            db.add(user)
            db.flush()
            logger.info("Created demo user", email=user.email, username=user.username, role=user.role)

        # Only seed history once per user
        history_count = db.query(HealthRecord).filter(HealthRecord.user_id == user.id).count()
        if history_count == 0:
            for idx, record in enumerate(DEMO_HISTORY.get(user.email, [])):
                report_json = json.dumps(record.get("report", {}))
                db.add(
                    HealthRecord(
                        user_id=user.id,
                        job_id=_new_job_id(),
                        severity=record["severity"],
                        confidence=record["confidence"],
                        symptoms_text=record.get("symptoms_text"),
                        medications_json=json.dumps(record.get("medications", [])),
                        xray_findings_json=json.dumps(record.get("xray_findings", [])),
                        report_json=report_json,
                        result_json=report_json,
                        created_at=_now() - timedelta(days=7 * (idx + 1)),
                    )
                )
            logger.info("Seeded demo health records", email=user.email, count=len(DEMO_HISTORY.get(user.email, [])))

        vitals_count = db.query(VitalSnapshot).filter(VitalSnapshot.user_id == user.id).count()
        if vitals_count == 0:
            for idx, vital in enumerate(DEMO_VITALS.get(user.email, [])):
                db.add(
                    VitalSnapshot(
                        user_id=user.id,
                        created_at=_now() - timedelta(days=2 * (idx + 1)),
                        **vital,
                    )
                )
            logger.info("Seeded demo vitals", email=user.email, count=len(DEMO_VITALS.get(user.email, [])))

        created_users.append(user)

    db.commit()
    return created_users


def clear_demo_users(db: Session) -> int:
    """Remove all demo users and their cascade-deleted data. Returns deletion count."""
    emails = {u["email"] for u in DEMO_USERS}
    users = db.query(User).filter(User.email.in_(emails)).all()
    count = len(users)
    for user in users:
        db.delete(user)
    db.commit()
    logger.info("Cleared demo users", count=count, emails=emails)
    return count