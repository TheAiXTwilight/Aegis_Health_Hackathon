"""Current ZIP export builder tests using real SQLAlchemy model instances.

The previous MagicMock fixture could not emulate SQLAlchemy's __table__ column
descriptor under current SQLAlchemy/Python versions. Real detached model
instances are lightweight and exercise the same serialisation contract.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import pytest

from app.db.models import AuditLog, HealthRecord, User, VitalSnapshot
from backend.zip_builder import _model_to_dict, build_user_zip


@pytest.fixture
def user() -> User:
    return User(
        id="user-test-001",
        email="priya@example.com",
        username="priya_sharma",
        display_name="Priya Sharma",
        password_hash="$2b$12$hashed",
        role="user",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        last_login_at=datetime(2026, 7, 3, tzinfo=timezone.utc),
    )


@pytest.fixture
def records() -> list[HealthRecord]:
    common = {
        "user_id": "user-test-001",
        "validation_status": "agreement",
        "medications_json": '["Paracetamol"]',
        "xray_findings_json": '["Consolidation"]',
        "created_at": datetime(2026, 7, 1, tzinfo=timezone.utc),
    }
    return [
        HealthRecord(
            id="rec-001",
            job_id="job-001",
            severity="HIGH",
            confidence=0.87,
            symptoms_text="Cough and fever",
            report_json='{"summary":"Possible pneumonia"}',
            result_json='{"severity":"HIGH"}',
            **common,
        ),
        HealthRecord(
            id="rec-002",
            job_id="job-002",
            severity="LOW",
            confidence=0.95,
            symptoms_text="Mild headache",
            report_json='{"summary":"Tension headache"}',
            result_json='{"severity":"LOW"}',
            **common,
        ),
    ]


@pytest.fixture
def vitals() -> list[VitalSnapshot]:
    return [
        VitalSnapshot(
            id="vit-001",
            user_id="user-test-001",
            systolic_bp=120,
            diastolic_bp=80,
            heart_rate=72,
            spo2=98.0,
            temperature_c=36.6,
            glucose_mg_dl=95.0,
            weight_kg=62.0,
            notes="Synthetic fixture",
            created_at=datetime(2026, 7, 3, tzinfo=timezone.utc),
        )
    ]


def read_zip(buf: io.BytesIO) -> zipfile.ZipFile:
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_model_to_dict_serializes_mapped_columns_and_datetimes(user):
    data = _model_to_dict(user)
    assert data["email"] == "priya@example.com"
    assert data["created_at"].startswith("2026-06-01T")


def test_zip_contains_required_entries_and_one_fhir_bundle_per_record(user, records, vitals):
    buf = build_user_zip(user=user, records=records, vitals=vitals)
    with read_zip(buf) as archive:
        names = set(archive.namelist())
        assert {"profile.json", "health_records.json", "vital_snapshots.json", "audit_metadata.json"} <= names
        assert {"fhir/bundle_rec-001.json", "fhir/bundle_rec-002.json"} <= names


def test_profile_and_records_are_safe_and_serialized(user, records, vitals):
    buf = build_user_zip(user=user, records=records, vitals=vitals)
    with read_zip(buf) as archive:
        profile = json.loads(archive.read("profile.json"))
        exported_records = json.loads(archive.read("health_records.json"))
        exported_vitals = json.loads(archive.read("vital_snapshots.json"))

    assert profile["display_name"] == "Priya Sharma"
    assert profile["email"] == "priya@example.com"
    assert "password_hash" not in profile
    assert "security_answer" not in profile
    assert len(exported_records) == 2
    assert all("report_json" not in item and "result_json" not in item for item in exported_records)
    assert exported_vitals[0]["heart_rate"] == 72


def test_fhir_bundles_have_collection_shape(user, records, vitals):
    buf = build_user_zip(user=user, records=records, vitals=vitals)
    with read_zip(buf) as archive:
        for name in archive.namelist():
            if name.startswith("fhir/"):
                bundle = json.loads(archive.read(name))
                assert bundle["resourceType"] == "Bundle"
                assert bundle["type"] == "collection"
                assert bundle["entry"]


def test_audit_metadata_and_empty_lists_are_valid(user):
    audit = AuditLog(
        id="audit-1",
        user_id="user-test-001",
        action="profile_update",
        resource_type="account",
        created_at=datetime(2026, 7, 3, tzinfo=timezone.utc),
    )
    populated = build_user_zip(user=user, records=[], vitals=[], audit_rows=[audit])
    with read_zip(populated) as archive:
        metadata = json.loads(archive.read("audit_metadata.json"))
        assert metadata["record_count"] == 0
        assert metadata["vital_snapshot_count"] == 0
        assert metadata["audit_log_entries"][0]["action"] == "profile_update"
        assert not [name for name in archive.namelist() if name.startswith("fhir/")]


def test_zip_buffer_is_valid_archive(user, records, vitals):
    buf = build_user_zip(user=user, records=records, vitals=vitals)
    buf.seek(0)
    assert zipfile.is_zipfile(buf)
