"""
tests/tools/test_zip_builder.py — Tests for ZIP export builder.

Place this at: tests/tools/test_zip_builder.py
"""
from __future__ import annotations

import json
import zipfile
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from backend.zip_builder import build_user_zip, _model_to_dict


# ── Helpers ──────────────────────────────────────────────────
def _make_mock(sqlalchemy_class, **kwargs):
    """Create a mock row approximating a SQLAlchemy model."""
    mock = MagicMock(spec=sqlalchemy_class)
    for key, val in kwargs.items():
        setattr(mock, key, val)
    # Simulate __table__.columns for _model_to_dict
    col_names = list(kwargs.keys())
    mock_cols = []
    for name in col_names:
        mc = MagicMock()
        mc.name = name
        mock_cols.append(mc)
    mock.__table__.columns = mock_cols
    return mock


def _read_zip_entry(buf, name):
    """Read a single entry from ZIP bytes as string."""
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        return zf.read(name).decode("utf-8")


def _list_zip_entries(buf):
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        return zf.namelist()


# ── Fixtures ─────────────────────────────────────────────────
@pytest.fixture
def mock_user():
    return _make_mock(
        None,  # No real model class needed — _model_to_dict uses __table__.columns
        id="user-test-001",
        email="priya@example.com",
        username="priya_sharma",
        display_name="Priya Sharma",
        role="user",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        last_login_at=datetime(2026, 7, 3, tzinfo=timezone.utc),
        is_active=True,
        password_hash="$2b$12$hashed",
    )


@pytest.fixture
def mock_records():
    return [
        _make_mock(
            None,
            id="rec-001",
            user_id="user-test-001",
            job_id="job-001",
            severity="HIGH",
            confidence=0.87,
            validation_status="agreement",
            symptoms_text="Cough and fever",
            medications_json='["Paracetamol"]',
            xray_findings_json='["Consolidation"]',
            report_json='{"summary":"Possible pneumonia"}',
            result_json='{"severity":"HIGH"}',
            created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        ),
        _make_mock(
            None,
            id="rec-002",
            user_id="user-test-001",
            job_id="job-002",
            severity="LOW",
            confidence=0.95,
            validation_status="agreement",
            symptoms_text="Mild headache",
            medications_json="[]",
            xray_findings_json="[]",
            report_json='{"summary":"Tension headache"}',
            result_json='{"severity":"LOW"}',
            created_at=datetime(2026, 6, 28, tzinfo=timezone.utc),
        ),
    ]


@pytest.fixture
def mock_vitals():
    return [
        _make_mock(
            None,
            id="vit-001",
            user_id="user-test-001",
            systolic_bp=120,
            diastolic_bp=80,
            heart_rate=72,
            spo2=98.0,
            temperature_c=36.6,
            glucose_mg_dl=95.0,
            weight_kg=62.0,
            notes="Feeling good",
            created_at=datetime(2026, 7, 3, tzinfo=timezone.utc),
        ),
    ]


# ── Tests ────────────────────────────────────────────────────
class TestZIPContents:
    """Verify the ZIP contains all required entries."""

    def test_has_profile_json(self, mock_user, mock_records, mock_vitals):
        buf = build_user_zip(user=mock_user, records=mock_records, vitals=mock_vitals)
        assert "profile.json" in _list_zip_entries(buf)

    def test_has_health_records_json(self, mock_user, mock_records, mock_vitals):
        buf = build_user_zip(user=mock_user, records=mock_records, vitals=mock_vitals)
        assert "health_records.json" in _list_zip_entries(buf)

    def test_has_vital_snapshots_json(self, mock_user, mock_records, mock_vitals):
        buf = build_user_zip(user=mock_user, records=mock_records, vitals=mock_vitals)
        assert "vital_snapshots.json" in _list_zip_entries(buf)

    def test_has_fhir_bundles(self, mock_user, mock_records, mock_vitals):
        buf = build_user_zip(user=mock_user, records=mock_records, vitals=mock_vitals)
        entries = _list_zip_entries(buf)
        fhir_entries = [e for e in entries if e.startswith("fhir/")]
        assert len(fhir_entries) == 2  # one per record

    def test_has_audit_metadata(self, mock_user, mock_records, mock_vitals):
        buf = build_user_zip(user=mock_user, records=mock_records, vitals=mock_vitals)
        assert "audit_metadata.json" in _list_zip_entries(buf)


class TestProfileJSON:
    def test_contains_user_fields(self, mock_user, mock_records, mock_vitals):
        buf = build_user_zip(user=mock_user, records=mock_records, vitals=mock_vitals)
        profile = json.loads(_read_zip_entry(buf, "profile.json"))
        assert profile["display_name"] == "Priya Sharma"
        assert profile["email"] == "priya@example.com"
        assert profile["role"] == "user"
        assert "password_hash" not in profile  # security: never export hashes

    def test_dates_are_iso_strings(self, mock_user, mock_records, mock_vitals):
        buf = build_user_zip(user=mock_user, records=mock_records, vitals=mock_vitals)
        profile = json.loads(_read_zip_entry(buf, "profile.json"))
        assert "T" in profile["created_at"]


class TestHealthRecordsJSON:
    def test_record_count(self, mock_user, mock_records, mock_vitals):
        buf = build_user_zip(user=mock_user, records=mock_records, vitals=mock_vitals)
        records = json.loads(_read_zip_entry(buf, "health_records.json"))
        assert len(records) == 2

    def test_no_report_json_leak(self, mock_user, mock_records, mock_vitals):
        """Report JSON is large and sent separately as FHIR — exclude from records list."""
        buf = build_user_zip(user=mock_user, records=mock_records, vitals=mock_vitals)
        records = json.loads(_read_zip_entry(buf, "health_records.json"))
        for rec in records:
            assert "report_json" not in rec
            assert "result_json" not in rec


class TestVitallSnapshotsJSON:
    def test_contains_vitals(self, mock_user, mock_records, mock_vitals):
        buf = build_user_zip(user=mock_user, records=mock_records, vitals=mock_vitals)
        vitals = json.loads(_read_zip_entry(buf, "vital_snapshots.json"))
        assert len(vitals) == 1
        assert vitals[0]["systolic_bp"] == 120
        assert vitals[0]["spo2"] == 98.0


class TestFHIRBundles:
    def test_bundles_are_valid_fhir(self, mock_user, mock_records, mock_vitals):
        buf = build_user_zip(user=mock_user, records=mock_records, vitals=mock_vitals)
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            for name in zf.namelist():
                if name.startswith("fhir/"):
                    bundle = json.loads(zf.read(name))
                    assert bundle["resourceType"] == "Bundle"
                    assert bundle["type"] == "collection"
                    assert "entry" in bundle


class TestAuditMetadata:
    def test_has_export_info(self, mock_user, mock_records, mock_vitals):
        buf = build_user_zip(user=mock_user, records=mock_records, vitals=mock_vitals)
        audit = json.loads(_read_zip_entry(buf, "audit_metadata.json"))
        assert audit["record_count"] == 2
        assert audit["vital_snapshot_count"] == 1
        assert audit["schema_version"] == "1.0.0"
        assert "exported_at" in audit
        assert "exported_by" in audit


class TestEdgeCases:
    def test_empty_records_and_vitals(self, mock_user):
        buf = build_user_zip(user=mock_user, records=[], vitals=[])
        entries = _list_zip_entries(buf)
        assert "profile.json" in entries
        assert "health_records.json" in entries
        assert "vital_snapshots.json" in entries
        assert "audit_metadata.json" in entries
        # No FHIR bundles when no records
        fhir = [e for e in entries if e.startswith("fhir/")]
        assert len(fhir) == 0

    def test_no_vitals(self, mock_user, mock_records):
        buf = build_user_zip(user=mock_user, records=mock_records, vitals=[])
        vitals = json.loads(_read_zip_entry(buf, "vital_snapshots.json"))
        assert vitals == []

    def test_zip_is_valid(self, mock_user, mock_records, mock_vitals):
        buf = build_user_zip(user=mock_user, records=mock_records, vitals=mock_vitals)
        buf.seek(0)
        assert zipfile.is_zipfile(buf) is True
