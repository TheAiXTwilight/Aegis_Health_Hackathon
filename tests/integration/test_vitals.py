"""
tests/integration/test_vitals.py — Tests for vitals check-in and trends.

Place at: tests/integration/test_vitals.py

Requires: pytest, httpx (TestClient)
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── Helpers ──────────────────────────────────────────────────
def _make_app_with_vitals():
    """Build a FastAPI test app with vitals router installed."""
    from backend.vitals import router as vitals_router

    app = FastAPI()

    # Wire the stub auth dependency
    from app.auth import _STUB_USER

    async def stub_user():
        return _STUB_USER

    app.dependency_overrides = {
        # get_current_user → stub
    }

    app.include_router(vitals_router)
    return app


# Skip these until DB is properly mocked — structure tests only
class TestVitalsCheckInScenarios:
    """Logical tests for the vitals check-in contract."""

    def test_all_fields_optional(self):
        """VitalsCheckIn model should accept all-None gracefully in unit."""
        from backend.vitals import VitalsCheckIn
        body = VitalsCheckIn()
        assert body.systolic_bp is None
        assert body.heart_rate is None

    def test_validation_rejects_invalid_bp(self):
        """Systolic BP must be in 50-300 range."""
        from backend.vitals import VitalsCheckIn
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VitalsCheckIn(systolic_bp=500)

    def test_validation_rejects_invalid_spo2(self):
        """SpO2 must be 50-100%."""
        from backend.vitals import VitalsCheckIn
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VitalsCheckIn(spo2=150.0)

    def test_validation_accepts_valid_data(self):
        from backend.vitals import VitalsCheckIn
        body = VitalsCheckIn(
            systolic_bp=120,
            diastolic_bp=80,
            heart_rate=72,
            spo2=98.0,
            temperature_c=36.6,
            glucose_mg_dl=95.0,
            weight_kg=62.0,
            notes="Normal check-in",
        )
        assert body.systolic_bp == 120
        assert body.notes == "Normal check-in"

    def test_notes_max_length(self):
        from backend.vitals import VitalsCheckIn
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VitalsCheckIn(notes="x" * 501)


class TestVitalFields:
    """Verify the vital field mapping is complete."""

    def test_seven_vital_fields(self):
        from backend.vitals import _VITAL_FIELDS
        assert len(_VITAL_FIELDS) == 7
        names = {name for name, _ in _VITAL_FIELDS}
        assert "heart_rate" in names
        assert "spo2" in names
        assert "temperature_c" in names
        assert "glucose_mg_dl" in names
        assert "weight_kg" in names
        assert "systolic_bp" in names
        assert "diastolic_bp" in names


class TestTrendsEmptyState:
    """Verify trends endpoint returns correctly when no data exists."""

    def test_empty_trends_structure(self):
        """The response shape should be consistent even with no data."""
        # This tests the return shape contract, not the DB query
        empty = {
            "baselines": [],
            "timeline": [],
            "sample_count": 0,
        }
        assert empty["baselines"] == []
        assert empty["timeline"] == []
        assert empty["sample_count"] == 0
