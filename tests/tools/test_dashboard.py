"""
tests/tools/test_dashboard.py — Tests for dashboard data builders.

Place at: tests/tools/test_dashboard.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from backend.dashboard import (
    _build_overall,
    _build_risk,
    _build_safety,
    _build_actions,
    _build_criticals,
    build_report_measurement_groups,
    _extract_contributing_factors,
    _extract_report_measurements,
    _json_list,
    _utc_iso,
)


def _record(**kw):
    defaults = {
        "id": "rec-001", "job_id": "job-001", "severity": "HIGH",
        "confidence": 0.87, "validation_status": "agreement",
        "symptoms_text": "Cough and fever",
        "medications_json": '["Paracetamol"]',
        "xray_findings_json": '["Consolidation"]',
        "report_json": '{"summary":"test"}',
        "result_json": '{"severity":"HIGH"}',
        "created_at": datetime(2026, 7, 1, tzinfo=timezone.utc),
    }
    defaults.update(kw)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


class TestDashboardMedications:
    def test_valid_medication_json(self):
        assert _json_list('["Metformin 500 mg", "Aspirin"]') == [
            "Metformin 500 mg",
            "Aspirin",
        ]

    def test_invalid_or_empty_medication_json(self):
        assert _json_list("not-json") == []
        assert _json_list("[]") == []


class TestDashboardTimestamps:
    def test_naive_sqlite_timestamp_is_serialized_as_utc(self):
        value = datetime(2026, 7, 10, 13, 15, 0)
        assert _utc_iso(value) == "2026-07-10T13:15:00Z"

    def test_aware_timestamp_is_normalized_to_utc(self):
        value = datetime(2026, 7, 10, 13, 15, 0, tzinfo=timezone.utc)
        assert _utc_iso(value) == "2026-07-10T13:15:00Z"

    def test_none_timestamp_stays_none(self):
        assert _utc_iso(None) is None


class TestDynamicReportData:
    def test_extracts_latest_report_vitals_and_labs(self):
        result_data = {
            "vitals": {
                "heart_rate": {"value": 112, "status": "high", "risk_score": 1},
                "spo2": {"value": 92, "status": "low", "risk_score": 1},
                "temperature_c": 38.2,
            },
            "lab_result": {
                "measurements": {
                    "haemoglobin": 10.4,
                    "glucose": 225,
                },
                "extra_measurements": {"vitamin_d": 12.4},
                "abnormal_values": [
                    "Low vitamin D: 12.4 ng/mL (commonly considered insufficient below 30)"
                ],
            },
        }

        measurements = _extract_report_measurements(result_data)
        by_key = {item["key"]: item for item in measurements}

        assert by_key["heart_rate"]["value"] == 112
        assert by_key["heart_rate"]["status"] == "high"
        assert by_key["spo2"]["status"] == "low"
        assert by_key["temperature_c"]["display_value"] == "38.2 °C"
        assert by_key["haemoglobin"]["source"] == "lab"
        assert by_key["glucose"]["risk_score"] == 1
        assert by_key["vitamin_d"]["status"] == "low"
        assert by_key["vitamin_d"]["risk_score"] == 1
        assert "Low vitamin D" in by_key["vitamin_d"]["note"]

    def test_measurement_groups_match_vitals_overview_categories(self):
        result_data = {
            "vitals": {
                "heart_rate": {"value": 142, "status": "critical_high", "risk_score": 2},
                "spo2": {"value": 88, "status": "critical_low", "risk_score": 2},
            },
            "lab_result": {
                "measurements": {"haemoglobin": 10.4, "glucose": 225},
                "extra_measurements": {"vitamin_d": 12.4},
                "abnormal_values": ["Low vitamin D: 12.4 ng/mL"],
            },
        }
        groups = build_report_measurement_groups(result_data)

        assert groups["counts"] == {
            "critical": 2,
            "under_observation": 2,
            "normal": 0,
        }
        assert groups["critical"][0]["name"] == "Heart Rate"
        assert all(item["risk_score"] == 1 for item in groups["under_observation"])

    def test_factors_come_from_latest_structured_result(self):
        record = _record(symptoms_text="Dizziness and fatigue")
        result_data = {
            "severity_result": {
                "reasons": ["Elevated glucose requires clinical review"],
            },
            "lab_result": {
                "abnormal_values": ["High glucose: 225 mg/dL"],
            },
            "xray_result": {"findings": []},
            "drug_result": {"interactions": [], "warnings": []},
            "symptom_result": {"symptoms": ["dizziness"]},
        }
        measurements = _extract_report_measurements(result_data)
        factors = _extract_contributing_factors(record, result_data, measurements)

        assert factors[0] == "Elevated glucose requires clinical review"
        assert "High glucose: 225 mg/dL" in factors
        assert all("heart rate" not in factor.lower() for factor in factors)

    def test_no_measurements_returns_no_fake_vitals(self):
        assert _extract_report_measurements({}) == []


class TestOverall:
    def test_no_record(self):
        result = _build_overall(None, [])
        assert result["status"] == "No Data"
        assert result["data_completeness_score"] == 0

    def test_with_record(self):
        result = _build_overall(_record(), [_record()])
        assert result["score"] > 0
        assert "status" in result
        assert result["data_completeness_score"] > 0

    def test_trend_first_assessment(self):
        result = _build_overall(_record(), [_record()])
        assert result["trend"] == "First assessment"

    def test_worsening_trend(self):
        r1 = _record(severity="LOW")
        r2 = _record(severity="HIGH")
        result = _build_overall(r2, [r2, r1])
        assert "Worsening" in result["trend"]

    def test_improving_trend(self):
        r1 = _record(severity="HIGH")
        r2 = _record(severity="LOW")
        result = _build_overall(r2, [r2, r1])
        assert "Improving" in result["trend"]


class TestRisk:
    def test_no_record(self):
        result = _build_risk(None, [])
        assert result["status"] == "Unknown"

    def test_high_severity(self):
        result = _build_risk(_record(severity="CRITICAL"), [])
        assert result["score"] >= 75

    def test_low_severity(self):
        result = _build_risk(_record(severity="LOW"), [])
        assert result["score"] <= 35

    def test_override_warning(self):
        result = _build_risk(_record(validation_status="override"), [])
        assert any("override" in f.lower() for f in result["factors"])


class TestSafety:
    def test_four_items(self):
        result = _build_safety(_record())
        assert len(result) == 4

    def test_no_record(self):
        result = _build_safety(None)
        assert any("No triage" in r["name"] for r in result)


class TestActions:
    def test_returns_list(self):
        result = _build_actions(_record())
        assert len(result) >= 1

    def test_critical_severity(self):
        result = _build_actions(_record(severity="CRITICAL"))
        assert any("emergency" in a.lower() for a in result)


class TestCriticals:
    def test_returns_cards(self):
        result = _build_criticals(_record())
        assert len(result) >= 1

    def test_critical_severity(self):
        result = _build_criticals(_record(severity="CRITICAL"))
        assert any("Critical" in c["badge"] for c in result)

