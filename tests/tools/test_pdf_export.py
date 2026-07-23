"""
tests/tools/test_pdf_export.py — Tests for dossier HTML rendering.

Place at: tests/tools/test_pdf_export.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from backend.pdf_export import (
    render_dossier_html,
    _severity_color,
    _severity_bg,
    _validation_banner_html,
    _record_hash,
    _heatmap_html,
)


def _make_mock_record(**overrides):
    defaults = {
        "id": "rec-test-001",
        "job_id": "job-test-001",
        "severity": "HIGH",
        "confidence": 0.87,
        "validation_status": "agreement",
        "symptoms_text": "Cough and fever for 3 days",
        "medications_json": '["Paracetamol 500mg"]',
        "xray_findings_json": '["Consolidation right lower lobe"]',
        "report_json": json.dumps({
            "severity": "HIGH",
            "text": (
                "### Patient Information\n"
                "- Name: Asha Patient\n"
                "- Date of Birth: 12/03/1990\n\n"
                "### Submitted Information\n"
                "- Symptoms: Cough and fever\n\n"
                "### Summary\n"
                "Possible pneumonia — urgent evaluation advised.\n\n"
                "### Findings\n"
                "- **Reported symptoms:** Cough and fever\n\n"
                "### Recommendations\n"
                "- Seek qualified clinical review."
            ),
            "citations": [
                {"title": "Community-Acquired Pneumonia Guidelines", "source": "IDSA 2019"},
                {"title": "Radiographic Patterns in Pneumonia", "source": "Radiology 2023"},
            ],
        }),
        "result_json": json.dumps({
            "patient": {
                "name": "Asha Patient",
                "dob": "12/03/1990",
                "sex": "Female",
                "blood_group": "B+",
                "allergies": "Penicillin",
            },
            "submitted": {
                "symptoms_text": "Cough and fever for 3 days",
                "medications": ["Paracetamol 500mg"],
                "lab_pdf_uploaded": True,
                "xray_image_uploaded": False,
                "audio_uploaded": False,
                "xray_findings": [],
            },
            "severity": "HIGH",
        }),
        "created_at": datetime(2026, 7, 1, 10, 30, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


def _make_mock_user():
    mock = MagicMock()
    mock.id = "user-test-001"
    mock.display_name = "Priya Sharma"
    return mock


class TestDossierHTML:
    def test_contains_patient_preview_fields(self):
        html = render_dossier_html(
            record=_make_mock_record(),
            user=_make_mock_user(),
        )
        assert "Patient Preview" in html
        assert "Asha Patient" in html
        assert "12/03/1990" in html
        assert "Female" in html
        assert "B+" in html
        assert "Penicillin" in html

    def test_contains_severity(self):
        html = render_dossier_html(
            record=_make_mock_record(severity="HIGH"),
            user=_make_mock_user(),
        )
        assert "HIGH" in html

    def test_formats_report_text_instead_of_dumping_json(self):
        html = render_dossier_html(
            record=_make_mock_record(),
            user=_make_mock_user(),
        )
        assert "Clinical Assessment" in html
        assert "Reported symptoms:" in html
        assert "<strong>Reported symptoms:</strong>" in html
        assert '"severity": "HIGH"' not in html
        assert "schema_version" not in html

    def test_submitted_information_matches_result_payload(self):
        html = render_dossier_html(
            record=_make_mock_record(),
            user=_make_mock_user(),
        )
        assert "Lab Report / PDF Processed" in html
        assert "X-ray Image Processed" in html
        assert "Paracetamol 500mg" in html
        assert "Cough and fever for 3 days" in html

    def test_contains_job_id(self):
        html = render_dossier_html(
            record=_make_mock_record(job_id="job-abc-123"),
            user=_make_mock_user(),
        )
        assert "job-abc-123" in html

    def test_contains_medical_disclaimer(self):
        html = render_dossier_html(
            record=_make_mock_record(),
            user=_make_mock_user(),
        )
        assert "MEDICAL DISCLAIMER" in html
        assert "does not constitute a medical diagnosis" in html

    def test_contains_citations(self):
        html = render_dossier_html(
            record=_make_mock_record(),
            user=_make_mock_user(),
        )
        assert "IDSA 2019" in html
        assert "Radiology 2023" in html

    def test_contains_record_hash(self):
        html = render_dossier_html(
            record=_make_mock_record(id="rec-unique-123"),
            user=_make_mock_user(),
        )
        assert _record_hash("rec-unique-123") in html

    def test_validation_agreement_banner(self):
        html = render_dossier_html(
            record=_make_mock_record(validation_status="agreement"),
            user=_make_mock_user(),
        )
        assert "in agreement" in html

    def test_validation_warning_banner(self):
        html = render_dossier_html(
            record=_make_mock_record(validation_status="warning"),
            user=_make_mock_user(),
        )
        assert "Clinician review advised" in html

    def test_validation_override_banner(self):
        html = render_dossier_html(
            record=_make_mock_record(validation_status="override"),
            user=_make_mock_user(),
        )
        assert "Urgent clinician review" in html

    def test_empty_medications(self):
        result = json.loads(_make_mock_record().result_json)
        result["submitted"]["medications"] = []
        html = render_dossier_html(
            record=_make_mock_record(
                medications_json="[]",
                result_json=json.dumps(result),
            ),
            user=_make_mock_user(),
        )
        assert "None reported" in html

    def test_no_unrelated_heatmap_when_report_has_no_heatmap(self):
        assert _heatmap_html(_make_mock_record(), {"xray_result": {}}) == ""


class TestSeverityColors:
    def test_low_is_green(self):
        assert "10b981" in _severity_color("LOW")

    def test_high_is_red(self):
        assert "ef4444" in _severity_color("HIGH")

    def test_critical_is_dark_red(self):
        assert "991b1b" in _severity_color("CRITICAL")

    def test_unknown_is_gray(self):
        assert "6b7280" in _severity_color("UNKNOWN")


class TestRecordHash:
    def test_consistent(self):
        h1 = _record_hash("test-123")
        h2 = _record_hash("test-123")
        assert h1 == h2
        assert len(h1) == 12

