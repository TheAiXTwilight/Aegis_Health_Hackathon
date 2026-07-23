"""
tests/test_fhir_bundle.py — Tests for FHIR R4 Bundle export.
"""
from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone

from backend.fhir_bundle import build_fhir_bundle


@pytest.fixture
def sample_record():
    return {
        "user_id": "user-abc-123",
        "display_name": "Priya Sharma",
        "record_id": "rec-xyz-789",
        "job_id": "job-001",
        "severity": "HIGH",
        "confidence": 0.87,
        "validation_status": "agreement",
        "symptoms_text": "Persistent dry cough for 5 days, mild fever",
        "medications_json": json.dumps(["Paracetamol 500mg", "Cetirizine 10mg"]),
        "xray_findings_json": json.dumps(["Consolidation right lower lobe"]),
        "report_json": json.dumps({"summary": "Possible pneumonia — urgent evaluation advised"}),
        "result_json": json.dumps({"severity": "HIGH", "confidence": 0.87}),
        "created_at": datetime(2026, 7, 1, 10, 30, 0, tzinfo=timezone.utc),
    }


# ── Helpers ──────────────────────────────────────────────────
def _collect_resource_types(bundle: dict) -> set[str]:
    return {e["resource"]["resourceType"] for e in bundle.get("entry", [])}


def _find_resource(bundle: dict, resource_type: str, predicate=None) -> dict | None:
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == resource_type:
            if predicate is None or predicate(resource):
                return resource
    return None


def _severity_filter(resource: dict) -> bool:
    return "triage severity" in resource.get("code", {}).get("text", "").lower()


# ── Tests ────────────────────────────────────────────────────
class TestFHIRBundleStructure:
    def test_bundle_resource_type(self, sample_record):
        bundle = build_fhir_bundle(**sample_record)
        assert bundle["resourceType"] == "Bundle"

    def test_bundle_type_is_collection(self, sample_record):
        assert build_fhir_bundle(**sample_record)["type"] == "collection"

    def test_bundle_has_id_and_timestamp(self, sample_record):
        bundle = build_fhir_bundle(**sample_record)
        assert bundle["id"].startswith("bundle-")
        assert bundle["timestamp"].endswith("Z")

    def test_bundle_has_entries(self, sample_record):
        assert len(build_fhir_bundle(**sample_record)["entry"]) > 0

    def test_entry_full_urls_are_absolute_fhir_urls(self, sample_record):
        bundle = build_fhir_bundle(**sample_record)
        assert all(
            entry["fullUrl"].startswith("https://aegis.local/fhir/")
            for entry in bundle["entry"]
        )


class TestRequiredResources:
    def test_all_six_minimum_resources(self, sample_record):
        types = _collect_resource_types(build_fhir_bundle(**sample_record))
        for required in ["Patient", "Encounter", "DiagnosticReport",
                          "Observation", "MedicationStatement", "DocumentReference"]:
            assert required in types, f"Missing required resource: {required}"


class TestPatientResource:
    def test_patient_name_and_identifier(self, sample_record):
        patient = _find_resource(build_fhir_bundle(**sample_record), "Patient")
        assert patient["name"][0]["text"] == "Priya Sharma"
        identifiers = [i["value"] for i in patient.get("identifier", [])]
        assert "user-abc-123" in identifiers


class TestConfidenceObservation:
    def test_confidence_is_exported_as_percent(self, sample_record):
        bundle = build_fhir_bundle(**sample_record)
        observation = _find_resource(
            bundle,
            "Observation",
            lambda resource: resource.get("code", {}).get("text") == "AI Model Confidence Score",
        )
        assert observation["valueQuantity"]["value"] == 87.0
        assert observation["valueQuantity"]["unit"] == "%"


class TestSeverityMapping:
    def test_high_maps_to_H(self, sample_record):
        sample_record["severity"] = "HIGH"
        obs = _find_resource(build_fhir_bundle(**sample_record), "Observation", _severity_filter)
        codes = [c["code"] for interp in obs.get("interpretation", [])
                 for c in interp.get("coding", [])]
        assert "H" in codes

    def test_low_maps_to_L(self, sample_record):
        sample_record["severity"] = "LOW"
        obs = _find_resource(build_fhir_bundle(**sample_record), "Observation", _severity_filter)
        codes = [c["code"] for interp in obs.get("interpretation", [])
                 for c in interp.get("coding", [])]
        assert "L" in codes


class TestEdgeCases:
    def test_no_symptoms_does_not_crash(self, sample_record):
        sample_record["symptoms_text"] = None
        bundle = build_fhir_bundle(**sample_record)
        assert "Patient" in _collect_resource_types(bundle)

    def test_empty_medications(self, sample_record):
        sample_record["medications_json"] = "[]"
        bundle = build_fhir_bundle(**sample_record)
        med_count = sum(1 for e in bundle["entry"]
                        if e["resource"]["resourceType"] == "MedicationStatement")
        assert med_count == 0

    def test_invalid_json_medications(self, sample_record):
        sample_record["medications_json"] = "not valid {{{"
        bundle = build_fhir_bundle(**sample_record)
        med_count = sum(1 for e in bundle["entry"]
                        if e["resource"]["resourceType"] == "MedicationStatement")
        assert med_count == 0

    def test_empty_xray(self, sample_record):
        sample_record["xray_findings_json"] = "[]"
        bundle = build_fhir_bundle(**sample_record)
        xray_obs = [e for e in bundle["entry"]
                    if e["resource"].get("code", {}).get("text") == "Chest X-ray Findings"]
        assert len(xray_obs) == 0


class TestDocumentReference:
    def test_has_attachment(self, sample_record):
        doc = _find_resource(build_fhir_bundle(**sample_record), "DocumentReference")
        assert doc["status"] == "current"
        assert len(doc["content"]) > 0
        assert "attachment" in doc["content"][0]


