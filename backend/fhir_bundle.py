"""
backend/fhir_bundle.py — FHIR R4 Bundle builder.

Constructs a valid FHIR R4 Bundle (type: collection) from:
    - User            → Patient resource
    - HealthRecord    → Encounter, DiagnosticReport, Observations,
                        MedicationStatements, DocumentReference

FHIR R4 references:
    https://hl7.org/fhir/R4/bundle.html
    https://hl7.org/fhir/R4/resourcelist.html
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4 as _uuid4


def _now_iso() -> str:
    """Return current UTC timestamp in FHIR datetime format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ref(resource_type: str, resource_id: str) -> dict[str, str]:
    """Build a FHIR Reference to another resource in the same bundle."""
    return {"reference": f"{resource_type}/{resource_id}"}


def _full_url(resource_type: str, resource_id: str) -> str:
    """Build an absolute, valid FHIR Bundle.entry.fullUrl."""
    return f"https://aegis.local/fhir/{resource_type}/{resource_id}"


# ── Public builder ───────────────────────────────────────────────
def build_fhir_bundle(
    *,
    user_id: str,
    display_name: str,
    record_id: str,
    job_id: str,
    severity: str,
    confidence: float,
    validation_status: str | None,
    symptoms_text: str | None,
    medications_json: str,
    xray_findings_json: str,
    report_json: str,
    result_json: str,
    created_at: str | datetime,
) -> dict[str, Any]:
    """
    Build a FHIR R4 Bundle from a single HealthRecord and its owning User.

    Returns a dict ready for JSON serialization as a FHIR Bundle
    with resourceType "Bundle", type "collection".
    """
    patient_id = f"Patient-{user_id}"
    encounter_id = f"Encounter-{record_id}"
    report_id = f"DiagnosticReport-{record_id}"

    entries: list[dict[str, Any]] = []

    # ── 1. Patient ────────────────────────────────────────────────
    entries.append({
        "fullUrl": _full_url("Patient", patient_id),
        "resource": {
            "resourceType": "Patient",
            "id": patient_id,
            "identifier": [{
                "system": "urn:ietf:rfc:4122",
                "value": user_id,
            }],
            "name": [{"text": display_name}],
            "active": True,
        },
    })

    # ── 2. Encounter ──────────────────────────────────────────────
    entries.append({
        "fullUrl": _full_url("Encounter", encounter_id),
        "resource": {
            "resourceType": "Encounter",
            "id": encounter_id,
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "AMB",
                "display": "ambulatory",
            },
            "subject": _ref("Patient", patient_id),
            "period": {
                "start": _format_fhir_date(created_at),
            },
            "reasonCode": [
                {
                    "text": (symptoms_text or "No symptoms recorded")[:500],
                }
            ],
        },
    })

    # ── 3. DiagnosticReport ───────────────────────────────────────
    entries.append({
        "fullUrl": _full_url("DiagnosticReport", report_id),
        "resource": {
            "resourceType": "DiagnosticReport",
            "id": report_id,
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "RAD",
                    "display": "Radiology",
                }],
                "text": "AI Triage Assessment",
            }],
            "code": {
                "text": "Aegis Health AI Triage Report",
            },
            "subject": _ref("Patient", patient_id),
            "encounter": _ref("Encounter", encounter_id),
            "effectiveDateTime": _format_fhir_date(created_at),
            "issued": _now_iso(),
            "conclusion": (
                f"Triage Severity: {severity} "
                f"(Confidence: {confidence:.0%})"
            ),
            "presentedForm": [
                {
                    "contentType": "application/json",
                    "data": _b64(json.dumps({
                        "severity": severity,
                        "confidence": confidence,
                        "validation_status": validation_status,
                    })),
                    "title": "Triage Result Summary",
                },
            ],
        },
    })

    # ── 4. Observation: Severity ──────────────────────────────────
    obs_severity_id = f"Observation-Severity-{record_id}"
    entries.append({
        "fullUrl": _full_url("Observation", obs_severity_id),
        "resource": {
            "resourceType": "Observation",
            "id": obs_severity_id,
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "survey",
                    "display": "Survey",
                }],
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "95421-9",
                    "display": "Triage severity score",
                }],
                "text": "AI Triage Severity",
            },
            "subject": _ref("Patient", patient_id),
            "encounter": _ref("Encounter", encounter_id),
            "effectiveDateTime": _format_fhir_date(created_at),
            "valueString": severity,
            "interpretation": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    "code": _severity_interpretation(severity),
                }],
            }],
        },
    })

    # ── 5. Observation: Confidence ────────────────────────────────
    obs_conf_id = f"Observation-Confidence-{record_id}"
    entries.append({
        "fullUrl": _full_url("Observation", obs_conf_id),
        "resource": {
            "resourceType": "Observation",
            "id": obs_conf_id,
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "survey",
                }],
            }],
            "code": {
                "text": "AI Model Confidence Score",
            },
            "subject": _ref("Patient", patient_id),
            "encounter": _ref("Encounter", encounter_id),
            "effectiveDateTime": _format_fhir_date(created_at),
            "valueQuantity": {
                "value": round(confidence * 100, 2),
                "unit": "%",
                "system": "http://unitsofmeasure.org",
                "code": "%",
            },
        },
    })

    # ── 6. Observation: Symptoms (if any) ─────────────────────────
    if symptoms_text:
        obs_sx_id = f"Observation-Symptoms-{record_id}"
        entries.append({
            "fullUrl": _full_url("Observation", obs_sx_id),
            "resource": {
                "resourceType": "Observation",
                "id": obs_sx_id,
                "status": "final",
                "category": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "exam",
                        "display": "Exam",
                    }],
                }],
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": "75325-1",
                        "display": "Symptom",
                    }],
                    "text": "Patient Reported Symptoms",
                },
                "subject": _ref("Patient", patient_id),
                "encounter": _ref("Encounter", encounter_id),
                "effectiveDateTime": _format_fhir_date(created_at),
                "valueString": symptoms_text[:1000],
            },
        })

    # ── 7. MedicationStatements (if any) ──────────────────────────
    try:
        medications: list[str] = json.loads(medications_json)
    except (json.JSONDecodeError, TypeError):
        medications = []

    for idx, med_name in enumerate(medications, start=1):
        med_id = f"MedicationStatement-{record_id}-{idx}"
        entries.append({
            "fullUrl": _full_url("MedicationStatement", med_id),
            "resource": {
                "resourceType": "MedicationStatement",
                "id": med_id,
                "status": "recorded",
                "subject": _ref("Patient", patient_id),
                "medicationCodeableConcept": {
                    "text": med_name,
                },
                "effectiveDateTime": _format_fhir_date(created_at),
            },
        })

    # ── 8. Observation: X-ray Findings (if any) ───────────────────
    try:
        xray_findings: list[str] = json.loads(xray_findings_json)
    except (json.JSONDecodeError, TypeError):
        xray_findings = []

    if xray_findings:
        xray_obs_id = f"Observation-XRay-{record_id}"
        entries.append({
            "fullUrl": _full_url("Observation", xray_obs_id),
            "resource": {
                "resourceType": "Observation",
                "id": xray_obs_id,
                "status": "final",
                "category": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "imaging",
                        "display": "Imaging",
                    }],
                }],
                "code": {
                    "text": "Chest X-ray Findings",
                },
                "subject": _ref("Patient", patient_id),
                "encounter": _ref("Encounter", encounter_id),
                "effectiveDateTime": _format_fhir_date(created_at),
                "valueString": "; ".join(xray_findings),
            },
        })

    # ── 9. DocumentReference for the report JSON ──────────────────
    doc_ref_id = f"DocumentReference-{record_id}"
    entries.append({
        "fullUrl": _full_url("DocumentReference", doc_ref_id),
        "resource": {
            "resourceType": "DocumentReference",
            "id": doc_ref_id,
            "status": "current",
            "type": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "60591-5",
                    "display": "Patient Summary Document",
                }],
                "text": "Aegis AI Triage Report",
            },
            "subject": _ref("Patient", patient_id),
            "date": _format_fhir_date(created_at),
            "content": [{
                "attachment": {
                    "contentType": "application/json",
                    "data": _b64(report_json),
                    "title": f"Triage Report — {job_id}",
                },
            }],
        },
    })

    # ── Assemble Bundle ──────────────────────────────────────────
    return {
        "resourceType": "Bundle",
        "id": f"bundle-{record_id}",
        "type": "collection",
        "timestamp": _now_iso(),
        "entry": entries,
    }


# ── Helpers ──────────────────────────────────────────────────────
def _format_fhir_date(value: str | datetime) -> str:
    """Convert a datetime or ISO string to FHIR datetime format."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Already a string — return as-is (should already be ISO)
    return value


def _severity_interpretation(severity: str) -> str:
    """Map Aegis severity labels to FHIR v3 ObservationInterpretation codes."""
    mapping: dict[str, str] = {
        "LOW": "L",
        "MEDIUM": "A",
        "MODERATE": "A",
        "HIGH": "H",
        "CRITICAL": "HH",
    }
    return mapping.get(severity.upper(), "A")


def _b64(data: str) -> str:
    """Base64-encode a string for FHIR attachment data fields."""
    import base64
    return base64.b64encode(data.encode("utf-8")).decode("ascii")


