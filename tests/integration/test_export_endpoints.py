"""New integration tests for owner-scoped FHIR and ZIP exports."""
from __future__ import annotations

import io
import json
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth import get_current_user
from app.db.models import AuditLog, HealthRecord, VitalSnapshot
from tests.integration._support import (
    install_db_override,
    make_session_factory,
    mutable_user_override,
    seed_user,
)


@pytest.fixture
def session_factory():
    return make_session_factory()


@pytest.fixture
def client(session_factory):
    from backend.exports import router as exports_router

    seed_user(session_factory, user_id="alice-id", email="alice@example.test", display_name="Alice")
    seed_user(
        session_factory,
        user_id="bob-id",
        email="bob@example.test",
        display_name="Bob",
        security_answer="Rex",
    )

    report = {"text": "Synthetic report", "severity": "HIGH", "citations": ["AHA-ACS-2024"]}
    result = {"report": report, "submitted": {"medications": ["aspirin"]}}
    with session_factory() as db:
        db.add(HealthRecord(
            id="alice-record",
            user_id="alice-id",
            job_id="alice-job",
            severity="HIGH",
            confidence=0.82,
            validation_status="agreement",
            symptoms_text="Synthetic chest pain",
            medications_json=json.dumps(["aspirin"]),
            xray_findings_json=json.dumps(["Cardiomegaly"]),
            report_json=json.dumps(report),
            result_json=json.dumps(result),
        ))
        db.add(HealthRecord(
            id="bob-record",
            user_id="bob-id",
            job_id="bob-job",
            severity="LOW",
            confidence=0.7,
            report_json=json.dumps({"text": "Bob report"}),
            result_json=json.dumps({"report": {"text": "Bob report"}}),
        ))
        db.add(VitalSnapshot(id="alice-vital", user_id="alice-id", heart_rate=72))
        db.add(AuditLog(user_id="alice-id", action="profile_update", resource_type="account"))
        db.commit()

    app = FastAPI()
    app.include_router(exports_router)
    install_db_override(app, session_factory)
    selected = {"value": "alice-id"}
    mutable_user_override(app, session_factory, selected, get_current_user)
    test_client = TestClient(app)
    test_client.selected_user = selected
    return test_client


@pytest.mark.parametrize("identifier", ["alice-record", "alice-job"])
def test_fhir_export_accepts_owned_record_id_or_job_id(client, identifier):
    response = client.get(f"/export/fhir/{identifier}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/fhir+json")
    assert response.headers["cache-control"] == "no-store"
    assert "attachment;" in response.headers["content-disposition"]

    bundle = response.json()
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert bundle["entry"]


def test_fhir_export_hides_other_users_record(client):
    client.selected_user["value"] = "bob-id"
    response = client.get("/export/fhir/alice-record")
    assert response.status_code == 404


def test_zip_export_contains_only_current_user_data_and_no_security_secret(client):
    response = client.get("/export/zip")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert "attachment;" in response.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        assert {"profile.json", "health_records.json", "vital_snapshots.json", "audit_metadata.json"} <= names
        assert "fhir/bundle_alice-record.json" in names

        profile = json.loads(archive.read("profile.json"))
        records = json.loads(archive.read("health_records.json"))
        vitals = json.loads(archive.read("vital_snapshots.json"))
        all_export_text = "\n".join(archive.read(name).decode("utf-8", errors="ignore") for name in names if name.endswith(".json"))

    assert profile["email"] == "alice@example.test"
    assert {record["job_id"] for record in records} == {"alice-job"}
    assert {vital["user_id"] for vital in vitals} == {"alice-id"}
    assert "bob@example.test" not in all_export_text
    assert "\"security_answer\"" not in all_export_text
    assert "security_answer_hash" not in all_export_text
    assert "password_hash" not in all_export_text


def test_zip_export_changes_with_authenticated_user(client):
    client.selected_user["value"] = "bob-id"
    response = client.get("/export/zip")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        profile = json.loads(archive.read("profile.json"))
        records = json.loads(archive.read("health_records.json"))
    assert profile["email"] == "bob@example.test"
    assert {record["job_id"] for record in records} == {"bob-job"}
