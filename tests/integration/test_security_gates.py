"""Strict expected-fail tests for source-derived P0 release gates.

They are intentionally not “made green” by weakening expectations. While a
known implementation gap remains, pytest reports XFAIL. Once the application
is fixed, remove the xfail decorator and retain the assertion as a normal
security regression test.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.mark.security_gate
@pytest.mark.xfail(strict=True, reason="Known P0: /admin/prewarm has no authentication or admin-role dependency.")
def test_admin_prewarm_rejects_anonymous_request(monkeypatch):
    from backend.model_registry import admin_router, model_registry

    monkeypatch.setattr(model_registry, "prewarm", AsyncMock(return_value={"ollama": True, "rag_embed": True}))
    app = FastAPI()
    app.include_router(admin_router)

    response = TestClient(app).post("/admin/prewarm")
    assert response.status_code == 401


@pytest.mark.security_gate
@pytest.mark.xfail(strict=True, reason="Known P0: queue status/result routes do not declare authenticated owner access.")
@pytest.mark.parametrize("endpoint_name", ["status", "result"])
def test_queue_phi_endpoints_require_current_user_dependency(endpoint_name):
    import backend.main as main

    endpoint = getattr(main, endpoint_name)
    parameters = inspect.signature(endpoint).parameters
    assert "user" in parameters


@pytest.mark.security_gate
@pytest.mark.xfail(strict=True, reason="Known P0: streaming and heatmap routes are public.")
def test_stream_and_heatmap_routes_declare_auth_dependency():
    import backend.main as main
    from backend.streaming import stream

    assert "user" in inspect.signature(stream).parameters
    assert "user" in inspect.signature(main.get_heatmap_by_job).parameters
    assert "user" in inspect.signature(main.get_heatmap_file).parameters


@pytest.mark.security_gate
@pytest.mark.xfail(strict=True, reason="Known P0: lab/xray validators accept any readable content.")
def test_lab_upload_validator_rejects_text_renamed_as_pdf(tmp_path):
    from backend.uploads import validate_lab_pdf

    spoofed = tmp_path / "payload.pdf"
    spoofed.write_text("<script>not a PDF</script>", encoding="utf-8")
    assert validate_lab_pdf(str(spoofed)) is not None


@pytest.mark.security_gate
@pytest.mark.xfail(strict=True, reason="Known P0: Permissions-Policy disables microphone despite VoiceRecorder feature.")
def test_security_headers_allow_first_party_microphone_after_user_consent():
    from backend.security import install_security

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    install_security(app)
    policy = TestClient(app).get("/health").headers["permissions-policy"]
    assert "microphone=()" not in policy


@pytest.mark.security_gate
@pytest.mark.xfail(strict=True, reason="Known P0: account deletion implementation currently accepts absent/wrong confirmation.")
def test_delete_account_implementation_contains_password_verification():
    from backend.account import delete_account

    source = inspect.getsource(delete_account)
    assert "verify_password_timing_safe" in source
    assert "pass  # Stub: skip verification" not in source
