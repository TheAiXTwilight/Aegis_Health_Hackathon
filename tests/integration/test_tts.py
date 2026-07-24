"""
tests/integration/test_tts.py — POST /tts/speak endpoint.

Follows the app-with-dependency-override pattern: builds a minimal
FastAPI app with only the tts_router installed, overrides
get_current_user so no real auth/DB is needed, and mocks
tools.tts_synthesizer.synthesize_speech so no real model is required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app():
    """Build a FastAPI test app with the tts router and a stub user."""
    from app.auth import get_current_user
    from backend.tts import router as tts_router

    app = FastAPI()
    app.include_router(tts_router)

    stub_user = SimpleNamespace(id="test-user-1", email="test@example.com")

    async def _stub_get_current_user():
        return stub_user

    app.dependency_overrides[get_current_user] = _stub_get_current_user
    return app


@pytest.fixture
def client():
    return TestClient(_make_app())


class TestSpeakEndpoint:
    def test_success_returns_wav_audio(self, client):
        with patch("backend.tts.synthesize_speech") as mock_synth:
            mock_synth.return_value = b"RIFF....WAVEfmt fake-wav-bytes"
            resp = client.post("/tts/speak", json={"text": "Hello world"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content == b"RIFF....WAVEfmt fake-wav-bytes"

    def test_empty_text_rejected_by_schema(self, client):
        resp = client.post("/tts/speak", json={"text": ""})
        assert resp.status_code == 422  # pydantic min_length=1 violation

    def test_missing_text_field_rejected(self, client):
        resp = client.post("/tts/speak", json={})
        assert resp.status_code == 422

    def test_model_missing_returns_503(self, client):
        from tools.tts_synthesizer import TTSError

        with patch("backend.tts.synthesize_speech") as mock_synth:
            mock_synth.side_effect = TTSError("no model", reason="model_missing")
            resp = client.post("/tts/speak", json={"text": "hello"})
        assert resp.status_code == 503
        assert "no model" in resp.json()["detail"]

    def test_text_too_long_returns_413(self, client):
        from tools.tts_synthesizer import TTSError

        with patch("backend.tts.synthesize_speech") as mock_synth:
            mock_synth.side_effect = TTSError("too long", reason="text_too_long")
            resp = client.post("/tts/speak", json={"text": "hello"})
        assert resp.status_code == 413

    def test_generic_synthesis_failure_returns_500(self, client):
        from tools.tts_synthesizer import TTSError

        with patch("backend.tts.synthesize_speech") as mock_synth:
            mock_synth.side_effect = TTSError("boom", reason="synthesis_failed")
            resp = client.post("/tts/speak", json={"text": "hello"})
        assert resp.status_code == 500

    def test_unauthenticated_request_rejected(self):
        """Without the dependency override, a real 401 should occur."""
        from backend.tts import router as tts_router

        app = FastAPI()
        app.include_router(tts_router)
        client = TestClient(app)
        resp = client.post("/tts/speak", json={"text": "hello"})
        assert resp.status_code == 401
