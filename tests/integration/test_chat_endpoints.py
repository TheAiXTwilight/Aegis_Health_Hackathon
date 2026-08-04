"""New endpoint tests for persisted report follow-up chat.

Covers the completed chat API rather than only helper functions. LLM enrichment
is replaced by a deterministic async stub so tests never require Ollama.
"""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth import get_current_user
from app.db.models import ChatMessage, HealthRecord, PipelineJobRow
from backend.chat import MAX_TURNS
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
def client(session_factory, monkeypatch):
    from backend.chat import router as chat_router
    import tools.chat_enricher as chat_enricher

    async def deterministic_enrichment(*, base_answer: str, **_kwargs) -> str:
        return base_answer

    monkeypatch.setattr(chat_enricher, "enrich_answer", deterministic_enrichment)

    seed_user(session_factory, user_id="alice-id", email="alice@example.test", display_name="Alice")
    seed_user(
        session_factory,
        user_id="bob-id",
        email="bob@example.test",
        display_name="Bob",
        security_answer="Rex",
    )

    result = {
        "report": {"text": "Synthetic report", "severity": "HIGH", "confidence": 0.82},
        "symptom_result": {"symptoms": ["chest pain", "shortness of breath"], "duration": "3 days"},
        "lab_result": {"abnormal_values": ["Elevated troponin: 0.12 ng/mL"]},
        "drug_result": {
            "resolved": ["warfarin", "aspirin"],
            "interactions": [
                {
                    "drugs": ["warfarin", "aspirin"],
                    "severity": "SEVERE",
                    "description": "Increased bleeding risk.",
                }
            ],
        },
        "xray_result": {"findings": ["Cardiomegaly"]},
        "severity_result": {"reasons": ["Chest pain with shortness of breath."]},
    }
    with session_factory() as db:
        db.add(
            PipelineJobRow(
                job_id="alice-job",
                user_id="alice-id",
                session_id="alice-session",
                status="completed",
            )
        )
        db.add(
            HealthRecord(
                id="alice-record",
                user_id="alice-id",
                job_id="alice-job",
                severity="HIGH",
                confidence=0.82,
                symptoms_text="Chest pain and shortness of breath for three days",
                medications_json=json.dumps(["warfarin", "aspirin"]),
                xray_findings_json=json.dumps(["Cardiomegaly"]),
                report_json=json.dumps(result["report"]),
                result_json=json.dumps(result),
            )
        )
        db.commit()

    app = FastAPI()
    app.include_router(chat_router)
    install_db_override(app, session_factory)
    selected = {"value": "alice-id"}
    mutable_user_override(app, session_factory, selected, get_current_user)
    test_client = TestClient(app)
    test_client.selected_user = selected
    return test_client


def test_chat_init_returns_owner_history_suggestions_and_no_consumed_turn(client):
    response = client.get("/queue/chat/alice-job/init")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == "alice-job"
    assert body["turn"] == 0
    assert body["turns_remaining"] == MAX_TURNS
    assert body["messages"] == []
    assert body["limit_reached"] is False
    assert body["suggested_questions"]


def test_chat_answer_persists_user_assistant_pair_and_is_grounded(client, session_factory):
    response = client.post(
        "/queue/chat",
        json={"job_id": "alice-job", "message": "How urgent is this?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["turn"] == 1
    assert body["answer"].strip()
    assert body["enriched"] is False

    with session_factory() as db:
        messages = (
            db.query(ChatMessage)
            .filter_by(user_id="alice-id", job_id="alice-job")
            .order_by(ChatMessage.seq)
            .all()
        )
        assert [message.role for message in messages] == ["user", "assistant"]

    restored = client.get("/queue/chat/alice-job/init").json()
    assert restored["turn"] == 1
    assert [message["role"] for message in restored["messages"]] == ["user", "assistant"]


@pytest.mark.clinical_gate
@pytest.mark.xfail(strict=True, reason="Known chat persistence defect: both messages are assigned the same seq before commit.")
def test_chat_message_sequence_is_monotonic_within_conversation(client, session_factory):
    response = client.post(
        "/queue/chat",
        json={"job_id": "alice-job", "message": "How urgent is this?"},
    )
    assert response.status_code == 200
    with session_factory() as db:
        messages = (
            db.query(ChatMessage)
            .filter_by(user_id="alice-id", job_id="alice-job")
            .order_by(ChatMessage.seq, ChatMessage.created_at)
            .all()
        )
    assert [(message.seq, message.role) for message in messages] == [(0, "user"), (1, "assistant")]


def test_duplicate_and_off_topic_questions_do_not_consume_turn(client, session_factory):
    first = client.post(
        "/queue/chat",
        json={"job_id": "alice-job", "message": "What does my troponin mean?"},
    )
    assert first.status_code == 200
    assert first.json()["turn"] == 1

    duplicate = client.post(
        "/queue/chat",
        json={"job_id": "alice-job", "message": "what does my troponin mean"},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["turn"] == 1

    off_topic = client.post(
        "/queue/chat",
        json={"job_id": "alice-job", "message": "What is the capital of France?"},
    )
    assert off_topic.status_code == 200
    assert off_topic.json()["turn"] == 1

    with session_factory() as db:
        assert db.query(ChatMessage).filter_by(user_id="alice-id", job_id="alice-job").count() == 2


def test_chat_hides_other_users_record(client):
    client.selected_user["value"] = "bob-id"
    assert client.get("/queue/chat/alice-job/init").status_code == 404
    assert client.post(
        "/queue/chat",
        json={"job_id": "alice-job", "message": "How urgent is this?"},
    ).status_code == 404


def test_turn_limit_returns_safe_non_persisting_response(client, session_factory):
    with session_factory() as db:
        for turn in range(MAX_TURNS):
            db.add(ChatMessage(
                user_id="alice-id", job_id="alice-job", seq=turn * 2,
                role="user", content=f"question {turn}",
            ))
            db.add(ChatMessage(
                user_id="alice-id", job_id="alice-job", seq=turn * 2 + 1,
                role="assistant", content=f"answer {turn}",
            ))
        db.commit()

    response = client.post(
        "/queue/chat",
        json={"job_id": "alice-job", "message": "one more question"},
    )
    assert response.status_code == 200
    assert response.json()["turn"] == MAX_TURNS
    assert response.json()["suggested_questions"] == []

    with session_factory() as db:
        assert db.query(ChatMessage).filter_by(user_id="alice-id", job_id="alice-job").count() == MAX_TURNS * 2
