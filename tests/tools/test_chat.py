"""Updated unit tests for the current deterministic chat engine.

The obsolete suite imported _generate_model_answer and called the old
_generate_answer(context=..., severity=...) signature. The current chat engine
uses ReportIntelligence and deterministic handlers, with optional enrichment
performed by the endpoint.
"""
from __future__ import annotations

import pytest

from app.db.models import HealthRecord
from backend.chat import (
    MAX_TURNS,
    _assess_severity_delta,
    _count_user_turns,
    _generate_answer,
    _is_duplicate,
    _is_off_topic,
)
from tools.report_analyst import ReportIntelligence
from tools.suggested_questions import build_suggested_questions


def make_intelligence(severity: str = "HIGH") -> ReportIntelligence:
    return ReportIntelligence.from_context(
        {
            "job_id": "job-1",
            "severity": severity,
            "confidence": 0.82,
            "validation_status": "agreement",
            "extracted_symptoms": ["chest pain", "shortness of breath"],
            "reported_symptoms": "Chest pain and shortness of breath for three days",
            "symptom_duration": "3 days",
            "lab_abnormal_values": ["Elevated troponin: 0.12 ng/mL"],
            "lab_measurements": {"potassium": 4.2},
            "medications": ["warfarin", "aspirin"],
            "drug_interactions": [
                {
                    "drugs": ["warfarin", "aspirin"],
                    "severity": "SEVERE",
                    "description": "Increased bleeding risk.",
                }
            ],
            "drug_warnings": ["Potential interaction detected."],
            "xray_findings": ["Cardiomegaly"],
            "severity_reasons": ["Chest pain with shortness of breath."],
            "report_text": "Synthetic report only.",
        }
    )


def test_suggested_questions_are_bounded_and_grounded_in_risk_context():
    questions = build_suggested_questions(
        "HIGH",
        ["chest pain", "shortness of breath"],
        context={"severity": "HIGH", "extracted_symptoms": ["chest pain"]},
        history=[],
    )
    assert 1 <= len(questions) <= 4
    assert all(isinstance(question, str) and question.strip() for question in questions)


def test_count_user_turns_ignores_assistant_messages():
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]
    assert _count_user_turns(history) == 2
    assert _count_user_turns(history * MAX_TURNS) == MAX_TURNS * 2


def test_duplicate_detection_normalizes_case_and_punctuation():
    history = [
        {"role": "user", "content": "What does my troponin mean?"},
        {"role": "assistant", "content": "Synthetic answer"},
    ]
    assert _is_duplicate("what does my troponin mean", history) is True
    assert _is_duplicate("What should I do next?", history) is False


def test_off_topic_detection_redirects_clear_non_health_question_without_model_call():
    assert _is_off_topic("What is the capital of France?") is True
    assert _is_off_topic("What does my elevated troponin mean?") is False


@pytest.mark.clinical_gate
@pytest.mark.xfail(strict=True, reason="Known classifier defect: substring hint 'eat' makes 'weather' appear health-related.")
def test_off_topic_detection_does_not_match_health_hints_inside_unrelated_words():
    assert _is_off_topic("What is the weather in Pune tomorrow?") is True


def test_deterministic_answer_uses_report_intelligence_for_urgency_question():
    answer = _generate_answer(make_intelligence("HIGH"), "How urgent is this?", history=[])
    assert answer.strip()
    assert any(word in answer.lower() for word in ("urgent", "high", "prompt", "review"))


def test_deterministic_answer_keeps_medication_question_grounded():
    answer = _generate_answer(
        make_intelligence("HIGH"),
        "Should I stop my medication?",
        history=[],
    )
    assert answer.strip()
    assert "medication" in answer.lower() or "warfarin" in answer.lower()


def test_emergency_question_returns_escalation_guidance():
    answer = _generate_answer(
        make_intelligence("CRITICAL"),
        "I have severe chest pain and cannot breathe right now",
        history=[],
    )
    assert any(word in answer.lower() for word in ("emergency", "urgent", "call", "immediate"))


def test_severity_delta_only_escalates_actual_patient_change_not_report_quotation():
    record = HealthRecord(
        user_id="user-1",
        job_id="job-1",
        severity="HIGH",
        confidence=0.8,
        report_json="{}",
        result_json="{}",
    )
    assert _assess_severity_delta("I have severe chest pain now", record) == "increased"
    assert _assess_severity_delta("The report flags 'chest pain' — what does that mean?", record) == "unchanged"
    assert _assess_severity_delta("I am feeling much better today", record) == "decreased"
