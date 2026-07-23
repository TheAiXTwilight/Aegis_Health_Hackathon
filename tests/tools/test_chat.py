"""
tests/tools/test_chat.py — Tests for chat answer generation and suggested questions.

Place at: tests/tools/test_chat.py
"""
from __future__ import annotations

import pytest
from backend.chat import (
    MAX_TURNS,
    _assess_severity_delta,
    _build_report_context,
    _count_user_turns,
    _generate_answer,
    _generate_model_answer,
    model_registry,
)
from tools.suggested_questions import build_suggested_questions


# ── Suggested questions ──────────────────────────────────────
class TestSuggestedQuestions:
    def test_always_returns_base_questions(self):
        result = build_suggested_questions("LOW", [])
        assert "When did the symptoms start?" in result
        assert "Are symptoms worsening?" in result

    def test_max_four_questions(self):
        result = build_suggested_questions(
            "HIGH",
            ["chest pain", "cough", "fever", "headache", "breath"]
        )
        assert len(result) <= 4

    def test_chest_pain_triggers_sob_question(self):
        result = build_suggested_questions("LOW", ["chest pain"])
        assert any("shortness of breath" in q for q in result)

    def test_high_severity_triggers_vitals_question(self):
        result = build_suggested_questions("HIGH", ["cough"])
        assert any("oxygen saturation" in q for q in result)

    def test_no_extras_for_low_without_keywords(self):
        result = build_suggested_questions("LOW", ["fatigue"])
        assert len(result) == 2  # only base questions

    def test_cough_triggers_productive_question(self):
        result = build_suggested_questions("MODERATE", ["cough"])
        assert any("dry or productive" in q for q in result)


# ── Severity delta ───────────────────────────────────────────
class TestSeverityDelta:
    def test_worsening_keywords(self):
        from app.db.models import HealthRecord
        mock = HealthRecord(severity="MODERATE", confidence=0.8,
                          report_json="{}", result_json="{}")
        assert _assess_severity_delta("my cough is getting worse", mock) == "increased"
        assert _assess_severity_delta("I have severe chest pain now", mock) == "increased"

    def test_improvement_keywords(self):
        from app.db.models import HealthRecord
        mock = HealthRecord(severity="HIGH", confidence=0.8,
                          report_json="{}", result_json="{}")
        assert _assess_severity_delta("feeling much better today", mock) == "decreased"
        assert _assess_severity_delta("symptoms resolved", mock) == "decreased"

    def test_unchanged(self):
        from app.db.models import HealthRecord
        mock = HealthRecord(severity="LOW", confidence=0.8,
                          report_json="{}", result_json="{}")
        assert _assess_severity_delta("what does moderate mean", mock) == "unchanged"
        assert _assess_severity_delta("can I exercise today", mock) == "unchanged"


# ── Answer generation ────────────────────────────────────────
class TestGenerateAnswer:
    def test_severity_question(self):
        ans = _generate_answer(context="", question="how urgent is this", history=[], severity="HIGH")
        assert "HIGH" in ans

    def test_medication_question(self):
        ans = _generate_answer(context="", question="should I stop my medication", history=[], severity="MODERATE")
        assert "medication" in ans.lower()

    def test_next_steps_low(self):
        ans = _generate_answer(context="", question="what should I do next", history=[], severity="LOW")
        assert "self-care" in ans.lower() or "monitoring" in ans.lower()

    def test_next_steps_critical(self):
        ans = _generate_answer(context="", question="what should I do next", history=[], severity="CRITICAL")
        assert "emergency" in ans.lower()

    def test_default_fallback(self):
        ans = _generate_answer(context="", question="tell me about the weather", history=[], severity="MODERATE")
        assert "MODERATE" in ans
        assert "triage" in ans.lower()

    def test_answer_uses_selected_report_lab_values(self):
        context = {
            "severity": "MEDIUM",
            "reported_symptoms": "vomiting",
            "lab_abnormal_values": ["Low vitamin D: 12.4 ng/mL"],
            "medications": ["Ondansetron"],
        }
        ans = _generate_answer(
            context=context,
            question="What was abnormal in my blood test?",
            history=[],
            severity="MEDIUM",
        )
        assert "vitamin D" in ans
        assert "12.4" in ans

    def test_answer_changes_with_selected_report(self):
        first = _generate_answer(
            context={"severity": "LOW", "reported_symptoms": "cough"},
            question="What symptoms are in this report?",
            history=[],
            severity="LOW",
        )
        second = _generate_answer(
            context={"severity": "MEDIUM", "reported_symptoms": "vomiting"},
            question="What symptoms are in this report?",
            history=[],
            severity="MEDIUM",
        )
        assert "cough" in first
        assert "vomiting" in second
        assert first != second


class TestConversationTurns:
    def test_limit_is_seven_user_questions(self):
        assert MAX_TURNS == 7

    def test_assistant_messages_do_not_consume_turns(self):
        history = []
        for index in range(7):
            history.append({"role": "user", "content": f"question {index}"})
            history.append({"role": "assistant", "content": f"answer {index}"})
        assert _count_user_turns(history) == 7


class TestReportContext:
    def test_context_comes_from_selected_record(self):
        import json
        from app.db.models import HealthRecord

        record = HealthRecord(
            job_id="selected-job",
            user_id="user-1",
            severity="MEDIUM",
            confidence=0.8,
            validation_status="warning",
            symptoms_text="vomiting",
            medications_json='["Ondansetron"]',
            xray_findings_json="[]",
            report_json='{"text":"Selected report"}',
            result_json=json.dumps({
                "submitted": {
                    "symptoms_text": "vomiting",
                    "medications": ["Ondansetron"],
                },
                "lab_result": {
                    "abnormal_values": ["Low vitamin D: 12.4 ng/mL"],
                    "measurements": {"vitamin_d": 12.4},
                },
                "severity_result": {
                    "reasons": ["Abnormal laboratory value detected"],
                },
                "report": {"text": "Selected report narrative"},
            }),
        )

        context = _build_report_context(record)
        assert context["job_id"] == "selected-job"
        assert context["reported_symptoms"] == "vomiting"
        assert context["medications"] == ["Ondansetron"]
        assert "Low vitamin D" in context["lab_abnormal_values"][0]


class TestModelAnswer:
    def test_local_model_receives_selected_report_context(self, monkeypatch):
        import asyncio

        captured = {}

        def allow_memory(_component):
            return None

        async def fake_generate(prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return {"response": "The selected report records vitamin D at 12.4 ng/mL."}

        monkeypatch.setattr(model_registry, "assert_memory_headroom", allow_memory)
        monkeypatch.setattr(model_registry, "ollama_generate", fake_generate)

        answer = asyncio.run(_generate_model_answer(
            context={
                "job_id": "job-selected",
                "severity": "MEDIUM",
                "lab_abnormal_values": ["Low vitamin D: 12.4 ng/mL"],
            },
            question="What was low?",
            history=[],
        ))

        assert "12.4" in answer
        assert "job-selected" in captured["prompt"]
        assert "Low vitamin D" in captured["prompt"]
        assert captured["kwargs"]["num_predict"] == 220

