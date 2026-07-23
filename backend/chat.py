# backend/chat.py
"""
Authenticated, report-grounded conversational follow-up chat.

Answer pipeline
───────────────
1. _build_report_context()   raw DB record → context dict
2. ReportIntelligence        context dict → typed analytical object
3. _detect_intents()         question → ranked intent list
4. _refine_intents()         drop framing intents (SEVERITY /
                             CONDITION_STATUS / SEVERITY_REASON) when
                             a specific body-system or lab intent
                             already covers the ground they'd restate,
                             AND drop generic catch-all primaries
                             (SYMPTOM_GENERAL / LAB_GENERAL) when a
                             more specific primary is present
5. _generate_answer()        intel + intents → deterministic answer
                             reads actual report values — never templates
6. enrich_answer()           deterministic answer + idle model →
                             +1 connecting sentence (optional, gated)
                             SKIPPED if model busy, adds nothing, or fails

POST /queue/chat           — ask about one selected completed report
POST /queue/rerun/{job_id} — lightweight re-score placeholder

No LLM is used for the primary answer path. The enricher only adds
ONE sentence for cross-domain questions (e.g. "how are my symptoms
connected to my labs?") and is fully silent on failure.

Fixes vs. earlier version:
  - queue_chat_init() no longer clears conversation history on open.
    Turn usage is PERSISTENT per (user, job_id): closing and reopening
    the chat panel preserves both the turn count and prior messages.
    A report that reached MAX_TURNS stays exhausted until a new
    report/assessment is generated — it does not "refill" on reopen.
  - _HEALTH_HINTS greatly expanded and _is_off_topic() short-circuits
    when the question contains any number. This stops legitimate lab
    questions like "My HbA1c (1c) is low" being flagged off-topic.
    ALSO now covers lifestyle, trend, recheck, specialist, ranges
    vocabulary so those suggested-question chips can never be
    incorrectly rejected as off-topic.
  - _value_mentioned_in_question() extracts numeric values the user
    mentions in their question (e.g. "glucose 245") so lab answers
    reflect the value asked about, not just the stored one.
  - _answer_lab_by_category() now accepts the question and warns the
    user when their mentioned value differs from the recorded one.
  - Compound answers join with a paragraph break instead of " | " for
    a more natural reading experience.
  - build_suggested_questions() now dedupes suggested chips against
    already-asked questions using near-duplicate (core-word overlap)
    matching, not just exact string equality — so a chip with slightly
    different wording can't resurface a question just asked.
  - DURATION moved to priority 8 (was 11). It's a more specific intent
    than SYMPTOM_GENERAL (priority 9), so on a question like "my
    symptoms have lasted 2 days — is that too long?" the duration-
    specific handler now sorts first and its narrow answer leads,
    instead of SYMPTOM_GENERAL dumping the full symptom list before
    the duration answer ever appears.
  - _refine_intents() drops framing intents (SEVERITY, CONDITION_STATUS,
    SEVERITY_REASON) whenever any primary fact intent (CARDIAC,
    RESPIRATORY, NEURO, METABOLIC, SYMPTOM_GENERAL, DURATION, XRAY,
    MEDICATION, DRUG_*, LAB_*, TREND, LIFESTYLE, RECHECK, RANGES,
    SPECIALIST) fires. This kills the stacked-closing bug where
    "could my chest tightness be serious?" produced CARDIAC's
    "seek prompt evaluation" closing PLUS SEVERITY's "Please seek
    prompt clinical evaluation" closing in the same bubble — two
    paraphrased action lines that _strip_duplicate_sentences() can't
    catch because they aren't textually identical. With this rule the
    primary handler runs alone; its own closing is the only one shown.
  - _refine_intents() also drops generic catch-all primaries
    (SYMPTOM_GENERAL, LAB_GENERAL) when a more specific primary is
    present. These generics exist as FALLBACKS for when nothing more
    specific fires — not as "and also dump everything else" companions.
    Kills the bug where "have my symptoms lasted 2 days?" produced
    DURATION's narrow duration answer PLUS SYMPTOM_GENERAL's full
    symptom-list dump in the same bubble. With this rule DURATION
    (specific) wins and SYMPTOM_GENERAL (generic fallback) is dropped.
  - _clean_reason() strips trailing punctuation from severity_reasons
    and drug_warnings before templating, killing the "detected.."
    stray double period seen when a stored reason already ended in ".".
    ALSO now detects "reassuring" reasons phrased as concerns (e.g.
    "No high-risk rules triggered") and returns "" so callers can skip
    the grammatically broken "driven by <good news>" phrasing.
  - New intents TREND, LIFESTYLE, RECHECK, RANGES, SPECIALIST with
    dedicated handlers. Every suggested-question chip that the
    tools/suggested_questions.py fallback pool can produce now has a
    matching answer handler — no chip falls through to the generic
    catchall.
  - _answer_symptom_general() now detects when the user asked about a
    SPECIFIC symptom (e.g. "why am I experiencing fever?") and answers
    about that symptom specifically, instead of dumping the full
    symptom list. Only falls back to the summary when the question is
    genuinely generic ("what symptoms are recorded?").
  - Answer formatting uses markdown-style bold and paragraph breaks
    so every answer reads as a scannable structured response rather
    than a wall of text.
  - User-facing messages no longer leak internal state ("no turn
    used", "Repeated —") — rephrased for a natural conversational
    tone.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.db.models import ChatMessage, HealthRecord, PipelineJobRow, User
from app.db.session import get_db
from tools.report_analyst import (
    ReportIntelligence,
    is_urgent,
    is_moderate,
    display_symptom,
    join_symptoms,
)
from tools.suggested_questions import build_suggested_questions

router = APIRouter(prefix="/queue", tags=["chat"])


# ── Request / Response models ─────────────────────────────────────

class ChatIn(BaseModel):
    job_id: str = Field(..., description="Selected completed report job ID")
    message: str = Field(..., min_length=1, max_length=1000)


class ChatOut(BaseModel):
    job_id: str
    turn: int
    answer: str
    severity_delta: str | None = None
    suggested_questions: list[str] = []
    enriched: bool = False


# ── Conversation store ────────────────────────────────────────────
#
# Chat history is persisted in the `chat_messages` table (see
# app/db/models.py) rather than kept in a module-level dict.
#
# Why: with multiple gunicorn/uvicorn workers, an in-process dict gives
# each worker its own independent copy of `_conversations`. Two turns
# from the same user's conversation can land on different workers, so
# each worker computes `used_turns` from a different partial view of
# the conversation — the turn badge then drifts or undercounts
# depending on which worker happened to answer which request. A DB
# table is shared, visible to every worker, and survives restarts.

MAX_TURNS = 7


def _load_history(db: Session, user_id: str, job_id: str) -> list[dict[str, Any]]:
    """Load full persisted conversation for (user_id, job_id), oldest first."""
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user_id, ChatMessage.job_id == job_id)
        .order_by(ChatMessage.seq.asc())
        .all()
    )
    return [
        {
            "role": row.role,
            "content": row.content,
            "severity_delta": row.severity_delta,
            "at": row.created_at.strftime("%Y-%m-%dT%H:%M:%SZ") if row.created_at else None,
        }
        for row in rows
    ]


def _next_seq(db: Session, user_id: str, job_id: str) -> int:
    """Next sequence number for a new message in this conversation."""
    last = (
        db.query(ChatMessage.seq)
        .filter(ChatMessage.user_id == user_id, ChatMessage.job_id == job_id)
        .order_by(ChatMessage.seq.desc())
        .first()
    )
    return (last[0] + 1) if last else 0


def _append_message(
    db: Session,
    user_id: str,
    job_id: str,
    role: str,
    content: str,
    severity_delta: str | None = None,
) -> None:
    """Persist one message to the conversation. Caller commits."""
    seq = _next_seq(db, user_id, job_id)
    db.add(
        ChatMessage(
            user_id=user_id,
            job_id=job_id,
            seq=seq,
            role=role,
            content=content,
            severity_delta=severity_delta,
        )
    )


# ── Intent detection ──────────────────────────────────────────────
# 29 intent buckets ordered from most-specific to most-generic.
# _detect_intents() scores all of them and returns every match —
# the answer engine dispatches up to 2 handlers for compound questions.

_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "EMERGENCY": (
        "can't breathe", "cannot breathe", "chest pain radiating",
        "collapsed", "unconscious", "passed out", "severe bleeding",
        "heart attack", "stroke", "not breathing",
    ),
    "DRUG_INTERACTION": (
        "interaction", "drug interaction", "combining", "mixing",
        "taking together", "safe together", "combine",
    ),
    "DRUG_WARNING": (
        "warning", "drug warning", "side effect", "adverse effect",
        "adverse reaction", "contraindication",
    ),
    "MEDICATION": (
        "medication", "medicine", "drug", "dose", "dosage",
        "pill", "tablet", "prescription", "prescribed",
    ),
    "XRAY": (
        "xray", "x-ray", "x ray", "imaging", "scan", "radiograph",
        "chest image", "chest film", "lung image",
    ),
    "LAB_GLUCOSE": (
        "glucose", "blood sugar", "sugar level", "hba1c", "a1c",
        "diabetes", "diabetic", "insulin",
    ),
    "LAB_THYROID": (
        "thyroid", "tsh", "t3", "t4", "hypothyroid", "hyperthyroid",
    ),
    "LAB_VITAMIN": (
        "vitamin d", "vitamin b", "vitamin", "deficiency",
    ),
    "LAB_BP": (
        "blood pressure", "bp", "hypertension", "systolic",
        "diastolic", "hypertensive",
    ),
    "LAB_CHOLESTEROL": (
        "cholesterol", "ldl", "hdl", "triglyceride", "lipid",
    ),
    "LAB_OXYGEN": (
        "oxygen", "spo2", "o2 saturation", "oxygen level",
        "oxygen saturation",
    ),
    "LAB_GENERAL": (
        "lab result", "lab value", "lab finding", "lab report",
        "blood test", "blood work", "test result", "measurement",
        "abnormal lab", "my labs", "the labs", "lab data",
    ),
    "CARDIAC": (
        "chest", "heart", "palpitation", "cardiac", "arrhythmia",
        "irregular heartbeat", "heart rate", "pulse",
    ),
    "RESPIRATORY": (
        "breath", "breathing", "shortness of breath", "wheez",
        "inhaler", "asthma", "copd", "lung",
    ),
    "NEURO": (
        "dizzy", "dizziness", "faint", "fainting", "balance",
        "lightheaded", "light-headed", "blackout", "black out",
        "headache", "migraine", "confusion",
    ),
    "METABOLIC": (
        "weight", "bmi", "obesity", "metabolism",
    ),
    # ── Lifestyle / trend / recheck / specialist / ranges ──────────
    # These cover the fallback suggested-question chips generated by
    # tools/suggested_questions.py. Without dedicated intents, those
    # chips would fall through to _answer_catchall (a wall-of-text
    # generic overview) OR be rejected as off-topic by _is_off_topic
    # (their vocabulary wasn't in _HEALTH_HINTS).
    "TREND": (
        "trend", "trended", "trending", "progress", "over time",
        "getting worse", "getting better", "compare reports",
        "across reports", "across my reports", "how has my health",
        "improved", "worsened", "changing",
    ),
    "LIFESTYLE": (
        "lifestyle", "diet", "eat", "eating", "exercise", "workout",
        "physical activity", "sleep", "stress", "habits", "prevent",
        "prevention", "healthy", "changes that could help",
        "changes to help", "changes could help",
    ),
    "RECHECK": (
        "recheck", "re-check", "retest", "re-test", "repeat test",
        "how often", "monitor", "monitoring", "when to check",
        "when should i check", "follow-up test", "when should i test",
    ),
    "RANGES": (
        "normal range", "reference range", "what is normal",
        "what's normal", "what range", "target range",
    ),
    "SPECIALIST": (
        "specialist", "which doctor", "what kind of doctor",
        "cardiologist", "endocrinologist", "neurologist",
        "pulmonologist", "which kind of doctor", "who should i see",
    ),
    "SYMPTOM_GENERAL": (
        "symptom", "feel", "feeling", "pain", "fatigue",
        "tired", "fever", "temperature", "nausea", "vomit",
        "diarrhea", "stomach", "ache",
    ),
    "SEVERITY": (
        "severity", "urgent", "serious", "risk", "critical",
        "dangerous", "concerning", "worried",
    ),
    "CONDITION_STATUS": (
        "how am i", "am i okay", "am i ok", "am i fine",
        "how is my condition", "my condition", "how bad",
        "is it serious", "is this bad", "is this good",
        "getting worse", "getting better", "good or bad",
        "better or worse",
    ),
    "SEVERITY_REASON": (
        "why is it", "what caused", "reason for", "why rated",
        "what is driving", "whats driving", "what's driving",
        "main factor", "critical factor", "key finding",
        "what makes it", "why high", "why critical",
        "why medium", "explain the rating",
    ),
    "NEXT_STEPS": (
        "what should i do", "what do i do", "next steps",
        "next step", "recommend", "action", "follow up",
        "what now", "do i need", "should i go",
        "should i call", "should i visit", "immediately",
        "right now", "do now",
    ),
    "DURATION": (
        "how long", "duration", "since when", "for how long",
        "when did", "started", "have lasted", "has lasted",
        "too long", "how many days", "how many hours",
    ),
    "CONFIDENCE": (
        "confidence", "how sure", "how certain", "how confident",
        "accuracy", "reliable", "trust", "certain",
    ),
}

# Lower priority number = more specific = sorts first in the intent list
# returned by _detect_intents(). Any ties are broken by dict insertion
# order, which is stable in Python 3.7+.
#
# DURATION is at 8 (not 11) because "my symptoms have lasted 2 days"
# matches both DURATION and SYMPTOM_GENERAL (9). DURATION is the more
# specific answer, so it must sort first — otherwise SYMPTOM_GENERAL
# leads and dumps the full symptom list before the duration answer.
#
# TREND, LIFESTYLE, RECHECK, RANGES, SPECIALIST are priority 9 (same
# tier as generic fallback primaries but not IN the generic-fallback
# set, so they don't get dropped by rule 3 in _refine_intents). They
# fire when their specific vocabulary is present and take precedence
# over the framing intents (SEVERITY/CONDITION_STATUS/SEVERITY_REASON).
_INTENT_PRIORITY: dict[str, int] = {
    "EMERGENCY": 0,
    "DRUG_INTERACTION": 1,
    "DRUG_WARNING": 2,
    "LAB_GLUCOSE": 3,
    "LAB_THYROID": 3,
    "LAB_VITAMIN": 3,
    "LAB_BP": 3,
    "LAB_CHOLESTEROL": 3,
    "LAB_OXYGEN": 3,
    "XRAY": 4,
    "MEDICATION": 5,
    "CARDIAC": 6,
    "RESPIRATORY": 6,
    "NEURO": 6,
    "METABOLIC": 6,
    "CONDITION_STATUS": 7,
    "SEVERITY_REASON": 7,
    "SEVERITY": 8,
    "DURATION": 8,          # more specific than SYMPTOM_GENERAL — must sort before it
    "TREND": 9,
    "LIFESTYLE": 9,
    "RECHECK": 9,
    "RANGES": 9,
    "SPECIALIST": 9,
    "LAB_GENERAL": 9,
    "SYMPTOM_GENERAL": 9,
    "NEXT_STEPS": 10,
    "CONFIDENCE": 11,
}

# Intents that produce a specific fact block AND end with their own
# severity/action closing. If any of these are present, the framing
# intents below are dropped — otherwise the primary handler's closing
# + the framing intent's closing stack in the same bubble, producing
# two paraphrased "seek care" lines that _strip_duplicate_sentences
# can't collapse (they aren't textually identical, just semantically
# redundant).
_PRIMARY_FACT_INTENTS = {
    "DRUG_INTERACTION", "DRUG_WARNING", "MEDICATION", "XRAY",
    "LAB_GLUCOSE", "LAB_THYROID", "LAB_VITAMIN", "LAB_BP",
    "LAB_CHOLESTEROL", "LAB_OXYGEN", "LAB_GENERAL",
    "CARDIAC", "RESPIRATORY", "NEURO", "METABOLIC", "SYMPTOM_GENERAL",
    "DURATION",
    "TREND", "LIFESTYLE", "RECHECK", "RANGES", "SPECIALIST",
}

# Intents that only restate severity / confidence / action framing.
# Suppressed when any primary fact intent is also present.
_FRAMING_INTENTS = {"SEVERITY", "CONDITION_STATUS", "SEVERITY_REASON"}

# Generic catch-all primaries. These exist as FALLBACKS for when nothing
# more specific fires (e.g. "how am I feeling?" → SYMPTOM_GENERAL alone,
# "explain the abnormal labs" → LAB_GENERAL alone). When a more specific
# primary is ALSO present (DURATION, CARDIAC, NEURO, LAB_GLUCOSE, ...)
# these generics get dropped so the answer stays focused on what the
# user actually asked about, instead of the specific answer being
# followed by a full symptom or lab dump the user didn't request.
_GENERIC_FALLBACK_PRIMARIES = {"SYMPTOM_GENERAL", "LAB_GENERAL"}


def _detect_intents(question: str) -> list[str]:
    """Return all matching intents, ordered by clinical specificity."""
    q = question.lower()
    matched: list[tuple[int, str]] = [
        (_INTENT_PRIORITY.get(intent, 99), intent)
        for intent, keywords in _INTENT_KEYWORDS.items()
        if any(kw in q for kw in keywords)
    ]
    matched.sort(key=lambda x: x[0])
    return [intent for _, intent in matched]


def _refine_intents(intents: list[str]) -> list[str]:
    """
    Drop redundant intents so each turn produces a focused answer to
    what the user actually asked, not a bundled data dump.

    Rules (applied in order):
      1. EMERGENCY alone if present — nothing else matters.
      2. NEXT_STEPS alone if present — its answer is fully self-contained
         action guidance. Adding LAB_GENERAL/SYMPTOM_GENERAL alongside
         produces a data dump that then repeats when the user asks about
         labs specifically.
      3. If a SPECIFIC primary is present (DURATION, CARDIAC, NEURO,
         LAB_GLUCOSE, TREND, LIFESTYLE, RECHECK, RANGES, SPECIALIST,
         ...) alongside a GENERIC fallback primary (SYMPTOM_GENERAL /
         LAB_GENERAL), drop the generic fallback.
      4. If any primary fact intent is present, drop framing intents
         (SEVERITY / CONDITION_STATUS / SEVERITY_REASON).
    """
    if not intents:
        return intents

    if "EMERGENCY" in intents:
        return ["EMERGENCY"]

    if "NEXT_STEPS" in intents:
        return ["NEXT_STEPS"]

    # Rule 3: drop generic catch-all primaries when a more specific
    # primary is present.
    specific_primaries = [
        i for i in intents
        if i in _PRIMARY_FACT_INTENTS and i not in _GENERIC_FALLBACK_PRIMARIES
    ]
    if specific_primaries:
        intents = [i for i in intents if i not in _GENERIC_FALLBACK_PRIMARIES]

    # Rule 4: drop framing intents when any primary is present.
    if any(i in _PRIMARY_FACT_INTENTS for i in intents):
        return [i for i in intents if i not in _FRAMING_INTENTS]

    return intents


# ── Value extraction from the question itself ─────────────────────
# If the user asks "What does my glucose 245 finding suggest?" we want
# the answer to acknowledge 245, not silently answer with the stored 117.

_QUESTION_VALUE_RE = re.compile(
    r"([a-z][a-z\s]{2,25}?)\s*(?:is|was|of|at|:|=|-)?\s*(\d+\.?\d*)",
    re.IGNORECASE,
)


def _value_mentioned_in_question(
    question: str, keywords: tuple[str, ...]
) -> str | None:
    """
    Return the numeric value the user mentioned near any of `keywords`.
    None if no keyword mention has a nearby number.
    """
    q = question.lower()
    for match in _QUESTION_VALUE_RE.finditer(q):
        preceding = match.group(1).lower()
        value = match.group(2)
        if any(kw in preceding for kw in keywords):
            return value
    return None


# ── Context builder ───────────────────────────────────────────────

def _json_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _build_report_context(record: HealthRecord) -> dict[str, Any]:
    """
    Build a bounded, structured context dict from the persisted report.
    Keys match exactly what ReportIntelligence.from_context() expects.
    """
    result = _json_dict(record.result_json)
    report = (
        result.get("report")
        if isinstance(result.get("report"), dict)
        else _json_dict(record.report_json)
    )
    submitted = result.get("submitted") if isinstance(result.get("submitted"), dict) else {}
    patient = result.get("patient") if isinstance(result.get("patient"), dict) else {}
    lab = result.get("lab_result") if isinstance(result.get("lab_result"), dict) else {}
    xray = result.get("xray_result") if isinstance(result.get("xray_result"), dict) else {}
    drug = result.get("drug_result") if isinstance(result.get("drug_result"), dict) else {}
    severity_result = result.get("severity_result") if isinstance(result.get("severity_result"), dict) else {}
    symptom_result = result.get("symptom_result") if isinstance(result.get("symptom_result"), dict) else {}

    medications = _string_list(submitted.get("medications"))
    if not medications:
        try:
            medications = _string_list(json.loads(record.medications_json or "[]"))
        except (TypeError, json.JSONDecodeError):
            medications = []

    xray_findings = _string_list(xray.get("findings")) or _string_list(
        submitted.get("xray_findings")
    )
    if not xray_findings:
        try:
            xray_findings = _string_list(json.loads(record.xray_findings_json or "[]"))
        except (TypeError, json.JSONDecodeError):
            xray_findings = []

    interactions: list[dict] = []
    for interaction in drug.get("interactions") or []:
        if isinstance(interaction, dict):
            interactions.append({
                "drugs": _string_list(interaction.get("drugs")),
                "severity": interaction.get("severity"),
                "description": interaction.get("description"),
            })

    return {
        "job_id": record.job_id,
        "patient": patient,
        "severity": record.severity,
        "confidence_percent": round((record.confidence or 0) * 100),
        "validation_status": record.validation_status,
        "reported_symptoms": (
            submitted.get("symptoms_text") or record.symptoms_text or "None reported"
        ),
        "extracted_symptoms": _string_list(symptom_result.get("symptoms")),
        "symptom_duration": symptom_result.get("duration"),
        "severity_indicators": _string_list(symptom_result.get("severity_indicators")),
        "severity_reasons": _string_list(severity_result.get("reasons")),
        "medications": medications,
        "drug_interactions": interactions,
        "drug_warnings": _string_list(drug.get("warnings")),
        "lab_abnormal_values": _string_list(lab.get("abnormal_values")),
        "lab_measurements": (
            lab.get("measurements")
            if isinstance(lab.get("measurements"), dict)
            else {}
        ),
        "xray_findings": xray_findings,
        "xray_notes": xray.get("free_text"),
        "report_text": str(report.get("text") or "")[:4000],
    }


# ── Guard helpers ─────────────────────────────────────────────────

def _count_user_turns(history: list[dict]) -> int:
    return sum(1 for m in history if m.get("role") == "user")


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]", "", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_duplicate(question: str, history: list[dict]) -> bool:
    norm = _normalize(question)
    if not norm:
        return False
    return any(
        _normalize(str(m.get("content") or "")) == norm
        for m in history
        if m.get("role") == "user"
    )


def _cached_answer(question: str, history: list[dict]) -> str | None:
    norm = _normalize(question)
    for idx, item in enumerate(history):
        if (
            item.get("role") == "user"
            and _normalize(str(item.get("content") or "")) == norm
            and idx + 1 < len(history)
            and history[idx + 1].get("role") == "assistant"
        ):
            return str(history[idx + 1].get("content") or "")
    return None


# Deliberately broad hint vocabulary. We would rather spend a turn on a
# borderline health question than wrongly reject a legitimate one.
#
# Universal coverage: every fallback suggested-question chip produced by
# tools/suggested_questions.py MUST be represented here, otherwise the
# app can suggest a chip it then rejects as off-topic — which is what
# was happening with "Are there lifestyle changes that could help?"
# (missing "lifestyle") and "How has my health trended?" (missing "trend").
_HEALTH_HINTS = (
    # General health
    "condition", "symptom", "report", "severity", "risk", "urgent",
    "serious", "concern", "concerning", "concerns", "main concern",
    "critical", "danger", "dangerous", "normal", "abnormal", "worse",
    "better", "worsening", "improving", "health", "confidence",
    "clinician", "doctor", "hospital", "treatment", "diagnosis",
    # Labs & measurements
    "lab", "blood", "test", "result", "measurement", "level", "value",
    "reading", "count",
    # Medications
    "medication", "medicine", "drug", "dose", "dosage", "pill", "tablet",
    "prescription", "interaction",
    # Imaging
    "xray", "x-ray", "x ray", "imaging", "scan", "radiograph",
    # Symptoms
    "feel", "feeling", "pain", "fever", "cough", "dizzy", "dizziness",
    "faint", "fainting", "breath", "breathing", "chest", "headache",
    "migraine", "fatigue", "tired", "tiredness", "nausea", "vomit",
    "ache", "swelling",
    # Vitals / metabolic
    "glucose", "sugar", "pressure", "bp", "vitamin", "thyroid", "tsh",
    "heart", "pulse", "oxygen", "spo2", "weight", "bmi", "temperature",
    "diabetes", "diabetic", "hba1c", "a1c", "cholesterol", "ldl", "hdl",
    "triglyceride", "creatinine", "haemoglobin", "hemoglobin",
    "potassium", "sodium", "troponin",
    # Lifestyle / prevention — covers the "lifestyle changes" fallback chip
    "lifestyle", "diet", "eat", "eating", "exercise", "workout", "activity",
    "sleep", "stress", "habits", "prevent", "prevention", "healthy",
    "changes", "help", "improve",
    # Trend / comparison — covers the "how has my health trended" chip
    "trend", "trended", "trending", "progress", "over time", "compare",
    "improved", "worsened", "changing", "across reports",
    # Recheck / monitoring — covers the "should any values be rechecked" chip
    "recheck", "retest", "monitor", "monitoring", "how often",
    "when should", "follow-up test",
    # Specialist / doctor referral — covers the "which kind of doctor" chip
    "specialist", "cardiologist", "endocrinologist", "neurologist",
    "pulmonologist", "which doctor", "which kind", "who should i see",
    # Reference ranges — covers the "what do the ranges mean" chip
    "range", "ranges", "reference", "normal range", "target",
    # Question intent words that generally accompany report questions
    "mean", "means", "explain", "why", "should",
    "recommend", "action", "follow up", "next", "next step",
)


def _is_off_topic(question: str) -> bool:
    """
    Return True only when a question has NO plausible connection to
    health, labs, medications, or the report itself.

    Deliberately permissive — when in doubt, allow the question to
    consume a turn. Two escape hatches keep legitimate lab/vital
    questions safe even if the hint list misses one:
      • Any number in the question implies a value question
      • Any short (<= 3 chars) message is treated as conversational
    """
    q = question.lower()
    if any(hint in q for hint in _HEALTH_HINTS):
        return False
    if len(q.strip()) <= 3:
        return False
    # Escape hatch: a question containing a number is almost always a
    # value question about a measurement. Never reject those as off-topic.
    if re.search(r"\d", q):
        return False
    return True


# ── Small text helper ─────────────────────────────────────────────

# Phrases that mean "everything is fine" but sometimes get stored in
# severity_reasons as if they were concerns. Using them verbatim in
# "This is mainly driven by <reason>" produces nonsense:
#   "This is mainly driven by No high-risk rules triggered."
# _clean_reason() returns "" for these so callers can skip the whole
# "driven by" phrasing instead of showing a broken sentence.
_REASSURING_REASON_CUES = (
    "no high-risk", "no high risk", "no significant", "no abnormal",
    "no critical", "no severe", "within normal", "no rules triggered",
    "no risk factors", "no concerns", "unremarkable",
)


def _clean_reason(reason: str) -> str:
    """
    Strip trailing punctuation so templates that append their own
    period don't produce "..".

    Also detects "reassuring" reasons phrased as concerns. When the
    severity_reasons list contains something like "No high-risk rules
    triggered" — that's actually a *reassurance* the system stored as
    a reason field. Using it verbatim in "this is mainly driven by
    <reason>" produces a nonsense sentence. Return an empty string
    in that case so callers can skip the "driven by" phrasing entirely.
    """
    cleaned = (reason or "").strip().rstrip(".!?;,: ")
    if not cleaned:
        return ""
    lower = cleaned.lower()
    if any(cue in lower for cue in _REASSURING_REASON_CUES):
        return ""
    return cleaned


# ── Deterministic answer composers ───────────────────────────────
# One function per intent bucket. Every function:
#   - Reads typed fields from ReportIntelligence (never raw dicts)
#   - Returns a complete answer with real values substituted in
#   - Uses \n\n paragraph breaks + **bold** so the frontend markdown
#     renderer produces a structured, scannable answer rather than a
#     wall of text
#   - Has a specific "not present" fallback — never a blank template

def _answer_emergency(_intel: ReportIntelligence, _q: str) -> str:
    return (
        "This sounds like it could be an emergency — please call emergency services right now "
        "rather than waiting."
    )


def _answer_drug_interaction(intel: ReportIntelligence, _q: str) -> str:
    if not intel.drug_interactions:
        return (
            f"Good news — no drug interactions are flagged for your medications "
            f"({intel.short_medication_summary()})."
        )
    top = intel.top_interaction()
    assert top is not None
    drugs = " and ".join(top.drugs[:2]) if top.drugs else "your medications"
    sev = f" It's rated **{top.severity}**." if top.severity else ""
    desc = _clean_reason(top.description or "a potential interaction")
    return f"This report flags an interaction between **{drugs}**: {desc}.{sev} Worth a quick check with a pharmacist or clinician."


def _answer_drug_warning(intel: ReportIntelligence, _q: str) -> str:
    if not intel.drug_warnings:
        return "There aren't any specific drug warnings recorded in this report."
    extra = (
        f" There are **{len(intel.drug_warnings)} warnings** in total worth reviewing."
        if len(intel.drug_warnings) > 1 else ""
    )
    return f"One thing flagged: {_clean_reason(intel.drug_warnings[0])}.{extra} Bring this up with your prescribing clinician."


def _answer_medication(intel: ReportIntelligence, _q: str) -> str:
    if not intel.medications:
        return "No medications are recorded in this report. Don't start or change anything without checking with a clinician first."
    suffix = ""
    if intel.drug_interactions:
        top = intel.top_interaction()
        assert top is not None
        suffix = f" One thing to know: {top.one_line()}."
    elif intel.drug_warnings:
        suffix = f" Also worth noting: {_clean_reason(intel.drug_warnings[0])}."
    return f"On record: **{intel.short_medication_summary()}**.{suffix} Check with your clinician before changing anything."


def _answer_xray(intel: ReportIntelligence, _q: str) -> str:
    if not intel.xray_findings:
        return "There's no X-ray or imaging data recorded in this report."
    findings = "; ".join(f.text for f in intel.xray_findings[:3])
    sig_count = sum(1 for f in intel.xray_findings if f.is_significant)
    note = (
        f" **{sig_count}** of these need prompt clinical correlation."
        if sig_count
        else " Worth a clinician's read to put these in context."
    )
    return f"**Imaging findings:** {findings}.{note}"


def _answer_lab_by_category(
    intel: ReportIntelligence,
    keywords: tuple[str, ...],
    label: str,
    clinical_note: str,
    question: str = "",
) -> str:
    """
    Generic named-lab answer with three sources checked in order:
      1. Numeric value the user mentioned in their question
      2. Abnormal lab findings from the structured lab result
      3. Normal (recorded) lab findings
    """
    mentioned = _value_mentioned_in_question(question, keywords) if question else None

    matching_abnormal = [
        lab for lab in intel.abnormal_labs
        if any(kw in lab.name.lower() for kw in keywords)
    ]
    normal_match = [
        lab for lab in intel.normal_labs
        if any(kw in lab.name.lower() for kw in keywords)
    ]

    # If the user mentioned a value, reconcile it with what's stored
    if mentioned:
        stored_val = None
        stored_source = None
        if matching_abnormal and matching_abnormal[0].value:
            stored_val = matching_abnormal[0].value
            stored_source = "abnormal"
        elif normal_match and normal_match[0].value:
            stored_val = normal_match[0].value
            stored_source = "normal"

        if stored_val and str(stored_val) != mentioned:
            return (
                f"You mentioned **{label}** of **{mentioned}**, but this report has a {stored_source} "
                f"recorded value of **{stored_val}**. If {mentioned} is a newer reading, share it with your clinician."
            )
        if not stored_val:
            return (
                f"You mentioned **{label}** of **{mentioned}**, but no structured {label} value "
                "is recorded in this report. Confirm the reading with your clinician."
            )

    if matching_abnormal:
        if len(matching_abnormal) == 1:
            lab = matching_abnormal[0]
            unit_ok = lab.unit and lab.unit.lower() not in ("none", "null")
            if lab.value and unit_ok:
                val = f" at **{lab.value}{lab.unit}**"
            elif lab.value:
                val = f" = **{lab.value}**"
            else:
                val = ""
            direction = lab.direction or "abnormal"
            return f"Your **{lab.name}**{val} is **{direction}**. {clinical_note}"
        summary = "; ".join(lab.one_line() for lab in matching_abnormal)
        return f"**Multiple {label} findings flagged:** {summary}. {clinical_note}"

    if normal_match:
        vals = "; ".join(lab.one_line() for lab in normal_match[:2])
        return f"**Recorded {label} values:** {vals}. None flagged as abnormal."

    return f"No {label} findings are recorded in this report."


def _answer_lab_glucose(intel: ReportIntelligence, q: str) -> str:
    return _answer_lab_by_category(
        intel,
        ("glucose", "sugar", "hba1c", "a1c", "diabet"),
        "glucose / diabetes",
        "Elevated glucose or HbA1c indicates blood sugar control needs attention.",
        question=q,
    )


def _answer_lab_thyroid(intel: ReportIntelligence, q: str) -> str:
    return _answer_lab_by_category(
        intel,
        ("thyroid", "tsh", "t3", "t4"),
        "thyroid",
        "Thyroid imbalances affect energy, weight, mood, and heart rate.",
        question=q,
    )


def _answer_lab_vitamin(intel: ReportIntelligence, q: str) -> str:
    return _answer_lab_by_category(
        intel,
        ("vitamin",),
        "vitamin",
        "Vitamin deficiencies can affect immunity, bone health, and energy.",
        question=q,
    )


def _answer_lab_bp(intel: ReportIntelligence, q: str) -> str:
    return _answer_lab_by_category(
        intel,
        ("blood pressure", "bp", "systolic", "diastolic", "hypertens"),
        "blood pressure",
        "Elevated blood pressure increases cardiovascular risk.",
        question=q,
    )


def _answer_lab_cholesterol(intel: ReportIntelligence, q: str) -> str:
    return _answer_lab_by_category(
        intel,
        ("cholesterol", "ldl", "hdl", "triglyceride", "lipid"),
        "cholesterol / lipids",
        "Abnormal lipid levels increase cardiovascular disease risk.",
        question=q,
    )


def _answer_lab_oxygen(intel: ReportIntelligence, q: str) -> str:
    return _answer_lab_by_category(
        intel,
        ("oxygen", "spo2", "o2", "saturation"),
        "oxygen saturation",
        "Low oxygen saturation can indicate respiratory or cardiac issues.",
        question=q,
    )


def _answer_lab_general(intel: ReportIntelligence, _q: str) -> str:
    if intel.abnormal_labs:
        count = len(intel.abnormal_labs)
        plural = "finding" if count == 1 else "findings"
        return f"**{count} abnormal {plural}**: {intel.short_lab_summary()}. Worth reviewing with a clinician."
    if intel.normal_labs:
        vals = "; ".join(lab.one_line() for lab in intel.normal_labs[:4])
        return f"Nothing flagged as abnormal. **Recorded values:** {vals}."
    return "There aren't any structured lab results recorded in this report."


def _answer_cardiac(intel: ReportIntelligence, _q: str) -> str:
    symptoms = intel.symptom_groups.cardiac
    cardiac_labs = [
        lab for lab in intel.abnormal_labs
        if any(kw in lab.name.lower() for kw in ("bp", "pressure", "cardiac", "troponin", "cholesterol"))
    ]
    if not symptoms and not cardiac_labs:
        return "No cardiac symptoms or related lab findings are recorded in this report."
    bits: list[str] = []
    if symptoms:
        bits.append(f"**{join_symptoms(symptoms, 3)}**")
    if cardiac_labs:
        bits.append(f"labs showing {'; '.join(lab.one_line() for lab in cardiac_labs[:2])}")
    body = "You've got " + " and ".join(bits) + " on record."
    urgency = (
        " That's part of the urgent rating — worth prompt evaluation."
        if is_urgent(intel.severity)
        else " Get checked if it worsens."
    )
    return body + urgency


def _answer_respiratory(intel: ReportIntelligence, _q: str) -> str:
    symptoms = intel.symptom_groups.respiratory
    resp_labs = [
        lab for lab in intel.abnormal_labs
        if any(kw in lab.name.lower() for kw in ("oxygen", "spo2", "co2"))
    ]
    if not symptoms and not resp_labs:
        return "No respiratory symptoms or related lab findings are recorded in this report."
    bits: list[str] = []
    if symptoms:
        bits.append(f"**{join_symptoms(symptoms, 3)}**")
    if resp_labs:
        bits.append(f"labs showing {'; '.join(lab.one_line() for lab in resp_labs[:2])}")
    body = "You've got " + " and ".join(bits) + f" on record — part of why this is rated **{intel.severity}**."
    urgency = " Sudden or severe breathing trouble is an emergency, not wait-and-see."
    return body + urgency


def _answer_neuro(intel: ReportIntelligence, _q: str) -> str:
    symptoms = intel.symptom_groups.neurological
    if not symptoms:
        return "There's no neurological symptom like dizziness or balance issues recorded in this report. If you're experiencing that now, it's worth getting checked promptly regardless."
    display_str = join_symptoms(symptoms, 4)
    acute = any(
        s.lower() in ("faint", "fainting", "blackout", "loss of balance", "collapse", "unconscious")
        for s in symptoms
    )
    if acute:
        return (
            f"You've got **{display_str}** on record. That combination can need urgent attention — "
            "go to the ER if it's happening now, especially with fainting or sudden balance loss."
        )
    return (
        f"**{display_str.capitalize()}** is on record and helped push this toward a "
        f"**{intel.severity}** rating. Common contributors are blood sugar swings, dehydration, "
        "low blood pressure, inner-ear issues, or medication side effects — worth flagging to a clinician."
    )


def _answer_metabolic(intel: ReportIntelligence, _q: str) -> str:
    symptoms = intel.symptom_groups.metabolic
    meta_labs = [
        lab for lab in intel.abnormal_labs
        if any(kw in lab.name.lower() for kw in ("bmi", "weight", "metabol"))
    ]
    if not symptoms and not meta_labs:
        return "No metabolic findings are recorded in this report."
    bits: list[str] = []
    if symptoms:
        bits.append(f"**{join_symptoms(symptoms, 3)}**")
    if meta_labs:
        bits.append(f"labs showing {'; '.join(lab.one_line() for lab in meta_labs[:2])}")
    return "You've got " + " and ".join(bits) + " on record. Worth a management plan with your clinician."


# ── Symptom keyword sets — used by _answer_symptom_general() to detect
# which specific symptom the user asked about. Broad on purpose so we
# match variations like "fever/temperature", "tired/fatigue", etc.
_SYMPTOM_MATCH_CUES = {
    "fever": ("fever", "temperature", "febrile"),
    "headache": ("headache", "migraine", "head pain"),
    "fatigue": ("fatigue", "tired", "tiredness", "exhausted", "low energy"),
    "nausea": ("nausea", "nauseous", "queasy", "vomit"),
    "pain": ("pain", "ache", "aching"),
    "cough": ("cough", "coughing"),
    "breath": ("breath", "breathing", "breathless", "short of breath"),
    "dizzy": ("dizzy", "dizziness", "lightheaded", "faint"),
    "chest": ("chest", "chest pain", "chest tightness"),
    "swelling": ("swelling", "swollen", "edema"),
}

# One-line clinical hint per matched symptom family. Kept generic on
# purpose — the goal is to give the user *some* context, not to diagnose.
_SYMPTOM_HINTS = {
    "fever": "Common contributors are viral or bacterial infection, inflammatory conditions, or (less often) medication reactions.",
    "headache": "Common contributors are dehydration, poor sleep, stress, eye strain, blood pressure changes, or sinus issues.",
    "fatigue": "Common contributors are poor sleep, anemia, thyroid issues, vitamin deficiencies, or ongoing infection.",
    "nausea": "Common contributors are infection, medication side effects, GI conditions, or pregnancy.",
    "pain": "Cause depends on location and pattern — worth describing to your clinician.",
    "cough": "Common contributors are viral infection, allergies, asthma, or acid reflux.",
    "breath": "Common contributors are asthma, infection, anxiety, deconditioning, or cardiac issues — this warrants prompt evaluation.",
    "dizzy": "Common contributors are dehydration, blood sugar swings, low blood pressure, inner-ear issues, or medication side effects.",
    "chest": "Chest symptoms always deserve a careful look — cardiac, respiratory, and musculoskeletal causes all overlap.",
    "swelling": "Common contributors are fluid retention, injury, allergic reaction, or cardiac/renal issues.",
}


def _find_asked_symptom(question: str, all_symptoms: list[str]) -> str | None:
    """
    If the question mentions a specific symptom from the recorded list
    (or a common alias), return that symptom. Otherwise return None.

    Priority: exact recorded-symptom substring match wins over alias.
    """
    q = question.lower()

    # First: match against actual recorded symptoms
    for symptom in all_symptoms:
        sym_lower = symptom.lower().strip()
        if not sym_lower:
            continue
        if sym_lower in q:
            return symptom
        first_word = sym_lower.split()[0] if sym_lower.split() else ""
        if first_word and len(first_word) >= 4 and first_word in q:
            return symptom

    # Fallback: alias match — e.g. "fever" in question, "fever" in cues
    for canonical, cues in _SYMPTOM_MATCH_CUES.items():
        if any(cue in q for cue in cues):
            # Find a recorded symptom that matches this canonical family
            for symptom in all_symptoms:
                if any(cue in symptom.lower() for cue in cues):
                    return symptom
            # No matching recorded symptom, but user asked about it —
            # return the canonical name so we still answer sensibly.
            return canonical

    return None


def _hint_for_symptom(symptom: str) -> str:
    """Return a one-line clinical hint for a symptom name."""
    sym_lower = symptom.lower()
    for family, cues in _SYMPTOM_MATCH_CUES.items():
        if any(cue in sym_lower for cue in cues):
            return _SYMPTOM_HINTS.get(family, "The exact cause depends on other factors that need clinician review.")
    return "The exact cause depends on other factors that need clinician review."


def _answer_symptom_general(intel: ReportIntelligence, q: str) -> str:
    """
    If the user asked about a specific symptom (e.g. "why am I experiencing fever?"),
    answer about THAT symptom. Only fall back to full-list summary if the question
    is genuinely generic ("what symptoms are recorded?").
    """
    all_s = intel.all_symptoms()
    if not all_s:
        return "No symptoms are recorded in this report."

    matched = _find_asked_symptom(q, all_s)

    if matched:
        hint = _hint_for_symptom(matched)
        duration = f" for **{intel.symptom_duration}**" if intel.symptom_duration else ""
        return (
            f"**{display_symptom(matched).capitalize()}** is on record{duration}. {hint}"
        )

    # Generic symptom question → summary of all
    duration = f" for **{intel.symptom_duration}**" if intel.symptom_duration else ""
    return (
        f"**Recorded symptoms:** {join_symptoms(all_s, 6)}{duration}. "
        "Ask about a specific one and I'll go deeper."
    )


def _answer_severity(intel: ReportIntelligence, _q: str) -> str:
    # Only runs when NO primary fact intent is present (see _refine_intents).
    confidence = f" ({intel.confidence_pct}% confidence)" if intel.confidence_pct else ""
    cleaned = _clean_reason(intel.severity_reasons[0]) if intel.severity_reasons else ""
    reason = f" — driven mainly by {cleaned}" if cleaned else ""
    if is_urgent(intel.severity):
        action = "Seek prompt clinical evaluation; go to emergency services if symptoms are severe."
    elif is_moderate(intel.severity):
        action = "Worth arranging a clinician review soon."
    else:
        action = "Keep an eye on things and follow up if anything changes."
    return f"This is rated **{intel.severity}**{confidence}{reason}. {action}"


def _answer_condition_status(intel: ReportIntelligence, _q: str) -> str:
    # Only runs when NO primary fact intent is present (see _refine_intents).
    verdict = {
        "CRITICAL": "Not well right now — this is rated **CRITICAL**.",
        "HIGH": "Not good — this is rated **HIGH** severity.",
        "MEDIUM": "A mixed picture — this is rated **MEDIUM** severity.",
        "MODERATE": "A mixed picture — this is rated **MODERATE** severity.",
        "LOW": "Reasonably stable — this is rated **LOW** severity.",
    }.get(intel.severity, f"This is rated **{intel.severity}**.")

    evidence: list[str] = []
    if intel.severity_reasons:
        cleaned = _clean_reason(intel.severity_reasons[0])
        if cleaned:
            evidence.append(cleaned)
    if intel.abnormal_labs:
        evidence.append(f"abnormal labs ({intel.short_lab_summary(n=2)})")
    if intel.symptom_groups.cardiac:
        evidence.append(f"cardiac symptoms ({join_symptoms(intel.symptom_groups.cardiac, 2)})")

    evidence_str = f" Based on {', '.join(evidence)}." if evidence else ""
    action = " Seek urgent care." if is_urgent(intel.severity) else " Keep monitoring and follow up if anything changes."
    return f"{verdict}{evidence_str}{action}"


def _answer_severity_reason(intel: ReportIntelligence, _q: str) -> str:
    # Only runs when NO primary fact intent is present (see _refine_intents).
    cleaned_reasons = [_clean_reason(r) for r in intel.severity_reasons[:3]]
    cleaned_reasons = [r for r in cleaned_reasons if r]

    if cleaned_reasons:
        reasons = "; ".join(cleaned_reasons)
        confidence = f" ({intel.confidence_pct}% confidence)" if intel.confidence_pct else ""
        return f"The **{intel.severity}**{confidence} rating comes down to: {reasons}."
    if intel.abnormal_labs:
        return f"No reason was itemized, but abnormal labs likely played a part: {intel.short_lab_summary()}."
    return f"No specific driver was itemized beyond the **{intel.severity}** rating itself."


def _answer_next_steps(intel: ReportIntelligence, _q: str) -> str:
    if is_urgent(intel.severity):
        return (
            f"**{intel.severity}** — seek urgent clinical evaluation now. Bring this report, "
            "and call emergency services if you have severe chest pain, trouble breathing, fainting, or sudden confusion."
        )
    if is_moderate(intel.severity):
        return f"**{intel.severity}** — arrange a clinician review within a few days. Go sooner if anything worsens."
    return f"**{intel.severity}** — nothing urgent needed right now. Follow up if symptoms persist or new ones show up."


def _answer_duration(intel: ReportIntelligence, _q: str) -> str:
    if not intel.symptom_duration:
        return "There's no symptom duration recorded in this report."
    if is_urgent(intel.severity):
        return f"**{intel.symptom_duration}**, combined with the **{intel.severity}** rating, warrants prompt evaluation."
    return f"**{intel.symptom_duration}** at **{intel.severity}** severity isn't automatically urgent, but mention it at your next visit."


def _answer_confidence(intel: ReportIntelligence, _q: str) -> str:
    if not intel.confidence_pct:
        return "No confidence score is recorded for this report."
    validation = intel.validation_status or "not specified"
    quality = (
        "high — strong signal in the data"
        if intel.confidence_pct >= 85
        else (
            "moderate — reliable but clinician review recommended"
            if intel.confidence_pct >= 65
            else "lower — treat as indicative, confirm with clinician"
        )
    )
    return f"**Confidence:** {intel.confidence_pct}% — {quality}. **Validation:** {validation}."


# ─── New handlers ────────────────────────────────────────────────
# Cover the fallback suggested-question chips from suggested_questions.py
# so no chip ever falls through to the generic catchall or gets rejected
# as off-topic.

def _answer_trend(intel: ReportIntelligence, _q: str) -> str:
    """
    Trend answer. Since chat is scoped to ONE report, we can't compare
    across reports here — but we give a clear, honest, actionable answer
    rather than a generic data dump.
    """
    return (
        f"This chat is scoped to a single report (**{intel.severity}** severity, "
        f"**{intel.confidence_pct}%** confidence), so I can't compare across your other reports from here.\n\n"
        "Open the **Dashboard** to see trend comparisons — it shows whether your severity, "
        "symptoms, and lab values have improved, worsened, or stayed stable across reports."
    )


def _answer_lifestyle(intel: ReportIntelligence, _q: str) -> str:
    """
    Lifestyle advice grounded in what's actually flagged in this report.
    Never generic — always tied to specific findings when possible.
    """
    tips: list[str] = []
    lab_names = " ".join(lab.name.lower() for lab in intel.abnormal_labs)

    if any(kw in lab_names for kw in ("glucose", "sugar", "hba1c", "a1c")):
        tips.append("**Blood sugar:** reduce refined carbs and sugary drinks, aim for 30 min of walking most days")
    if any(kw in lab_names for kw in ("cholesterol", "ldl", "triglyceride", "lipid")):
        tips.append("**Cholesterol:** cut saturated fat, add more fibre (oats, legumes, vegetables), regular cardio")
    if any(kw in lab_names for kw in ("bp", "pressure", "hypertens")):
        tips.append("**Blood pressure:** reduce sodium, limit alcohol, aim for 7–9 hours sleep, manage stress")
    if "vitamin d" in lab_names:
        tips.append("**Vitamin D:** 15–20 min of sunlight most days, include fatty fish or fortified foods")
    if any(kw in lab_names for kw in ("vitamin b12", "b12")):
        tips.append("**B12:** include eggs, dairy, meat, or a supplement if vegetarian")
    if any(kw in lab_names for kw in ("iron", "ferritin", "haemoglobin", "hemoglobin")):
        tips.append("**Iron:** leafy greens, lentils, red meat, take with Vitamin C for absorption")
    if any(kw in lab_names for kw in ("thyroid", "tsh", "t3", "t4")):
        tips.append("**Thyroid:** consistent sleep, manage stress, avoid crash diets, take medication as prescribed")

    if intel.symptom_groups.cardiac:
        tips.append("**Cardiac symptoms:** avoid heavy caffeine, monitor blood pressure, don't ignore chest discomfort")
    if intel.symptom_groups.respiratory:
        tips.append("**Breathing:** avoid smoke and strong fumes, stay hydrated, use a humidifier if air is dry")
    if intel.symptom_groups.neurological:
        neuro_label = " / ".join(
            s.capitalize() for s in intel.symptom_groups.neurological[:2]
        )
        tips.append(f"**{neuro_label}:** stay hydrated, don't skip meals, sleep regularly, limit screen strain")

    if not tips:
        return (
            "Nothing specific to target in this report, but general basics still apply:\n\n"
            "• Regular movement (30 min most days)\n"
            "• Balanced meals with vegetables, protein, and whole grains\n"
            "• 7–9 hours sleep\n"
            "• Hydration\n"
            "• Limit alcohol and smoking\n\n"
            "Ask your clinician what's most relevant for you."
        )

    header = "Based on what's flagged in your report:\n\n"
    body = "\n".join(f"• {t}" for t in tips[:4])
    footer = "\n\nDiscuss any major changes with your clinician first."
    return header + body + footer


def _answer_recheck(intel: ReportIntelligence, _q: str) -> str:
    """When to recheck the flagged values."""
    if not intel.abnormal_labs:
        return (
            "No abnormal labs are flagged in this report, so no specific rechecks are needed right now.\n\n"
            "Routine annual labs are still worth doing."
        )

    items: list[str] = []
    for lab in intel.abnormal_labs[:4]:
        name_lower = lab.name.lower()
        if any(kw in name_lower for kw in ("hba1c", "a1c")):
            items.append(f"• **{lab.name}** — every 3 months while managing, every 6 months once stable")
        elif "glucose" in name_lower:
            items.append(f"• **{lab.name}** — recheck fasting glucose in 4–6 weeks")
        elif any(kw in name_lower for kw in ("cholesterol", "ldl", "hdl", "lipid", "triglyceride")):
            items.append(f"• **{lab.name}** — recheck in 8–12 weeks after any diet or medication change")
        elif any(kw in name_lower for kw in ("thyroid", "tsh", "t3", "t4")):
            items.append(f"• **{lab.name}** — recheck in 6–8 weeks after any dose change, otherwise every 6 months")
        elif any(kw in name_lower for kw in ("vitamin d", "vitamin b12", "b12", "ferritin", "iron")):
            items.append(f"• **{lab.name}** — recheck in 8–12 weeks after starting supplements")
        elif any(kw in name_lower for kw in ("bp", "pressure")):
            items.append(f"• **{lab.name}** — home monitor weekly, clinic recheck in 4 weeks")
        else:
            items.append(f"• **{lab.name}** — ask your clinician for a specific recheck interval")

    header = "For the values flagged in this report:\n\n"
    body = "\n".join(items)
    footer = "\n\nYour clinician may adjust these intervals based on your history."
    return header + body + footer


def _answer_ranges(intel: ReportIntelligence, _q: str) -> str:
    """Explain reference ranges using the actual labs in the report."""
    if not intel.abnormal_labs and not intel.normal_labs:
        return "No lab values are recorded in this report to compare against reference ranges."

    intro = (
        "Reference ranges are the 'expected' band for a healthy population — "
        "they're not one-size-fits-all, and being slightly outside doesn't always mean disease.\n\n"
    )

    lines: list[str] = []
    if intel.abnormal_labs:
        lines.append("**Flagged in your report:**")
        for lab in intel.abnormal_labs[:3]:
            direction = lab.direction or "outside range"
            lines.append(f"• {lab.name} — {direction}")
        lines.append("")

    if intel.normal_labs:
        lines.append("**Within range:**")
        for lab in intel.normal_labs[:3]:
            lines.append(f"• {lab.name} — normal")

    footer = "\n\nYour clinician can explain what each range means for your specific situation."
    return intro + "\n".join(lines) + footer


def _answer_specialist(intel: ReportIntelligence, _q: str) -> str:
    """Suggest specialist based on what's flagged."""
    referrals: list[str] = []

    lab_names = " ".join(lab.name.lower() for lab in intel.abnormal_labs)
    if any(kw in lab_names for kw in ("thyroid", "tsh", "t3", "t4", "glucose", "hba1c", "insulin")):
        referrals.append("**Endocrinologist** — for thyroid or blood sugar findings")
    if any(kw in lab_names for kw in ("cholesterol", "ldl", "triglyceride", "bp", "pressure")) or intel.symptom_groups.cardiac:
        referrals.append("**Cardiologist** — for heart or lipid findings")
    if intel.symptom_groups.respiratory or any(kw in lab_names for kw in ("oxygen", "spo2")):
        referrals.append("**Pulmonologist** — for breathing or oxygen concerns")
    if intel.symptom_groups.neurological:
        referrals.append("**Neurologist** — for dizziness, headache, or balance symptoms")
    if any(kw in lab_names for kw in ("creatinine", "urea", "bun", "kidney")):
        referrals.append("**Nephrologist** — for kidney function findings")
    if any(kw in lab_names for kw in ("alt", "ast", "bilirubin", "liver", "ggt", "alp")):
        referrals.append("**Hepatologist / Gastroenterologist** — for liver findings")

    if not referrals:
        return (
            "Start with your **primary care doctor / GP** — they can review the full picture "
            "and refer you to a specialist if anything needs deeper evaluation."
        )

    header = "Based on what's in this report:\n\n"
    body = "\n".join(f"• {r}" for r in referrals[:3])
    footer = "\n\nYour **primary care doctor** can coordinate any referrals."
    return header + body + footer


def _answer_catchall(intel: ReportIntelligence, _q: str) -> str:
    """
    Generic overview — used only when no intent matches at all.
    Uses paragraph breaks + bold labels for scannability instead of a
    single dense sentence.
    """
    confidence = f"{intel.confidence_pct}% confidence" if intel.confidence_pct else "confidence not recorded"
    return (
        f"**Severity:** {intel.severity} ({confidence})\n"
        f"**Symptoms:** {intel.short_symptom_summary()}\n"
        f"**Abnormal labs:** {intel.short_lab_summary()}\n"
        f"**Medications:** {intel.short_medication_summary()}\n\n"
        "Ask about any specific lab value, symptom, medication, or finding and I'll go deeper."
    )


# ── Dispatch table ────────────────────────────────────────────────

_INTENT_HANDLERS: dict[str, Any] = {
    "EMERGENCY": _answer_emergency,
    "DRUG_INTERACTION": _answer_drug_interaction,
    "DRUG_WARNING": _answer_drug_warning,
    "MEDICATION": _answer_medication,
    "XRAY": _answer_xray,
    "LAB_GLUCOSE": _answer_lab_glucose,
    "LAB_THYROID": _answer_lab_thyroid,
    "LAB_VITAMIN": _answer_lab_vitamin,
    "LAB_BP": _answer_lab_bp,
    "LAB_CHOLESTEROL": _answer_lab_cholesterol,
    "LAB_OXYGEN": _answer_lab_oxygen,
    "LAB_GENERAL": _answer_lab_general,
    "CARDIAC": _answer_cardiac,
    "RESPIRATORY": _answer_respiratory,
    "NEURO": _answer_neuro,
    "METABOLIC": _answer_metabolic,
    "SYMPTOM_GENERAL": _answer_symptom_general,
    "SEVERITY": _answer_severity,
    "CONDITION_STATUS": _answer_condition_status,
    "SEVERITY_REASON": _answer_severity_reason,
    "NEXT_STEPS": _answer_next_steps,
    "DURATION": _answer_duration,
    "CONFIDENCE": _answer_confidence,
    "TREND": _answer_trend,
    "LIFESTYLE": _answer_lifestyle,
    "RECHECK": _answer_recheck,
    "RANGES": _answer_ranges,
    "SPECIALIST": _answer_specialist,
}


def _sentence_key(sentence: str) -> str:
    """Normalize a sentence for near-duplicate comparison."""
    words = re.findall(r"[a-z0-9]+", sentence.lower())
    stop = {"a", "an", "the", "is", "this", "and", "to", "that", "for", "of", "on"}
    return " ".join(w for w in words if w not in stop)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _strip_duplicate_sentences(candidate: str, already: list[str]) -> str:
    """Remove sentences from `candidate` that are near-duplicates of any
    sentence already present in `already`."""
    prior_keys: set[str] = set()
    for prior_answer in already:
        for sent in _split_sentences(prior_answer):
            key = _sentence_key(sent)
            if len(key) >= 8:
                prior_keys.add(key)

    kept: list[str] = []
    for sent in _split_sentences(candidate):
        key = _sentence_key(sent)
        if len(key) >= 8 and key in prior_keys:
            continue
        kept.append(sent)
        if len(key) >= 8:
            prior_keys.add(key)

    return " ".join(kept).strip()


def _generate_answer(
    intel: ReportIntelligence,
    question: str,
    history: list[dict],
) -> str:
    """
    Detect intents → refine → dispatch to handler(s) → return
    deterministic answer.
    """
    intents = _refine_intents(_detect_intents(question))

    if not intents:
        return _answer_catchall(intel, question)

    if intents[0] == "EMERGENCY":
        return _answer_emergency(intel, question)

    answers: list[str] = []
    seen: set[str] = set()

    for intent in intents[:2]:
        handler = _INTENT_HANDLERS.get(intent)
        if not handler or intent in seen:
            continue
        seen.add(intent)
        result = handler(intel, question)
        if not result:
            continue
        deduped = _strip_duplicate_sentences(result, answers)
        if deduped:
            answers.append(deduped)

    if not answers:
        return _answer_catchall(intel, question)

    # Paragraph break reads more naturally than " | " between answers
    return "\n\n".join(answers) if len(answers) > 1 else answers[0]


# ── Severity delta ────────────────────────────────────────────────

# Unambiguous escalation phrases — safe to match anywhere in the message.
_ESCALATION_PHRASES = (
    "getting worse", "feels worse", "is worse", "much worse",
    "can't breathe", "cannot breathe", "collapsed", "passed out",
    "unconscious", "just started", "new pain", "sudden", "suddenly",
)
_IMPROVEMENT_CUES = (
    "better", "improving", "improved", "resolved", "fine now", "feeling better",
)

# Symptom/severity words that only count as escalation when the user is
# making a first-person, present-state claim about themselves — NOT when
# they're quoting or asking about report text. e.g. "I have severe chest
# pain now" is a real claim; "The report flags 'chest pain' — what does
# that mean?" is a question referencing the report's own wording and must
# not trip the badge.
_CONDITIONAL_ESCALATION_WORDS = ("severe", "chest pain", "worse", "worsening")
_FIRST_PERSON_CLAIM_CUES = (
    "i have", "i've got", "i am having", "i'm having", "having ",
    "i feel", "i'm feeling", "my ", "i am experiencing", "i'm experiencing",
)
# If the message is clearly quoting/referencing the report rather than
# self-reporting, conditional words never count even with first-person
# framing nearby (protects "the report flags '...' — can you explain?").
_QUOTING_CUES = (
    "the report", "report flags", "report says", "it flags", "it says",
    "says '", 'says "', "flags '", 'flags "', "can you explain", "what does that mean",
)


def _assess_severity_delta(message: str, record: HealthRecord) -> str | None:
    msg = message.lower()

    if any(p in msg for p in _ESCALATION_PHRASES):
        return "increased"

    is_quoting = any(c in msg for c in _QUOTING_CUES)
    if not is_quoting:
        has_claim_framing = any(c in msg for c in _FIRST_PERSON_CLAIM_CUES)
        has_conditional_word = any(w in msg for w in _CONDITIONAL_ESCALATION_WORDS)
        if has_claim_framing and has_conditional_word:
            return "increased"

    if any(f in msg for f in _IMPROVEMENT_CUES):
        return "decreased"
    return "unchanged"


# ── Helpers ───────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_job_and_record(
    job_id: str, user: User, db: Session
) -> tuple[PipelineJobRow, HealthRecord]:
    job_row = (
        db.query(PipelineJobRow).filter_by(job_id=job_id, user_id=user.id).first()
    )
    if not job_row:
        raise HTTPException(status_code=404, detail="Job not found")
    record = (
        db.query(HealthRecord).filter_by(job_id=job_id, user_id=user.id).first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Record not found for this job")
    return job_row, record


def _build_suggestions(
    intel: ReportIntelligence,
    record: HealthRecord,
    history: list[dict],
) -> list[str]:
    """Build suggested questions from the rich intel object."""
    context = {
        "severity": intel.severity,
        "confidence_percent": intel.confidence_pct,
        "extracted_symptoms": intel.all_symptoms(),
        "reported_symptoms": record.symptoms_text or "",
        "lab_abnormal_values": [lab.raw_text for lab in intel.abnormal_labs],
        "lab_measurements": {
            lab.name: lab.value for lab in intel.normal_labs if lab.value
        },
        "medications": intel.medications,
        "drug_interactions": [
            {"drugs": i.drugs, "severity": i.severity, "description": i.description}
            for i in intel.drug_interactions
        ],
        "drug_warnings": intel.drug_warnings,
        "xray_findings": [f.text for f in intel.xray_findings],
        "severity_reasons": intel.severity_reasons,
        "symptom_duration": intel.symptom_duration,
        "severity_indicators": intel.severity_indicators,
    }
    suggested = build_suggested_questions(
        record.severity,
        intel.all_symptoms(),
        context=context,
        history=history,
    )
    return suggested or ["What are the most critical factors in my report?"]


# ── Routes ────────────────────────────────────────────────────────

@router.get("/chat/{job_id}/init")
def queue_chat_init(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return the chat state — including any previously-used turns and
    prior messages — when the chat panel opens.
    """
    _, record = _get_job_and_record(job_id, user, db)

    history: list[dict] = _load_history(db, user.id, job_id)

    context = _build_report_context(record)
    intel = ReportIntelligence.from_context(context)

    used_turns = _count_user_turns(history)
    limit_reached = used_turns >= MAX_TURNS

    suggested = [] if limit_reached else _build_suggestions(intel, record, history)

    replayed_messages = [
        {
            "role": m.get("role"),
            "content": m.get("content"),
            "severity_delta": m.get("severity_delta"),
        }
        for m in history
        if m.get("role") in ("user", "assistant")
    ]

    return {
        "job_id": job_id,
        "turn": used_turns,
        "turns_remaining": max(0, MAX_TURNS - used_turns),
        "messages": replayed_messages,
        "suggested_questions": suggested,
        "limit_reached": limit_reached,
    }


@router.post("/chat")
async def queue_chat(
    payload: ChatIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatOut:
    """
    Answer up to MAX_TURNS questions using the selected user-owned report.
    """
    _, record = _get_job_and_record(payload.job_id, user, db)

    history = _load_history(db, user.id, payload.job_id)
    used_turns = _count_user_turns(history)

    if used_turns >= MAX_TURNS:
        return ChatOut(
            job_id=payload.job_id,
            turn=MAX_TURNS,
            answer=(
                f"You've reached the {MAX_TURNS}-question follow-up limit for this report.\n\n"
                "Start a new assessment if your symptoms or clinical information have changed."
            ),
            severity_delta=None,
            suggested_questions=[],
            enriched=False,
        )

    context = _build_report_context(record)
    intel = ReportIntelligence.from_context(context)

    # ── Duplicate — replay without consuming a turn ───────────────
    if _is_duplicate(payload.message, history):
        prior = _cached_answer(payload.message, history)
        return ChatOut(
            job_id=payload.job_id,
            turn=used_turns,
            answer=(
                "You asked this a moment ago — here's the same answer:\n\n"
                + (prior or "See the answer above.")
            ),
            severity_delta=None,
            suggested_questions=_build_suggestions(intel, record, history),
            enriched=False,
        )

    # ── Off-topic — redirect without consuming a turn ─────────────
    if _is_off_topic(payload.message):
        return ChatOut(
            job_id=payload.job_id,
            turn=used_turns,
            answer=(
                "That doesn't look health-related. I can help with your symptoms, "
                "lab results, medications, X-ray findings, severity rating, trends, "
                "lifestyle guidance, or specialist recommendations."
            ),
            severity_delta=None,
            suggested_questions=_build_suggestions(intel, record, history),
            enriched=False,
        )

    # ── Step 1: deterministic answer — always correct, always fast ─
    intents = _refine_intents(_detect_intents(payload.message))
    base_answer = _generate_answer(intel, payload.message, history)
    severity_delta = _assess_severity_delta(payload.message, record)

    # ── Step 2: optional enrichment — one grounded sentence ────────
    final_answer = base_answer
    was_enriched = False

    try:
        from tools.chat_enricher import enrich_answer
        from backend.model_registry import model_registry
        enriched = await enrich_answer(
            base_answer=base_answer,
            intel=intel,
            question=payload.message,
            intents=intents,
            model_registry=model_registry,
        )
        if enriched != base_answer:
            final_answer = enriched
            was_enriched = True
    except Exception as exc:
        logger.debug(
            "chat · enrichment unavailable — using deterministic answer",
            error=str(exc),
        )

    _append_message(db, user.id, payload.job_id, "user", payload.message)
    _append_message(db, user.id, payload.job_id, "assistant", final_answer, severity_delta)
    db.commit()

    history = history + [
        {"role": "user", "content": payload.message, "severity_delta": None},
        {"role": "assistant", "content": final_answer, "severity_delta": severity_delta},
    ]

    logger.info(
        "Chat turn answered",
        job_id=payload.job_id,
        turn=used_turns + 1,
        intents=intents,
        enriched=was_enriched,
    )

    return ChatOut(
        job_id=payload.job_id,
        turn=used_turns + 1,
        answer=final_answer,
        severity_delta=severity_delta,
        suggested_questions=_build_suggestions(intel, record, history),
        enriched=was_enriched,
    )


@router.post("/rerun/{job_id}")
def rerun_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the existing score. Full pipeline is not rerun from chat."""
    _, record = _get_job_and_record(job_id, user, db)
    return {
        "job_id": job_id,
        "rerun": True,
        "severity": record.severity,
        "confidence": record.confidence,
    }