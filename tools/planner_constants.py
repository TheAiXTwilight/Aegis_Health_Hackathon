"""
tools/planner_constants.py — PlanValidator safety-override configuration.

This module owns the configuration that controls when PlanValidator
forces use_rag=True regardless of planner output.

Separation of concerns:
    tools/plan_validator.py    — the override mechanism
    tools/planner_constants.py — the override configuration (this file)

Changing a threshold or adding a clinical term requires editing only
this file. The mechanism in PlanValidator is unchanged.

This module is NOT a tool. It has no run() method, no lifecycle, and
no execution semantics. It is imported directly where needed:

    from tools.planner_constants import RAG_FORCE_SYMPTOM_TERMS

It is NOT re-exported from tools/__init__.py.

frozenset is used for all set constants — immutable by type, not
merely by convention. frozenset provides constant-time membership
testing for the configured trigger set (term in RAG_FORCE_SYMPTOM_TERMS);
the overall substring scanning cost in PlanValidator depends on input
text length, not on set size.

Term selection policy:
    Terms must be clinically specific enough to stand alone as
    high-acuity signals without a modifier. Generic terms whose
    clinical significance depends on context are excluded.

    Each term in RAG_FORCE_SYMPTOM_TERMS must satisfy both criteria:

        1. Independently interpretable as a high-acuity clinical signal
           without requiring a qualifying phrase or contextual
           interpretation.

        2. Safe to match using simple case-insensitive substring search.
           Terms that require contextual interpretation (e.g. "severe",
           "acute", "sudden", "cardiac") must not be included because
           substring matching cannot determine what they are modifying.

    Examples of correct triggering:
        "severe chest pain"          → matches "chest pain"         ✓
        "severe shortness of breath" → matches "shortness of breath" ✓
        "elevated troponin levels"   → matches "troponin"           ✓

    Examples of non-triggering (correct behaviour):
        "severe headache"            → no term matches              ✓
        "cardiac history"            → no term matches              ✓
        "cardiac rehab"              → no term matches              ✓
        "severe anxiety"             → no term matches              ✓

    Known coverage gap:
        "cardiac arrest" does not currently trigger RAG — it contains
        neither "chest pain", "troponin", "heart attack", nor any other
        listed term. If future requirements indicate that cardiac arrest
        should force RAG, adding "cardiac arrest" as a compound phrase
        would satisfy the current selection criteria.

    Note on synonyms:
        "unconscious" may not match all patient phrasings
        ("passed out", "blacked out", "loss of consciousness", "LOC").
        Synonym expansion is deferred to Phase 3 when structured
        symptom extraction is available.

Fallback plan rationale (Decision 50):
    When planner reasoning is unavailable, use_rag defaults to True
    as a safety-first policy. Retrieving unnecessary evidence is
    acceptable; omitting evidence in a planner-failure scenario is not.
"""

from __future__ import annotations


# ── Symptom terms that force use_rag=True ─────────────────────────
#
# Searched as substrings (case-insensitive) in state.raw_symptoms_text.
#
# Selection criteria (both must be satisfied):
#   1. Independently high-acuity without a modifier.
#   2. Safe to match via case-insensitive substring search without
#      contextual disambiguation.
#
# Excluded intentionally:
#   "severe"    — modifier; depends on what it modifies
#   "acute"     — modifier; same reason
#   "sudden"    — modifier; same reason
#   "cardiac"   — adjective; matches "cardiac history", "cardiac rehab",
#                 "cardiac clinic" — not independently high-acuity

RAG_FORCE_SYMPTOM_TERMS: frozenset[str] = frozenset({
    # Cardiac — specific presentations only, not the generic adjective
    "chest pain",
    "chest tightness",
    "chest pressure",
    "heart attack",
    "troponin",
    # Respiratory — British spelling canonical, US spelling included
    # for robustness against patient-reported text
    "shortness of breath",
    "dyspnoea",
    "dyspnea",
    "breathlessness",
    # Neurological
    "stroke",
    "seizure",
    "unconscious",
})


# ── X-ray findings that force use_rag=True ────────────────────────
#
# Matched case-insensitively against state.xray_findings_raw entries.
# These findings correspond directly to HIGH severity X-ray rules in
# SeverityScorer, making evidence retrieval clinically important.
#
# Entries use the canonical checklist vocabulary from XRayResult.findings.

RAG_FORCE_XRAY_FINDINGS: frozenset[str] = frozenset({
    "pneumothorax",
    "pulmonary edema",
    "cardiomegaly",
})


# ── Polypharmacy threshold ─────────────────────────────────────────
#
# When len(state.medications_raw) > RAG_FORCE_POLYPHARMACY_THRESHOLD,
# use_rag is forced True regardless of planner output.
#
# Named for the clinical concept (polypharmacy) not the storage
# mechanism (count) — intent is obvious six months later.
# Threshold is > not >= so that exactly 3 medications do not trigger.

RAG_FORCE_POLYPHARMACY_THRESHOLD: int = 3