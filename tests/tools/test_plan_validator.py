"""
tests/tools/test_plan_validator.py — PlanValidator.

PlanValidator has a single responsibility: validate use_rag.
No mocking required — synchronous pure Python.

Coverage:
    - use_rag=False with no safety signals → respected (not forced)
    - use_rag=False + critical symptom term → forced True + was_repaired
    - use_rag=False + critical xray finding → forced True
    - use_rag=False + polypharmacy exceeded → forced True
    - use_rag=True → not counted as repair
    - use_rag=False at exactly threshold → not forced (> not >=)
    - use_rag=False above threshold → forced
    - Routine symptom text does not trigger RAG forcing
    - validation_errors populated on repair
    - was_repaired=False when no repair needed
    - is_fallback preserved
    - Fresh ExecutionPlan returned (input not mutated)
    - model_validator invariant enforced by schema
    - "severe" alone does not force RAG (regression guard)
    - "cardiac" alone does not force RAG (regression guard)
    - Compound phrases containing trigger terms still force RAG
"""

from __future__ import annotations

import pytest

from schemas.plan import ExecutionPlan
from schemas.state import AegisState
from tools.plan_validator import PlanValidator
from tools.planner_constants import RAG_FORCE_POLYPHARMACY_THRESHOLD


# ── Helpers ───────────────────────────────────────────────────────

def _plan(
    use_rag:      bool = False,
    reasoning:    str  = "test",
    is_fallback:  bool = False,
    was_repaired: bool = False,
) -> ExecutionPlan:
    return ExecutionPlan(
        use_rag      = use_rag,
        reasoning    = reasoning,
        is_fallback  = is_fallback,
        was_repaired = was_repaired,
    )


def _validate(raw: ExecutionPlan, state: AegisState) -> ExecutionPlan:
    return PlanValidator().validate(raw, state)


# ── No safety signals — planner decision respected ────────────────

def test_use_rag_false_no_signals_not_forced():
    state = AegisState(raw_symptoms_text="mild headache")
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is False
    assert result.was_repaired is False


def test_use_rag_false_empty_state_not_forced():
    result = _validate(_plan(use_rag=False), AegisState())
    assert result.use_rag is False
    assert result.was_repaired is False


def test_use_rag_false_routine_symptoms_not_forced():
    state = AegisState(raw_symptoms_text="runny nose and mild cough")
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is False


# ── Safety signals — RAG forced ───────────────────────────────────

def test_chest_pain_forces_rag():
    state = AegisState(raw_symptoms_text="chest pain for 2 days")
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is True
    assert result.was_repaired is True


def test_shortness_of_breath_forces_rag():
    state = AegisState(raw_symptoms_text="shortness of breath")
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is True


def test_troponin_in_symptoms_forces_rag():
    state = AegisState(raw_symptoms_text="elevated troponin levels noted")
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is True


def test_dyspnoea_british_spelling_forces_rag():
    """British canonical spelling triggers RAG correctly."""
    state = AegisState(raw_symptoms_text="progressive dyspnoea on exertion")
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is True


def test_dyspnea_us_spelling_forces_rag():
    """US spelling also triggers RAG — included for patient-reported text."""
    state = AegisState(raw_symptoms_text="dyspnea since yesterday")
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is True


def test_pneumothorax_finding_forces_rag():
    state = AegisState(xray_findings_raw=["Pneumothorax"])
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is True
    assert result.was_repaired is True


def test_pulmonary_edema_finding_forces_rag():
    state = AegisState(xray_findings_raw=["Pulmonary Edema"])
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is True


def test_cardiomegaly_finding_forces_rag():
    state = AegisState(xray_findings_raw=["Cardiomegaly"])
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is True


def test_polypharmacy_above_threshold_forces_rag():
    meds  = ["drug"] * (RAG_FORCE_POLYPHARMACY_THRESHOLD + 1)
    state = AegisState(medications_raw=meds)
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is True
    assert result.was_repaired is True


def test_polypharmacy_at_threshold_not_forced():
    """Exactly at threshold — > not >= so threshold itself does not trigger."""
    meds  = ["drug"] * RAG_FORCE_POLYPHARMACY_THRESHOLD
    state = AegisState(medications_raw=meds)
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is False


def test_polypharmacy_below_threshold_not_forced():
    meds  = ["drug"] * (RAG_FORCE_POLYPHARMACY_THRESHOLD - 1)
    state = AegisState(medications_raw=meds)
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is False


# ── use_rag already True — not a repair ──────────────────────────

def test_use_rag_already_true_not_repaired():
    state  = AegisState(raw_symptoms_text="chest pain")
    result = _validate(_plan(use_rag=True), state)
    assert result.use_rag is True
    assert result.was_repaired is False
    assert result.validation_errors == []


# ── validation_errors ─────────────────────────────────────────────

def test_validation_errors_populated_on_repair():
    state  = AegisState(raw_symptoms_text="chest pain")
    result = _validate(_plan(use_rag=False), state)
    assert len(result.validation_errors) >= 1


def test_validation_errors_empty_on_no_repair():
    state  = AegisState(raw_symptoms_text="mild headache")
    result = _validate(_plan(use_rag=False), state)
    assert result.validation_errors == []


# ── is_fallback preservation ──────────────────────────────────────

def test_is_fallback_preserved_true():
    state  = AegisState()
    result = _validate(_plan(is_fallback=True), state)
    assert result.is_fallback is True


def test_is_fallback_preserved_false():
    state  = AegisState()
    result = _validate(_plan(is_fallback=False), state)
    assert result.is_fallback is False


# ── Fresh plan returned — input not mutated ───────────────────────

def test_returns_fresh_plan():
    state = AegisState(raw_symptoms_text="chest pain")
    raw   = _plan(use_rag=False)
    result = _validate(raw, state)
    assert result is not raw


def test_input_not_mutated():
    state = AegisState(raw_symptoms_text="chest pain")
    raw   = _plan(use_rag=False)
    _validate(raw, state)
    assert raw.use_rag is False


# ── Model validator invariant ─────────────────────────────────────

def test_schema_rejects_fallback_and_repaired():
    """Schema-level invariant: both True simultaneously is a bug."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ExecutionPlan(
            use_rag      = True,
            reasoning    = "test",
            is_fallback  = True,
            was_repaired = True,
        )


# ── Case-insensitive xray matching ────────────────────────────────

def test_xray_finding_case_insensitive():
    """Finding 'pneumothorax' (lowercase) must still trigger RAG."""
    state  = AegisState(xray_findings_raw=["pneumothorax"])
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is True


# ── Regression guards — excluded terms ───────────────────────────

def test_severe_modifier_alone_does_not_force_rag():
    """
    'severe' as a standalone modifier does not force RAG.

    Clinical significance of 'severe' depends entirely on what it
    modifies. A substring match cannot make that determination.
    'severe headache' is not a high-acuity trigger in the same way
    'chest pain' is.

    Guards against future regression where someone reintroduces
    'severe' as a standalone term in RAG_FORCE_SYMPTOM_TERMS.
    """
    state = AegisState(raw_symptoms_text="severe headache for 2 days")
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is False


def test_severe_chest_pain_forces_rag_via_chest_pain_term():
    """
    'severe chest pain' triggers RAG because 'chest pain' is in
    RAG_FORCE_SYMPTOM_TERMS — not because 'severe' is present.

    Demonstrates that the modifier is irrelevant to the trigger
    decision. The clinical signal ('chest pain') does the work.
    """
    state = AegisState(raw_symptoms_text="severe chest pain")
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is True  # triggered by "chest pain", not "severe"


def test_cardiac_adjective_alone_does_not_force_rag():
    """
    'cardiac' as a standalone adjective does not force RAG.

    'cardiac history', 'cardiac rehab', 'cardiac clinic' are not
    independently high-acuity signals. Specific terms ('chest pain',
    'troponin', 'heart attack') cover genuine cardiac emergencies
    without the false positives that bare 'cardiac' produces.

    Guards against future regression where someone reintroduces
    'cardiac' as a standalone term in RAG_FORCE_SYMPTOM_TERMS.
    """
    state = AegisState(
        raw_symptoms_text="cardiac history, on medication review"
    )
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is False


def test_cardiac_emergency_forces_rag_via_specific_term():
    """
    Demonstrates that specific cardiac indicators trigger RAG correctly
    after removing the generic 'cardiac' adjective from the trigger list.

    'elevated troponin' triggers RAG because 'troponin' is in
    RAG_FORCE_SYMPTOM_TERMS. The word 'cardiac' appearing elsewhere in
    the text does not contribute to the trigger.

    Note: 'cardiac arrest' as a standalone phrase would NOT trigger RAG
    with the current trigger list — it contains neither 'chest pain',
    'troponin', 'heart attack', nor any other listed term. If future
    requirements indicate that cardiac arrest should force RAG, adding
    'cardiac arrest' as a compound phrase would satisfy the current
    selection criteria.
    """
    state = AegisState(raw_symptoms_text="elevated troponin, query MI")
    result = _validate(_plan(use_rag=False), state)
    assert result.use_rag is True  # triggered by "troponin", not "cardiac"