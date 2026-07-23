"""
tests/scenarios/test_scenario_c.py — Scenario C: Token budget stress.

Uses a large symptom text, 20 medications, and multiple xray findings
to push the context builder toward its token budget limit.

The pipeline executes exactly once via a module-scoped async fixture.
All test functions read from the single resulting (state, report_text)
tuple — one Ollama call for the entire module.

Inputs:
    raw_symptoms_text  — ~800-token symptom description (repeated prose)
    medications_raw    — 20 medications (mix of known and unknown)
    xray_findings_raw  — four positive findings

Expected:
    pipeline completes without raising (no OOM)
    pipeline_complete = True
    report present (sections validated by ReportGenerator)
    if truncation fired: core_fields_truncated is False
    tools_run ∩ tools_failed = ∅

The truncation invariant (core never truncated) is tested
conditionally — we do not require truncation to always fire since
it depends on exact token budget math, but when it does the
invariant must hold. Deterministic truncation coverage lives in
tests/tools/test_report_generator.py::test_build_context_sets_core_truncation_when_budget_tiny.

Marked @pytest.mark.ollama — requires live Ollama with aegis-llama loaded.
Run with: pytest tests/scenarios/test_scenario_c.py -v -m ollama
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from agents.pipeline import AegisPipeline
from schemas.state import AegisState


# ── Stress inputs ─────────────────────────────────────────────────

_LONG_SYMPTOMS = (
    "Patient presents with a complex multi-system presentation. "
    "Chief complaint: severe chest pain radiating to the left arm, "
    "associated with profuse diaphoresis and nausea for the past 6 hours. "
    "Additionally reports progressive shortness of breath on minimal exertion "
    "over the past 2 weeks, orthopnoea requiring 3 pillows, and paroxysmal "
    "nocturnal dyspnoea on 3 occasions this week. "
    "Past medical history is significant for hypertension diagnosed 15 years ago, "
    "type 2 diabetes mellitus for 10 years, hypercholesterolaemia, "
    "and a previous myocardial infarction 5 years ago with subsequent "
    "percutaneous coronary intervention to the left anterior descending artery. "
    "Social history: 30 pack-year smoking history, quit 5 years ago. "
    "Occasional alcohol use. Retired engineer. "
    "Review of systems positive for: ankle oedema bilateral for 3 weeks, "
    "reduced urine output, fatigue, anorexia, and mild confusion in the "
    "evenings over the past 4 days. "
    "Denies fever, rigors, haemoptysis, haematemesis, melaena, or recent travel. "
    "Family history: father died of MI at age 58, mother has atrial fibrillation. "
    "On examination the patient is pale, diaphoretic, and distressed. "
    "BP 88/56 mmHg, HR 118 bpm irregular, RR 28/min, SpO2 89% on room air. "
    "Elevated JVP at 6 cm above sternal angle. "
    "Bibasal crackles on auscultation. S3 gallop present. "
    "Pitting oedema to mid-thigh bilaterally. "
    "Abdomen: hepatomegaly 4 cm below costal margin, mild ascites. "
    "ECG shows sinus tachycardia with new left bundle branch block "
    "and ST elevation in leads V1 through V4. "
) * 2  # repeated to approach 800-token budget

_MEDICATIONS = [
    "aspirin", "warfarin", "metformin", "lisinopril", "atorvastatin",
    "metoprolol", "amlodipine", "furosemide", "spironolactone", "clopidogrel",
    "ramipril", "digoxin", "amiodarone", "ibuprofen", "insulin glargine",
    "pantoprazole", "allopurinol", "levothyroxine", "sertraline", "omeprazole",
]

_XRAY_FINDINGS = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Pulmonary Edema",
    "Consolidation",
]


# ── Module-scoped fixture — one pipeline execution ─────────────────

@pytest_asyncio.fixture(scope="module")
async def scenario_c():
    """
    Run Scenario C once. All tests in this module share the result.
    Returns (state, report_text).
    """
    state = AegisState(
        raw_symptoms_text=_LONG_SYMPTOMS,
        medications_raw=_MEDICATIONS,
        xray_findings_raw=_XRAY_FINDINGS,
    )
    pipeline = AegisPipeline()
    tokens: list[str] = []
    async for token in pipeline.run(state):
        tokens.append(token)
    return state, "".join(tokens)


# ── Pipeline completion ────────────────────────────────────────────

@pytest.mark.ollama
def test_c_pipeline_completes(scenario_c):
    """Pipeline must complete without raising under token budget stress."""
    state, _ = scenario_c
    assert state.pipeline_complete is True


# ── Report ─────────────────────────────────────────────────────────

@pytest.mark.ollama
def test_c_report_present(scenario_c):
    """
    Report existence implies all six sections passed validation.
    FatalPipelineError would have been raised — and state.report left
    None — if any required section were missing.
    """
    state, _ = scenario_c
    assert state.report is not None


@pytest.mark.ollama
def test_c_report_text_contains_summary(scenario_c):
    """Smoke-check on streamed token content."""
    _, report_text = scenario_c
    assert "### Summary" in report_text


@pytest.mark.ollama
def test_c_report_severity_matches_scorer(scenario_c):
    state, _ = scenario_c
    assert state.report is not None
    assert state.severity_result is not None
    assert state.report.severity == state.severity_result.level


# ── Truncation invariant ───────────────────────────────────────────

@pytest.mark.ollama
def test_c_core_not_truncated_when_truncation_fired(scenario_c):
    """
    Invariant: if any truncation flag fired, core must not be the victim.
    Core fields (severity, key symptoms) are unconditionally protected
    by the token budget logic in _build_context().

    Tested conditionally — truncation is not guaranteed to fire since
    it depends on exact token budget math. When it does fire the
    core-fields-protected invariant must hold.

    Deterministic coverage of this path:
        tests/tools/test_report_generator.py
        ::test_build_context_sets_core_truncation_when_budget_tiny
    """
    state, _ = scenario_c
    if state.enrichment_fields_truncated or state.core_fields_truncated:
        assert state.core_fields_truncated is False, (
            "Core fields were truncated — severity or key symptoms may be "
            "missing from the report context"
        )


# ── Severity ───────────────────────────────────────────────────────

@pytest.mark.ollama
def test_c_severity_result_present(scenario_c):
    state, _ = scenario_c
    assert state.severity_result is not None


# ── Pipeline lifecycle ─────────────────────────────────────────────

@pytest.mark.ollama
def test_c_tools_run_and_failed_disjoint(scenario_c):
    state, _ = scenario_c
    overlap = set(state.tools_run) & set(state.tools_failed)
    assert overlap == set()