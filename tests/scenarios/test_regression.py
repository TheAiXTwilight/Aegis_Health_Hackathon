"""
tests/scenarios/test_regression.py — Data-driven clinical regression suite.

Parametrized against tests/scenarios/clinical_cases.json.

Each case builds AegisState with structured fixtures (LabReportResult,
DrugInteractionResult, XRayResult) injected directly — no reliance on
placeholder text parsing for rule-triggering data. SymptomExtractor
runs on raw_symptoms_text for symptom-driven rule cases
(RULE_CHEST_PAIN_AND_SOB, RULE_PROLONGED_SYMPTOMS) since those rules
read structured symptom output. All other rules are driven by
pre-built structured fixtures.

Pipeline execution:
    One parametrized test function per case. Each case runs the
    pipeline exactly once and asserts all conditions in that single
    execution — 12 Ollama calls total for 12 cases.

JSON validation at load time:
    Every expected_rule value is validated against ALL_RULE_CONSTANTS
    using ValueError (not assert) so validation executes regardless of
    Python optimization flags (-O / -OO). Caught by pytest --collect-only
    before any test runs.

Marked @pytest.mark.ollama — requires live Ollama with aegis-llama loaded.
Run with: pytest tests/scenarios/test_regression.py -v -m ollama
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents.pipeline import AegisPipeline
from schemas.drugs import (
    DrugInteraction,
    DrugInteractionResult,
    DrugInteractionSeverity,
)
from schemas.lab import LabReportResult
from schemas.state import AegisState
from schemas.xray import XRayResult
from tools.severity_scorer import ALL_RULE_CONSTANTS


_CASES_PATH        = Path(__file__).parent / "clinical_cases.json"
_RULE_CONSTANT_SET = frozenset(ALL_RULE_CONSTANTS)


# ── JSON load + validation ─────────────────────────────────────────

def _load_cases() -> list[dict[str, Any]]:
    """
    Load and validate clinical_cases.json at import time.

    Raises ValueError (not AssertionError) if any expected_rule is not
    a known rule constant. ValueError is never silenced by -O or -OO,
    making this a reliable data integrity check caught by
    pytest --collect-only before any test runs.
    """
    cases = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    for case in cases:
        rule = case["expected_rule"]
        if rule not in _RULE_CONSTANT_SET:
            raise ValueError(
                f"clinical_cases.json: unknown rule {rule!r} "
                f"in case {case['name']!r}. "
                f"Known rules: {sorted(_RULE_CONSTANT_SET)}"
            )
    return cases


_CASES = _load_cases()


# ── State builder ──────────────────────────────────────────────────

def _build_state(case: dict[str, Any]) -> AegisState:
    """
    Build AegisState from a clinical case definition.

    Structured fixtures are injected directly for lab values, drug
    interactions, and xray findings — no reliance on placeholder text
    parsing for rule-triggering data. SymptomExtractor runs on
    raw_symptoms_text which is adequate for the simple clinical phrases
    used in symptom-driven regression cases.
    """
    state = AegisState(
        raw_symptoms_text=case["symptoms"],
        medications_raw=case.get("medications", []),
    )

    # Inject structured lab result when measurements or abnormal values present
    if case.get("lab_measurements") or case.get("lab_abnormal"):
        state.lab_result = LabReportResult(
            measurements=case.get("lab_measurements", {}),
            abnormal_values=case.get("lab_abnormal", []),
        )

    # Inject structured xray result when findings present
    if case.get("xray_findings"):
        state.xray_result = XRayResult(
            findings=case["xray_findings"],
        )

    # Inject structured drug result when interactions are defined
    raw_interactions = case.get("drug_interactions", [])
    if raw_interactions:
        interactions = [
            DrugInteraction(
                drugs=i["drugs"],
                severity=DrugInteractionSeverity(i["severity"]),
                description=i["description"],
            )
            for i in raw_interactions
        ]
        all_drugs = list({d for i in interactions for d in i.drugs})
        state.drug_result = DrugInteractionResult(
            resolved=all_drugs,
            unresolved=[],
            interactions=interactions,
            warnings=[f"{len(interactions)} interaction(s) detected."],
            confidence=1.0,
        )

    return state


# ── Single parametrized test — one pipeline call per case ──────────

@pytest.mark.ollama
@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
async def test_regression_case(case: dict[str, Any]):
    """
    Run the full pipeline once per case and assert all conditions.

    One Ollama call per case — 12 total for 12 cases.

    Assertions:
        severity_result present
        severity level          matches expected_level
        highest_priority_rule   matches expected_rule
        expected_rule           in triggered_rules
        report                  is not None (all six sections passed)
        pipeline_complete       is True
    """
    state = _build_state(case)

    pipeline = AegisPipeline()
    async for _ in pipeline.run(state):
        pass

    name = case["name"]

    assert state.severity_result is not None, (
        f"Case '{name}': severity_result is None — "
        f"SeverityScorer may have failed, tools_failed={state.tools_failed}"
    )

    assert state.severity_result.level == case["expected_level"], (
        f"Case '{name}': "
        f"expected level={case['expected_level']!r}, "
        f"got level={state.severity_result.level!r}, "
        f"triggered_rules={state.severity_result.triggered_rules}"
    )

    assert state.severity_result.highest_priority_rule == case["expected_rule"], (
        f"Case '{name}': "
        f"expected rule={case['expected_rule']!r}, "
        f"got rule={state.severity_result.highest_priority_rule!r}, "
        f"triggered_rules={state.severity_result.triggered_rules}"
    )

    assert case["expected_rule"] in state.severity_result.triggered_rules, (
        f"Case '{name}': "
        f"expected {case['expected_rule']!r} in triggered_rules, "
        f"got {state.severity_result.triggered_rules}"
    )

    assert state.report is not None, (
        f"Case '{name}': report is None — "
        f"pipeline may have raised FatalPipelineError, "
        f"tools_failed={state.tools_failed}"
    )

    assert state.pipeline_complete is True, (
        f"Case '{name}': pipeline_complete is False, "
        f"tools_failed={state.tools_failed}"
    )