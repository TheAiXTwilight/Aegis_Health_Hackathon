"""
tests/tools/test_rule_validator.py — RuleValidator.

No mocking required. RuleValidator is synchronous logic in async wrapper.
Tests build state.report directly with fabricated text containing the
### Severity section.

Coverage:
    - state.report=None → ToolError (non-fatal)
    - state.severity_result=None → ToolError (non-fatal)
    - state.severity_result=ToolError → ToolError (non-fatal)
    - Narrative matches deterministic → AGREEMENT
    - Deterministic=HIGH, narrative=LOW → OVERRIDE, overridden=True
    - Deterministic=HIGH, narrative=MEDIUM → OVERRIDE, overridden=True
    - Deterministic=LOW, narrative=HIGH → WARNING (not override)
    - Deterministic=MEDIUM, narrative=LOW → WARNING
    - Deterministic=MEDIUM, narrative=HIGH → WARNING
    - No extractable level → WARNING, slm_narrative_level=None
    - Missing ### Severity section → WARNING
    - deterministic_level always equals state.severity_result.level
    - overridden=True only when status==OVERRIDE
    - disagreement_reason=None on AGREEMENT
    - schema_version == "1.0"
    - _extract_narrative_level: word boundary matching
    - _extract_narrative_level: only searches Severity section
    - _classify: all branches covered
"""

from __future__ import annotations


from schemas.errors import ToolError
from schemas.report import TriageReport
from schemas.severity import SeverityResult
from schemas.state import AegisState
from schemas.validation import ValidationStatus
from tools.rule_validator import RuleValidator, _classify, _extract_narrative_level


# ── Helpers ───────────────────────────────────────────────────────

_DISCLAIMER = "Clinical decision support only."


def _report_text(severity_content: str) -> str:
    """Build a full six-section report with given ### Severity content."""
    sections = {
        "### Summary":         "Patient presents with reported symptoms.",
        "### Findings":        "No abnormal findings.",
        "### Evidence":        "No evidence retrieved.",
        "### Severity":        severity_content,
        "### Recommendations": "Follow up as needed.",
        "### Disclaimer":      _DISCLAIMER,
    }
    return "\n\n".join(
        f"{h}\n{c}" for h, c in sections.items()
    )


def _report(level: str, severity_content: str) -> TriageReport:
    return TriageReport(
        severity   = level,          # type: ignore[arg-type]
        confidence = 0.8,
        text       = _report_text(severity_content),
        citations  = [],
        disclaimer = _DISCLAIMER,
    )


def _sev(level: str) -> SeverityResult:
    return SeverityResult(
        level                 = level,  # type: ignore[arg-type]
        confidence            = 0.9,
        triggered_rules       = ["RULE_DEFAULT_LOW"],
        highest_priority_rule = "RULE_DEFAULT_LOW",
        reasons               = ["Test."],
        contributing_tools    = [],
    )


def _state(deterministic: str, severity_content: str) -> AegisState:
    state = AegisState()
    state.severity_result = _sev(deterministic)
    state.report          = _report(deterministic, severity_content)
    return state


# ── Guard conditions ──────────────────────────────────────────────

async def test_no_report_returns_nonfatal_tool_error():
    state = AegisState()
    state.severity_result = _sev("LOW")
    result = await RuleValidator().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False


async def test_no_severity_result_returns_nonfatal_tool_error():
    state = AegisState()
    state.report = _report("LOW", "Severity: LOW.")
    result = await RuleValidator().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False


async def test_severity_result_is_tool_error_returns_nonfatal():
    state = AegisState()
    state.severity_result = ToolError(tool="SeverityScorer", reason="fail")
    state.report          = _report("LOW", "Severity: LOW.")
    result = await RuleValidator().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False


# ── AGREEMENT ─────────────────────────────────────────────────────

async def test_agreement_low():
    result = await RuleValidator().run(_state("LOW", "Severity: LOW."))
    assert result.status == ValidationStatus.AGREEMENT
    assert result.overridden is False
    assert result.slm_narrative_level == "LOW"
    assert result.disagreement_reason is None


async def test_agreement_medium():
    result = await RuleValidator().run(_state("MEDIUM", "Severity: MEDIUM."))
    assert result.status == ValidationStatus.AGREEMENT


async def test_agreement_high():
    result = await RuleValidator().run(_state("HIGH", "Severity: HIGH."))
    assert result.status == ValidationStatus.AGREEMENT
    assert result.overridden is False


# ── OVERRIDE ──────────────────────────────────────────────────────

async def test_override_high_deterministic_low_narrative():
    result = await RuleValidator().run(_state("HIGH", "Severity: LOW."))
    assert result.status == ValidationStatus.OVERRIDE
    assert result.overridden is True
    assert result.slm_narrative_level == "LOW"
    assert result.deterministic_level == "HIGH"


async def test_override_high_deterministic_medium_narrative():
    result = await RuleValidator().run(_state("HIGH", "Severity: MEDIUM."))
    assert result.status == ValidationStatus.OVERRIDE
    assert result.overridden is True
    assert result.slm_narrative_level == "MEDIUM"


async def test_override_disagreement_reason_populated():
    result = await RuleValidator().run(_state("HIGH", "Severity: LOW."))
    assert result.disagreement_reason is not None
    assert len(result.disagreement_reason) > 0


# ── WARNING ───────────────────────────────────────────────────────

async def test_warning_low_deterministic_high_narrative():
    """Overclaiming HIGH is not a safety hazard — WARNING not OVERRIDE."""
    result = await RuleValidator().run(_state("LOW", "Severity: HIGH."))
    assert result.status == ValidationStatus.WARNING
    assert result.overridden is False


async def test_warning_medium_deterministic_low_narrative():
    result = await RuleValidator().run(_state("MEDIUM", "Severity: LOW."))
    assert result.status == ValidationStatus.WARNING
    assert result.overridden is False


async def test_warning_medium_deterministic_high_narrative():
    result = await RuleValidator().run(_state("MEDIUM", "Severity: HIGH."))
    assert result.status == ValidationStatus.WARNING
    assert result.overridden is False


async def test_warning_when_no_level_in_section():
    result = await RuleValidator().run(
        _state("LOW", "The assessment is complete. No specific level stated.")
    )
    assert result.status == ValidationStatus.WARNING
    assert result.slm_narrative_level is None
    assert result.disagreement_reason is not None


async def test_warning_when_severity_section_missing():
    state = AegisState()
    state.severity_result = _sev("LOW")
    state.report = TriageReport(
        severity   = "LOW",
        confidence = 0.8,
        text       = (
            "### Summary\nOk.\n"
            "### Findings\nNone.\n"
            "### Evidence\nNone.\n"
            "### Recommendations\nFollow up.\n"
            "### Disclaimer\nDisclaimer."
        ),
        citations  = [],
        disclaimer = _DISCLAIMER,
    )
    result = await RuleValidator().run(state)
    assert result.status == ValidationStatus.WARNING


# ── deterministic_level ───────────────────────────────────────────

async def test_deterministic_level_matches_severity_result():
    result = await RuleValidator().run(_state("MEDIUM", "Severity: MEDIUM."))
    assert result.deterministic_level == "MEDIUM"


# ── overridden invariant ──────────────────────────────────────────

async def test_overridden_false_on_agreement():
    result = await RuleValidator().run(_state("LOW", "Severity: LOW."))
    assert result.overridden is False


async def test_overridden_false_on_warning():
    result = await RuleValidator().run(_state("LOW", "Severity: HIGH."))
    assert result.overridden is False


async def test_overridden_true_only_on_override():
    result = await RuleValidator().run(_state("HIGH", "Severity: LOW."))
    assert result.overridden is True
    assert result.status == ValidationStatus.OVERRIDE


# ── schema_version ────────────────────────────────────────────────

async def test_schema_version():
    result = await RuleValidator().run(_state("LOW", "Severity: LOW."))
    assert result.schema_version == "1.0"


# ── _extract_narrative_level — unit tests ─────────────────────────

def test_extract_high():
    assert _extract_narrative_level(_report_text("Severity: HIGH.")) == "HIGH"


def test_extract_medium():
    assert _extract_narrative_level(_report_text("Severity: MEDIUM.")) == "MEDIUM"


def test_extract_low():
    assert _extract_narrative_level(_report_text("Severity: LOW.")) == "LOW"


def test_extract_none_when_no_level():
    assert _extract_narrative_level(_report_text("Assessment complete.")) is None


def test_extract_none_when_no_section():
    assert _extract_narrative_level("### Summary\nOk.") is None


def test_word_boundary_high_not_matched_in_highest():
    """'highest' must not match HIGH."""
    text = _report_text("The highest priority finding is noted.")
    assert _extract_narrative_level(text) != "HIGH"


def test_word_boundary_case_sensitive():
    """Extraction is uppercase only — 'low' does not match LOW."""
    text = _report_text("The risk is low for this patient.")
    assert _extract_narrative_level(text) is None


def test_only_searches_severity_section():
    """HIGH in ### Summary must not be extracted — only ### Severity counts."""
    full = (
        "### Summary\n"
        "This patient has HIGH risk factors.\n\n"
        "### Findings\nNone.\n\n"
        "### Evidence\nNone.\n\n"
        "### Severity\n"
        "Severity: LOW. No acute rules triggered.\n\n"
        "### Recommendations\nFollow up.\n\n"
        "### Disclaimer\nDisclaimer."
    )
    assert _extract_narrative_level(full) == "LOW"


# ── _classify — unit tests ────────────────────────────────────────

def test_classify_agreement():
    assert _classify("LOW", "LOW").status == ValidationStatus.AGREEMENT


def test_classify_agreement_high():
    assert _classify("HIGH", "HIGH").status == ValidationStatus.AGREEMENT


def test_classify_override_high_low():
    r = _classify("HIGH", "LOW")
    assert r.status == ValidationStatus.OVERRIDE
    assert r.overridden is True


def test_classify_override_high_medium():
    r = _classify("HIGH", "MEDIUM")
    assert r.status == ValidationStatus.OVERRIDE
    assert r.overridden is True


def test_classify_warning_none():
    r = _classify("LOW", None)
    assert r.status == ValidationStatus.WARNING
    assert r.slm_narrative_level is None


def test_classify_warning_low_high():
    r = _classify("LOW", "HIGH")
    assert r.status == ValidationStatus.WARNING
    assert r.overridden is False


def test_classify_warning_medium_low():
    assert _classify("MEDIUM", "LOW").status == ValidationStatus.WARNING


def test_classify_disagreement_reason_none_on_agreement():
    assert _classify("HIGH", "HIGH").disagreement_reason is None