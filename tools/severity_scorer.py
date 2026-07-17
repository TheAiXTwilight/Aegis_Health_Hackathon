"""
tools/severity_scorer.py — Deterministic priority-based rule engine.

Architecture:
    - Declarative _RULES table: 13 clinical rules, priority-sorted.
    - Named _check_* functions: no lambdas, individually testable.
    - RuleContext: minimal (state + any_high_fired).
    - RULE_DEFAULT_LOW: evaluator fallback, not in _RULES.
    - ALL_RULE_CONSTANTS: auto-derived from _RULES + RULE_DEFAULT_LOW.
    - SeverityResult.confidence: from highest-priority fired rule.
    - contributing_tools: ordered deduplication, final triggered only.

Evaluation order:
    Rules evaluated in descending priority order.
    ctx.any_high_fired updated immediately when a HIGH rule fires.
    RULE_PROLONGED_SYMPTOMS and RULE_MODERATE_DRUG_INTERACTION
    check ctx.any_high_fired at evaluation time — they never fire
    when any HIGH rule has already fired.

Async interface:
    SeverityScorer.score is async because it is passed to
    AegisPipeline._run_step() which unconditionally awaits tool_fn(state).
    The scoring logic itself is synchronous — async is a pipeline
    contract requirement, not an intrinsic scorer requirement.

Resilience:
    If a check_fn raises unexpectedly, that rule is skipped and
    evaluation continues with remaining rules. This is intentional —
    a bug in one rule suppresses only that rule rather than failing
    scoring entirely. The exception is logged for diagnosis.

Logging note:
    loguru reserves the kwarg `level` for setting log severity.
    Structured fields must not use `level` as a keyword name. We use
    `severity_level` instead when emitting the result's level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal

from loguru import logger

from schemas.drugs import DrugInteractionSeverity
from schemas.errors import ToolError
from schemas.severity import SeverityResult
from schemas.state import AegisState
from tools.lab_constants import (
    LAB_KEY_HAEMOGLOBIN,
    LAB_KEY_POTASSIUM,
    LAB_KEY_TROPONIN,
)
from tools.lab_thresholds import (
    CRITICAL_HAEMOGLOBIN_G_DL,
    CRITICAL_POTASSIUM_MMOL_L,
    CRITICAL_TROPONIN_NG_ML,
)
from tools.tool_names import (
    TOOL_DRUG_INTERACTION_CHECKER,
    TOOL_LAB_REPORT_PARSER,
    TOOL_SEVERITY_SCORER,
    TOOL_SYMPTOM_EXTRACTOR,
    TOOL_XRAY_PROCESSOR,
)

# FIX #10: text finding analyzer is a contributing tool for
# synthesized text-finding rules. Import defensively — if the tool
# constant is not defined yet in tool_names.py, fall back to a stable
# literal so this module still imports cleanly.
try:
    from tools.tool_names import TOOL_TEXT_FINDING_ANALYZER
except ImportError:  # pragma: no cover
    TOOL_TEXT_FINDING_ANALYZER = "text_finding_analyzer"


# FIX #8: severity calibrator applies multi-signal confidence and
# escalation adjustments AFTER the base rule engine has determined
# the initial (level, confidence). Import defensively — if the
# module is missing, calibration is skipped and severity behaves
# exactly as before this fix.
try:
    from tools.severity_calibrator import calibrate_severity
    _CALIBRATOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CALIBRATOR_AVAILABLE = False
    calibrate_severity = None  # type: ignore

# ── Rule constants ─────────────────────────────────────────────────

RULE_CHEST_PAIN_AND_SOB        = "RULE_CHEST_PAIN_AND_SOB"
RULE_CRITICAL_LAB_TROPONIN     = "RULE_CRITICAL_LAB_TROPONIN"
RULE_CRITICAL_LAB_HAEMOGLOBIN  = "RULE_CRITICAL_LAB_HAEMOGLOBIN"
RULE_CRITICAL_LAB_POTASSIUM    = "RULE_CRITICAL_LAB_POTASSIUM"
RULE_XRAY_PNEUMOTHORAX         = "RULE_XRAY_PNEUMOTHORAX"
RULE_XRAY_PULMONARY_EDEMA      = "RULE_XRAY_PULMONARY_EDEMA"
RULE_SEVERE_DRUG_INTERACTION   = "RULE_SEVERE_DRUG_INTERACTION"
RULE_ABNORMAL_LAB_ANY          = "RULE_ABNORMAL_LAB_ANY"
RULE_XRAY_CARDIOMEGALY         = "RULE_XRAY_CARDIOMEGALY"
RULE_XRAY_PLEURAL_EFFUSION     = "RULE_XRAY_PLEURAL_EFFUSION"
RULE_XRAY_CONSOLIDATION        = "RULE_XRAY_CONSOLIDATION"
RULE_PROLONGED_SYMPTOMS        = "RULE_PROLONGED_SYMPTOMS"
RULE_MODERATE_DRUG_INTERACTION = "RULE_MODERATE_DRUG_INTERACTION"
RULE_DEFAULT_LOW               = "RULE_DEFAULT_LOW"

# ── FIX #10: Text finding rule prefix ──────────────────────────────
# Rules synthesized from tools.text_finding_analyzer._TEXT_PATTERNS
# use this prefix so downstream code (e.g. reason explanations) can
# recognise them universally without hardcoding individual pattern IDs.
RULE_TEXT_FINDING_PREFIX = "RULE_TEXT_FINDING_"


# ── Internal structures ────────────────────────────────────────────

@dataclass
class RuleContext:
    """
    Minimal evaluation context passed to every check function.

    state           — full pipeline state (read-only by convention)
    any_high_fired  — True once any HIGH-level rule fires during
                      the current evaluation pass. Updated immediately
                      by the evaluator after each HIGH rule fires.
                      Read by RULE_PROLONGED_SYMPTOMS and
                      RULE_MODERATE_DRUG_INTERACTION only.
    """
    state: AegisState
    any_high_fired: bool = False


@dataclass
class Rule:
    """One entry in the declarative rule table."""
    constant:          str
    priority:          int
    level:             Literal["LOW", "MEDIUM", "HIGH"]
    check_fn:          Callable[[RuleContext], bool]
    reason:            str
    contributing_tool: str | None
    rule_confidence:   float


# ── Symptom search helpers ─────────────────────────────────────────

_CHEST_PAIN_TERMS: set[str] = {
    "chest pain",
    "chest tightness",
    "chest pressure",
}

_SOB_TERMS: set[str] = {
    "shortness of breath",
    "sob",
    "dyspnoea",
    "dyspnea",
    "breathlessness",
    "difficulty breathing",
}


def _symptoms_contain(symptoms: list[str], terms: set[str]) -> bool:
    """Return True if any extracted symptom matches any term."""
    lowered = [s.lower() for s in symptoms]
    return any(
        any(term in symptom for term in terms)
        for symptom in lowered
    )


def _text_contains_all(text: str, *term_sets: set[str]) -> bool:
    """Return True if text contains at least one term from every set."""
    lowered = text.lower()
    return all(
        any(term in lowered for term in terms)
        for terms in term_sets
    )


# ── Named check functions ──────────────────────────────────────────
# Each function receives RuleContext and returns bool.
# Guard pattern: check for None and ToolError before accessing fields.
# Structured output preferred over raw text where available.

def _check_chest_pain_and_sob(ctx: RuleContext) -> bool:
    """
    HIGH — Chest pain with shortness of breath.

    Prefers structured symptom_result when available.
    Falls back to raw_symptoms_text if extraction failed or unavailable.
    """
    sym = ctx.state.symptom_result

    if sym and not isinstance(sym, ToolError) and sym.symptoms:
        has_chest_pain = _symptoms_contain(sym.symptoms, _CHEST_PAIN_TERMS)
        has_sob        = _symptoms_contain(sym.symptoms, _SOB_TERMS)
        return has_chest_pain and has_sob

    text = ctx.state.raw_symptoms_text or ""
    if not text:
        return False

    return _text_contains_all(text, _CHEST_PAIN_TERMS, _SOB_TERMS)


def _check_critical_troponin(ctx: RuleContext) -> bool:
    """HIGH — Troponin above critical threshold."""
    lab = ctx.state.lab_result
    if not lab or isinstance(lab, ToolError):
        return False
    value = lab.measurements.get(LAB_KEY_TROPONIN)
    return value is not None and value > CRITICAL_TROPONIN_NG_ML


def _check_critical_haemoglobin(ctx: RuleContext) -> bool:
    """HIGH — Haemoglobin below critical threshold."""
    lab = ctx.state.lab_result
    if not lab or isinstance(lab, ToolError):
        return False
    value = lab.measurements.get(LAB_KEY_HAEMOGLOBIN)
    return value is not None and value < CRITICAL_HAEMOGLOBIN_G_DL


def _check_critical_potassium(ctx: RuleContext) -> bool:
    """HIGH — Potassium above critical threshold."""
    lab = ctx.state.lab_result
    if not lab or isinstance(lab, ToolError):
        return False
    value = lab.measurements.get(LAB_KEY_POTASSIUM)
    return value is not None and value > CRITICAL_POTASSIUM_MMOL_L


def _check_xray_finding(ctx: RuleContext, term: str) -> bool:
    """
    Shared helper for X-ray finding checks.
    Not referenced directly by _RULES — called only by named wrappers.
    """
    xray = ctx.state.xray_result
    if not xray or isinstance(xray, ToolError):
        return False
    return any(term.lower() in f.lower() for f in xray.findings)


def _check_xray_pneumothorax(ctx: RuleContext) -> bool:
    """HIGH — Pneumothorax present in X-ray findings."""
    return _check_xray_finding(ctx, "pneumothorax")


def _check_xray_pulmonary_oedema(ctx: RuleContext) -> bool:
    """HIGH — Pulmonary oedema present in X-ray findings."""
    return _check_xray_finding(ctx, "pulmonary edema")


def _check_xray_cardiomegaly(ctx: RuleContext) -> bool:
    """MEDIUM — Cardiomegaly present in X-ray findings."""
    return _check_xray_finding(ctx, "cardiomegaly")


def _check_xray_pleural_effusion(ctx: RuleContext) -> bool:
    """MEDIUM — Pleural effusion present in X-ray findings."""
    return _check_xray_finding(ctx, "pleural effusion")


def _check_xray_consolidation(ctx: RuleContext) -> bool:
    """MEDIUM — Consolidation present in X-ray findings."""
    return _check_xray_finding(ctx, "consolidation")


def _check_severe_drug(ctx: RuleContext) -> bool:
    """HIGH — At least one severe drug interaction present."""
    drug = ctx.state.drug_result
    if not drug or isinstance(drug, ToolError):
        return False
    return any(
        i.severity == DrugInteractionSeverity.SEVERE
        for i in drug.interactions
    )


def _check_abnormal_lab_any(ctx: RuleContext) -> bool:
    """MEDIUM — Any abnormal lab value flagged by LabReportParser."""
    lab = ctx.state.lab_result
    if not lab or isinstance(lab, ToolError):
        return False
    return bool(lab.abnormal_values)


def _check_prolonged_symptoms(ctx: RuleContext) -> bool:
    """
    MEDIUM — Prolonged symptom duration detected.
    Does not fire if any HIGH rule has already fired.
    """
    if ctx.any_high_fired:
        return False
    sym = ctx.state.symptom_result
    if not sym or isinstance(sym, ToolError):
        return False
    duration = (sym.duration or "").lower()
    return "week" in duration or "month" in duration


def _check_moderate_drug(ctx: RuleContext) -> bool:
    """
    MEDIUM — At least one moderate drug interaction present.
    Does not fire if any HIGH rule has already fired.
    """
    if ctx.any_high_fired:
        return False
    drug = ctx.state.drug_result
    if not drug or isinstance(drug, ToolError):
        return False
    return any(
        i.severity == DrugInteractionSeverity.MODERATE
        for i in drug.interactions
    )

# ── FIX #10: Text finding rule support ─────────────────────────────
# Generic check function: reads state.text_finding_matched_patterns
# (populated by tools.report_generator via tools.text_finding_analyzer)
# and returns True if this rule's pattern_id was matched.
#
# Universal design: ONE check function serves ALL text-finding rules
# via closure over pattern_id. No per-pattern conditionals anywhere
# in this file — the pattern registry in text_finding_analyzer is the
# single source of truth.

def _make_text_finding_check(pattern_id: str) -> Callable[[RuleContext], bool]:
    """Return a check function bound to a specific text-finding pattern id."""
    def _check(ctx: RuleContext) -> bool:
        matched = getattr(ctx.state, "text_finding_matched_patterns", None)
        if not matched:
            return False
        try:
            return pattern_id in matched
        except Exception:
            return False
    _check.__name__ = f"_check_text_finding_{pattern_id}"
    return _check


# Confidence + priority + severity level per severity_class.
# These are the ONLY biomarker/pattern-agnostic knobs. Adjusting a
# value here changes behavior for ALL text-finding patterns of that
# severity class — single point of tuning.
_TEXT_FINDING_RULE_PROFILE: dict[str, dict] = {
    "critical":   {"level": "HIGH",   "priority": 175, "confidence": 0.94},
    "urgent":     {"level": "HIGH",   "priority": 145, "confidence": 0.90},
    "concerning": {"level": "MEDIUM", "priority": 85,  "confidence": 0.82},
}


def _synthesize_text_finding_rules() -> List["Rule"]:
    """
    Build Rule objects from the text_finding_analyzer pattern registry.

    Only patterns whose severity_class is 'concerning', 'urgent', or
    'critical' produce rules. Reassuring/informational patterns are
    intentionally excluded — they do not warrant escalation.

    Import is deferred to avoid a circular import between
    severity_scorer and text_finding_analyzer.

    Fail-safe: if the analyzer module is unavailable for any reason
    (missing file, import error), returns an empty list so the base
    13 rules continue to function exactly as before.
    """
    try:
        from tools.text_finding_analyzer import get_registered_patterns
        patterns = get_registered_patterns()
    except Exception:
        logger.exception(
            "severity_scorer · text_finding_analyzer unavailable · "
            "text-finding rules disabled for this run"
        )
        return []

    synthesized: List[Rule] = []
    for p in patterns:
        cls = str(p.get("severity_class") or "").lower()
        profile = _TEXT_FINDING_RULE_PROFILE.get(cls)
        if not profile:
            continue

        pid = str(p.get("id") or "").strip()
        if not pid:
            continue

        # Human-readable reason: prefer the pattern's observation text,
        # falling back to a generic description if the pattern is silent
        # (e.g., reassuring patterns — which we already excluded above).
        reason = (
            str(p.get("observation") or "").strip()
            or f"Text finding pattern '{pid}' matched on peripheral smear or clinical impression."
        )

        synthesized.append(
            Rule(
                constant          = f"{RULE_TEXT_FINDING_PREFIX}{pid.upper()}",
                priority          = int(profile["priority"]),
                level             = profile["level"],
                check_fn          = _make_text_finding_check(pid),
                reason            = reason,
                contributing_tool = TOOL_TEXT_FINDING_ANALYZER,
                rule_confidence   = float(profile["confidence"]),
            )
        )

    if synthesized:
        logger.info(
            "severity_scorer · synthesized text-finding rules",
            count=len(synthesized),
        )

    return synthesized


# ── Rule table ─────────────────────────────────────────────────────
# Sorted once at module load by priority descending.
# Evaluation proceeds top to bottom — order is deterministic.
# sorted() is kept as a safety net against accidental reordering.

_RULES: List[Rule] = sorted(
    [
        Rule(
            constant          = RULE_CHEST_PAIN_AND_SOB,
            priority          = 190,
            level             = "HIGH",
            check_fn          = _check_chest_pain_and_sob,
            reason            = "Chest pain with shortness of breath detected.",
            contributing_tool = TOOL_SYMPTOM_EXTRACTOR,
            rule_confidence   = 0.97,
        ),
        Rule(
            constant          = RULE_CRITICAL_LAB_TROPONIN,
            priority          = 180,
            level             = "HIGH",
            check_fn          = _check_critical_troponin,
            reason            = "Critical troponin level detected.",
            contributing_tool = TOOL_LAB_REPORT_PARSER,
            rule_confidence   = 0.99,
        ),
        Rule(
            constant          = RULE_CRITICAL_LAB_HAEMOGLOBIN,
            priority          = 170,
            level             = "HIGH",
            check_fn          = _check_critical_haemoglobin,
            reason            = "Critical haemoglobin level detected.",
            contributing_tool = TOOL_LAB_REPORT_PARSER,
            rule_confidence   = 0.98,
        ),
        Rule(
            constant          = RULE_CRITICAL_LAB_POTASSIUM,
            priority          = 160,
            level             = "HIGH",
            check_fn          = _check_critical_potassium,
            reason            = "Critical potassium level detected.",
            contributing_tool = TOOL_LAB_REPORT_PARSER,
            rule_confidence   = 0.98,
        ),
        Rule(
            constant          = RULE_XRAY_PNEUMOTHORAX,
            priority          = 150,
            level             = "HIGH",
            check_fn          = _check_xray_pneumothorax,
            reason            = "Pneumothorax detected on X-ray.",
            contributing_tool = TOOL_XRAY_PROCESSOR,
            rule_confidence   = 0.97,
        ),
        Rule(
            constant          = RULE_XRAY_PULMONARY_EDEMA,
            priority          = 140,
            level             = "HIGH",
            check_fn          = _check_xray_pulmonary_oedema,
            reason            = "Pulmonary oedema detected on X-ray.",
            contributing_tool = TOOL_XRAY_PROCESSOR,
            rule_confidence   = 0.95,
        ),
        Rule(
            constant          = RULE_SEVERE_DRUG_INTERACTION,
            priority          = 130,
            level             = "HIGH",
            check_fn          = _check_severe_drug,
            reason            = "Severe drug interaction detected.",
            contributing_tool = TOOL_DRUG_INTERACTION_CHECKER,
            rule_confidence   = 0.95,
        ),
        Rule(
            constant          = RULE_ABNORMAL_LAB_ANY,
            priority          = 90,
            level             = "MEDIUM",
            check_fn          = _check_abnormal_lab_any,
            reason            = "Abnormal laboratory values detected.",
            contributing_tool = TOOL_LAB_REPORT_PARSER,
            rule_confidence   = 0.86,
        ),
        Rule(
            constant          = RULE_XRAY_CARDIOMEGALY,
            priority          = 80,
            level             = "MEDIUM",
            check_fn          = _check_xray_cardiomegaly,
            reason            = "Cardiomegaly detected on X-ray.",
            contributing_tool = TOOL_XRAY_PROCESSOR,
            rule_confidence   = 0.84,
        ),
        Rule(
            constant          = RULE_XRAY_PLEURAL_EFFUSION,
            priority          = 75,
            level             = "MEDIUM",
            check_fn          = _check_xray_pleural_effusion,
            reason            = "Pleural effusion detected on X-ray.",
            contributing_tool = TOOL_XRAY_PROCESSOR,
            rule_confidence   = 0.82,
        ),
        Rule(
            constant          = RULE_XRAY_CONSOLIDATION,
            priority          = 70,
            level             = "MEDIUM",
            check_fn          = _check_xray_consolidation,
            reason            = "Consolidation detected on X-ray.",
            contributing_tool = TOOL_XRAY_PROCESSOR,
            rule_confidence   = 0.80,
        ),
        Rule(
            constant          = RULE_PROLONGED_SYMPTOMS,
            priority          = 60,
            level             = "MEDIUM",
            check_fn          = _check_prolonged_symptoms,
            reason            = "Prolonged symptom duration detected.",
            contributing_tool = TOOL_SYMPTOM_EXTRACTOR,
            rule_confidence   = 0.75,
        ),
        Rule(
            constant          = RULE_MODERATE_DRUG_INTERACTION,
            priority          = 50,
            level             = "MEDIUM",
            check_fn          = _check_moderate_drug,
            reason            = "Moderate drug interaction detected.",
            contributing_tool = TOOL_DRUG_INTERACTION_CHECKER,
            rule_confidence   = 0.78,
        ),
        # ── FIX #10: Text-finding rules synthesized from the pattern
        # registry in tools.text_finding_analyzer. Unpacked here so
        # they participate in the same priority-sorted evaluation
        # pipeline as the base rules. Any pattern added to that
        # registry with severity_class ∈ {concerning, urgent, critical}
        # is automatically included — no code change here required.
        *_synthesize_text_finding_rules(),
    ],
    key=lambda r: r.priority,
    reverse=True,
)


# ── Auto-derived constants list ────────────────────────────────────

ALL_RULE_CONSTANTS: List[str] = (
    [r.constant for r in _RULES] + [RULE_DEFAULT_LOW]
)


# ── Evaluator ──────────────────────────────────────────────────────

class SeverityScorer:
    """
    Stateless evaluator. One instance per call, no shared mutable state.

    score() is async because it is passed to AegisPipeline._run_step(),
    which unconditionally awaits tool_fn(state). The scoring logic
    itself is synchronous — async is a pipeline contract requirement.
    """

    TOOL_NAME = TOOL_SEVERITY_SCORER

    async def score(self, state: AegisState) -> SeverityResult:
        ctx   = RuleContext(state=state)
        fired: List[Rule] = []

        for rule in _RULES:
            try:
                if rule.check_fn(ctx):
                    fired.append(rule)
                    if rule.level == "HIGH":
                        ctx.any_high_fired = True
            except Exception:
                # Intentional: a bug in one check_fn skips that rule
                # only, evaluation continues. Logged for diagnosis.
                logger.exception(
                    "severity_scorer · check_fn raised unexpectedly · "
                    "rule skipped",
                    rule=rule.constant,
                    session_id=getattr(state, "session_id", None),
                )

        if not fired:
            logger.debug(
                "severity_scorer · no rules fired · DEFAULT_LOW",
                session_id=getattr(state, "session_id", None),
            )
            return SeverityResult(
                level                 = "LOW",
                confidence            = 0.80,
                triggered_rules       = [RULE_DEFAULT_LOW],
                highest_priority_rule = RULE_DEFAULT_LOW,
                reasons               = ["No high-risk rules triggered."],
                contributing_tools    = [],
            )

        highest = fired[0]

        contributing_tools: List[str] = []
        for r in fired:
            if (
                r.contributing_tool
                and r.contributing_tool not in contributing_tools
            ):
                contributing_tools.append(r.contributing_tool)

                # ── Baseline values from the base rule engine ─────────────
        base_level = highest.level
        base_confidence = highest.rule_confidence
        base_reasons = [r.reason for r in fired]

        # ── FIX #8: multi-signal severity calibration ─────────────
        # Apply confidence adjustments and (optionally) escalate the
        # severity level based on symptom-lab correlation, corroborating
        # rule clusters, text-finding matches, and diagnostic ambiguity.
        # Fail-safe: if the calibrator is unavailable or errors, baseline
        # values are used and severity behaves exactly as before FIX #8.
        final_level = base_level
        final_confidence = base_confidence
        final_reasons = list(base_reasons)

        if _CALIBRATOR_AVAILABLE and calibrate_severity is not None:
            try:
                calibrated = calibrate_severity(
                    state=state,
                    fired_rules=fired,
                    current_level=base_level,
                    current_confidence=base_confidence,
                    current_reasons=base_reasons,
                )
                final_level = calibrated.get("level", base_level)
                final_confidence = calibrated.get("confidence", base_confidence)
                final_reasons = calibrated.get("reasons", base_reasons)
            except Exception:
                logger.exception(
                    "severity_scorer · calibration raised; using baseline values",
                    session_id=getattr(state, "session_id", None),
                )

        result = SeverityResult(
            level                 = final_level,
            confidence            = final_confidence,
            triggered_rules       = [r.constant for r in fired],
            highest_priority_rule = highest.constant,
            reasons               = final_reasons,
            contributing_tools    = contributing_tools,
        )

        # NOTE: `level` is reserved by loguru — use `severity_level`.
        logger.info(
            "severity_scorer · complete",
            severity_level=result.level,
            highest_priority_rule=result.highest_priority_rule,
            triggered_count=len(fired),
            base_level=base_level,
            base_confidence=round(float(base_confidence), 3),
            final_confidence=round(float(final_confidence), 3),
            session_id=getattr(state, "session_id", None),
        )

        return result


async def score(state: AegisState) -> SeverityResult:
    """Canonical functional entrypoint."""
    return await SeverityScorer().score(state)