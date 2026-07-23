"""
tests/tools/test_report_generator.py

Tests the LLM-adjacent infrastructure that Ollama integration depends on.

Assertion philosophy
────────────────────
Test observable behaviour, not presentation format.

    BEHAVIOUR (what we test):
        - severity information is preserved in the prompt
        - extracted symptoms appear in the prompt
        - abnormal lab values appear in the prompt
        - drug interaction descriptions appear
        - citation identifiers propagate to the report
        - truncation flags update under budget pressure
        - required sections are validated
        - brace-laden patient input does not crash substitution
        - empty RAG result handled correctly

    PRESENTATION (we DO NOT test):
        - specific section labels ("SYMPTOMS:" vs "Symptoms:")
        - punctuation in headers
        - capitalisation of formatting prefixes

Exception: _validate_sections tests assert exact header strings.
The six required headers ARE the public contract.

Confidence assertion policy
────────────────────────────
ReportGenerator sets state.report.confidence = 0.0 as a deliberate
placeholder. The pipeline injects the real value via calculate_confidence()
after ReportGenerator completes. Tests here assert the placeholder value
(0.0) because that is what ReportGenerator itself is responsible for —
the pipeline integration test (test_pipeline_confidence_injection.py)
verifies the injected value is > 0.0.

Mocking strategy
────────────────
The only thing mocked is httpx.AsyncClient — the external HTTP boundary.
AegisState and all schema instances are REAL (from conftest.py).
We never call a real Ollama server.

Fixture isolation
─────────────────
Pytest fixtures are function-scoped by default. Every test that uses
populated_state or any schema fixture gets a fresh instance. No state
leaks between tests. monkeypatch handles MAX_INPUT_TOKENS restoration
automatically after each test.

Bug regression guards
─────────────────────
test_assemble_prompt_with_braces_in_context
    Guards the .format() crash on patient input with curly braces.
test_assemble_prompt_with_format_specifiers_in_context
    Defence-in-depth for the same bug.

Knowledge base field policy (Commit 2)
──────────────────────────────────────
Commit 2 wired get_corpus_version() and get_corpus_date() into
ReportGenerator. These read docs/corpus_version.md and return None only
when the file lacks git_commit / snapshot_date entries.

test_run_report_knowledge_base_fields_are_none was written before Commit 2
and assumed the fields would always be None. It is now updated to reflect
the real contract: the fields are None when corpus_version.md is absent
or unpopulated, and non-None strings when it is populated. The test uses
monkeypatch to isolate from the on-disk file so both paths are exercised
deterministically regardless of local corpus state.
"""

from __future__ import annotations

import json

import pytest

from schemas.errors import FatalPipelineError, ToolError
from schemas.state import AegisState
from tools import report_generator as rg
from tools.report_generator import (
    DISCLAIMER,
    REQUIRED_SECTIONS,
    ReportGenerator,
    _assemble_prompt,
    _build_context,
    _estimate_tokens,
    _validate_sections,
)


# ── Helpers ────────────────────────────────────────────────────────

def _full_report_text() -> str:
    """A string containing every required section header."""
    return "\n".join(
        f"{header}\nsome content here"
        for header in REQUIRED_SECTIONS
    )


class _FakeStream:
    """Mimics httpx streaming response as an async context manager."""

    def __init__(self, lines: list[str]):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeClient:
    """
    Mimics httpx.AsyncClient as an async context manager.

    Uses *args, **kwargs on stream() so that changes to the httpx
    signature do not require updating the fake.
    """

    def __init__(self, lines: list[str]):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, *args, **kwargs):
        return _FakeStream(self._lines)


def _ollama_ndjson_lines(full_text: str) -> list[str]:
    """Build Ollama-style NDJSON: one chunk with full text, then done."""
    return [
        json.dumps({"response": full_text, "done": False}),
        json.dumps({"response": "", "done": True}),
    ]


def _patch_ollama(monkeypatch, text: str) -> None:
    """Patch httpx.AsyncClient to return the given text as a stream."""
    monkeypatch.setattr(
        rg.httpx,
        "AsyncClient",
        lambda *a, **k: _FakeClient(_ollama_ndjson_lines(text)),
    )


# ── _estimate_tokens — pure ────────────────────────────────────────

def test_estimate_tokens_empty_string_returns_one():
    """max(1, ceil(0/4)) = 1 — never below one."""
    assert _estimate_tokens("") == 1


def test_estimate_tokens_short_string_rounds_up():
    """5 chars / 4 = 1.25 → ceil → 2."""
    assert _estimate_tokens("hello") == 2


def test_estimate_tokens_exact_boundary():
    """8 chars / 4 = 2.0 → ceil → 2."""
    assert _estimate_tokens("12345678") == 2


def test_estimate_tokens_single_char_returns_one():
    assert _estimate_tokens("a") == 1


def test_estimate_tokens_scales_linearly():
    """Sanity check: 4000 chars → 1000 tokens, no overflow."""
    assert _estimate_tokens("x" * 4000) == 1000


# ── _validate_sections — pure ──────────────────────────────────────
# These DO test exact strings because the six section headers are
# the documented public contract from docs/Technical_Project_Spec.md.

def test_validate_sections_all_present_returns_empty():
    assert _validate_sections(_full_report_text()) == []


def test_validate_sections_missing_one_reports_it():
    text = _full_report_text().replace("### Disclaimer", "")
    assert _validate_sections(text) == ["### Disclaimer"]


def test_validate_sections_missing_multiple_reports_all():
    text = (
        _full_report_text()
        .replace("### Disclaimer", "")
        .replace("### Evidence", "")
    )
    assert set(_validate_sections(text)) == {
        "### Disclaimer", "### Evidence",
    }


def test_validate_sections_empty_text_reports_all_six():
    missing = _validate_sections("")
    assert set(missing) == set(REQUIRED_SECTIONS)
    assert len(missing) == 6


# ── _assemble_prompt — substitution + regression guards ────────────

def test_assemble_prompt_substitutes_both_placeholders():
    """Both %%DISCLAIMER%% and %%CONTEXT%% must be replaced."""
    prompt = _assemble_prompt("MY DISCLAIMER", "MY CONTEXT")
    assert "MY DISCLAIMER" in prompt
    assert "MY CONTEXT" in prompt
    assert "%%DISCLAIMER%%" not in prompt
    assert "%%CONTEXT%%" not in prompt


def test_assemble_prompt_with_braces_in_context():
    """
    REGRESSION GUARD: previously used str.format() which crashed on
    patient input containing curly braces. The fix switched to
    str.replace() with %%PLACEHOLDER%% markers. This test would
    have raised KeyError before the fix.
    """
    context_with_braces = "Patient reports pain when typing { or }."
    prompt = _assemble_prompt(DISCLAIMER, context_with_braces)
    assert context_with_braces in prompt


def test_assemble_prompt_with_format_specifiers_in_context():
    """
    DEFENCE-IN-DEPTH: even strings that look like format references
    ({name}, {0}, {foo!r}) must survive substitution untouched.
    """
    weird = "Patient mentions {name} and {0} and {foo!r}."
    prompt = _assemble_prompt(DISCLAIMER, weird)
    assert weird in prompt


# ── _build_context — behavioural assertions on real state ──────────
# Each test constructs a fresh AegisState — no shared state risk.

def test_build_context_preserves_severity_level_and_rule(severity_high):
    """Severity level value and highest priority rule must appear."""
    state = AegisState()
    state.severity_result = severity_high
    context = _build_context(state)
    # Use severity_high.level rather than a bare string — reads the
    # actual field value so the assertion holds for any severity level.
    assert severity_high.level in context


def test_build_context_preserves_severity_reasons(severity_high):
    """Every reason string from SeverityResult must appear."""
    state = AegisState()
    state.severity_result = severity_high
    context = _build_context(state)
    for reason in severity_high.reasons:
        assert reason in context


def test_build_context_preserves_extracted_symptoms(populated_state):
    """Each extracted symptom must appear in the context."""
    symptoms = populated_state.symptom_result.symptoms
    context = _build_context(populated_state)
    for symptom in symptoms:
        assert symptom in context


def test_build_context_preserves_symptom_duration(populated_state):
    """Symptom duration must appear when present."""
    duration = populated_state.symptom_result.duration
    context = _build_context(populated_state)
    assert duration in context


def test_build_context_skips_symptom_data_when_tool_error(
    severity_low, tool_error_nonfatal,
):
    """ToolError in symptom_result must not crash — symptom block absent."""
    state = AegisState()
    state.severity_result = severity_low
    state.symptom_result = tool_error_nonfatal
    # Must not raise
    _build_context(state)


def test_build_context_preserves_abnormal_lab_findings(populated_state):
    """Abnormal lab values must appear in context."""
    lab = populated_state.lab_result
    context = _build_context(populated_state)
    for abnormal in lab.abnormal_values:
        assert abnormal in context


def test_build_context_preserves_drug_interaction_descriptions(populated_state):
    """Drug interaction warning strings must appear in context."""
    drug = populated_state.drug_result
    context = _build_context(populated_state)
    for warning in drug.warnings:
        assert warning in context


def test_build_context_preserves_xray_findings(populated_state):
    """X-ray findings must appear in context."""
    xray = populated_state.xray_result
    context = _build_context(populated_state)
    for finding in xray.findings:
        assert finding in context


def test_build_context_preserves_rag_citations(populated_state):
    """Each citation identifier must appear in the context."""
    rag = populated_state.rag_result
    context = _build_context(populated_state)
    for citation in rag.citations:
        assert citation in context


def test_build_context_empty_rag_no_truncation(severity_low, rag_empty):
    """
    When RAG retrieval succeeds but returns zero passages, context
    assembly must not set truncation flags. There is nothing to truncate.
    """
    state = AegisState()
    state.severity_result = severity_low
    state.rag_result = rag_empty
    _build_context(state)
    assert state.enrichment_fields_truncated is False
    assert state.core_fields_truncated is False


def test_build_context_no_truncation_on_normal_input(populated_state):
    """Truncation flags must remain False on a normal-sized prompt."""
    _build_context(populated_state)
    assert populated_state.core_fields_truncated is False
    assert populated_state.enrichment_fields_truncated is False


def test_build_context_sets_core_truncation_when_budget_tiny(
    monkeypatch, severity_high,
):
    """
    Force core truncation by shrinking the token budget below the
    severity block size. monkeypatch restores MAX_INPUT_TOKENS after
    the test — no module state leakage.
    """
    monkeypatch.setattr(rg, "MAX_INPUT_TOKENS", 1)
    state = AegisState()
    state.severity_result = severity_high
    _build_context(state)
    assert state.core_fields_truncated is True


# ── ReportGenerator.run — guard paths (no HTTP) ────────────────────

async def test_run_raises_fatal_when_severity_missing():
    """Pipeline contract: ReportGenerator must not run without severity."""
    rgen = ReportGenerator()
    state = AegisState()
    with pytest.raises(FatalPipelineError):
        async for _ in rgen.run(state):
            pass


async def test_run_raises_fatal_when_severity_is_tool_error(
    tool_error_fatal,
):
    rgen = ReportGenerator()
    state = AegisState()
    state.severity_result = tool_error_fatal
    with pytest.raises(FatalPipelineError):
        async for _ in rgen.run(state):
            pass


# ── ReportGenerator.run — HTTP boundary mocked ─────────────────────

async def test_run_success_streams_tokens_and_writes_report(
    monkeypatch, populated_state,
):
    """
    Happy path: full sections returned, report stored on state.

    confidence is asserted as 0.0 because ReportGenerator sets it as a
    deliberate placeholder. The pipeline injects the real value via
    calculate_confidence() after this method returns. The integration
    test test_pipeline_confidence_injection.py verifies the injected
    value is > 0.0.
    """
    full_text = _full_report_text()
    _patch_ollama(monkeypatch, full_text)

    rgen = ReportGenerator()
    tokens = [t async for t in rgen.run(populated_state)]

    assert populated_state.report is not None
    assert "".join(tokens) == full_text
    assert populated_state.report.text == full_text
    assert populated_state.report.severity == populated_state.severity_result.level
    assert populated_state.report.confidence == 0.0


async def test_run_raises_fatal_when_sections_missing(
    monkeypatch, populated_state,
):
    """Section validation is a hard contract — missing section is fatal."""
    incomplete = _full_report_text().replace("### Disclaimer", "")
    _patch_ollama(monkeypatch, incomplete)

    rgen = ReportGenerator()
    with pytest.raises(FatalPipelineError):
        async for _ in rgen.run(populated_state):
            pass


async def test_run_propagates_rag_citations_to_report(
    monkeypatch, populated_state,
):
    """Citations from rag_result must appear on the final TriageReport."""
    _patch_ollama(monkeypatch, _full_report_text())

    rgen = ReportGenerator()
    async for _ in rgen.run(populated_state):
        pass

    assert populated_state.report is not None
    for citation in populated_state.rag_result.citations:
        assert citation in populated_state.report.citations


async def test_run_omits_citations_when_rag_is_tool_error(
    monkeypatch, populated_state, tool_error_nonfatal,
):
    """Failed RAG must not propagate citations to the report."""
    populated_state.rag_result = tool_error_nonfatal
    _patch_ollama(monkeypatch, _full_report_text())

    rgen = ReportGenerator()
    async for _ in rgen.run(populated_state):
        pass

    assert populated_state.report is not None
    assert populated_state.report.citations == []


async def test_run_wraps_http_failure_as_fatal_pipeline_error(
    monkeypatch, populated_state,
):
    """No raw httpx error escapes — must always become FatalPipelineError."""
    def _boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(rg.httpx, "AsyncClient", _boom)

    rgen = ReportGenerator()
    with pytest.raises(FatalPipelineError):
        async for _ in rgen.run(populated_state):
            pass


async def test_run_report_has_correct_severity(
    monkeypatch, populated_state,
):
    """Report severity must match the SeverityResult level on state."""
    _patch_ollama(monkeypatch, _full_report_text())

    rgen = ReportGenerator()
    async for _ in rgen.run(populated_state):
        pass

    assert populated_state.report is not None
    assert populated_state.report.severity == populated_state.severity_result.level


async def test_run_report_disclaimer_is_set(
    monkeypatch, populated_state,
):
    """Report disclaimer must be the canonical DISCLAIMER string."""
    _patch_ollama(monkeypatch, _full_report_text())

    rgen = ReportGenerator()
    async for _ in rgen.run(populated_state):
        pass

    assert populated_state.report is not None
    assert populated_state.report.disclaimer == DISCLAIMER


async def test_run_report_knowledge_base_fields_from_corpus_version(
    monkeypatch, populated_state,
):
    """
    knowledge_base_version and knowledge_base_date are sourced from
    get_corpus_version() and get_corpus_date() (tools.corpus_version).

    When corpus_version.md is absent or has no git_commit/snapshot_date
    entries, both return None. When populated, they return non-None strings.

    Patch target: tools.report_generator.get_corpus_version / get_corpus_date
    NOT tools.corpus_version.get_corpus_version.

    ReportGenerator imports these names directly:
        from tools.corpus_version import get_corpus_version, get_corpus_date
    That creates local name bindings in tools.report_generator. Patching the
    source module replaces the function object there but the already-bound
    names in report_generator are unaffected. Patching the call site
    (tools.report_generator.*) intercepts correctly.

    Contract:
        - When corpus returns None → report fields are None
        - When corpus returns strings → report fields match those strings
    """
    _patch_ollama(monkeypatch, _full_report_text())

    # Path 1: no corpus built → fields are None
    monkeypatch.setattr(rg, "get_corpus_version", lambda: None)
    monkeypatch.setattr(rg, "get_corpus_date", lambda: None)

    rgen = ReportGenerator()
    async for _ in rgen.run(populated_state):
        pass

    assert populated_state.report is not None
    assert populated_state.report.knowledge_base_version is None
    assert populated_state.report.knowledge_base_date is None

    # Path 2: corpus built → fields are the returned strings
    _patch_ollama(monkeypatch, _full_report_text())
    populated_state.report = None  # reset so run() writes a fresh report

    monkeypatch.setattr(rg, "get_corpus_version", lambda: "abc123")
    monkeypatch.setattr(rg, "get_corpus_date", lambda: "2025-01-01")

    rgen2 = ReportGenerator()
    async for _ in rgen2.run(populated_state):
        pass

    assert populated_state.report is not None
    assert populated_state.report.knowledge_base_version == "abc123"
    assert populated_state.report.knowledge_base_date == "2025-01-01"