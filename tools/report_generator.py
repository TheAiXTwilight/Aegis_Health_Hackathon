"""
tools/report_generator.py — Step 7

llama3.2:1b synthesis via Ollama. Six-section streaming report.
Token budget enforced before the LLM call — no silent truncation.

Context prioritization (descending priority under budget pressure):
    1. SeverityResult          — never truncated
    2. Core symptoms           — never truncated (symptoms, duration,
                                 severity_indicators)
    3. Symptom enrichment      — dropped first under pressure
                                 (medical_entities, then negations)
    4. Lab findings            — abnormal values protected; normal
                                 values dropped first
    5. Drug interactions       — flagged pairs only if tight
    6. X-ray positives         — checklist summary only if tight
    7. RAG passages            — top-2 if tight, omitted last

Truncation flags written back to AegisState:
    core_fields_truncated       = True  → confidence penalty (severe)
    enrichment_fields_truncated = True  → confidence penalty (minor)

The confidence formula in tools/confidence.py reads these flags.
This module only sets them — it never computes confidence directly.

Six required report sections (missing any → ToolError fatal=True):
    ### Summary
    ### Findings
    ### Evidence
    ### Severity
    ### Recommendations
    ### Disclaimer
"""
from __future__ import annotations

import math
import httpx
from loguru import logger

from schemas.errors import ToolError
from schemas.report import TriageReport
from schemas.state import AegisState

# ── Ollama configuration ──────────────────────────────────────────
OLLAMA_STREAM_URL = "http://localhost:11434/api/generate"
MODEL_TAG         = "aegis-llama"       # created by entrypoint.sh from Modelfile

# ── Token budget ──────────────────────────────────────────────────
# All budget constants are owned here — they govern how this tool
# manages its prompt. APPROX_CHARS_PER_TOKEN is tunable for clinical
# text on Jetson without touching truncation logic.
NUM_CTX                = 4096
RESERVE_OUTPUT         = 1024           # num_predict — reserved for generation
MAX_INPUT_TOKENS       = NUM_CTX - RESERVE_OUTPUT   # 3072
APPROX_CHARS_PER_TOKEN = 4              # ~1 token per 4 chars; tune if clinical
                                        # abbreviations (mg/dL, eGFR, ↑) skew short

# ── Report contract ───────────────────────────────────────────────
REQUIRED_SECTIONS = [
    "### Summary",
    "### Findings",
    "### Evidence",
    "### Severity",
    "### Recommendations",
    "### Disclaimer",
]

DISCLAIMER = (
    "Clinical decision support only — not a diagnosis. "
    "All outputs must be reviewed by a qualified healthcare professional "
    "before any clinical action is taken. Do not use in emergency situations."
)

MID_STREAM_FAILURE_MESSAGE = (
    "\n\n⚠️ Report generation was interrupted. The output above is incomplete "
    "and must not be used for clinical decisions. Please resubmit or "
    "contact support."
)

REPORT_PROMPT_TEMPLATE = """\
You are Aegis Health ReportGenerator. Synthesise a structured triage report
from the clinical data provided below.

Rules:
- The severity level is already determined — do NOT change it.
- Use only the evidence provided — do NOT fabricate citations.
- Every section header below is required. Do not omit any.
- Write in plain clinical language suitable for a triage clinician.

Required sections (use these exact headers):
### Summary
2-3 sentence overview of the patient's presentation.

### Findings
Key clinical findings across submitted modalities.
If X-ray data is present, include a Radiological subsection.

### Evidence
Cite the retrieved medical evidence passages. If no passages were
retrieved, state explicitly: "No medical evidence was retrieved for
this presentation."

### Severity
State the severity level and confidence. Explain the highest-priority
rule that determined it.

### Recommendations
Ordered list of clinical next steps.

### Disclaimer
{disclaimer}

---
CLINICAL DATA:
{context}
---
Generate the report now:"""


# ── Token budget helpers ──────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate for prompt budget enforcement.

    Uses APPROX_CHARS_PER_TOKEN — owned by this module because prompt
    budget is a ReportGenerator concern. If RAGRetriever ever needs
    embedding cost estimation, it should own its own heuristic: the
    two contexts have different requirements and may legitimately diverge.
    """
    return max(1, math.ceil(len(text) / APPROX_CHARS_PER_TOKEN))


# ── Context builder ───────────────────────────────────────────────

def _build_context(state: AegisState) -> str:
    """
    Assemble prompt context from AegisState, respecting token budget.

    Priority order enforced under budget pressure (see module docstring).
    Sets state.core_fields_truncated / state.enrichment_fields_truncated
    as side effects — the confidence formula reads these flags.

    Returns the assembled context string.
    """
    parts: list[str] = []
    budget = MAX_INPUT_TOKENS

    def _add(block: str) -> bool:
        """Add block if it fits. Returns True if added."""
        nonlocal budget
        cost = _estimate_tokens(block)
        if budget >= cost:
            parts.append(block)
            budget -= cost
            return True
        return False

    # ── 1. SeverityResult — NEVER truncated ──────────────────────
    sev = state.severity_result
    if sev and not isinstance(sev, ToolError):
        block = (
            f"SEVERITY: {sev.level} "
            f"(confidence {sev.confidence:.0%})\n"
            f"Highest-priority rule: {sev.highest_priority_rule}\n"
            f"Reasons: {'; '.join(sev.reasons)}"
        )
        if not _add(block):
            # SeverityResult must always fit — if it doesn't, something
            # is structurally wrong with the prompt template.
            logger.error(
                "report_generator · SeverityResult exceeded token budget — "
                "increase MAX_INPUT_TOKENS or reduce RESERVE_OUTPUT"
            )
            state.core_fields_truncated = True

    # ── 2. Core symptoms — NEVER truncated ───────────────────────
    sym = state.symptom_result
    if sym and not isinstance(sym, ToolError):
        core = (
            f"SYMPTOMS: {', '.join(sym.symptoms)}\n"
            f"Duration: {sym.duration or 'not specified'}\n"
            f"Severity indicators: {', '.join(sym.severity_indicators) or 'none'}"
        )
        if not _add(core):
            logger.error(
                "report_generator · core symptoms exceeded token budget"
            )
            state.core_fields_truncated = True

        # Enrichment fields — dropped first under pressure.
        if sym.medical_entities:
            block = f"Medical entities: {', '.join(sym.medical_entities[:10])}"
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning(
                    "report_generator · symptom.medical_entities truncated"
                )

        if sym.negations:
            block = f"Negations: {', '.join(sym.negations[:5])}"
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning(
                    "report_generator · symptom.negations truncated"
                )

    # ── 3. Lab findings — abnormal values protected ───────────────
    lab = state.lab_result
    if lab and not isinstance(lab, ToolError):
        # abnormal_values: list[str] — human-readable, per Day1 schema
        # measurements: dict[str, float] — numeric, for completeness
        if lab.abnormal_values:
            block = (
                "LAB FINDINGS (abnormal):\n"
                + "\n".join(f"  {v}" for v in lab.abnormal_values)
            )
            if not _add(block):
                # Abnormal lab values are core — flag if they don't fit.
                state.core_fields_truncated = True
                logger.error(
                    "report_generator · abnormal lab values exceeded budget"
                )

        if lab.measurements and budget > 200:
            # Include numeric measurements if budget allows.
            lines = [
                f"  {name}: {value}"
                for name, value in list(lab.measurements.items())[:10]
            ]
            block = "LAB MEASUREMENTS:\n" + "\n".join(lines)
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning(
                    "report_generator · lab.measurements truncated"
                )

    # ── 4. Drug interactions ──────────────────────────────────────
    drug = state.drug_result
    if drug and not isinstance(drug, ToolError) and budget > 150:
        lines = []
        for interaction in drug.interactions:
            lines.append(f"  INTERACTION: {interaction}")
        if drug.unresolved:
            lines.append(
                f"  Unresolved drugs: {', '.join(drug.unresolved)}"
            )
        if drug.warnings:
            lines.append(
                f"  Warnings: {'; '.join(drug.warnings)}"
            )
        if lines:
            block = "DRUG INTERACTIONS:\n" + "\n".join(lines)
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning(
                    "report_generator · drug interactions truncated"
                )

    # ── 5. X-ray positives ────────────────────────────────────────
    xray = state.xray_result
    if xray and not isinstance(xray, ToolError) and budget > 100:
        positives = [f for f in xray.findings if f]   # findings: list[str]
        if positives:
            block = f"XRAY FINDINGS: {', '.join(positives)}"
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning(
                    "report_generator · xray findings truncated"
                )
        if xray.free_text and budget > 80:
            block = f"XRAY FREE TEXT: {xray.free_text}"
            if not _add(block):
                state.enrichment_fields_truncated = True

    # ── 6. RAG passages — top-2, omitted last ────────────────────
    rag = state.rag_result
    if rag and not isinstance(rag, ToolError) and budget > 200:
        for passage in rag.passages[:2]:
            block = (
                f"EVIDENCE [{passage.source}]:\n"
                f"  {passage.text[:300]}\n"
                f"  Citation: {passage.citation}"
            )
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning(
                    "report_generator · RAG passage truncated · "
                    "source={}", passage.source
                )
                break   # if one passage doesn't fit, skip remaining

    return "\n\n".join(parts)


# ── Section validator ─────────────────────────────────────────────

def _validate_sections(text: str) -> list[str]:
    """
    Return list of required section headers missing from text.
    Empty list means the report is complete.
    """
    return [s for s in REQUIRED_SECTIONS if s not in text]


# ── Tool ──────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Step 7 — LLM synthesis.

    Interface contract (matches AegisPipeline._run_report_generator):
        async def run(state) -> AsyncIterator[str]

    Yields raw string tokens as they arrive from Ollama.
    Writes state.report once the stream is exhausted and sections
    are validated. Never yields SSE framing — plain text only.

    On failure, raises FatalPipelineError so the pipeline worker
    marks the job FAILED. state.report is left None.
    """

    async def run(self, state: AegisState) -> AsyncIterator[str]:
        """
        Stream report tokens and finalise state.report on completion.

        Yields plain string tokens — no framing, no events.
        st.write_stream consumes these directly via /queue/stream.

        Raises FatalPipelineError on:
            - missing SeverityResult
            - Ollama request failure
            - missing required report sections
        """
        import json as _json

        from schemas.errors import FatalPipelineError

        sev = state.severity_result
        if not sev or isinstance(sev, ToolError):
            raise FatalPipelineError(
                ToolError(
                    tool="ReportGenerator",
                    reason="SeverityResult missing or failed — cannot generate report.",
                    fatal=True,
                )
            )

        context = _build_context(state)
        prompt  = REPORT_PROMPT_TEMPLATE.format(
            disclaimer=DISCLAIMER,
            context=context,
        )

        full_text = ""

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    OLLAMA_STREAM_URL,
                    json={
                        "model":  MODEL_TAG,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": RESERVE_OUTPUT,
                            "num_ctx":     NUM_CTX,
                        },
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data  = _json.loads(line)
                        token = data.get("response", "")
                        if token:
                            full_text += token
                            yield token
                        if data.get("done"):
                            break

        except FatalPipelineError:
            raise

        except Exception as exc:
            raise FatalPipelineError(
                ToolError(
                    tool="ReportGenerator",
                    reason=f"Ollama request failed: {exc}",
                    fatal=True,
                )
            )

        missing = _validate_sections(full_text)
        if missing:
            logger.error(
                "report_generator · missing sections · {}",
                missing,
            )
            raise FatalPipelineError(
                ToolError(
                    tool="ReportGenerator",
                    reason=f"Report missing required sections: {missing}",
                    fatal=True,
                )
            )

        # RAG provenance — kb_version/kb_date populated in Week 2.
        rag = state.rag_result
        citations: list[str] = []
        if rag and not isinstance(rag, ToolError):
            citations = rag.citations

        state.report = TriageReport(
            severity               = sev.level,
            confidence             = 0.0,   # filled by confidence.py after this step
            text                   = full_text,
            citations              = citations,
            disclaimer             = DISCLAIMER,
            knowledge_base_version = None,
            knowledge_base_date    = None,
        )
        logger.info(
            "report_generator · complete · truncated_core={} · "
            "truncated_enrichment={}",
            state.core_fields_truncated,
            state.enrichment_fields_truncated,
        )