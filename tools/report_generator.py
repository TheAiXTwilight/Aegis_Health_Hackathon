"""
tools/report_generator.py — Step 7: LLM report synthesis.

Changes from original:
    - OLLAMA_BASE_URL read from environment variable.
    - FatalPipelineError import at top level.
    - DrugInteraction objects formatted from structured fields.
    - TOOL_REPORT_GENERATOR from tool_names.py.
    - Async generator return annotation corrected to AsyncGenerator.
"""

from __future__ import annotations

import json as _json
import math
import os
from typing import AsyncGenerator

import httpx
from loguru import logger

from schemas.errors import FatalPipelineError, ToolError
from schemas.report import TriageReport
from schemas.state import AegisState
from tools.tool_names import TOOL_REPORT_GENERATOR


# ── Ollama configuration ──────────────────────────────────────────
# OLLAMA_BASE_URL read at module import time.
# For production: set in environment before startup.
# For tests: mock the httpx call rather than the env var.

OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_STREAM_URL = OLLAMA_BASE_URL + "/api/generate"
MODEL_TAG         = "aegis-llama"


# ── Token budget ──────────────────────────────────────────────────

NUM_CTX                = 4096
RESERVE_OUTPUT         = 1024
MAX_INPUT_TOKENS       = NUM_CTX - RESERVE_OUTPUT
APPROX_CHARS_PER_TOKEN = 4


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
    "before any clinical action is taken. "
    "Do not use in emergency situations."
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
    return max(1, math.ceil(len(text) / APPROX_CHARS_PER_TOKEN))


# ── Context builder ───────────────────────────────────────────────

def _build_context(state: AegisState) -> str:
    """
    Assemble prompt context from AegisState respecting token budget.

    SIDE EFFECT: sets state.core_fields_truncated and
    state.enrichment_fields_truncated. Must be called before
    calculate_confidence().
    """
    parts:  list[str] = []
    budget: int       = MAX_INPUT_TOKENS

    def _add(block: str) -> bool:
        nonlocal budget
        cost = _estimate_tokens(block)
        if budget >= cost:
            parts.append(block)
            budget -= cost
            return True
        return False

    # 1. SeverityResult — never truncated
    sev = state.severity_result
    if sev and not isinstance(sev, ToolError):
        block = (
            f"SEVERITY: {sev.level} (confidence {sev.confidence:.0%})\n"
            f"Highest-priority rule: {sev.highest_priority_rule}\n"
            f"Reasons: {'; '.join(sev.reasons)}"
        )
        if not _add(block):
            logger.error(
                "report_generator · SeverityResult exceeded token budget"
            )
            state.core_fields_truncated = True

    # 2. Core symptoms — never truncated
    sym = state.symptom_result
    if sym and not isinstance(sym, ToolError):
        core = (
            f"SYMPTOMS: {', '.join(sym.symptoms)}\n"
            f"Duration: {sym.duration or 'not specified'}\n"
            f"Severity indicators: "
            f"{', '.join(sym.severity_indicators) or 'none'}"
        )
        if not _add(core):
            logger.error(
                "report_generator · core symptoms exceeded token budget"
            )
            state.core_fields_truncated = True

        if sym.medical_entities:
            block = (
                f"Medical entities: "
                f"{', '.join(sym.medical_entities[:10])}"
            )
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

    # 3. Lab findings
    lab = state.lab_result
    if lab and not isinstance(lab, ToolError):
        if lab.abnormal_values:
            block = (
                "LAB FINDINGS (abnormal):\n"
                + "\n".join(f"  {v}" for v in lab.abnormal_values)
            )
            if not _add(block):
                state.core_fields_truncated = True
                logger.error(
                    "report_generator · abnormal lab values exceeded budget"
                )

        if lab.measurements and budget > 200:
            lines = [
                f"  {name}: {value}"
                for name, value in list(lab.measurements.items())[:10]
            ]
            block = "LAB MEASUREMENTS:\n" + "\n".join(lines)
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning("report_generator · lab.measurements truncated")

    # 4. Drug interactions — structured DrugInteraction objects
    drug = state.drug_result
    if drug and not isinstance(drug, ToolError) and budget > 150:
        lines = []
        for interaction in drug.interactions:
            lines.append(
                f"  {interaction.severity.value.upper()}: "
                f"{interaction.description}"
            )
        if drug.unresolved:
            lines.append(
                f"  Unresolved drugs: {', '.join(drug.unresolved)}"
            )
        if drug.warnings:
            lines.append(f"  Warnings: {'; '.join(drug.warnings)}")
        if lines:
            block = "DRUG INTERACTIONS:\n" + "\n".join(lines)
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning(
                    "report_generator · drug interactions truncated"
                )

    # 5. X-ray findings
    xray = state.xray_result
    if xray and not isinstance(xray, ToolError) and budget > 100:
        positives = [f for f in xray.findings if f]
        if positives:
            block = f"XRAY FINDINGS: {', '.join(positives)}"
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning("report_generator · xray findings truncated")

        if xray.free_text and budget > 80:
            block = f"XRAY FREE TEXT: {xray.free_text}"
            if not _add(block):
                state.enrichment_fields_truncated = True

    # 6. RAG passages — top-2, omitted last
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
                    "report_generator · RAG passage truncated",
                    source=passage.source,
                )
                break

    return "\n\n".join(parts)


# ── Section validator ─────────────────────────────────────────────

def _validate_sections(text: str) -> list[str]:
    return [s for s in REQUIRED_SECTIONS if s not in text]


# ── Tool ──────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Step 7 — LLM synthesis via Ollama.
    Yields raw string tokens. Writes state.report on completion.
    Raises FatalPipelineError on any failure.
    """

    TOOL_NAME = TOOL_REPORT_GENERATOR

    async def run(
        self, state: AegisState
    ) -> AsyncGenerator[str, None]:
        sev = state.severity_result
        if not sev or isinstance(sev, ToolError):
            raise FatalPipelineError(
                ToolError(
                    tool=TOOL_REPORT_GENERATOR,
                    reason=(
                        "SeverityResult missing or failed — "
                        "cannot generate report."
                    ),
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
                    tool=TOOL_REPORT_GENERATOR,
                    reason=f"Ollama request failed: {exc}",
                    fatal=True,
                )
            )

        missing = _validate_sections(full_text)
        if missing:
            logger.error(
                "report_generator · missing sections",
                missing=missing,
            )
            raise FatalPipelineError(
                ToolError(
                    tool=TOOL_REPORT_GENERATOR,
                    reason=f"Report missing required sections: {missing}",
                    fatal=True,
                )
            )

        rag = state.rag_result
        citations: list[str] = []
        if rag and not isinstance(rag, ToolError):
            citations = rag.citations

        state.report = TriageReport(
            severity               = sev.level,
            confidence             = 0.0,   # filled by calculate_confidence() in pipeline
            text                   = full_text,
            citations              = citations,
            disclaimer             = DISCLAIMER,
            knowledge_base_version = None,
            knowledge_base_date    = None,
        )

        logger.info(
            "report_generator · complete",
            truncated_core=state.core_fields_truncated,
            truncated_enrichment=state.enrichment_fields_truncated,
        )