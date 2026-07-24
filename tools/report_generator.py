"""
tools/report_generator.py — Step 7: LLM report synthesis.


Changes from original:
    - OLLAMA_BASE_URL read from environment variable.
    - FatalPipelineError import at top level.
    - DrugInteraction objects formatted from structured fields.
    - TOOL_REPORT_GENERATOR from tool_names.py.
    - Async generator return annotation corrected to AsyncGenerator.
    - Prompt assembled via .replace() not .format() so that patient input
      containing curly braces (e.g. "pain when typing { or }") cannot
      crash the prompt build with KeyError.
    - Prompt template reframed as technical document formatting to avoid
      safety refusals from small Llama 3.2 models. Includes few-shot
      example to anchor the markdown structure.

Refactor (backend-first content generation):
    _build_deterministic_report now emits the COMPLETE 11-section report
    including Reported Symptoms & Clinical History, grouped Findings,
    Critical Observations & Flags, Personalized Recommendations, and
    Care Plan. Frontend and PDF exporter both consume this as-is via
    RAW_HTML tokens for pixel-identical rendering.

    Classification/grouping is delegated to
    backend.dashboard.build_report_measurement_groups so there is a
    single source of truth for measurement risk classification.

    Advice content is resolved through tools.biomarker_knowledge, which
    provides a curated knowledge base plus a universal fallback so no
    flagged item is ever silently skipped.

    Extended enrichment applied on top of the dashboard extractor:
        - Units taken from the actual lab PDF (lab_result.units), with
          CANONICAL_UNITS only as fallback.
        - Reference ranges from the actual lab PDF (lab_result.reference_ranges),
          with REFERENCE_RANGES fallback.
        - Missing display names normalised through a medical abbreviation
          formatter (TSH, LDL, SGOT, HbA1c, etc.).
        - Un-classified items re-scored against reference ranges so
          thyroid, lipids, vitamins, and other extras also get high/low
          flags in the report body.
        - X-ray findings appended as flagged items so they participate
          in Findings, Critical Observations, Personalized Recommendations,
          and Care Plan.
        - Text findings from the lab report (morphology, smear, impressions)
          rendered in a dedicated "Peripheral Smear & Morphology" subsection.
        - All items sorted by clinical priority tier (Critical → Out-of-range
          → Borderline → Normal → Reported), then alphabetical within a tier.
        - Numeric biomarker mentions inside the patient's typed symptoms /
          medical history are extracted and injected into the same
          measurements pipeline. This lets a user typing "BP 160/100,
          glucose 220, TSH 12" get the same personalised findings, advice,
          and care plan as if they had uploaded a lab PDF. PDF-extracted
          values always win in case of a duplicate.

    Advice consolidation:
        - When an item has NO curated knowledge-base entry and would fall
          through to the universal fallback, its Care Plan text is folded
          into a single "share these results with your healthcare provider"
          line per bucket. Personalized Recommendations still shows one
          knowledge-based line per known biomarker.

    Care Plan density controls (new):
        - Fuzzy deduplication merges near-identical bucket lines
          (e.g. "Recheck CBC in 2–3 months" absorbs "Recheck in 2–3
          months" and "Recheck total protein in 2–3 months") using a
          token-overlap heuristic so buckets don't repeat the same
          advice five different ways.
        - Line caps limit each bucket to a maximum of 6 items. If more
          exist, they are summarised as "and N more clinical actions —
          please review with your physician."
"""


from __future__ import annotations


import asyncio
import json as _json
import math
import os
import re as _re
from typing import AsyncGenerator


import httpx
from loguru import logger
from sympy import limit


from schemas.errors import FatalPipelineError, ToolError
from schemas.report import TriageReport
from schemas.state import AegisState
from tools.tool_names import TOOL_REPORT_GENERATOR
from tools.corpus_version import get_corpus_date, get_corpus_version
from tools.biomarker_knowledge import resolve_advice
from tools.lab_thresholds import CANONICAL_UNITS, REFERENCE_RANGES


# ── Ollama configuration ──────────────────────────────────────────


OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_STREAM_URL = OLLAMA_BASE_URL + "/api/generate"
MODEL_TAG         = "aegis-llama"


# ── Token budget ──────────────────────────────────────────────────


NUM_CTX                = 4096
RESERVE_OUTPUT         = 2200
MAX_INPUT_TOKENS       = NUM_CTX - RESERVE_OUTPUT
APPROX_CHARS_PER_TOKEN = 4


# ── Section-reveal pacing ──────────────────────────────────────────
# The deterministic report is built as one complete string, then
# streamed to the client. Previously it went out in a single yield,
# so any "pause" the UI appeared to show between sections (Patient
# Information → Date of Birth → Reported Symptoms, etc.) was really
# just an artifact of arbitrary TCP/HTTP chunk boundaries landing
# inside an already-finished string — not a real, controllable pause.
#
# Yielding one section at a time with a small sleep between them
# gives an actual, intentional pause at section boundaries instead.
# Override via env var, e.g. AEGIS_REPORT_SECTION_PAUSE_SECONDS=0.15
SECTION_REVEAL_PAUSE_SECONDS = float(
    os.getenv("AEGIS_REPORT_SECTION_PAUSE_SECONDS", "0.2")
)


# ── Report contract ───────────────────────────────────────────────


REQUIRED_SECTIONS = [
    "### Patient Information",
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


_RULE_EXPLANATIONS = {
    "RULE_CHEST_PAIN_AND_SOB": "Combined chest pain and shortness of breath reported",
    "RULE_CRITICAL_LAB_TROPONIN": "Critical elevation detected in cardiac troponin levels",
    "RULE_CRITICAL_LAB_HAEMOGLOBIN": "Critical low hemoglobin (severe anemia) detected",
    "RULE_CRITICAL_LAB_POTASSIUM": "Critical serum potassium abnormality (hyperkalemia/hypokalemia)",
    "RULE_XRAY_PNEUMOTHORAX": "Pneumothorax (collapsed lung) detected on chest X-ray",
    "RULE_XRAY_PULMONARY_EDEMA": "Pulmonary edema (fluid in lung tissue) detected on chest X-ray",
    "RULE_SEVERE_DRUG_INTERACTION": "Severe contraindicated drug interaction identified in medication list",
    "RULE_ABNORMAL_LAB_ANY": "Biomarker outside reference range detected in lab results",
    "RULE_XRAY_CARDIOMEGALY": "Cardiomegaly (enlarged heart silhouette) detected on chest X-ray",
    "RULE_XRAY_PLEURAL_EFFUSION": "Pleural effusion (fluid around lungs) detected on chest X-ray",
    "RULE_XRAY_CONSOLIDATION": "Pulmonary consolidation (opacification/infection) detected on chest X-ray",
    "RULE_PROLONGED_SYMPTOMS": "Persistent symptoms lasting longer than 14 days without resolution",
    "RULE_MODERATE_DRUG_INTERACTION": "Moderate medication interaction requiring monitoring",
    "RULE_DEFAULT_LOW": "No high-risk biomarker, imaging, or pharmacological flags triggered",
}


def _explain_rule(rule_id: str) -> str:
    """
    Return a human-readable explanation for a severity rule constant.

    Universal handling:
      1. Exact match in _RULE_EXPLANATIONS (curated base rules).
      2. Text-finding rules (RULE_TEXT_FINDING_*): strip prefix and
         format the pattern ID as a readable phrase. This handles any
         current or future pattern added to
         tools.text_finding_analyzer._TEXT_PATTERNS without requiring
         an entry here.
      3. Any other unknown rule: strip 'RULE_' prefix and title-case.
    """
    if not rule_id:
        return "Not available"

    rid = rule_id.strip()

    # Curated explanations first
    if rid in _RULE_EXPLANATIONS:
        return _RULE_EXPLANATIONS[rid]

    # Text-finding rules: universally derived from pattern id
    text_prefix = "RULE_TEXT_FINDING_"
    if rid.startswith(text_prefix):
        pattern_id = rid[len(text_prefix):]
        # Convert "SMEAR_REACTIVE_LYMPHOCYTE" → "Smear Reactive Lymphocyte"
        readable = pattern_id.replace("_", " ").lower().title()
        return f"Interpretive finding: {readable}"

    # Generic fallback
    return rid.replace("RULE_", "").replace("_", " ").title()


MID_STREAM_FAILURE_MESSAGE = (
    "\n\n⚠️ Report generation was interrupted. The output above is incomplete "
    "and must not be used for clinical decisions. Please resubmit or "
    "contact support."
)


REPORT_PROMPT_TEMPLATE = """\
You are a clinical document formatting assistant.

Your task is to write a clear, detailed, patient-facing health report for
PDF export using ONLY the structured data provided below.

Important role boundary:
- You are NOT diagnosing the patient.
- You are NOT prescribing medication.
- You are NOT deciding clinical severity.
- The deterministic rule engine has already produced the severity.
- Your job is to explain the provided information clearly and safely.

Write in a professional, calm, patient-friendly tone.

CRITICAL OUTPUT RULES:
- Output exactly EIGHT sections.
- Use the exact markdown section headers listed below.
- Keep the section order exactly as listed.
- Do not skip any section.
- Do not rename any section.
- Do not add extra section headers.
- Do not invent facts that are not present in the structured data.
- Do not add body systems, organs, X-rays, lab tests, or medications unless they appear in the structured data.
- Do not repeat placeholder phrases such as "Not provided, No data provided, Not available".
- If something is missing, write one natural sentence explaining that the information was not provided.
- Use clean markdown formatting with normal spacing.
- Do not insert excessive blank lines or empty bullets.
- Each section should contain enough detail to be useful in a downloadable patient report.
- Do not provide a diagnosis.
- Do not prescribe medication, dosage, or treatment.
- Do not say that medical review is unnecessary.
- Severity must match the deterministic severity in the structured data.
- The Disclaimer section must use the required disclaimer text exactly.

Required section headers, in exact order:

### Patient Information
### Summary
### Findings
### Evidence
### Severity
### Recommendations
### Disclaimer

STRUCTURED DATA:
%%CONTEXT%%

REQUIRED DISCLAIMER TEXT:
%%DISCLAIMER%%

Now write the full report in markdown.

Start exactly with:
### Patient Information

End with the Disclaimer section.
"""


def _assemble_prompt(disclaimer: str, context: str | None) -> str:
    return (
        REPORT_PROMPT_TEMPLATE
        .replace("%%DISCLAIMER%%", disclaimer)
        .replace("%%CONTEXT%%", context)
    )


# ── Token budget helpers ──────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / APPROX_CHARS_PER_TOKEN))


# ── Context builder ───────────────────────────────────────────────


def _build_context(state: AegisState) -> str:
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

    patient_name = getattr(state, "patient_name", None)
    patient_dob = getattr(state, "patient_dob", None)
    patient_sex = getattr(state, "patient_sex", None)
    patient_blood_group = getattr(state, "patient_blood_group", None)
    patient_weight_kg = getattr(state, "patient_weight_kg", None)
    patient_height_cm = getattr(state, "patient_height_cm", None)
    patient_allergies = getattr(state, "patient_allergies", None)
    patient_conditions = getattr(state, "patient_medical_conditions", []) or []

    patient_lines: list[str] = [
        f"Name: {patient_name or 'Not provided'}",
        f"Date of Birth: {patient_dob or 'Not provided'}",
        f"Sex: {patient_sex or 'Not provided'}",
        f"Blood group: {patient_blood_group or 'Not provided'}",
        f"Weight: {f'{patient_weight_kg:g} kg' if patient_weight_kg is not None else 'Not provided'}",
        f"Height: {f'{patient_height_cm:g} cm' if patient_height_cm is not None else 'Not provided'}",
        f"Allergies: {patient_allergies or 'Not provided'}",
        f"Existing medical conditions: {', '.join(patient_conditions) if patient_conditions else 'Not provided'}",
    ]

    if not _add("PATIENT PROFILE:\n" + "\n".join(patient_lines)):
        state.core_fields_truncated = True
        logger.error("report_generator · patient profile exceeded token budget")

    submitted_symptoms = (
        getattr(state, "submitted_symptoms_text", None)
        or state.raw_symptoms_text
    )

    submission_lines: list[str] = []

    if submitted_symptoms:
        submission_lines.append(
            f"Symptoms / medical history submitted: {submitted_symptoms}"
        )

    if state.audio_file_path:
        submission_lines.append("Voice input was submitted.")

    if state.medications_raw:
        submission_lines.append(
            "Medication list submitted: " + ", ".join(state.medications_raw)
        )

    if state.lab_pdf_path:
        submission_lines.append("A lab report PDF was uploaded for analysis.")

    if state.xray_image_path:
        submission_lines.append("An X-ray image was uploaded for analysis.")

    if state.xray_findings_raw:
        submission_lines.append(
            "User-selected X-ray findings: " + ", ".join(state.xray_findings_raw)
        )

    if state.xray_free_text_raw:
        submission_lines.append(
            f"Additional X-ray notes submitted: {state.xray_free_text_raw}"
        )

    if submission_lines:
        if not _add("USER SUBMISSION:\n" + "\n".join(submission_lines)):
            state.core_fields_truncated = True
            logger.error("report_generator · submitted information exceeded token budget")

    summary_inputs: list[str] = []

    if submitted_symptoms:
        summary_inputs.append(f"Primary symptom text: {submitted_symptoms}")

    if state.lab_pdf_path:
        summary_inputs.append("Lab PDF available for tool analysis.")

    if state.xray_image_path or state.xray_findings_raw or state.xray_free_text_raw:
        summary_inputs.append("X-ray-related input available for tool analysis.")

    if state.medications_raw:
        summary_inputs.append("Medication list available for interaction analysis.")

    if summary_inputs:
        if not _add("CASE OVERVIEW INPUTS:\n" + "\n".join(summary_inputs)):
            state.enrichment_fields_truncated = True
            logger.warning("report_generator · case overview inputs truncated")

    voice = state.voice_result
    if voice and not isinstance(voice, ToolError):
        transcript = getattr(voice, "transcript", None)
        if transcript:
            block = f"VOICE TRANSCRIPTION:\n{transcript}"
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning("report_generator · voice transcript truncated")

    sev = state.severity_result
    if sev and not isinstance(sev, ToolError):
        block = (
            f"SEVERITY: {sev.level} (confidence {sev.confidence:.0%})\n"
            f"Highest-priority rule: {_explain_rule(sev.highest_priority_rule)}\n"
            f"Reasons: {'; '.join(sev.reasons)}"
        )
        if not _add(block):
            logger.error(
                "report_generator · SeverityResult exceeded token budget"
            )
            state.core_fields_truncated = True

    sym = state.symptom_result
    if sym and not isinstance(sym, ToolError):
        block = (
            f"EXTRACTED SYMPTOMS: {', '.join(sym.symptoms) or 'None extracted'}\n"
            f"Duration: {sym.duration or 'Not specified'}\n"
            f"Severity indicators: {', '.join(sym.severity_indicators) or 'None'}"
        )
        if not _add(block):
            state.core_fields_truncated = True
            logger.error(
                "report_generator · symptom findings exceeded token budget"
            )

    lab = state.lab_result
    if lab and not isinstance(lab, ToolError):
        if lab.abnormal_values:
            block = (
                "LAB FINDINGS:\n"
                + "\n".join(f"- {v}" for v in lab.abnormal_values)
            )
            if not _add(block):
                state.core_fields_truncated = True
                logger.error(
                    "report_generator · abnormal lab values exceeded budget"
                )
        elif state.lab_pdf_path:
            block = "LAB FINDINGS:\n- A lab report was analyzed, but no clearly abnormal values were extracted."
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning("report_generator · lab summary truncated")

    drug = state.drug_result
    if drug and not isinstance(drug, ToolError):
        lines: list[str] = []

        for interaction in drug.interactions:
            lines.append(
                f"- {interaction.severity.value.upper()}: {interaction.description}"
            )

        if drug.unresolved:
            lines.append(
                "- Unresolved medications: " + ", ".join(drug.unresolved)
            )

        if drug.warnings:
            lines.append(
                "- Warnings: " + "; ".join(drug.warnings)
            )

        if not lines and state.medications_raw:
            lines.append("- Medication list reviewed with no significant interactions reported.")

        if lines:
            block = "DRUG FINDINGS:\n" + "\n".join(lines)
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning(
                    "report_generator · drug findings truncated"
                )

    xray = state.xray_result
    if xray and not isinstance(xray, ToolError):
        positives = [f for f in xray.findings if f]

        if positives:
            if positives == ["Normal / No significant findings"]:
                block = (
                    "X-RAY FINDINGS:\n"
                    "- X-ray analysis reported no significant abnormal findings."
                )
            else:
                block = (
                    "X-RAY FINDINGS:\n"
                    + "\n".join(f"- Clinical finding: Evidence suggests {finding}." for finding in positives)
                )

            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning("report_generator · xray findings truncated")

        if xray.free_text and budget > 80:
            block = f"X-RAY INTERPRETATION NOTE:\n{xray.free_text}"
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning("report_generator · xray free text truncated")

    rag = state.rag_result
    if rag and not isinstance(rag, ToolError) and budget > 200:
        if rag.passages:
            passage_blocks: list[str] = []
            for passage in rag.passages[:2]:
                passage_blocks.append(
                    f"- Source: {passage.source}\n"
                    f"  Summary: {passage.text[:280]}\n"
                    f"  Citation: {passage.citation or 'None'}"
                )

            block = "EVIDENCE FINDINGS:\n" + "\n".join(passage_blocks)
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning("report_generator · RAG passages truncated")
        else:
            block = "EVIDENCE FINDINGS:\n- No evidence passages were retrieved."
            if not _add(block):
                state.enrichment_fields_truncated = True
                logger.warning("report_generator · RAG empty-result summary truncated")

    elif isinstance(rag, ToolError):
        block = (
            "EVIDENCE FINDINGS:\n"
            f"- Local evidence retrieval was unavailable: {rag.reason}"
        )
        if not _add(block):
            state.enrichment_fields_truncated = True
            logger.warning("report_generator · RAG error summary truncated")

    return "\n\n".join(parts)


# ── Section validator ─────────────────────────────────────────────


def _validate_sections(text: str) -> list[str]:
    return [s for s in REQUIRED_SECTIONS if s not in text]


def _clean_report_text(text: str) -> str:
    import re

    cleaned = text.strip()
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\n-\s*\n", "\n", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-•]\s*$\n?", "", cleaned)
    cleaned = re.sub(r"(?m)^•\s*", "- ", cleaned)

    return cleaned.strip()


def _repair_report_sections(text: str) -> str:
    import re

    repaired = text.strip()

    replacements = [
        (r"(?im)^patient information\s*$", "### Patient Information"),
        (r"(?im)^patient\s*$", "### Patient Information"),

        (r"(?im)^summary\s*$", "### Summary"),
        (r"(?im)^findings\s*$", "### Findings"),
        (r"(?im)^evidence\s*$", "### Evidence"),
        (r"(?im)^severity\s*$", "### Severity"),
        (r"(?im)^recommendations\s*$", "### Recommendations"),
        (r"(?im)^disclaimer\s*$", "### Disclaimer"),
    ]

    for pattern, replacement in replacements:
        repaired = re.sub(pattern, replacement, repaired)

    repaired = re.sub(r"(?i)(### Summary)([A-Za-z])", r"\1\n\2", repaired)
    repaired = re.sub(r"(?i)(### Findings)([A-Za-z])", r"\1\n\2", repaired)
    repaired = re.sub(r"(?i)(### Evidence)([A-Za-z])", r"\1\n\2", repaired)
    repaired = re.sub(r"(?i)(### Severity)([A-Za-z])", r"\1\n\2", repaired)
    repaired = re.sub(r"(?i)(### Recommendations)([A-Za-z])", r"\1\n\2", repaired)
    repaired = re.sub(r"(?i)(### Disclaimer)([A-Za-z])", r"\1\n\2", repaired)

    return repaired


# ── HTML rendering helpers (RAW_HTML blocks) ──────────────────────
# These build the pixel-identical HTML the frontend previously produced
# for Findings, Critical Observations, Personalized Recommendations,
# and Care Plan sections. Wrapped in <!--RAW_HTML_START--> markers so
# both frontend renderMarkdown() and backend markdown_to_html_enriched()
# inject them verbatim.


def _escape_html(value: object) -> str:
    from html import escape
    return escape(str(value or ""))


RAW_HTML_START = "<!--RAW_HTML_START-->"
RAW_HTML_END = "<!--RAW_HTML_END-->"


# ── Display-name formatter (medical abbreviations) ────────────────
# Dashboard's raw ".title()" produces "Tsh", "Sgot Ast", "Hdl Cholesterol".
# We normalise those to proper medical casing here.

_UPPERCASE_ABBR: dict[str, str] = {
    "tsh": "TSH",
    "t3": "T3",
    "t4": "T4",
    "wbc": "WBC",
    "rbc": "RBC",
    "hdl": "HDL",
    "ldl": "LDL",
    "vldl": "VLDL",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "sgpt": "SGPT",
    "sgot": "SGOT",
    "alt": "ALT",
    "ast": "AST",
    "bun": "BUN",
    "hba1c": "HbA1c",
    "a1c": "HbA1c",
    "spo2": "SpO2",
    "bp": "BP",
    "hb": "Hb",
    "hgb": "Hb",
    "crp": "CRP",
    "esr": "ESR",
    "psa": "PSA",
    "tibc": "TIBC",
    "alp": "ALP",
    "ggt": "GGT",
    "ldh": "LDH",
    "ck": "CK",
    "bnp": "BNP",
    "inr": "INR",
    "aptt": "aPTT",
    "pt": "PT",
    "d": "D",
    "b12": "B12",
    "c": "C",
    "e": "E",
    "k": "K",
    "na": "Na",
    "mpv": "MPV",
    "pdw": "PDW",
    "pcv": "PCV",
    "rdw": "RDW",
    "cv": "CV",
    "tlc": "TLC",
    "fbs": "FBS",
    "rbs": "RBS",
    "ft3": "FT3",
    "ft4": "FT4",
    "cea": "CEA",
    "afp": "AFP",
    "lh": "LH",
    "fsh": "FSH",
    "ca": "Ca",
    "cl": "Cl",
    "mg": "Mg",
    "hcg": "hCG",
    "ag": "A/G",
    "a/g": "A/G",
}


def _pretty_display_name(raw_name: str, canonical_key: str = "") -> str:
    """
    Produce a proper display name from either the raw dashboard 'name'
    or the canonical key. Preserves medical abbreviations (TSH, LDL,
    SGOT, HbA1c, etc.) and title-cases the rest.

    Tokens are split on BOTH whitespace AND forward-slash so that
    compound tokens like "A/G" are handled correctly — "a/g" is looked
    up as a whole token in _UPPERCASE_ABBR before attempting
    character-level casing.
    """
    # Priority: if canonical_key maps to a known display name in
    # biomarker_knowledge, use that directly — it is always correct.
    # This short-circuits the tokenizer entirely for known biomarkers.
    try:
        from tools.biomarker_knowledge import get_display_name as _get_kb_display
        kb_name = _get_kb_display(canonical_key)
        if kb_name and canonical_key:
            return kb_name
    except Exception:
        pass

    source = (raw_name or canonical_key or "").strip()
    if not source:
        return ""

    # Normalise separators to spaces
    source = source.replace("_", " ")
    tokens = [t for t in _re.split(r"\s+", source) if t]

    out_tokens: list[str] = []
    for tok in tokens:
        lower = tok.lower()
        if lower in _UPPERCASE_ABBR:
            # Whole token is a known abbreviation
            out_tokens.append(_UPPERCASE_ABBR[lower])
        elif "/" in tok:
            # Slash-separated compound token (e.g. "A/G", "A/g", "a/g")
            # Look it up as a whole first; if not found, case each part
            parts = tok.split("/")
            cased_parts: list[str] = []
            for part in parts:
                part_lower = part.lower()
                if part_lower in _UPPERCASE_ABBR:
                    cased_parts.append(_UPPERCASE_ABBR[part_lower])
                else:
                    cased_parts.append(part[:1].upper() + part[1:].lower() if part else "")
            out_tokens.append("/".join(cased_parts))
        else:
            out_tokens.append(tok[:1].upper() + tok[1:].lower())
    return " ".join(out_tokens)


# ── Unit resolver (lab PDF unit is authoritative) ─────────────────

def _resolve_unit(item: dict, lab_units: dict | None = None) -> str:
    """
    Return item's unit with strict priority:
      1. Unit already on the item (from dashboard extractor / lab PDF)
      2. lab_result.units[key] (extracted directly from the lab report PDF)
      3. CANONICAL_UNITS fallback
    """
    unit = str(item.get("unit") or "").strip()
    if unit:
        return unit
    key = str(item.get("key") or "").lower()
    if lab_units and key in lab_units:
        return str(lab_units[key])
    return CANONICAL_UNITS.get(key, "")


# ── Reclassification against reference ranges ─────────────────────
# Dashboard only classifies ~15 biomarkers. Everything else comes back
# as status="reported" / risk_score=None. We re-score those extras
# against reference ranges — preferring ranges from the actual lab PDF
# over the hardcoded fallback — so thyroid, lipids, vitamins, and
# other extras also get high/low flags in the report body.

def _classify_against_ranges(item: dict, lab_ranges: dict | None = None) -> dict:
    """
    If the item lacks classification, compute status/risk_score from
    reference ranges. Prefers lab_result.reference_ranges (extracted
    from the actual PDF) over the hardcoded REFERENCE_RANGES.

    Returns the item (mutated in-place) for chainability.
    """
    status = str(item.get("status") or "").lower()
    if status not in ("", "reported"):
        return item  # already classified by dashboard — respect it

    key = str(item.get("key") or "").lower()

    # ── Smart range lookup ────────────────────────────────────────
    # PDF may store ranges under "vitamin d" but item key is
    # "vitamin_d" or vice versa. Try all key variants before
    # giving up so no biomarker is silently skipped due to a
    # space/underscore mismatch.
    def _find_range(k: str) -> dict | None:
        variants = [
            k,                   # exact as-is
            k.replace("_", " "), # underscore → space
            k.replace(" ", "_"), # space → underscore
        ]
        # Priority 1: ranges from the actual lab report PDF
        for v in variants:
            rng = (lab_ranges or {}).get(v)
            if rng:
                return rng
        # Priority 2: REFERENCE_RANGES fallback
        for v in variants:
            rng = REFERENCE_RANGES.get(v)
            if rng:
                return rng
        return None
    # ─────────────────────────────────────────────────────────────

    rng = _find_range(key)
    if not rng:
        # No numeric range available anywhere (neither the lab's own PDF
        # nor the hardcoded fallback table) — this does NOT mean the
        # value is normal, it means we can't determine direction/severity
        # numerically. Leaving status="reported"/risk_score=None here
        # makes the item invisible to every downstream abnormality check
        # (including the clinical picture biomarker fallback), so a real
        # abnormality in an uncommon/regional biomarker could silently
        # never surface anywhere. Mark it distinctly so callers can
        # still choose to show it — flagged for clinician review, just
        # without a computed high/low direction — rather than losing it.
        item["status"] = "unclassified"
        item["risk_score"] = 1
        return item

    value = item.get("value")
    try:
        val = float(value)
    except (TypeError, ValueError):
        return item

    low = rng.get("low")
    high = rng.get("high")

    new_status: str | None = None
    new_risk = 0

    # Two-sided range
    if low is not None and high is not None:
        if val > high:
            new_status, new_risk = "high", 1
        elif val < low:
            new_status, new_risk = "low", 1
        else:
            span = high - low
            margin = span * 0.1 if span > 0 else 0

            # ── FIX #1: skip low-end borderline when low == 0 ──
            near_high = val >= high - margin
            near_low = (low > 0) and (val <= low + margin)
            if near_high or near_low:
                new_status, new_risk = "borderline", 1
            else:
                new_status, new_risk = "normal", 0
            # ────────────────────────────────────────────────────

    # Upper-limit only
    elif high is not None:
        if val >= high:
            new_status, new_risk = "high", 1
        elif val >= high * 0.9:
            new_status, new_risk = "borderline", 1
        else:
            new_status, new_risk = "normal", 0

    # Lower-limit only
    elif low is not None:
        if val <= low:
            new_status, new_risk = "low", 1
        # ── FIX #1: skip low-end borderline when low == 0 ──
        elif low > 0 and val <= low * 1.1:
            new_status, new_risk = "borderline", 1
        else:
            new_status, new_risk = "normal", 0
        # ────────────────────────────────────────────────────

    if new_status is not None:
        item["status"] = new_status
        item["risk_score"] = new_risk
        item["normal_low"] = low
        item["normal_high"] = high

    return item


def _enrich_measurement(
    item: dict,
    lab_units: dict | None = None,
    lab_ranges: dict | None = None,
) -> dict:
    """
    Full enrichment on top of the dashboard extractor:
      - Reclassify against reference ranges (lab PDF preferred)
      - Attach display range
      - Fill missing unit
      - Normalise display name
      - Rebuild display_value
      - Normalize display_value and reference ranges to canonical units
        (DISPLAY ONLY — raw values untouched)
    """
    item = dict(item)
    item = _classify_against_ranges(item, lab_ranges)

    key = str(item.get("key") or "").lower()

    # Preserve original parser unit BEFORE any resolution
    original_parser_unit = str(
        item.get("_raw_parser_unit") or item.get("unit") or ""
    ).strip()

    # ─────────────────────────────────────────────────────────────
    # Reference range lookup
    # ─────────────────────────────────────────────────────────────
    def _find_range(k: str) -> dict | None:
        variants = [k, k.replace("_", " "), k.replace(" ", "_")]
        for v in variants:
            rng = (lab_ranges or {}).get(v)
            if rng:
                return rng
        for v in variants:
            rng = REFERENCE_RANGES.get(v)
            if rng:
                return rng
        return None

    rng_for_meta = _find_range(key)

    if item.get("normal_low") is None or item.get("normal_high") is None:
        if rng_for_meta:
            if item.get("normal_low") is None:
                item["normal_low"] = rng_for_meta.get("low")
            if item.get("normal_high") is None:
                item["normal_high"] = rng_for_meta.get("high")

    if rng_for_meta:
        if "upper_is_nominal" in rng_for_meta:
            item["upper_is_nominal"] = bool(rng_for_meta.get("upper_is_nominal"))
        if "lower_is_nominal" in rng_for_meta:
            item["lower_is_nominal"] = bool(rng_for_meta.get("lower_is_nominal"))

    # ─────────────────────────────────────────────────────────────
    # Fill unit (lab PDF wins)
    # ─────────────────────────────────────────────────────────────
    if not item.get("unit"):
        item["unit"] = _resolve_unit(item, lab_units)

    # Pretty display name
    raw_name = str(item.get("name") or "")
    pretty = _pretty_display_name(raw_name, key)
    if pretty:
        item["name"] = pretty

    # ─────────────────────────────────────────────────────────────
    # Rebuild raw display_value first
    # ─────────────────────────────────────────────────────────────
    value = item.get("value")
    unit  = item.get("unit") or ""

    if value is not None:
        num_str = f"{value:.2f}".rstrip("0").rstrip(".")
        item["display_value"] = f"{num_str} {unit}".strip()

    # ─────────────────────────────────────────────────────────────
    # DISPLAY-ONLY magnitude-aware normalization
    # ─────────────────────────────────────────────────────────────
    if value is not None and original_parser_unit:
        try:
            from tools.unit_normalizer import normalize_display_value

            norm = normalize_display_value(
                key,
                value,
                original_parser_unit   # ✅ CRITICAL FIX
            )

            if norm.was_converted and norm.value is not None:
                norm_str = f"{norm.value:.2f}".rstrip("0").rstrip(".")
                item["display_value"] = f"{norm_str} {norm.canonical_unit}".strip()
                item["unit"] = norm.canonical_unit

                # Normalize range bounds for display consistency
                if item.get("normal_low") is not None:
                    low_norm = normalize_display_value(
                        key,
                        item["normal_low"],
                        original_parser_unit
                    )
                    if low_norm.value is not None:
                        item["normal_low"] = low_norm.value

                if item.get("normal_high") is not None:
                    high_norm = normalize_display_value(
                        key,
                        item["normal_high"],
                        original_parser_unit
                    )
                    if high_norm.value is not None:
                        item["normal_high"] = high_norm.value

        except Exception:
            pass  # Display normalization is non-critical

    return item


# ── Symptoms-text value extractor ─────────────────────────────────
# Scans free-text medical history / symptoms for common biomarker
# mentions with numeric values (e.g. "BP 160/100", "glucose 220",
# "TSH 12", "SpO2 88%", "temp 102F"). Extracted values are injected
# into the same measurements pipeline as PDF-parsed values, so they
# get classified, flagged, and receive advice identically.
#
# This makes symptoms-only submissions materially more useful: users
# without lab reports can still receive personalised Findings /
# Critical Observations / Care Plan content from what they typed.
#
# Priority policy: PDF-extracted values always win. If the same
# biomarker exists in lab_result.measurements AND is mentioned in
# text, the text mention is skipped (dedup by canonical key).

# Regex patterns keyed to canonical biomarker keys. Each pattern is
# designed to be permissive about phrasing but strict about capturing
# a plausible number. Group 1 (or explicitly named group) must be the
# numeric value.

_SYMPTOM_VALUE_PATTERNS: list[tuple[str, str, _re.Pattern[str]]] = [
    # (canonical_key, display_name_hint, pattern)

    # ── Blood pressure — 2 values ("BP 160/100", "BP was 140 over 90")
    # Special-cased because it produces TWO measurements.
    (
        "bp_pair",
        "BP",
        _re.compile(
            r"(?:\b(?:bp|blood\s*pressure|b\.?p\.?)\b[^0-9]{0,20})"
            r"(?P<sys>\d{2,3})"
            r"\s*(?:/|over|on)\s*"
            r"(?P<dia>\d{2,3})",
            _re.IGNORECASE,
        ),
    ),

    # ── Heart rate / pulse ──
    (
        "heart_rate",
        "Heart Rate",
        _re.compile(
            r"(?:\b(?:heart\s*rate|pulse|hr|pulse\s*rate)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|around|about|approximately))?\s*)"
            r"(\d{2,3})(?!\s*/)"
            r"\s*(?:bpm|beats)?",
            _re.IGNORECASE,
        ),
    ),

    # ── SpO2 / oxygen saturation ──
    (
        "spo2",
        "SpO2",
        _re.compile(
            r"(?:\b(?:spo2|sp02|o2\s*sat(?:uration)?|oxygen\s*sat(?:uration)?|oxygen\s*level|oxygen)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|dropped\s*to|around|about))?\s*)"
            r"(\d{1,3})"
            r"\s*%?",
            _re.IGNORECASE,
        ),
    ),

    # ── Temperature (assume F unless "C" suffix; handled downstream) ──
    (
        "temperature",
        "Temperature",
        _re.compile(
            r"(?:\b(?:temp(?:erature)?|fever|body\s*temp)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|around|about|reached))?\s*)"
            r"(\d{2,3}(?:\.\d)?)"
            r"\s*(?:°|deg(?:rees)?)?\s*[fFcC]?",
            _re.IGNORECASE,
        ),
    ),

    # ── Respiratory rate ──
    (
        "respiratory_rate",
        "Respiratory Rate",
        _re.compile(
            r"(?:\b(?:respiratory\s*rate|breathing\s*rate|resp\s*rate|rr)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|around|about))?\s*)"
            r"(\d{1,2})",
            _re.IGNORECASE,
        ),
    ),

    # ── Glucose / sugar ──
    (
        "glucose",
        "Glucose",
        _re.compile(
            r"(?:\b(?:glucose|blood\s*glucose|blood\s*sugar|sugar|fasting\s*(?:blood\s*)?(?:sugar|glucose)|fbs|rbs|random\s*sugar)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|came\s*out\s*to|around|about|approximately|level))?\s*)"
            r"(\d{2,4})"
            r"\s*(?:mg\s*/?\s*dl)?",
            _re.IGNORECASE,
        ),
    ),

    # ── HbA1c ──
    (
        "hba1c",
        "HbA1c",
        _re.compile(
            r"(?:\b(?:hba1c|a1c|glycated\s*h(?:a?e)moglobin)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|came\s*out\s*to|around|about|level))?\s*)"
            r"(\d{1,2}(?:\.\d{1,2})?)"
            r"\s*%?",
            _re.IGNORECASE,
        ),
    ),

    # ── TSH ──
    (
        "tsh",
        "TSH",
        _re.compile(
            r"(?:\b(?:tsh|thyroid\s*stimulating\s*hormone)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|came\s*out\s*to|around|about|level))?\s*)"
            r"(\d{1,3}(?:\.\d{1,3})?)",
            _re.IGNORECASE,
        ),
    ),

    # ── Haemoglobin / Hb ──
    (
        "haemoglobin",
        "Haemoglobin",
        _re.compile(
            r"(?:\b(?:h(?:a?)?emoglobin|hb|hgb)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|around|about|level))?\s*)"
            r"(\d{1,2}(?:\.\d{1,2})?)"
            r"\s*(?:g\s*/?\s*dl)?",
            _re.IGNORECASE,
        ),
    ),

    # ── Total cholesterol ──
    (
        "total_cholesterol",
        "Total Cholesterol",
        _re.compile(
            r"(?:\b(?:total\s*cholesterol|cholesterol|serum\s*cholesterol)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|around|about|level))?\s*)"
            r"(\d{2,4})",
            _re.IGNORECASE,
        ),
    ),

    # ── LDL cholesterol ──
    (
        "ldl_cholesterol",
        "LDL Cholesterol",
        _re.compile(
            r"(?:\bldl(?:\s*cholesterol)?\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|around|about|level))?\s*)"
            r"(\d{2,4})",
            _re.IGNORECASE,
        ),
    ),

    # ── HDL cholesterol ──
    (
        "hdl_cholesterol",
        "HDL Cholesterol",
        _re.compile(
            r"(?:\bhdl(?:\s*cholesterol)?\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|around|about|level))?\s*)"
            r"(\d{2,4})",
            _re.IGNORECASE,
        ),
    ),

    # ── Triglycerides ──
    (
        "triglycerides",
        "Triglycerides",
        _re.compile(
            r"(?:\b(?:triglycerides?|tg)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|around|about|level))?\s*)"
            r"(\d{2,4})",
            _re.IGNORECASE,
        ),
    ),

    # ── Creatinine ──
    (
        "creatinine",
        "Creatinine",
        _re.compile(
            r"(?:\b(?:creatinine|creat)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|around|about|level))?\s*)"
            r"(\d{1,2}(?:\.\d{1,2})?)",
            _re.IGNORECASE,
        ),
    ),

    # ── Potassium ──
    (
        "potassium",
        "Potassium",
        _re.compile(
            r"(?:\b(?:potassium|k\+?)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|around|about|level))?\s*)"
            r"(\d{1,2}(?:\.\d{1,2})?)",
            _re.IGNORECASE,
        ),
    ),

    # ── Sodium ──
    (
        "sodium",
        "Sodium",
        _re.compile(
            r"(?:\b(?:sodium|na\+?)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|around|about|level))?\s*)"
            r"(\d{2,3})",
            _re.IGNORECASE,
        ),
    ),

    # ── Vitamin D ──
    (
        "vitamin_d",
        "Vitamin D",
        _re.compile(
            r"(?:\b(?:vitamin\s*d|vit\s*d|25\s*(?:oh|-oh)?\s*vitamin\s*d)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|around|about|level))?\s*)"
            r"(\d{1,3}(?:\.\d)?)",
            _re.IGNORECASE,
        ),
    ),

    # ── Vitamin B12 ──
    (
        "vitamin_b12",
        "Vitamin B12",
        _re.compile(
            r"(?:\b(?:vitamin\s*b\s*12|vit\s*b\s*12|b12)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|around|about|level))?\s*)"
            r"(\d{2,4})",
            _re.IGNORECASE,
        ),
    ),

    # ── CRP ──
    (
        "crp",
        "CRP",
        _re.compile(
            r"(?:\b(?:crp|c[-\s]?reactive\s*protein|hs[-\s]?crp)\b"
            r"(?:\s*(?:is|was|of|at|:|-|=|around|about|level))?\s*)"
            r"(\d{1,3}(?:\.\d{1,2})?)",
            _re.IGNORECASE,
        ),
    ),
]


def _sanity_check_value(key: str, value: float) -> bool:
    """
    Reject implausible values that would create noise. This is a
    permissive check — anything within roughly 10× a normal-range
    bound is accepted so we still catch pathological cases.
    """
    if value <= 0:
        return False

    # Rough plausibility ceilings per biomarker
    ceilings = {
        "bp_systolic":      300,   # extreme hypertensive crisis
        "bp_diastolic":     200,
        "heart_rate":       260,
        "spo2":             100,
        "temperature":      115,   # F — anything above is implausible
        "respiratory_rate": 80,
        "glucose":          1500,
        "hba1c":            25,
        "tsh":              200,
        "haemoglobin":      25,
        "total_cholesterol": 1000,
        "ldl_cholesterol":  700,
        "hdl_cholesterol":  200,
        "triglycerides":    3000,
        "creatinine":       25,
        "potassium":        12,
        "sodium":            200,
        "vitamin_d":        500,
        "vitamin_b12":      5000,
        "crp":              500,
    }
    ceiling = ceilings.get(key)
    if ceiling is not None and value > ceiling:
        return False
    return True


def _extract_measurements_from_symptoms_text(
    text: str,
    exclude_keys: set[str],
) -> list[dict]:
    """
    Extract biomarker values mentioned in free-text medical history.

    Args:
        text: The full symptoms / medical history text.
        exclude_keys: Canonical keys already present in lab measurements.
                      These are skipped so PDF-extracted values win.

    Returns:
        A list of measurement dicts in the same shape the dashboard
        extractor produces, so downstream enrichment works identically.
        Each item is marked with source="symptoms_text" for traceability.

    Notes:
        - BP mentions produce TWO items (bp_systolic + bp_diastolic).
        - Values that fail _sanity_check_value are discarded.
        - Multiple mentions of the same biomarker → first mention wins.
    """
    if not text or not isinstance(text, str):
        return []

    normalised = _re.sub(r"\s+", " ", text.strip())
    if len(normalised) < 3:
        return []

    extracted: list[dict] = []
    seen_keys: set[str] = set()

    for canonical_key, display_hint, pattern in _SYMPTOM_VALUE_PATTERNS:
        for match in pattern.finditer(normalised):
            # ── Special case: BP produces 2 measurements ──
            if canonical_key == "bp_pair":
                try:
                    sys_val = float(match.group("sys"))
                    dia_val = float(match.group("dia"))
                except (ValueError, IndexError):
                    continue

                if (
                    "bp_systolic" not in exclude_keys
                    and "bp_systolic" not in seen_keys
                    and _sanity_check_value("bp_systolic", sys_val)
                ):
                    extracted.append({
                        "key":           "bp_systolic",
                        "name":          "BP Systolic",
                        "value":         sys_val,
                        "unit":          "mmHg",
                        "display_value": f"{sys_val:g} mmHg",
                        "status":        "reported",
                        "risk_score":    None,
                        "deviation_score": None,
                        "source":        "symptoms_text",
                        "category":      None,
                        "note":          "Value mentioned in patient's medical history",
                    })
                    seen_keys.add("bp_systolic")

                if (
                    "bp_diastolic" not in exclude_keys
                    and "bp_diastolic" not in seen_keys
                    and _sanity_check_value("bp_diastolic", dia_val)
                ):
                    extracted.append({
                        "key":           "bp_diastolic",
                        "name":          "BP Diastolic",
                        "value":         dia_val,
                        "unit":          "mmHg",
                        "display_value": f"{dia_val:g} mmHg",
                        "status":        "reported",
                        "risk_score":    None,
                        "deviation_score": None,
                        "source":        "symptoms_text",
                        "category":      None,
                        "note":          "Value mentioned in patient's medical history",
                    })
                    seen_keys.add("bp_diastolic")
                # For BP we only take the first match to avoid duplicates
                break

            # ── Regular single-value biomarker ──
            if canonical_key in exclude_keys or canonical_key in seen_keys:
                break

            try:
                raw_value = match.group(1)
            except (IndexError, TypeError):
                continue

            try:
                value = float(raw_value)
            except ValueError:
                continue

            if not _sanity_check_value(canonical_key, value):
                continue

            extracted.append({
                "key":           canonical_key,
                "name":          display_hint,
                "value":         value,
                "unit":          "",
                "display_value": str(raw_value),
                "status":        "reported",
                "risk_score":    None,
                "deviation_score": None,
                "source":        "symptoms_text",
                "category":      None,
                "note":          "Value mentioned in patient's medical history",
            })
            seen_keys.add(canonical_key)
            break  # first mention wins

    if extracted:
        logger.info(
            "report_generator · extracted values from symptoms text",
            count=len(extracted),
            keys=[e["key"] for e in extracted],
        )

    return extracted



# ── Canonical key → category map (authoritative routing) ──────────
# When a biomarker resolves to a known canonical key via
# biomarker_knowledge.resolve_canonical_key(), we route it using this
# map instead of substring matching. This eliminates collisions like
# "White Blood Cell Protein Index" wrongly matching "white blood cell".
#
# Every canonical key that appears in biomarker_knowledge._DISPLAY_NAMES
# should have an entry here. Unknown canonical keys fall through to
# keyword matching.
_CANONICAL_KEY_CATEGORY: dict[str, str] = {
    # ── Vital Signs ──
    "heart_rate":         "Vital Signs",
    "spo2":               "Vital Signs",
    "temperature_c":      "Vital Signs",
    "temperature_f":      "Vital Signs",
    "temperature":        "Vital Signs",
    "respiratory_rate":   "Vital Signs",
    "bp_systolic":        "Vital Signs",
    "bp_diastolic":       "Vital Signs",
    "systolic_bp":        "Vital Signs",
    "diastolic_bp":       "Vital Signs",

    # ── Metabolic Panel ──
    "glucose":            "Metabolic Panel",
    "hba1c":              "Metabolic Panel",
    "insulin":            "Metabolic Panel",

    # ── Thyroid Panel ──
    "tsh":                "Thyroid Panel",
    "t3":                 "Thyroid Panel",
    "t4":                 "Thyroid Panel",
    "free_t3":            "Thyroid Panel",
    "free_t4":            "Thyroid Panel",
    "anti_tpo":           "Thyroid Panel",

    # ── Lipid Profile ──
    "total_cholesterol":  "Lipid Profile",
    "ldl_cholesterol":    "Lipid Profile",
    "hdl_cholesterol":    "Lipid Profile",
    "vldl_cholesterol":   "Lipid Profile",
    "non_hdl_cholesterol":"Lipid Profile",
    "triglycerides":      "Lipid Profile",
    "chol_hdl_ratio":     "Lipid Profile",
    "ldl_hdl_ratio":      "Lipid Profile",

    # ── Blood Analysis ──
    "haemoglobin":        "Blood Analysis",
    "wbc":                "Blood Analysis",
    "rbc":                "Blood Analysis",
    "platelets":          "Blood Analysis",
    "hematocrit":         "Blood Analysis",
    "mcv":                "Blood Analysis",
    "mch":                "Blood Analysis",
    "mchc":               "Blood Analysis",
    "mpv":                "Blood Analysis",
    "pdw":                "Blood Analysis",
    "rdw_cv":             "Blood Analysis",
    "rdw":                "Blood Analysis",
    "neutrophils":        "Blood Analysis",
    "lymphocytes":        "Blood Analysis",
    "monocytes":          "Blood Analysis",
    "eosinophils":        "Blood Analysis",
    "basophils":          "Blood Analysis",

    # ── Kidney & Liver Function ──
    "creatinine":         "Kidney & Liver Function",
    "urea":               "Kidney & Liver Function",
    "bun":                "Kidney & Liver Function",
    "uric_acid":          "Kidney & Liver Function",
    "sgpt_alt":           "Kidney & Liver Function",
    "sgot_ast":           "Kidney & Liver Function",
    "alp":                "Kidney & Liver Function",
    "ggt":                "Kidney & Liver Function",
    "bilirubin":          "Kidney & Liver Function",
    "bilirubin_direct":   "Kidney & Liver Function",
    "bilirubin_indirect": "Kidney & Liver Function",
    "albumin":            "Kidney & Liver Function",
    "globulin":           "Kidney & Liver Function",
    "total_protein":      "Kidney & Liver Function",
    "ag_ratio":           "Kidney & Liver Function",

    # ── Vitamins ──
    "vitamin_d":          "Vitamins",
    "vitamin_b12":        "Vitamins",
    "folate":             "Vitamins",
    "iron":               "Vitamins",
    "ferritin":           "Vitamins",
    "tibc":               "Vitamins",
    "transferrin":        "Vitamins",
    "transferrin_saturation": "Vitamins",

    # ── Electrolytes ──
    "sodium":             "Electrolytes",
    "potassium":          "Electrolytes",
    "chloride":           "Electrolytes",
    "calcium":            "Electrolytes",
    "magnesium":          "Electrolytes",
    "phosphorus":         "Electrolytes",

    # ── Cardiac Markers ──
    "troponin":           "Cardiac Markers",
    "bnp":                "Cardiac Markers",

    # ── Inflammation Markers ──
    "crp":                "Inflammation Markers",
    "esr":                "Inflammation Markers",

    # ── Coagulation ──
    "pt":                 "Coagulation",
    "inr":                "Coagulation",
    "aptt":               "Coagulation",
    "d_dimer":            "Coagulation",
    "fibrinogen":         "Coagulation",

    # ── Hormones ──
    "cortisol":           "Hormones",
    "testosterone":       "Hormones",
    "estrogen":           "Hormones",
    "progesterone":       "Hormones",
    "prolactin":          "Hormones",
    "lh":                 "Hormones",
    "fsh":                "Hormones",
    "homocysteine":       "Hormones",

    # ── Tumor Markers ──
    "psa":                "Tumor Markers",
    "ca_125":             "Tumor Markers",
    "ca_19_9":            "Tumor Markers",
    "cea":                "Tumor Markers",
    "afp":                "Tumor Markers",
    "beta_hcg":           "Tumor Markers",

    # ── Urine Analysis ──
    "urine_ph":           "Urine Analysis",
    "specific_gravity":   "Urine Analysis",
    "urine_protein":      "Urine Analysis",
    "urine_glucose":      "Urine Analysis",
    "urine_ketones":      "Urine Analysis",
    "urine_wbc":          "Urine Analysis",
    "urine_rbc":          "Urine Analysis",
    "urine_epithelial":   "Urine Analysis",
}


# Category routing for the Findings section. Mirrors the previous
# frontend groupFindingsByCategory() behaviour so the visual output
# stays identical.
_CATEGORY_KEYWORDS = [
    ("Vital Signs",              ["bp", "blood pressure", "heart rate", "pulse", "spo2", "oxygen", "temperature", "temp", "respirator"]),
    ("Metabolic Panel",          ["glucose", "sugar", "hba1c", "a1c", "insulin", "fbs", "rbs"]),
    ("Thyroid Panel",            ["tsh", "t3", "t4", "thyroid", "anti-tpo", "anti tpo"]),
    ("Lipid Profile",            ["cholesterol", "triglyceride", "hdl", "ldl", "vldl", "lipid", "chol"]),
    ("Blood Analysis",           ["hemoglobin", "haemoglobin", "platelet", "rbc", "wbc", "white blood cell", "hematocrit", "mcv", "mch", "mchc", "mpv", "pdw", "rdw", "pcv", "leucocyte", "leukocyte", "tlc", "total leucocyte", "neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"]),
    ("Kidney & Liver Function",  ["creatinine", "urea", "bun", "sgpt", "sgot", "alt", "ast", "bilirubin", "albumin", "uric", "globulin", "protein", "alp", "ggt", "egfr", "a/g"]),
    ("Vitamins",                 ["vitamin", "folate", "ferritin", "iron", "tibc", "transferrin"]),
    ("Imaging / Radiology",      ["xray", "x-ray", "consolidation", "effusion", "pneumonia", "nodule", "opacity", "infiltrat", "cardiomegaly", "pneumothorax", "edema", "atelectasis", "fracture", "mass"]),
    ("Medications & Interactions", ["drug", "interaction"]),
    ("Inflammation Markers",     ["crp", "esr"]),
    ("Cardiac Markers",          ["troponin", "bnp"]),
    ("Coagulation",              ["prothrombin", "aptt", "d-dimer", "d dimer", "inr", "fibrinogen"]),
    ("Electrolytes",             ["sodium", "potassium", "chloride", "calcium", "phosphorus", "magnesium"]),
    ("Hormones",                 ["cortisol", "testosterone", "estrogen", "progesterone", "prolactin", "homocysteine"]),
    ("Tumor Markers",            ["psa", "ca 125", "ca 19", "cea", "afp", "hcg"]),
    ("Urine Analysis",           ["urine", "specific gravity", "pus cells", "epithelial"]),
]


def _classify_category(item: dict) -> str:
    """
    Route a measurement item into a display category.

    Universal three-layer waterfall:

      1. Explicit hints (source/category) — for imaging + medications.

      2. Canonical key lookup — if the item's key resolves to a known
         canonical biomarker in biomarker_knowledge, use the
         authoritative _CANONICAL_KEY_CATEGORY map. This eliminates
         substring collisions like "White Blood Cell Protein Index"
         wrongly matching "white blood cell".

      3. Whole-word keyword fallback — for biomarkers unknown to the
         knowledge base, match _CATEGORY_KEYWORDS using WORD BOUNDARIES
         so short keywords like "k" or "mg" don't collide with longer
         words that happen to contain them.

      4. Default: "Other Clinical Findings".
    """
    # ── Layer 1: Explicit source/category hints ────────────────────
    source = str(item.get("source") or "").lower()
    category_hint = str(item.get("category") or "").lower()

    if category_hint == "medication" or source == "medication":
        return "Medications & Interactions"
    if category_hint == "imaging" or source == "imaging":
        return "Imaging / Radiology"

    # ── Layer 2: Canonical key → category (authoritative) ──────────
    # Try the item's key AND its raw name against the canonical resolver
    # so we catch both dashboard-classified items and text-extracted
    # items where the "key" may already be a canonical form.
    raw_key = str(item.get("key") or "").strip()
    raw_name = str(item.get("name") or "").strip()

    # First try the key as-is (it may already be canonical)
    if raw_key and raw_key in _CANONICAL_KEY_CATEGORY:
        return _CANONICAL_KEY_CATEGORY[raw_key]

    # Then try resolving name → canonical → category
    try:
        from tools.biomarker_knowledge import resolve_canonical_key
        for candidate in (raw_name, raw_key):
            if not candidate:
                continue
            canonical = resolve_canonical_key(candidate)
            if canonical and canonical in _CANONICAL_KEY_CATEGORY:
                return _CANONICAL_KEY_CATEGORY[canonical]
    except Exception:
        pass

    # ── Layer 3: Whole-word keyword fallback ───────────────────────
    # Use word boundaries so "vitamin" matches "Vitamin D" but NOT
    # "protein induced by vitamin K absence".
    name_lower = (raw_name or raw_key or "").lower()
    if not name_lower:
        return "Other Clinical Findings"

    for label, keywords in _CATEGORY_KEYWORDS:
        for kw in keywords:
            # Escape keyword for regex, use word boundaries
            pattern = r"\b" + _re.escape(kw) + r"\b"
            if _re.search(pattern, name_lower):
                return label

    return "Other Clinical Findings"


_STATUS_LABEL = {
    "critical_low":  "Critical Low",
    "critical_high": "Critical High",
    "critical":      "Critical",
    "high":          "High",
    "low":           "Low",
    "borderline":    "Borderline",
    "abnormal":      "Abnormal",
    "normal":        "Normal",
    "reported":      "",
}

_STATUS_COMPARATOR = {
    "critical_low":  "↓↓",
    "critical_high": "↑↑",
    "critical":      "",
    "high":          "↑",
    "low":           "↓",
    "borderline":    "≈",
    "abnormal":      "",
    "normal":        "✓",
    "reported":      "",
}


# ── Clinical priority tiers ───────────────────────────────────────
# Every list in the report is sorted by these tiers so the reader
# always sees Critical → out-of-range → Borderline → Normal → Reported.
# Lower number = higher priority (comes first).

_TIER_CRITICAL   = 0   # critical_low, critical_high, critical
_TIER_OUT_RANGE  = 1   # high, low, abnormal
_TIER_BORDERLINE = 2   # borderline (under observation)
_TIER_NORMAL     = 3   # normal
_TIER_REPORTED   = 4   # reported (no threshold available)


def _status_tier(item: dict) -> int:
    """Return the clinical priority tier for an item's status."""
    status = str(item.get("status") or "").lower()
    if status in ("critical_low", "critical_high", "critical"):
        return _TIER_CRITICAL
    if status in ("high", "low", "abnormal"):
        return _TIER_OUT_RANGE
    if status == "borderline":
        return _TIER_BORDERLINE
    if status == "normal":
        return _TIER_NORMAL
    return _TIER_REPORTED


def _item_sort_key(item: dict) -> tuple:
    """
    Universal sort key: worst clinical concern first.
    Tier ascending (critical=0 comes first), then name alphabetical.
    """
    return (_status_tier(item), (item.get("name") or "").lower())


def _format_range_string(item: dict) -> str:
    """
    Return the normal range string for display.

    Universal rule set (metadata-driven, no biomarker-specific logic):

      1. If the item's range metadata marks the upper bound as nominal
         (upper_is_nominal=True) and a lower bound exists, show "≥ low".
      2. If the lower bound is nominal (lower_is_nominal=True) and an
         upper bound exists, show "≤ high".
      3. If both bounds are meaningful, show "low–high".
      4. If only upper exists, show "<high".
      5. If only lower exists, show ">low".
      6. If nothing available, return empty string.

    The nominal flags are optional metadata on REFERENCE_RANGES entries.
    Any biomarker whose upper (or lower) reference bound is artificial
    or clinically non-meaningful can be tagged in lab_thresholds.py
    without any code change here.

    Classification is UNAFFECTED — the flags only change display.
    """
    low = item.get("normal_low")
    high = item.get("normal_high")
    upper_nominal = bool(item.get("upper_is_nominal"))
    lower_nominal = bool(item.get("lower_is_nominal"))

    # Rule 1: upper nominal → show as lower threshold only
    if upper_nominal and not lower_nominal and low is not None:
        return f"≥ {low}"

    # Rule 2: lower nominal → show as upper threshold only
    if lower_nominal and not upper_nominal and high is not None:
        return f"≤ {high}"

    # Rule 3: standard two-sided range
    if low is not None and high is not None:
        return f"{low}–{high}"

    # Rule 4: upper-only
    if high is not None:
        return f"<{high}"

    # Rule 5: lower-only
    if low is not None:
        return f">{low}"

    # Rule 6: no bounds
    return ""


def _render_vital_line(item: dict) -> str:
    """
    Render one measurement line matching the frontend's vital-line HTML.

    Special-cases X-ray / imaging findings (no numeric value) so they
    still appear as clearly-flagged lines in the Findings section.
    """
    name = _escape_html(item.get("name") or item.get("key") or "")
    display_value = str(item.get("display_value") or "").strip()
    raw_unit = str(item.get("unit") or "")
    unit = _escape_html(raw_unit)
    value = item.get("value")
    source = str(item.get("source") or "").lower()
    category_hint = str(item.get("category") or "").lower()

    # X-ray / imaging findings: no numeric value; render with status only
    if source == "imaging" or category_hint == "imaging":
        status = str(item.get("status") or "").lower()
        status_label = _STATUS_LABEL.get(status, "Detected")
        comparator = _STATUS_COMPARATOR.get(status, "")
        comp_prefix = f"{_escape_html(comparator)} " if comparator else ""
        return (
            f'<li class="vital-line"><span class="vital-name">{name}</span>'
            f'<span class="vital-sep">•</span>'
            f'<span class="vital-status">{comp_prefix}{_escape_html(status_label)}</span>'
            f'<span class="vital-sep">•</span>'
            f'<span class="vital-range">Detected on X-ray imaging</span></li>'
        )

    if value is None or display_value == "":
        return f'<li class="vital-line-plain"><span class="vital-name">{name}:</span>&nbsp;&nbsp;{_escape_html(display_value)}</li>'

    # Use the already-normalized display_value for the numeric portion
    # (NOT the raw item["value"]). _enrich_measurement's magnitude/unit
    # conversion via normalize_display_value() is DISPLAY-ONLY and never
    # mutates item["value"] (by design — raw values must stay untouched
    # for threshold comparisons elsewhere). Rebuilding the number from
    # `value` here silently discards that conversion — e.g. platelets
    # would show the raw 147000 instead of the converted 147.
    # display_value is "{number} {unit}"; strip the trailing raw unit
    # (if present) to recover just the number.
    numeric_str = display_value
    if raw_unit and numeric_str.endswith(raw_unit):
        numeric_str = numeric_str[: -len(raw_unit)].strip()
    if not numeric_str:
        numeric_str = f"{value:.2f}".rstrip("0").rstrip(".")

    status = str(item.get("status") or "").lower()
    status_label = _STATUS_LABEL.get(status, "")
    comparator = _STATUS_COMPARATOR.get(status, "")

    unit_part = f'&nbsp;<span class="vital-unit">{unit}</span>' if unit else ""

    status_part = ""
    if status_label:
        comp_prefix = f"{_escape_html(comparator)} " if comparator else ""
        status_part = f'<span class="vital-sep">•</span><span class="vital-status">{comp_prefix}{_escape_html(status_label)}</span>'

    range_str = _format_range_string(item)
    range_part = ""
    if range_str:
        rng_unit = f" {unit}" if unit else ""
        range_part = f'<span class="vital-sep">•</span><span class="vital-range">Normal {_escape_html(range_str)}{rng_unit}</span>'

    return (
        f'<li class="vital-line"><span class="vital-name">{name}:</span>&nbsp;&nbsp;'
        f'<span class="vital-value">{_escape_html(numeric_str)}</span>{unit_part}{status_part}{range_part}</li>'
    )


def _collect_all_measurements(result_data_like: dict) -> list[dict]:
    """
    Use dashboard's extractor as the base source, then enrich each item
    with units/ranges from the actual lab report PDF, and append X-ray
    findings (both AI-detected and user-selected) as flagged items so
    they participate in Findings, Critical Observations, Personalized
    Recommendations, and Care Plan.

    Also extracts biomarker mentions from the patient's typed symptoms /
    medical history text (e.g. "BP 160/100, TSH 12") so symptoms-only
    submissions still receive personalised findings and care plans.

    Final list is sorted by clinical priority tier so the report
    always reads Critical → out-of-range → Borderline → Normal.
    """
    try:
        from backend.dashboard import _extract_report_measurements
    except Exception as exc:
        logger.warning("report_generator · dashboard extractor unavailable", error=str(exc))
        return []
    try:
        raw = _extract_report_measurements(result_data_like) or []
    except Exception:
        logger.exception("report_generator · dashboard extractor raised")
        raw = []

    # Pull units and ranges from the actual lab report PDF
    lab_result = result_data_like.get("lab_result") or {}
    if isinstance(lab_result, dict):
        lab_units = lab_result.get("units") or {}
        lab_ranges = lab_result.get("reference_ranges") or {}
    else:
        lab_units = {}
        lab_ranges = {}

    # Stamp each item with the original PDF unit BEFORE enrichment
    # so _enrich_measurement can normalize display values correctly
    # even when the dashboard extractor has already set a canonical unit.
    for it in raw:
        key = str(it.get("key") or "").lower()
        if not it.get("_raw_parser_unit"):
            it["_raw_parser_unit"] = str(
                lab_units.get(key) or it.get("unit") or ""
            ).strip()

    enriched = [_enrich_measurement(it, lab_units, lab_ranges) for it in raw]

    # ── Extract biomarker mentions from typed symptoms/medical history ──
    # PDF-extracted values always win; text mentions are only added for
    # biomarkers that don't already appear in the dashboard extraction.
    existing_keys = {
        str(item.get("key") or "").lower() for item in enriched if item.get("key")
    }
    submitted_for_text = result_data_like.get("submitted") or {}
    symptoms_text = ""
    if isinstance(submitted_for_text, dict):
        symptoms_text = str(submitted_for_text.get("symptoms_text") or "").strip()

    if symptoms_text:
        text_items = _extract_measurements_from_symptoms_text(
            symptoms_text, existing_keys
        )
        if text_items:
            # Enrich text-extracted items through the same pipeline so they
            # get proper units, ranges, classification, and display names.
            enriched.extend(
                _enrich_measurement(it, lab_units, lab_ranges) for it in text_items
            )

    # Track imaging items we've already added (avoid dupes between
    # AI-detected findings and user-selected checkboxes).
    seen_xray_keys: set[str] = set()

    # Append AI-detected X-ray findings (tier: Critical — highest priority)
    xray_result = result_data_like.get("xray_result") or {}
    if isinstance(xray_result, dict):
        xray_findings = xray_result.get("findings") or []
        for finding in xray_findings:
            if not finding:
                continue
            lowered = finding.lower()
            if lowered in ("normal", "normal / no significant findings"):
                continue
            finding_key = _re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
            if finding_key in seen_xray_keys:
                continue
            seen_xray_keys.add(finding_key)
            enriched.append({
                "key": finding_key,
                "name": _pretty_display_name(finding, finding_key) or finding,
                "value": None,
                "unit": "",
                "display_value": finding,
                "status": "critical",  # X-ray findings are top clinical priority
                "risk_score": 2,
                "deviation_score": None,
                "source": "imaging",
                "category": "imaging",
                "note": f"Detected on chest X-ray: {finding}",
            })

    # Also include user-selected X-ray findings from the medical form
    submitted = result_data_like.get("submitted") or {}
    if isinstance(submitted, dict):
        user_xray = submitted.get("xray_findings") or []
        for finding in user_xray:
            if not finding:
                continue
            lowered = finding.lower()
            if lowered in ("normal", "normal / no significant findings"):
                continue
            finding_key = _re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
            if finding_key in seen_xray_keys:
                continue
            seen_xray_keys.add(finding_key)
            enriched.append({
                "key": finding_key,
                "name": _pretty_display_name(finding, finding_key) or finding,
                "value": None,
                "unit": "",
                "display_value": finding,
                "status": "critical",  # X-ray findings are top clinical priority
                "risk_score": 2,
                "deviation_score": None,
                "source": "imaging",
                "category": "imaging",
                "note": f"User-selected X-ray finding: {finding}",
            })

    # Sort by clinical priority tier, then alphabetical within tier.
    enriched.sort(key=_item_sort_key)

    return enriched


def _is_flagged(item: dict) -> bool:
    """Anything that is not Normal and not just Reported is 'flagged'."""
    return _status_tier(item) <= _TIER_BORDERLINE


def _build_result_data_like(state: AegisState) -> dict:
    """
    Build the same dict shape /queue/result/{job_id} would return, but
    only the fields the dashboard extractor needs. Kept local so we do
    not depend on API-layer wiring during pipeline execution.
    """
    submitted = {
        "symptoms_text": (
            getattr(state, "submitted_symptoms_text", None)
            or getattr(state, "raw_symptoms_text", None)
        ),
        "medications": list(getattr(state, "medications_raw", []) or []),
        "xray_findings": list(getattr(state, "xray_findings_raw", []) or []),
        "xray_free_text": getattr(state, "xray_free_text_raw", None),
        "lab_pdf_uploaded": bool(getattr(state, "lab_pdf_path", None)),
        "xray_image_uploaded": bool(getattr(state, "xray_image_path", None)),
        "audio_uploaded": bool(getattr(state, "audio_file_path", None)),
    }
    patient = {
        "name": getattr(state, "patient_name", None),
        "dob": getattr(state, "patient_dob", None),
        "sex": getattr(state, "patient_sex", None),
        "blood_group": getattr(state, "patient_blood_group", None),
        "weight_kg": getattr(state, "patient_weight_kg", None),
        "height_cm": getattr(state, "patient_height_cm", None),
        "allergies": getattr(state, "patient_allergies", None),
        "medical_conditions": list(getattr(state, "patient_medical_conditions", []) or []),
    }

    lab = getattr(state, "lab_result", None)
    lab_dict = lab.model_dump(mode="json") if (lab and not isinstance(lab, ToolError)) else None

    xray = getattr(state, "xray_result", None)
    xray_dict = xray.model_dump(mode="json") if (xray and not isinstance(xray, ToolError)) else None

    return {
        "submitted": submitted,
        "patient": patient,
        "lab_result": lab_dict,
        "xray_result": xray_dict,
    }


def _category_max_tier(items: list[dict]) -> int:
    """Highest-priority (lowest-number) tier present in a group."""
    return min((_status_tier(i) for i in items), default=_TIER_REPORTED)


def _render_text_findings_html(text_findings: list[str]) -> str:
    """
    Render the "Peripheral Smear & Morphology" subsection body from
    the lab report's interpretive text findings.
    """
    if not text_findings:
        return ""

    parts: list[str] = []
    parts.append('<h4 class="findings-heading">Peripheral Smear & Morphology</h4>')
    parts.append('<ul class="findings-list">')
    for finding in text_findings:
        # Split at the first colon so we can bold the label
        if ":" in finding:
            label, _, content = finding.partition(":")
            label = label.strip()
            content = content.strip()
            parts.append(
                f'<li class="vital-line-plain">'
                f'<span class="vital-name">{_escape_html(label)}:</span>&nbsp;&nbsp;'
                f'{_escape_html(content)}</li>'
            )
        else:
            parts.append(
                f'<li class="vital-line-plain">{_escape_html(finding)}</li>')
    parts.append('</ul>')
    return "".join(parts)


def _render_findings_html(
    all_items: list[dict],
    text_findings: list[str] | None = None,
) -> str:
    """
    Build the full Findings HTML block:
      - Flagged Vitals Snapshot at the top (Critical → Out-of-range → Borderline)
      - Per-category groups ordered by their worst tier
      - Items within each category also ordered by tier then name
      - Every measurement is shown (not just flagged ones)
      - Peripheral Smear & Morphology subsection at the end (from text_findings)
    """
    if not all_items and not text_findings:
        return ""

    html_parts: list[str] = []

    if all_items:
        # Group by category
        grouped: dict[str, list[dict]] = {}
        for item in all_items:
            cat = _classify_category(item)
            grouped.setdefault(cat, []).append(item)

        # Sort items within each category by clinical priority tier
        for cat in grouped:
            grouped[cat].sort(key=_item_sort_key)

        # Sort categories by their worst-tier item (categories with critical
        # findings appear first)
        order_within_tier = [
            "Vital Signs", "Cardiac Markers", "Metabolic Panel", "Thyroid Panel",
            "Lipid Profile", "Blood Analysis", "Kidney & Liver Function",
            "Vitamins", "Electrolytes", "Inflammation Markers", "Coagulation",
            "Hormones", "Tumor Markers", "Urine Analysis",
            "Imaging / Radiology", "Medications & Interactions",
            "Other Clinical Findings",
        ]
        ordered = [c for c in order_within_tier if c in grouped]
        ordered.sort(key=lambda c: _category_max_tier(grouped[c]))

                # Flagged snapshot — Critical → Out-of-range → Borderline
        # Universal UX pattern: this is a SUMMARY view. The same items
        # also appear below within their respective clinical categories
        # (e.g. Vitamins, Blood Analysis, Thyroid Panel) with full
        # context. A one-line hint below the header prevents readers
        # from perceiving the intentional duplication as a bug.
        flagged = [it for it in all_items if _is_flagged(it)]
        flagged.sort(key=_item_sort_key)
        if flagged:
            html_parts.append('<h4 class="findings-heading">Flagged Vitals Snapshot</h4>')
            html_parts.append(
                '<p class="findings-hint" style="font-size:0.9em;color:#666;'
                'margin:0 0 8px 0;font-style:italic;">'
                'Summary of all flagged results — full details appear below '
                'within each clinical category.'
                '</p>'
            )
            html_parts.append('<ul class="findings-list">')
            for it in flagged:
                html_parts.append(_render_vital_line(it))
            html_parts.append('</ul>')

        # Every category with every item
        for cat in ordered:
            html_parts.append(f'<h4 class="findings-heading">{_escape_html(cat)}</h4>')
            html_parts.append('<ul class="findings-list">')
            for it in grouped[cat]:
                html_parts.append(_render_vital_line(it))
            html_parts.append('</ul>')

    # Peripheral Smear & Morphology at the end
    if text_findings:
        html_parts.append(_render_text_findings_html(text_findings))

    return "".join(html_parts)


def _build_interpretation(item: dict) -> str:
    """One-liner interpretation for the Critical Observations section."""
    name = _escape_html(item.get("name") or item.get("key") or "")
    status = str(item.get("status") or "").lower()
    display_value = _escape_html(item.get("display_value") or "")
    source = str(item.get("source") or "").lower()
    category_hint = str(item.get("category") or "").lower()

    # X-ray items: no numeric value, phrase it as an imaging finding
    if source == "imaging" or category_hint == "imaging":
        return f'<span class="vital-name">{name}</span> was <span class="vital-status-word">DETECTED</span> on chest X-ray imaging and requires clinical review.'

    if status in {"critical_low", "critical_high", "critical"}:
        return f'<span class="vital-name">{name}</span> is at a <span class="vital-status-word">CRITICAL</span> level of {display_value}.'
    if status == "high":
        return f'<span class="vital-name">{name}</span> is <span class="vital-status-word">HIGHER</span> than the normal range at {display_value}.'
    if status == "low":
        return f'<span class="vital-name">{name}</span> is <span class="vital-status-word">LOWER</span> than the normal range at {display_value}.'
    if status == "borderline":
        return f'<span class="vital-name">{name}</span> is at a <span class="vital-status-word">BORDERLINE</span> value of {display_value}.'
    return f'<span class="vital-name">{name}</span>: {display_value}'


def _render_critical_observations_html(
    flagged: list[dict],
    extra_observations: list[str] | None = None,
) -> str:
    """
    Render observations in strict tier order: Critical → Out-of-range → Borderline.

    Extra observations (from text finding analyzer) are appended after
    the biomarker-derived observations, in a dedicated "Peripheral
    Smear & Interpretive Findings" sub-block so readers can distinguish
    them from the numeric findings.

    Density control: numeric observations cap at 8 items with a summary
    line for the remainder. Extra observations render in full because
    they represent interpretive clinical patterns (typically 0–5 items).
    """
    if not flagged and not extra_observations:
        return ""

    ordered = sorted(flagged, key=_item_sort_key)

    max_items = 8
    if len(ordered) <= max_items:
        display = ordered
        remainder = 0
    else:
        display = ordered[:max_items]
        remainder = len(ordered) - max_items

    parts: list[str] = []

    # Numeric biomarker observations
    if display:
        parts.append("<ul>")
        for it in display:
            parts.append(f'<li class="vital-line-plain">{_build_interpretation(it)}</li>')

        if remainder > 0:
            remainder_names = [
                str(it.get("name") or it.get("key") or "").strip()
                for it in ordered[max_items:]
            ]
            remainder_names = [n for n in remainder_names if n]
            names_joined = ", ".join(remainder_names)
            parts.append(
                f'<li class="vital-line-plain">'
                f'<span class="vital-name">And {remainder} additional flagged finding'
                f'{"s" if remainder != 1 else ""}</span> — {_escape_html(names_joined)} — '
                f'see the Findings section above for full details.</li>'
            )
        parts.append("</ul>")

    # Text finding observations (from FIX #3 analyzer)
    if extra_observations:
        parts.append(
            '<p class="findings-hint" style="font-size:0.9em;color:#666;'
            'margin:8px 0 4px 0;font-style:italic;">'
            'Additional interpretive findings from the peripheral smear '
            'and clinical impressions:'
            '</p>'
        )
        parts.append("<ul>")
        for obs in extra_observations:
            parts.append(f'<li class="vital-line-plain">{_escape_html(obs)}</li>')
        parts.append("</ul>")

    return "".join(parts)


# ── Advice classification (knowledge vs. fallback) ────────────────
# A flagged item is either "known" (has a curated BIOMARKER_KNOWLEDGE
# entry) or "unknown" (falls through to the universal fallback). We
# render one knowledge line per known item in Personalized Recs, but
# consolidate unknown items into a single "share with your provider"
# line per bucket in the Care Plan so the report doesn't flood with
# 15 near-identical "Share the X result" / "Ask about repeat X" lines.


def _resolve_and_classify(
    item: dict,
    patient_context: dict | None = None,
) -> tuple[dict, bool]:
    """
    Return (advice_dict, is_known). is_known=True when advice came
    from BIOMARKER_KNOWLEDGE, False when from universal fallback.

    If patient_context is provided (from FIX #4 adapter), the advice
    is adapted for patient allergies, conditions, and other context
    before being returned. The adaptation is fail-safe: any error
    returns the original unadapted advice.
    """
    advice = resolve_advice(item)
    is_known = str(advice.get("source") or "").lower() == "knowledge"

    if patient_context:
        try:
            from tools.patient_context_adapter import adapt_advice_for_patient
            advice = adapt_advice_for_patient(advice, patient_context, item)
        except Exception:
            logger.exception(
                "report_generator · patient context adaptation failed; "
                "using unadapted advice"
            )

    return advice, is_known


def _render_recommendations_html(
    flagged: list[dict],
    patient_context: dict | None = None,
    clinical_narratives: list[str] | None = None,
    clinical_picture_html: str = "",
) -> str:
    """
    Render Personalized Recommendations (Section 6 blocks 2 and 3).

    Visual hierarchy:
      Block 2 — Clinical Pattern Summary   SECONDARY emphasis
                 Collapsed by default (typically 2-6 lines)
                 Deduplicated against Block 1 narratives
      Block 3 — Biomarker-Specific Recs    TERTIARY emphasis
                 First 6 lines visible, remainder collapsible

    Deduplication:
      Block 2 narratives that already appear verbatim in the
      clinical_picture_html (Block 1) are suppressed here so the
      same finding is never stated twice in Section 6.

    Fail-safe:
      - If <details> is not supported by PDF engine, content still
        renders because we never put content ONLY inside <details>;
        the element degrades to expanded.
      - Any exception returns empty string.

    Args:
        flagged:               Flagged measurement items (sorted).
        patient_context:       From normalize_patient_context().
        clinical_narratives:   Narrative strings from clinical_synthesis.
        clinical_picture_html: Rendered HTML of Block 1 — used to
                               extract already-surfaced narratives for
                               deduplication. Empty string = no dedup.
    """
    try:
        if not flagged and not clinical_narratives:
            return ""

        # ── Extract narratives already surfaced in Block 1 ────────────
        # Dedup uses both exact match and token-overlap (≥70%) so
        # paraphrased versions of the same finding are suppressed.
        block1_narratives: set[str] = set()
        block1_tokens: list[set[str]] = []

        _NARRATIVE_STOP = frozenset({
            "a", "an", "the", "and", "or", "of", "in", "on", "is",
            "are", "was", "with", "for", "to", "that", "this", "your",
            "may", "be", "as", "by", "at", "from", "it", "its",
        })

        def _narrative_tokens(text: str) -> set[str]:
            words = _re.findall(r"[a-zA-Z]+", text.lower())
            return {w for w in words if w not in _NARRATIVE_STOP and len(w) > 2}

        def _narrative_already_surfaced(candidate: str) -> bool:
            """True if candidate matches or strongly overlaps a Block 1 narrative."""
            cand_lower = candidate.strip().lower()
            if cand_lower in block1_narratives:
                return True
            cand_tokens = _narrative_tokens(candidate)
            if not cand_tokens:
                return False
            for b1_tokens in block1_tokens:
                if not b1_tokens:
                    continue
                smaller = min(len(cand_tokens), len(b1_tokens))
                if smaller == 0:
                    continue
                overlap = len(cand_tokens & b1_tokens) / smaller
                if overlap >= 0.70:
                    return True
            return False

        if clinical_picture_html:
            attr_match = _re.search(
                r"data-surfaced-narratives='([^']*)'",
                clinical_picture_html,
            )
            if attr_match:
                try:
                    import json as _json_inner
                    parsed = _json_inner.loads(
                        attr_match.group(1).replace("&#39;", "'")
                    )
                    for n in parsed:
                        if n:
                            block1_narratives.add(str(n).strip().lower())
                            block1_tokens.append(_narrative_tokens(str(n)))
                except Exception:
                    pass

        parts: list[str] = []

        # ════════════════════════════════════════════════════════════
        # BLOCK 2: Clinical Pattern Summary (secondary emphasis)
        # ════════════════════════════════════════════════════════════
        if clinical_narratives:
            # Dedupe within Block 2 itself (preserve order)
            seen_b2: set[str] = set()
            unique_b2: list[str] = []
            for n in clinical_narratives:
                key = str(n).strip().lower()
                if not key:
                    continue
                if key in seen_b2:
                    continue
                # Suppress if already surfaced in Block 1 (exact or overlap)
                if _narrative_already_surfaced(str(n).strip()):
                    continue
                seen_b2.add(key)
                unique_b2.append(str(n).strip())

            if unique_b2:
                n_b2 = len(unique_b2)
                b2_label = (
                    f'Additional Clinical Patterns ({n_b2})'
                )

                # Collapsed by default — secondary role
                # Auto-expand when only 1-2 patterns (hiding 1-2 lines
                # is more annoying than helpful)
                use_open = n_b2 <= 2

                parts.append(
                    '<div class="clinical-pattern-summary" '
                    'style="margin:0 0 12px 0;">'
                )

                if use_open:
                    parts.append(
                        '<details open style="margin:8px 0 6px 0;">'
                    )
                else:
                    parts.append(
                        '<details style="margin:8px 0 6px 0;">'
                    )

                parts.append(
                    f'<summary style="cursor:pointer;font-weight:600;'
                    f'color:#1565c0;font-size:0.92em;list-style:none;'
                    f'padding:2px 0;">'
                    f'📋 {_escape_html(b2_label)}'
                    f'</summary>'
                )
                parts.append(
                    '<ul style="margin:8px 0 0 0;padding-left:20px;">'
                )
                for narrative in unique_b2:
                    parts.append(
                        f'<li style="margin-bottom:5px;color:#37474f;'
                        f'line-height:1.5;font-size:0.95em;">'
                        f'{_escape_html(narrative)}</li>'
                    )
                parts.append('</ul>')
                parts.append('</details>')
                parts.append('</div>')

        # ════════════════════════════════════════════════════════════
        # BLOCK 3: Biomarker-Specific Recommendations (tertiary)
        # ════════════════════════════════════════════════════════════
        if flagged:
            ordered = sorted(flagged, key=_item_sort_key)

            known_lines:        list[str] = []
            known_source_names: list[str] = []
            seen_known:         set[str]  = set()
            unknown_names:      list[str] = []
            seen_unknown:       set[str]  = set()

            for it in ordered:
                advice, is_known = _resolve_and_classify(it, patient_context)
                rec  = (advice.get("recommendation") or "").strip()
                name = str(it.get("name") or it.get("key") or "").strip()

                if is_known and rec:
                    if rec not in seen_known:
                        seen_known.add(rec)
                        known_lines.append(rec)
                        known_source_names.append(name)
                elif name:
                    if name.lower() not in seen_unknown:
                        seen_unknown.add(name.lower())
                        unknown_names.append(name)

            # ── Density cap: first 6 visible, remainder collapsible ───
            _VISIBLE_CAP = 6
            visible_lines:    list[str] = known_lines[:_VISIBLE_CAP]
            remainder_lines:  list[str] = known_lines[_VISIBLE_CAP:]
            remainder_names:  list[str] = known_source_names[_VISIBLE_CAP:]

            # Unknown-item consolidated line (always in visible block)
            visible_extra: list[str] = []
            if unknown_names:
                joined = ", ".join(unknown_names)
                visible_extra.append(
                    f"The following results also fall outside the reference "
                    f"range and should be reviewed by your physician for "
                    f"personalised guidance: {joined}."
                )

            # Closing line (always visible)
            closing = (
                "Bring this report and all flagged values to your clinician "
                "at your next visit for personalised assessment."
            )

            # Build visible block
            b3_visible: list[str] = list(visible_lines) + visible_extra
            if closing not in seen_known:
                b3_visible.append(closing)

            # Build remainder block (collapsible)
            b3_remainder: list[str] = list(remainder_lines)

            # Only render Block 3 if there is something to show
            if b3_visible or b3_remainder:
                parts.append(
                    '<div class="biomarker-recs" '
                    'style="margin:0 0 8px 0;">'
                )

                # Section heading (tertiary — smaller, plain)
                if clinical_narratives:
                    # Only add heading if Block 2 already rendered
                    # (otherwise Block 3 is the only block and needs
                    # no additional label — the section title suffices)
                    parts.append(
                        '<p style="font-size:0.90em;color:#546e7a;'
                        'font-weight:600;margin:0 0 6px 0;">'
                        'Biomarker-Specific Recommendations'
                        '</p>'
                    )

                # Visible lines (always shown)
                if b3_visible:
                    parts.append(
                        '<ul style="margin:0 0 6px 0;padding-left:20px;">'
                    )
                    for rec in b3_visible:
                        parts.append(
                            f'<li style="margin-bottom:5px;color:#37474f;'
                            f'line-height:1.5;font-size:0.94em;">'
                            f'{_escape_html(rec)}</li>'
                        )
                    parts.append('</ul>')

                # Remainder lines (collapsible)
                if b3_remainder:
                    n_rem = len(b3_remainder)
                    parts.append(
                        f'<details style="margin-top:4px;">'
                        f'<summary style="cursor:pointer;font-size:0.88em;'
                        f'color:#607d8b;font-weight:600;list-style:none;'
                        f'padding:3px 0;">'
                        f'Show {n_rem} more recommendation'
                        f'{"s" if n_rem != 1 else ""} '
                        f'({", ".join(remainder_names[:3])}'
                        f'{"..." if len(remainder_names) > 3 else ""})'
                        f'</summary>'
                    )
                    parts.append(
                        '<ul style="margin:6px 0 0 0;padding-left:20px;">'
                    )
                    for rec in b3_remainder:
                        parts.append(
                            f'<li style="margin-bottom:5px;color:#37474f;'
                            f'line-height:1.5;font-size:0.94em;">'
                            f'{_escape_html(rec)}</li>'
                        )
                    parts.append('</ul>')
                    parts.append('</details>')

                parts.append('</div>')

        if not parts:
            return ""

        return "".join(parts)

    except Exception:
        logger.exception(
            "report_generator · _render_recommendations_html failed; "
            "returning empty"
        )
        return ""


# ── Care Plan bucket density controls ─────────────────────────────
# Two helpers to make Care Plan buckets less dense:
#   1. _dedupe_bucket — merges near-identical advice lines using a
#      token-overlap heuristic. If two lines share ≥ 60% of their
#      significant words, keep only the more informative (longer) one.
#   2. _cap_bucket   — limits each bucket to a maximum of N items,
#      appending a summary line when more exist.

# Words considered "insignificant" for the token-overlap heuristic
# (articles, prepositions, common verbs that appear in most advice).
# Medical domain terms added so bucket lines like
# "Complete metabolic panel" and "Complete lipid panel" are
# correctly identified as near-duplicates of each other.
_STOP_WORDS: frozenset[str] = frozenset({
    # ── English function words ─────────────────────────────────────
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to",
    "for", "with", "without", "by", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "should", "could", "may", "might", "must",
    "can", "your", "you", "please", "if", "when", "then", "than",
    "any", "some", "all", "each", "every", "no", "not",
    # ── Common clinical action verbs ──────────────────────────────
    "consider", "consult", "discuss", "check", "recheck", "repeat",
    "monitor", "review", "evaluate", "avoid", "maintain", "ensure",
    "recommend", "recommended", "advised", "advice", "complete",
    "schedule", "arrange", "book", "refer", "follow", "start",
    "continue", "track", "seek", "obtain", "request", "order",
    # ── Generic clinical nouns (present in almost every line) ─────
    "test", "testing", "workup", "results", "result",
    "physician", "doctor", "clinician", "provider", "healthcare",
    "care", "management", "medical", "clinical", "specialist",
    "additional", "further", "next", "specific", "personal",
    "visit", "appointment", "consultation",
    # ── Time qualifiers ───────────────────────────────────────────
    "weeks", "week", "months", "month", "days", "day", "years", "year",
    # ── NEW: Medical domain stopwords ─────────────────────────────
    # These appear in almost every care-plan line and carry no
    # discriminating information for overlap detection.
    "panel",        # "Complete metabolic panel" ≈ "Complete lipid panel"
    "level",        # "Check vitamin D level" ≈ "Check TSH level"
    "levels",       # same
    "count",        # "Recheck platelet count" ≈ "Recheck WBC count"
    "study",        # "Complete bone density study" ≈ "Complete sleep study"
    "studies",      # same
    "assessment",   # "Schedule cardiovascular assessment" ≈ "Schedule renal assessment"
    "screening",    # "Complete diabetes screening" ≈ "Complete lipid screening"
    "function",     # "Review kidney function" ≈ "Review liver function"
    "profile",      # "Check lipid profile" ≈ "Check thyroid profile"
    "serum",        # "Check serum iron" ≈ "Check serum B12"
    "blood",        # "Complete blood count" — "blood" adds no signal
    "full",         # "Complete full iron panel" — "full" adds no signal
    "baseline",     # "Establish baseline" appears in many lines
    "ongoing",      # "Ongoing monitoring" — too generic
    "routine",      # "Schedule routine follow-up"
    "current",      # "Review current medications"
    "based",        # "Based on results"
    "given",        # "Given your condition"
    "per",          # "Per specialist guidance"
    "after",        # "After starting treatment"
    "before",       # "Before any prescription"
    "within",       # "Within 1-2 weeks"
    "interval",     # "At regular intervals"
    "retest",       # synonym of recheck — adds no signal
    "recheck",      # already listed above, belt-and-braces
})


def _tokenize_advice(text: str) -> set[str]:
    """Extract significant word tokens for overlap comparison."""
    words = _re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def _dedupe_bucket(lines: list[str], overlap_threshold: float = 0.5) -> list[str]:
    """
    Merge near-identical bucket lines by token overlap.

    Threshold reduced from 0.6 → 0.5 so lines that share half their
    significant words are treated as duplicates. This is appropriate
    now that the expanded _STOP_WORDS strips generic medical terms
    (panel, level, count, etc.), leaving only the truly discriminating
    tokens to compare.

    If line B's significant tokens are a ≥50% subset of line A's tokens
    (or vice-versa), the shorter one is dropped. Preserves the longer,
    more informative line so nothing meaningful is lost.
    """
    if not lines or len(lines) < 2:
        return list(lines)

    kept: list[tuple[str, set[str]]] = []

    for line in lines:
        tokens = _tokenize_advice(line)
        if not tokens:
            # No significant tokens — always keep (safe fallback)
            kept.append((line, tokens))
            continue

        merge_index = -1
        replace_existing = False

        for idx, (existing_line, existing_tokens) in enumerate(kept):
            if not existing_tokens:
                continue
            # Compute overlap in both directions
            smaller = min(len(tokens), len(existing_tokens))
            if smaller == 0:
                continue
            overlap = len(tokens & existing_tokens) / smaller

            if overlap >= overlap_threshold:
                # Near-duplicate — decide which to keep
                merge_index = idx
                # Keep the longer (more informative) line
                if len(line) > len(existing_line):
                    replace_existing = True
                break

        if merge_index == -1:
            kept.append((line, tokens))
        elif replace_existing:
            kept[merge_index] = (line, tokens)
        # else: drop this line, keep the existing longer one

    return [line for line, _ in kept]


def _cap_bucket(lines: list[str], max_items: int = 5) -> list[str]:
    """
    Limit a bucket to max_items entries. If more exist, append a summary
    line noting how many were consolidated.

    Default reduced from 6 → 5 to improve care plan density.
    """
    if len(lines) <= max_items:
        return list(lines)

    kept = lines[:max_items]
    remainder = len(lines) - max_items
    kept.append(
        f"And {remainder} additional related clinical action{'s' if remainder != 1 else ''} — "
        f"please review the full findings with your physician."
    )
    return kept

# ── Care plan action clustering ───────────────────────────────────
# Groups care-plan lines that share a common action verb and a common
# object suffix (e.g. "panel", "test", "count") into one merged line.
#
# Structure:
#   verb_pattern:  regex that matches the action verb at start of line
#   object_suffix: word that must appear in the line to qualify
#   merge_template: f-string template; {items} = joined object list
#
# Registry-driven: add a new entry to add a new cluster type.
# No biomarker-specific logic — purely structural / grammatical.

_CLUSTER_RULES: list[dict] = [
    {
        "id":            "complete_panels",
        "verb_pattern":  _re.compile(r"(?i)^complete\b"),
        "object_suffix": "panel",
        "merge_template": "Complete {items} panels as clinically indicated.",
    },
    {
        "id":            "recheck_counts",
        "verb_pattern":  _re.compile(r"(?i)^(recheck|repeat)\b"),
        "object_suffix": "count",
        "merge_template": "Recheck {items} counts as clinically indicated.",
    },
    {
        "id":            "recheck_levels",
        "verb_pattern":  _re.compile(r"(?i)^(recheck|repeat)\b"),
        "object_suffix": "level",
        "merge_template": "Recheck {items} levels as clinically indicated.",
    },
    {
        "id":            "schedule_studies",
        "verb_pattern":  _re.compile(r"(?i)^(schedule|arrange|book)\b"),
        "object_suffix": "study",
        "merge_template": "Schedule {items} studies as clinically indicated.",
    },
]


def _cluster_bucket(lines: list[str]) -> list[str]:
    """
    Cluster care-plan lines that share a common action verb and object
    suffix pattern into single merged lines before deduplication and
    capping.

    Example:
        ["Complete metabolic panel.", "Complete lipid panel.",
         "Complete thyroid panel."]
        →
        ["Complete metabolic, lipid, and thyroid panels as clinically indicated."]

    Only clusters of 2+ lines are merged. Single-line matches pass
    through unchanged.

    Fail-safe: any error returns the original lines unchanged.
    """
    try:
        # Build a working copy; track which indices have been consumed
        remaining: list[str | None] = list(lines)
        output: list[str] = []

        for rule in _CLUSTER_RULES:
            verb_pat    = rule["verb_pattern"]
            obj_suffix  = rule["object_suffix"].lower()
            template    = rule["merge_template"]

            # Collect indices that match this rule
            match_indices: list[int] = []
            for i, line in enumerate(remaining):
                if line is None:
                    continue
                if not verb_pat.match(line):
                    continue
                if obj_suffix not in line.lower():
                    continue
                match_indices.append(i)

            if len(match_indices) < 2:
                # Not enough to cluster — leave as-is
                continue

            # Extract the "object" from each matching line.
            # Strategy: strip the verb prefix and the object_suffix word
            # and everything after it to get just the distinguishing noun.
            objects: list[str] = []
            for i in match_indices:
                line = remaining[i]
                # Remove leading verb
                stripped = verb_pat.sub("", line).strip()
                # Find and trim at the object_suffix word
                idx_suf = stripped.lower().find(obj_suffix)
                if idx_suf > 0:
                    obj_fragment = stripped[:idx_suf].strip().strip(",").strip()
                else:
                    obj_fragment = stripped.strip(".").strip()
                if obj_fragment:
                    objects.append(obj_fragment)
                # Mark consumed regardless — even if extraction failed,
                # we don't want orphaned partial lines
                remaining[i] = None

            if not objects:
                # Extraction failed for all — restore lines
                for i in match_indices:
                    remaining[i] = lines[i]
                continue

            # Join extracted objects naturally
            if len(objects) == 1:
                joined = objects[0]
            elif len(objects) == 2:
                joined = f"{objects[0]} and {objects[1]}"
            else:
                joined = ", ".join(objects[:-1]) + f", and {objects[-1]}"

            merged_line = template.format(items=joined)
            output.append(merged_line)

        # Append all non-consumed lines in original order
        for line in remaining:
            if line is not None:
                output.append(line)

        return output if output else list(lines)

    except Exception:
        logger.warning(
            "report_generator · _cluster_bucket failed; returning original lines"
        )
        return list(lines)


def _render_care_plan_html(
    flagged: list[dict],
    extra_care_plan: dict[str, list[str]] | None = None,
    patient_context: dict | None = None,
) -> str:
    """Build the 5-bucket care plan with unknown-item consolidation
    plus fuzzy deduplication and per-bucket line caps.

    Iterates flagged items in clinical priority order (Critical →
    Out-of-range → Borderline). For each item:
      - Known items (KB advice): each bucket entry is added individually
        (adapted for patient context if provided).
      - Unknown items (fallback advice): names are collected into a
        single consolidated line per bucket.

    Additionally merges extra_care_plan (from FIX #3 text finding
    analyzer + FIX #4 safety escalations) into the same buckets before
    dedup/cap so text-derived and safety-derived care actions get the
    same density controls as biomarker-derived care actions.
    """
    if not flagged and not (extra_care_plan and any(extra_care_plan.values())):
        return ""

    ordered = sorted(flagged, key=_item_sort_key)

    known_buckets: dict[str, list[str]] = {
        "immediate": [], "short_term": [], "lifestyle": [],
        "follow_up": [], "long_term": [],
    }
    known_seen: dict[str, set[str]] = {k: set() for k in known_buckets}

    unknown_names: list[str] = []
    unknown_seen: set[str] = set()

    for it in ordered:
        advice, is_known = _resolve_and_classify(it, patient_context)
        name = str(it.get("name") or it.get("key") or "").strip()

        if is_known:
            care = advice.get("care_plan") or {}
            for bucket_key in known_buckets:
                value = care.get(bucket_key)
                if not value:
                    continue
                # care_plan bucket may be str (KB advice) or list[str]
                # (condition-adapted advice from _apply_condition_rules).
                # Normalise to list before iterating so both shapes work.
                if isinstance(value, list):
                    entries = [str(v).strip() for v in value if v]
                else:
                    entries = [str(value).strip()]
                for text in entries:
                    if text and text not in known_seen[bucket_key]:
                        known_seen[bucket_key].add(text)
                        known_buckets[bucket_key].append(text)
        else:
            if name and name.lower() not in unknown_seen:
                unknown_seen.add(name.lower())
                unknown_names.append(name)

    final_buckets: dict[str, list[str]] = {
        k: list(v) for k, v in known_buckets.items()
    }

    if unknown_names:
        joined = ", ".join(unknown_names)
        consolidated_immediate = (
            f"Share these additional results with your healthcare provider "
            f"at your next visit for review: {joined}."
        )
        consolidated_followup = (
            f"Ask your physician whether repeat testing or additional workup "
            f"is recommended for these results: {joined}."
        )
        if consolidated_immediate not in final_buckets["immediate"]:
            final_buckets["immediate"].append(consolidated_immediate)
        if consolidated_followup not in final_buckets["follow_up"]:
            final_buckets["follow_up"].append(consolidated_followup)

    # Merge extra care plan actions (FIX #3 text findings + FIX #4 safety)
    if extra_care_plan:
        for bucket_key in final_buckets:
            extras = extra_care_plan.get(bucket_key) or []
            for text in extras:
                if not text:
                    continue
                text = str(text).strip()
                if text and text not in final_buckets[bucket_key]:
                    final_buckets[bucket_key].append(text)

    for bucket_key in final_buckets:
        clustered = _cluster_bucket(final_buckets[bucket_key])
        merged    = _dedupe_bucket(clustered)
        capped    = _cap_bucket(merged)
        final_buckets[bucket_key] = capped

    if all(not v for v in final_buckets.values()):
        return ""

    label_map = [
        ("immediate",  "Immediate Actions (next 1–2 weeks)"),
        ("short_term", "Short-Term Actions (2–4 weeks)"),
        ("lifestyle",  "Lifestyle & Ongoing Habits"),
        ("follow_up",  "Follow-Up Testing Schedule"),
        ("long_term",  "Long-Term Monitoring (3+ months)"),
    ]

    parts: list[str] = []
    for key, label in label_map:
        items = final_buckets[key]
        if not items:
            continue
        parts.append('<div class="care-plan-block">')
        parts.append(f'<div class="care-plan-subhead">{_escape_html(label)}</div>')
        parts.append('<ul class="care-plan-list">')
        for it in items:
            parts.append(f'<li class="vital-line-plain">{_escape_html(it)}</li>')
        parts.append('</ul></div>')

    return "".join(parts)

def _render_safety_alerts_html(safety_warnings: list[str]) -> str:
    """
    Render top-of-report Safety Alerts block from combined patient-context
    and safety-escalation warnings. Deduplicated, order-preserved.

    Universal design: no biomarker-specific text; renders whatever
    warnings the patient_context_adapter produced.
    """
    if not safety_warnings:
        return ""

    # Preserve order while deduping
    seen: set[str] = set()
    unique: list[str] = []
    for w in safety_warnings:
        key = str(w).strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(key)

    if not unique:
        return ""

    parts: list[str] = []
    parts.append('<div class="safety-alerts" style="border-left:4px solid #d32f2f;'
                 'padding:10px 14px;background:#fff5f5;margin:0 0 12px 0;'
                 'border-radius:4px;">')
    parts.append('<h4 style="margin:0 0 8px 0;color:#c62828;">⚠️ Safety Alerts</h4>')
    parts.append('<ul style="margin:0;padding-left:20px;">')
    for warning in unique:
        parts.append(f'<li style="margin-bottom:6px;">{_escape_html(warning)}</li>')
    parts.append('</ul></div>')
    return "".join(parts)


def _render_clinical_picture_html(picture: dict) -> str:
    """
    Render the FIX #9 clinical picture summary block.

    Visual role: PLAIN TEXT — no box, no colored background, no border.
    Matches the visual weight of Findings, Care Plan, and other
    narrative sections in the report. Consistency over artificial
    emphasis; the content itself carries its meaning.

    Only visual affordances retained:
      • Bold headings for scannability
      • Small emoji glyph on the main heading (matches other sections)
      • Confidence percentages inline with each pattern

    Deduplication support:
      Narratives surfaced here are exposed via a hidden data attribute
      on a lightweight wrapper span so _render_recommendations_html()
      can dedupe Block 2 without a second synthesizer pass.

    Returns empty string when nothing meaningful to display.
    Fail-safe: any error returns empty string.
    """
    try:
        if not picture:
            return ""

        confident    = picture.get("confident_findings")    or []
        differential = picture.get("differential_findings") or []
        summary      = str(picture.get("clinical_picture_summary") or "").strip()

        if not confident and not differential and not summary:
            return ""

        # Collect surfaced narratives for downstream dedup
        surfaced_narratives: list[str] = []
        for p in confident + differential:
            n = str(p.get("narrative") or "").strip()
            if n:
                surfaced_narratives.append(n)

        import json as _json_inner
        surfaced_json = _json_inner.dumps(surfaced_narratives)

        parts: list[str] = []

        # ── Hidden marker span for dedup (no visual effect) ───────────
        # Zero dimensions; carries the data attribute for
        # _render_recommendations_html() to read.
        parts.append(
            '<span class="clinical-picture-marker" '
            'data-surfaced-narratives=\'' + surfaced_json.replace("'", "&#39;") + '\' '
            'style="display:none;"></span>'
        )

        # ── Main heading (plain, matches other section sub-headings) ──
        parts.append(
            '<p style="margin:0 0 6px 0;font-weight:700;'
            'font-size:1.0em;">'
            'Clinical Picture Summary'
            '</p>'
        )

        # ── Summary paragraph ─────────────────────────────────────────
        if summary:
            parts.append(
                f'<p style="margin:0 0 10px 0;line-height:1.55;">'
                f'{_escape_html(summary)}</p>'
            )

        # ── Confident findings ────────────────────────────────────────
        if confident:
            parts.append(
                '<p style="margin:6px 0 4px 0;font-weight:600;'
                'font-size:0.95em;">'
                'Most Likely Interpretations'
                '</p>'
            )
            parts.append('<ul style="margin:0 0 10px 0;padding-left:22px;">')
            for p in confident:
                line = _format_scored_pattern_line(p)
                parts.append(
                    f'<li style="margin-bottom:4px;line-height:1.5;">'
                    f'{_escape_html(line)}</li>'
                )
            parts.append('</ul>')

        # ── Differential findings ─────────────────────────────────────
        # Same collapse logic as before — content behavior unchanged,
        # only the visual container is now plain.
        if differential:
            n_diff = len(differential)
            diff_label = (
                f'Also Consider — Differential ({n_diff} '
                f'pattern{"s" if n_diff != 1 else ""})'
            )

            if n_diff == 1:
                parts.append(
                    '<p style="margin:6px 0 4px 0;font-weight:600;'
                    'font-size:0.95em;">'
                    f'{_escape_html(diff_label)}'
                    '</p>'
                )
                parts.append('<ul style="margin:0 0 10px 0;padding-left:22px;">')
                for p in differential:
                    line = _format_scored_pattern_line(p)
                    parts.append(
                        f'<li style="margin-bottom:4px;line-height:1.5;">'
                        f'{_escape_html(line)}</li>'
                    )
                parts.append('</ul>')
            else:
                parts.append(
                    '<details style="margin:6px 0 10px 0;" open>'
                    if n_diff <= 3 else
                    '<details style="margin:6px 0 10px 0;">'
                )
                parts.append(
                    f'<summary style="cursor:pointer;font-weight:600;'
                    f'font-size:0.95em;list-style:none;padding:2px 0;">'
                    f'{_escape_html(diff_label)} '
                    f'<span style="font-weight:400;font-size:0.88em;'
                    f'color:#607d8b;">(click to expand)</span>'
                    f'</summary>'
                )
                parts.append('<ul style="margin:6px 0 0 0;padding-left:22px;">')
                for p in differential:
                    line = _format_scored_pattern_line(p)
                    parts.append(
                        f'<li style="margin-bottom:4px;line-height:1.5;">'
                        f'{_escape_html(line)}</li>'
                    )
                parts.append('</ul>')
                parts.append('</details>')

        return "".join(parts)

    except Exception:
        logger.exception(
            "report_generator · _render_clinical_picture_html failed; "
            "returning empty"
        )
        return ""


def _format_scored_pattern_line(p: dict) -> str:
    """
    Format a single scored pattern for display.
    Prefers the pattern's narrative; falls back to a derived title.
    """
    narrative = str(p.get("narrative") or "").strip()
    if not narrative:
        pid = str(p.get("id") or "").strip()
        narrative = f"Pattern: {pid.replace('_', ' ').title()}"

    confidence = float(p.get("confidence") or 0.0)
    confidence_pct = int(round(confidence * 100))
    return f"{narrative} (confidence {confidence_pct}%)"


# ── Tool ──────────────────────────────────────────────────────────


class ReportGenerator:
    """
    Step 8 — deterministic report synthesis.

    Builds a stable markdown report from structured pipeline results.
    """

    TOOL_NAME = TOOL_REPORT_GENERATOR

    def _build_deterministic_report(self, state: AegisState) -> str:
        def clean(value) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            return str(value).strip()

        def tool_ok(value) -> bool:
            return value is not None and not isinstance(value, ToolError)

        def as_percent(value) -> str:
            try:
                return f"{round(float(value) * 100)}%"
            except Exception:
                return "Not available"

        def unique_list(values) -> list[str]:
            result = []
            seen = set()
            if not values:
                return result
            for item in values:
                item = clean(item)
                key = item.lower()
                if item and key not in seen:
                    seen.add(key)
                    result.append(item)
            return result

        def format_list(values) -> str:
            values = unique_list(values)
            if not values:
                return ""
            return ", ".join(values)

        sev = state.severity_result

        level = clean(getattr(sev, "level", "")) or "UNKNOWN"
        confidence = as_percent(getattr(sev, "confidence", None))
        reasons = unique_list(getattr(sev, "reasons", []) or [])

        # ─── Patient profile ───
        patient_name = clean(getattr(state, "patient_name", ""))
        patient_dob = clean(getattr(state, "patient_dob", ""))
        patient_sex = clean(getattr(state, "patient_sex", ""))
        patient_blood_group = clean(getattr(state, "patient_blood_group", ""))
        patient_weight_kg = getattr(state, "patient_weight_kg", None)
        patient_height_cm = getattr(state, "patient_height_cm", None)
        patient_allergies = clean(getattr(state, "patient_allergies", ""))
        patient_conditions = unique_list(getattr(state, "patient_medical_conditions", []) or [])

        submitted_text = (
            clean(getattr(state, "submitted_symptoms_text", ""))
            or clean(getattr(state, "raw_symptoms_text", ""))
        )
        # Fix B: normalize whitespace in submitted_text for display only.
        # Collapses newlines, tabs, and runs of spaces from the patient's
        # typed input into single spaces so Section 2 and Section 3
        # render as clean prose rather than visually broken paragraphs.
        # Applied here at the display layer — state attributes are
        # untouched so the parser + symptom extraction pipeline sees
        # the original text.
        import re as _re
        if submitted_text:
            submitted_text = _re.sub(r"\s+", " ", submitted_text).strip()

        voice_transcript = ""
        if tool_ok(getattr(state, "voice_result", None)):
            voice_transcript = clean(getattr(state.voice_result, "transcript", ""))

        medications_list = list(getattr(state, "medications_raw", []) or [])
        medications_text = format_list(medications_list)

        xray_findings_submitted = list(getattr(state, "xray_findings_raw", []) or [])
        xray_free_text = clean(getattr(state, "xray_free_text_raw", ""))

        has_lab_result = tool_ok(getattr(state, "lab_result", None))
        has_xray_result = tool_ok(getattr(state, "xray_result", None))
        has_drug_result = tool_ok(getattr(state, "drug_result", None))
        has_rag_result = tool_ok(getattr(state, "rag_result", None))

        content_rich_input = has_lab_result or has_xray_result or has_drug_result
        symptom_only_input = not content_rich_input

        review_items = []
        if has_lab_result:
            review_items.append("uploaded lab report")
        if has_xray_result:
            review_items.append("uploaded X-ray image")
        if has_drug_result:
            review_items.append("medication review")

        if review_items:
            review_materials_text = format_list(review_items)
        else:
            review_materials_text = "submitted information"

        # Pull text_findings from lab_result if available
        lab_text_findings: list[str] = []
        if has_lab_result:
            tf = getattr(state.lab_result, "text_findings", None) or []
            if isinstance(tf, list):
                lab_text_findings = [str(x) for x in tf if x]

        # ─── Section 1: Patient Information ───
        patient_lines = []
        if patient_name:
            patient_lines.append(f"- Name: {patient_name}")
        if patient_dob:
            patient_lines.append(f"- Date of Birth: {patient_dob}")
        if patient_sex:
            patient_lines.append(f"- Sex: {patient_sex}")
        if patient_blood_group:
            patient_lines.append(f"- Blood group: {patient_blood_group}")
        if patient_weight_kg is not None:
            patient_lines.append(f"- Weight: {patient_weight_kg:g} kg")
        if patient_height_cm is not None:
            patient_lines.append(f"- Height: {patient_height_cm:g} cm")
        if patient_conditions:
            patient_lines.append(f"- Existing medical conditions: {format_list(patient_conditions)}")

        if patient_allergies:
            patient_lines.append(f"- Allergies: {patient_allergies}")
        else:
            patient_lines.append("- Allergies: Not provided")

        # ─── Section 2: Reported Symptoms & Clinical History ───
        history_lines: list[str] = []
        if submitted_text:
            history_lines.append(f"- Primary reported concern: {submitted_text}")
        if voice_transcript:
            history_lines.append(f"- Voice-recorded description: {voice_transcript}")
        elif getattr(state, "audio_file_path", None):
            history_lines.append("- Voice input: Recorded description was submitted.")
        if medications_text:
            history_lines.append(f"- Current medications: {medications_text}")
        if has_lab_result or getattr(state, "lab_pdf_path", None):
            history_lines.append("- Lab report: Uploaded and processed by the system.")
        if getattr(state, "xray_image_path", None) or has_xray_result:
            history_lines.append("- X-ray image: Uploaded and processed by the system.")
        if xray_findings_submitted:
            history_lines.append(f"- X-ray findings selected: {format_list(xray_findings_submitted)}")
        if xray_free_text:
            history_lines.append(f"- X-ray notes: {xray_free_text}")

        if not history_lines:
            history_lines.append(
                "- No symptoms, medications, lab reports, or imaging were submitted for this session."
            )

        # ─── Section 3: Summary ───
        summary_lines = [
            f"The submitted case was assessed as **{level} severity** by the clinical rule engine."
        ]

        if submitted_text:
            summary_lines.append(
                f"The patient's main concern was: {submitted_text}."
            )

        if symptom_only_input:
            summary_lines.append(
                "This assessment is based on the reported symptoms, since no lab reports, imaging, or medication data were submitted in this session."
            )
        else:
            rich_inputs = []
            if has_lab_result:
                rich_inputs.append("lab report")
            if has_xray_result:
                rich_inputs.append("X-ray image")
            if has_drug_result:
                rich_inputs.append("medication review")

            summary_lines.append(
                f"This report includes processed information from the {format_list(rich_inputs)}. "
                "Detailed clinical findings are shown in the Findings section below."
            )

        if level.upper() == "LOW":
            summary_lines.append(
                "Overall, the current information supports routine monitoring unless warning signs develop."
            )
        elif level.upper() in {"MODERATE", "MEDIUM"}:
            if has_lab_result:
                summary_lines.append(
                    "Some values in the lab report were outside their expected ranges. A timely clinical review with a qualified doctor is recommended."
                )
            elif has_xray_result:
                summary_lines.append(
                    "An imaging finding was flagged in the X-ray. A timely clinical review with a qualified doctor is recommended."
                )
            else:
                summary_lines.append(
                    "The reported symptoms suggest that a clinical review with a qualified doctor is advised, especially if symptoms continue or worsen."
                )
        else:
            summary_lines.append(
                "The current information suggests that urgent clinical evaluation may be needed. Please do not delay seeking medical care."
            )

        summary_lines.append(
            "This is a preliminary decision-support summary and does not replace medical advice from a qualified healthcare professional."
        )

        # ─── Sections 4–7: dynamic content via dashboard extractor + knowledge base ───
        result_data_like = _build_result_data_like(state)
        all_measurements = _collect_all_measurements(result_data_like)
        flagged_items = [it for it in all_measurements if _is_flagged(it)]
        flagged_items.sort(key=_item_sort_key)

        # ── FIX #3: Text finding analysis ──────────────────────────
        # Scan interpretive text findings (peripheral smear morphology,
        # impressions, parasite screening, etc.) for clinically
        # meaningful patterns.
        try:
            from tools.text_finding_analyzer import analyze_text_findings
            text_analysis = analyze_text_findings(
                lab_text_findings, all_measurements,
            )
        except Exception:
            logger.exception("report_generator · text finding analysis failed")
            text_analysis = {
                "observations": [],
                "care_plan_additions": {
                    "immediate": [], "short_term": [], "lifestyle": [],
                    "follow_up": [], "long_term": [],
                },
                "severity_boost": 0,
                "correlated_findings": [],
                "matched_pattern_ids": [],
            }

        try:
            state.text_finding_severity_boost = int(
                text_analysis.get("severity_boost") or 0
            )
            state.text_finding_matched_patterns = list(
                text_analysis.get("matched_pattern_ids") or []
            )
        except Exception:
            pass

        # ── FIX #4: Patient context normalization + safety escalations ──
        # Normalize patient allergies + conditions into structured form,
        # then compute cross-cutting safety escalations that combine
        # patient context with biomarker findings.
        patient_context: dict = {}
        safety_result: dict = {
            "safety_warnings": [],
            "severity_boost": 0,
            "care_plan_appends": {},
            "matched_rule_ids": [],
        }
        try:
            from tools.patient_context_adapter import (
                normalize_patient_context,
                compute_safety_escalations,
            )
            patient_context = normalize_patient_context(state)
            safety_result = compute_safety_escalations(
                all_measurements, patient_context,
            )
        except Exception:
            logger.exception(
                "report_generator · patient context adapter failed"
            )

        # ── FIX #7 Part 2: Clinical synthesis for Sections 6 & 7 ────────
        # Detect cross-cutting clinical patterns (viral fever, iron
        # deficiency anemia, metabolic syndrome, uncontrolled diabetes,
        # etc.) and produce synthesized narratives + care plan actions.
        # These are ADDITIVE to per-biomarker KB advice — nothing is
        # removed. Fail-safe: any error returns empty synthesis.
        # Extract symptoms list once for reuse in both synthesis and
        # dynamic recommendations (Section 10 hook further below).
        symptoms_list: list[str] = []
        sym_for_synth = state.symptom_result
        if sym_for_synth and not isinstance(sym_for_synth, ToolError):
            symptoms_list = list(getattr(sym_for_synth, "symptoms", []) or [])
        if not symptoms_list and submitted_text:
            symptoms_list = [submitted_text]

        clinical_synthesis: dict = {
            "recommendation_narratives": [],
            "care_plan_narratives": {
                "immediate": [], "short_term": [], "lifestyle": [],
                "follow_up": [], "long_term": [],
            },
            "matched_pattern_ids": [],
        }
        try:
            from tools.clinical_synthesis import synthesize_personalized_insights
            clinical_synthesis = synthesize_personalized_insights(
                flagged_items=flagged_items,
                symptoms=symptoms_list,
                patient_context=patient_context,
                text_pattern_ids=list(
                    getattr(state, "text_finding_matched_patterns", []) or []
                ),
                severity_level=level,
            )
        except Exception:
            logger.exception(
                "report_generator · clinical synthesis failed; "
                "sections 6 & 7 render without narratives"
            )

        # ── FIX #9: Cross-registry clinical picture synthesis ───────────
        # Consumes matched pattern IDs from all three pattern sources
        # (text findings, clinical synthesis, safety escalations),
        # scores each by confidence, and produces a unified clinical
        # picture block rendered above per-biomarker recommendations.
        #
        # Fail-safe: on any error returns empty result → nothing renders.
        clinical_picture: dict = {
            "confident_findings": [],
            "differential_findings": [],
            "clinical_picture_summary": "",
            "confidence_scores": {},
            "all_scored_patterns": [],
        }
        try:
            from tools.clinical_picture_synthesizer import synthesize_clinical_picture
            clinical_picture = synthesize_clinical_picture(
                flagged_items=flagged_items,
                symptoms=symptoms_list,
                patient_context=patient_context,
                text_pattern_ids=list(
                    getattr(state, "text_finding_matched_patterns", []) or []
                ),
                clinical_pattern_ids=list(
                    clinical_synthesis.get("matched_pattern_ids") or []
                ),
                safety_rule_ids=list(
                    safety_result.get("matched_rule_ids") or []
                ),
            )
        except Exception:
            logger.exception(
                "report_generator · clinical picture synthesis failed"
            )

        # ── Persist structured clinical picture on state for dashboard ──
        # Rendered HTML is consumed inside the report body, but the
        # dashboard's ClinicalPictureSummaryCard needs the raw structured
        # dict (confident_findings, differential_findings). Stash on
        # state so the pipeline layer can persist it into result_json.
        try:
            state.clinical_picture = clinical_picture
        except Exception as exc:
            logger.warning(
                "report_generator · failed to stash clinical_picture on state",
                error=str(exc),
            )

        clinical_picture_html = _render_clinical_picture_html(clinical_picture)

        # Merge care plan additions from THREE sources into one dict:
        #   1. FIX #3 text finding analysis
        #   2. FIX #4 safety escalations
        #   3. FIX #7 Part 2 clinical synthesis (new)
        # All go through the same dedupe/cap pipeline inside
        # _render_care_plan_html.
        combined_care_plan_additions: dict[str, list[str]] = {
            "immediate": [], "short_term": [], "lifestyle": [],
            "follow_up": [], "long_term": [],
        }
        for bucket in combined_care_plan_additions:
            combined_care_plan_additions[bucket].extend(
                text_analysis.get("care_plan_additions", {}).get(bucket, []) or []
            )
            combined_care_plan_additions[bucket].extend(
                safety_result.get("care_plan_appends", {}).get(bucket, []) or []
            )
            combined_care_plan_additions[bucket].extend(
                clinical_synthesis.get("care_plan_narratives", {}).get(bucket, []) or []
            )

        # Collect all safety warnings for top-of-report rendering.
        # Sources: safety escalations + per-item patient adaptations.
        aggregated_safety_warnings: list[str] = list(
            safety_result.get("safety_warnings") or []
        )
        # Per-item warnings are collected inside the recommendations render;
        # to surface them at the top we do a lightweight pre-pass here.
        try:
            from tools.patient_context_adapter import adapt_advice_for_patient
            for it in flagged_items:
                adv = resolve_advice(it)
                adapted = adapt_advice_for_patient(adv, patient_context, it)
                for w in (adapted.get("safety_warnings") or []):
                    if w and w not in aggregated_safety_warnings:
                        aggregated_safety_warnings.append(w)
        except Exception:
            logger.exception(
                "report_generator · pre-pass safety warning collection failed"
            )

        findings_html_body = _render_findings_html(all_measurements, lab_text_findings)
        critical_obs_html = _render_critical_observations_html(
            flagged_items,
            extra_observations=text_analysis.get("observations") or [],
        )
        recommendations_personal_html = _render_recommendations_html(
            flagged_items,
            patient_context=patient_context,
            clinical_narratives=clinical_synthesis.get("recommendation_narratives") or [],
            clinical_picture_html=clinical_picture_html,
        )
        care_plan_html = _render_care_plan_html(
            flagged_items,
            extra_care_plan=combined_care_plan_additions,
            patient_context=patient_context,
        )

        # Pre-render safety alerts block (goes at top of report)
        safety_alerts_html = _render_safety_alerts_html(
            aggregated_safety_warnings
        )

        # ─── Section 4: Findings ───
        # Emits the RAW_HTML block matching the previous frontend output,
        # plus a graceful fallback line when no measurements exist.
        findings_lines: list[str] = []
        if findings_html_body:
            findings_lines.append(f"{RAW_HTML_START}{findings_html_body}{RAW_HTML_END}")
        else:
            if not (has_lab_result or has_xray_result or has_drug_result):
                findings_lines.append(
                    "- No structured clinical measurements were available in this session."
                )
                if submitted_text:
                    findings_lines.append(f"- Reported concern: {submitted_text}")
            else:
                findings_lines.append(
                    "- No individually flagged measurements were extracted from the submitted reports."
                )

        # ─── Section 5: Critical Observations & Flags ───
        critical_obs_lines: list[str] = []
        if critical_obs_html:
            critical_obs_lines.append(f"{RAW_HTML_START}{critical_obs_html}{RAW_HTML_END}")
        else:
            critical_obs_lines.append(
                "- No individual measurements were flagged as critical, high, low, or borderline in this session."
            )

        # ─── Section 6: Personalized Recommendations ───
        # FIX #9: If clinical picture synthesis produced meaningful
        # output (confident findings and/or differential), render it as
        # a "Clinical Picture Summary" block ABOVE the per-biomarker
        # recommendations. If empty, this line adds nothing.
        personal_rec_lines: list[str] = []
        if clinical_picture_html:
            personal_rec_lines.append(
                f"{RAW_HTML_START}{clinical_picture_html}{RAW_HTML_END}"
            )
        if recommendations_personal_html:
            personal_rec_lines.append(
                f"{RAW_HTML_START}{recommendations_personal_html}{RAW_HTML_END}"
            )
        if not personal_rec_lines:
            personal_rec_lines.append(
                "- No biomarker-specific recommendations were generated because no measurements were flagged."
            )

        # ─── Section 7: Care Plan ───
        care_plan_lines: list[str] = []
        if care_plan_html:
            care_plan_lines.append(f"{RAW_HTML_START}{care_plan_html}{RAW_HTML_END}")
        else:
            care_plan_lines.append(
                "- No structured care plan was generated because no measurements were flagged. Follow the general recommendations below."
            )

        # ─── Section 8: Evidence ───
        evidence_lines = []
        if has_rag_result:
            citations = getattr(state.rag_result, "citations", []) or []
            if citations:
                evidence_lines.append("- Local medical evidence was retrieved to support this report.")
                for citation in citations[:5]:
                    evidence_lines.append(f"- Reference: {citation}")
            else:
                evidence_lines.append(
                    "- Evidence retrieval completed, but no matching references were returned."
                )
        else:
            evidence_lines.append(
                "- No supporting medical evidence was retrieved for this report."
            )

        evidence_lines.append(
            "- This report is based on the submitted patient information and the tool outputs available during this run."
        )

        # ─── Section 9: Severity (user-friendly, no jargon) ───
        severity_lines = [
            f"- Overall severity: **{level}**",
            f"- Confidence in this assessment: {confidence}",
        ]

        friendly_reason = None
        if reasons:
            first_reason = reasons[0].lower()
            if "abnormal lab" in first_reason or "biomarker" in first_reason:
                friendly_reason = "One or more lab values were outside the expected reference range."
            elif "chest pain" in first_reason and "breath" in first_reason:
                friendly_reason = "Chest pain along with breathing difficulty was reported, which needs prompt review."
            elif "troponin" in first_reason:
                friendly_reason = "Cardiac troponin level was elevated, which requires urgent clinical review."
            elif "haemoglobin" in first_reason or "hemoglobin" in first_reason:
                friendly_reason = "Haemoglobin was critically low, suggesting significant anemia."
            elif "potassium" in first_reason:
                friendly_reason = "Potassium level was outside safe limits."
            elif "pneumothorax" in first_reason or "pulmonary" in first_reason or "cardiomegaly" in first_reason or "effusion" in first_reason or "consolidation" in first_reason:
                friendly_reason = "An imaging finding on the X-ray was flagged for clinical review."
            elif "drug" in first_reason or "interaction" in first_reason:
                friendly_reason = "A significant medication interaction was identified in the submitted medication list."
            elif "prolonged" in first_reason:
                friendly_reason = "Symptoms have persisted longer than the usual expected duration."
            elif "default" in first_reason or "no high-risk" in first_reason:
                friendly_reason = "No high-risk features were detected from the submitted information."
            else:
                friendly_reason = reasons[0]

        if friendly_reason:
            severity_lines.append(f"- Why this severity: {friendly_reason}")

        severity_lines.append(
            "- This assessment reflects only the information provided. Additional tests, vital signs, or a physical examination may change the outcome."
        )

        # ─── Section 10: Recommendations (clinical actions) ───
        # FIX #7: Dynamic recommendation synthesis using top flagged
        # findings, reported symptoms, patient context, text-finding
        # patterns, and severity level. Fail-safe: any error inside
        # build_dynamic_recommendations() falls back to static
        # severity-based recommendations so Section 10 is never empty.

        # Extract symptoms list for the synthesizer
        symptoms_for_recs: list[str] = []
        sym = state.symptom_result
        if sym and not isinstance(sym, ToolError):
            symptoms_for_recs = list(getattr(sym, "symptoms", []) or [])
        # Also include the raw submitted symptom text as a single "symptom"
        # so pattern matching can search it (falls back to submitted_text
        # when structured extraction is empty).
        if not symptoms_for_recs and submitted_text:
            symptoms_for_recs = [submitted_text]

        # Text pattern IDs — populated by FIX #3 / stored on state via FIX #10
        text_pattern_ids: list[str] = list(
            getattr(state, "text_finding_matched_patterns", []) or []
        )

        try:
            from tools.dynamic_recommendations import build_dynamic_recommendations
            recommendation_lines = build_dynamic_recommendations(
                severity_level=level,
                flagged_items=flagged_items,
                symptoms=symptoms_for_recs,
                patient_context=patient_context,
                text_pattern_ids=text_pattern_ids,
                input_is_symptom_only=symptom_only_input,
                review_materials_text=review_materials_text,
            )
        except Exception:
            logger.exception(
                "report_generator · dynamic recommendations failed; using static fallback"
            )
            # Inline static fallback (same as pre-FIX #7 behavior)
            if level.upper() == "LOW":
                if symptom_only_input:
                    recommendation_lines = [
                        "- Continue monitoring your symptoms over the next 24 to 48 hours.",
                        "- Maintain hydration, adequate rest, and supportive care.",
                        "- Track temperature, cough, breathing comfort, appetite, and overall energy level.",
                        "- Consult a qualified healthcare professional if symptoms persist, worsen, or new symptoms appear.",
                        "- Seek urgent medical care if breathing difficulty, chest pain, confusion, fainting, bluish lips, severe weakness, dehydration, persistent high fever, or rapid worsening develops.",
                        "- If lab reports, X-ray images, or medication details become available later, repeat the assessment with those included.",
                    ]
                else:
                    recommendation_lines = [
                        "- Follow up with a qualified healthcare professional if symptoms persist or worsen.",
                        "- Discuss the submitted lab, imaging, or medication findings with a clinician for full interpretation.",
                        "- Continue monitoring symptoms and overall condition over the next 24 to 48 hours.",
                        "- Seek urgent medical care if red-flag symptoms such as breathing difficulty, chest pain, confusion, fainting, or rapid worsening develop.",
                    ]
            elif level.upper() in {"MODERATE", "MEDIUM"}:
                recommendation_lines = [
                    "- Arrange a timely review with a qualified healthcare professional.",
                    "- Monitor symptoms closely and do not delay care if symptoms progress.",
                    "- Keep a record of temperature, breathing symptoms, medications taken, and any new symptoms.",
                    "- Seek urgent medical attention if red-flag symptoms such as breathing difficulty, chest pain, confusion, fainting, or severe worsening occur.",
                    f"- Bring the {review_materials_text} and this generated summary to the clinician for review.",
                ]
            else:
                recommendation_lines = [
                    "- Seek urgent medical evaluation as soon as possible.",
                    "- Do not rely on automated triage alone for high-risk symptoms.",
                    "- Share the submitted symptoms, uploaded reports, and this generated summary with a qualified clinician.",
                    "- If symptoms are severe or rapidly worsening, use emergency medical services immediately.",
                ]

        # ─── Section 11: Disclaimer (short, single line) ───
        disclaimer_lines = [
            "Clinical decision support only — not a diagnosis. All outputs must be reviewed by a qualified healthcare professional before any clinical action is taken. Do not use in emergency situations.",
        ]

        sections = [
            ("Patient Information", patient_lines),
            ("Reported Symptoms & Clinical History", history_lines),
            ("Summary", summary_lines),
            ("Findings", findings_lines),
            ("Critical Observations & Flags", critical_obs_lines),
            ("Personalized Recommendations", personal_rec_lines),
            ("Care Plan", care_plan_lines),
            ("Evidence", evidence_lines),
            ("Severity", severity_lines),
            ("Recommendations", recommendation_lines),
            ("Disclaimer", disclaimer_lines),
        ]

        output_parts = []

        # ── FIX #4: Top-of-report Safety Alerts ────────────────────
        # Rendered BEFORE Patient Information so critical patient-
        # context conflicts (allergies vs. treatment implications,
        # anticoagulation vs. low platelets, etc.) are impossible to
        # miss. If there are no warnings, this block emits nothing.
        if safety_alerts_html:
            output_parts.append(f"{RAW_HTML_START}{safety_alerts_html}{RAW_HTML_END}")
            output_parts.append("")

        for title, lines in sections:
            output_parts.append(f"### {title}")
            output_parts.extend(lines)
            output_parts.append("")

        return "\n".join(output_parts).strip()

    async def run(self, state: AegisState) -> AsyncGenerator[str, None]:
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

        full_text = self._build_deterministic_report(state)
        clean_text = _clean_report_text(full_text)

        missing = _validate_sections(clean_text)

        if missing:
            raise FatalPipelineError(
                ToolError(
                    tool=TOOL_REPORT_GENERATOR,
                    reason=f"Deterministic report missing sections: {missing}",
                    fatal=True,
                )
            )

        rag = state.rag_result
        citations: list[str] = []

        if rag and not isinstance(rag, ToolError):
            citations = getattr(rag, "citations", []) or []

        state.report = TriageReport(
            severity=sev.level,
            confidence=getattr(sev, "confidence", 0.0) or 0.0,
            text=clean_text,
            citations=citations,
            disclaimer=DISCLAIMER,
            knowledge_base_version=get_corpus_version(),
            knowledge_base_date=get_corpus_date(),
        )

        logger.info(
            "report_generator · deterministic report complete",
            truncated_core=state.core_fields_truncated,
            truncated_enrichment=state.enrichment_fields_truncated,
        )

        # ── Stream section-by-section instead of one giant chunk ───
        # state.report above already holds the full clean_text, so
        # nothing downstream (validation, TriageReport, caching) is
        # affected by how we choose to stream it here — this only
        # changes what the client sees arrive over time.
        sections = _re.split(r"(?=^### )", clean_text, flags=_re.MULTILINE)
        for section in sections:
            if not section.strip():
                continue
            yield section
            if SECTION_REVEAL_PAUSE_SECONDS > 0:
                await asyncio.sleep(SECTION_REVEAL_PAUSE_SECONDS)