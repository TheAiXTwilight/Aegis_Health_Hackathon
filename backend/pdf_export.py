"""
backend/pdf_export.py — Offline-safe PDF clinical dossier generator.

Generates a structured clinical dossier PDF from a completed triage
record. No CDN dependencies — all rendering is server-side.

Uses WeasyPrint for HTML→PDF (pure Python, offline-safe).

GET /export/pdf/{job_id} — Download clinical dossier PDF

Refactor (backend-first content generation):
    The Clinical Assessment body is now a direct render of
    report.text, which the ReportGenerator produces as a complete
    11-section markdown document (including Reported Symptoms &
    Clinical History, grouped Findings, Critical Observations,
    Personalized Recommendations, and Care Plan). All enrichment
    logic has been removed from this file — the PDF is now a pure
    renderer of the same markdown consumed by the frontend preview,
    guaranteeing byte-for-byte content parity.

Layout refinement:
    Card backgrounds and borders have been removed so the PDF
    reads as a clean clinical document on white A4 paper. Sections
    flow naturally as text with headings, not as boxed cards.

Pagination control:
    - Patient Information and Submitted Information blocks stay
      together (page-break-inside: avoid) so they never split
      mid-block across pages.
    - Care Plan buckets flow naturally without forcing new pages,
      but bucket headings stay glued to their content
      (page-break-after: avoid) to prevent orphaned subheadings.
"""
from __future__ import annotations

import base64
import json
import hashlib
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from weasyprint import HTML

from app.auth import get_current_user
from app.db.models import User, HealthRecord
from app.db.session import get_db

router = APIRouter(prefix="/export", tags=["export"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _record_hash(record_id: str) -> str:
    """Short hash for record integrity verification."""
    return hashlib.sha256(record_id.encode()).hexdigest()[:12]


def _severity_color(severity: str) -> str:
    colors = {
        "LOW": "#10b981",
        "MEDIUM": "#f59e0b",
        "MODERATE": "#f59e0b",
        "HIGH": "#ef4444",
        "CRITICAL": "#991b1b",
    }
    return colors.get(severity.upper(), "#6b7280")


def _severity_bg(severity: str) -> str:
    bgs = {
        "LOW": "#d1fae5",
        "MEDIUM": "#fef3c7",
        "MODERATE": "#fef3c7",
        "HIGH": "#fee2e2",
        "CRITICAL": "#fecaca",
    }
    return bgs.get(severity.upper(), "#f3f4f6")


def _safe(value: object, fallback: str = "Not provided") -> str:
    """Return an escaped display value suitable for the PDF HTML."""
    if value is None or value == "":
        return fallback
    return escape(str(value))


def _inline_markdown(value: str) -> str:
    """Render the small inline Markdown subset emitted by TriageReport."""
    rendered = escape(value)
    rendered = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"`(.+?)`", r"<code>\1</code>", rendered)
    return rendered


def _normalise_report_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Some older persisted records contain literal backslash-n sequences.
    if "\\n" in text:
        text = text.replace("\\n", "\n")
    return text.strip()


# ── RAW_HTML-aware markdown renderer ──────────────────────────────
# The ReportGenerator emits <!--RAW_HTML_START-->...<!--RAW_HTML_END-->
# blocks for the Findings, Critical Observations, Personalized
# Recommendations, and Care Plan sections. We stash those blocks first
# and inject them verbatim so the PDF renders the same HTML the frontend
# renders — guaranteeing byte-for-byte content parity.

_RAW_TOKEN_RE = re.compile(r"<!--RAW_HTML_START-->([\s\S]*?)<!--RAW_HTML_END-->")


def _markdown_to_html(text: str) -> str:
    """Convert report Markdown (with optional RAW_HTML blocks) to safe, readable offline HTML."""
    text = _normalise_report_text(text)
    if not text:
        return '<p class="muted">No report narrative available.</p>'

    # Stash raw HTML blocks so they survive markdown processing untouched.
    raw_blocks: list[str] = []

    def _stash(match: re.Match) -> str:
        raw_blocks.append(match.group(1))
        return f"\n\n__RAW_HTML_TOKEN_{len(raw_blocks) - 1}__\n\n"

    text = _RAW_TOKEN_RE.sub(_stash, text)

    html_parts: list[str] = []
    list_type: str | None = None

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            html_parts.append(f"</{list_type}>")
            list_type = None

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line or line in {"-", "*", "•", "```"}:
            close_list()
            continue

        # RAW_HTML token: inject verbatim
        tok = re.match(r"^__RAW_HTML_TOKEN_(\d+)__$", line)
        if tok:
            close_list()
            idx = int(tok.group(1))
            if 0 <= idx < len(raw_blocks):
                html_parts.append(raw_blocks[idx])
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            close_list()
            level = min(len(heading.group(1)) + 1, 4)
            html_parts.append(
                f'<h{level} class="report-heading">'
                f"{_inline_markdown(heading.group(2))}</h{level}>"
            )
            continue

        bullet = re.match(r"^[-*•]\s+(.+)$", line)
        if bullet:
            if list_type != "ul":
                close_list()
                html_parts.append("<ul>")
                list_type = "ul"
            html_parts.append(f"<li>{_inline_markdown(bullet.group(1))}</li>")
            continue

        numbered = re.match(r"^\d+[.)]\s+(.+)$", line)
        if numbered:
            if list_type != "ol":
                close_list()
                html_parts.append("<ol>")
                list_type = "ol"
            html_parts.append(f"<li>{_inline_markdown(numbered.group(1))}</li>")
            continue

        close_list()
        html_parts.append(f"<p>{_inline_markdown(line)}</p>")

    close_list()
    return "".join(html_parts)


def _report_markdown(report: dict) -> str:
    """Return the narrative field without serialising the whole report JSON."""
    text = report.get("text")
    if isinstance(text, str) and text.strip():
        return _normalise_report_text(text)

    parts: list[str] = []
    summary = report.get("summary") or report.get("conclusion") or report.get("narrative")
    if summary:
        parts.extend(["## Summary", str(summary)])

    findings = report.get("findings") or report.get("key_findings")
    if findings:
        parts.append("## Key Findings")
        if isinstance(findings, list):
            parts.extend(f"- {finding}" for finding in findings)
        else:
            parts.append(str(findings))

    recommendations = report.get("recommendations")
    if recommendations:
        parts.append("## Recommendations")
        if isinstance(recommendations, list):
            parts.extend(f"- {recommendation}" for recommendation in recommendations)
        else:
            parts.append(str(recommendations))

    return "\n".join(parts) or "No report narrative available."


def _extract_markdown_section(text: str, section_title: str) -> str:
    """Extract one heading section from the generated narrative."""
    wanted = section_title.strip().lower()
    captured: list[str] = []
    collecting = False

    for raw_line in _normalise_report_text(text).split("\n"):
        heading = re.match(r"^#{1,6}\s+(.+)$", raw_line.strip())
        if heading:
            title = re.sub(r"[*_`]", "", heading.group(1)).strip().lower()
            if collecting:
                break
            collecting = title == wanted
            continue
        if collecting:
            captured.append(raw_line)

    return "\n".join(captured).strip()


def _without_markdown_sections(text: str, section_titles: set[str]) -> str:
    """Remove sections already represented in the Patient Preview block."""
    unwanted = {title.strip().lower() for title in section_titles}
    kept: list[str] = []
    skipping = False

    for raw_line in _normalise_report_text(text).split("\n"):
        heading = re.match(r"^#{1,6}\s+(.+)$", raw_line.strip())
        if heading:
            title = re.sub(r"[*_`]", "", heading.group(1)).strip().lower()
            skipping = title in unwanted
        if not skipping:
            kept.append(raw_line)

    cleaned = "\n".join(kept).strip()
    return cleaned or text


def _display_list(values: object, fallback: str = "None reported") -> str:
    if not isinstance(values, list) or not values:
        return fallback
    return ", ".join(_safe(value, "") for value in values if value not in (None, "")) or fallback


def _validation_banner_html(status: str | None) -> str:
    if not status:
        return ""
    banners = {
        "agreement": (
            '<div style="background:#d1fae5;border-left:4px solid #10b981;padding:12px 16px;'
            'border-radius:8px;margin:16px 0;font-size:14px;">'
            '<strong>✓ Safety Validation:</strong> Rule-based and AI assessments are in agreement.'
            '</div>'
        ),
        "warning": (
            '<div style="background:#fef3c7;border-left:4px solid #f59e0b;padding:12px 16px;'
            'border-radius:8px;margin:16px 0;font-size:14px;">'
            '<strong>⚠ Validation Warning:</strong> Minor discrepancies detected between '
            'rule-based and AI assessments. Clinician review advised.'
            '</div>'
        ),
        "override": (
            '<div style="background:#fee2e2;border-left:4px solid #ef4444;padding:12px 16px;'
            'border-radius:8px;margin:16px 0;font-size:14px;">'
            '<strong>⚠ Validation Override:</strong> Significant disagreement between '
            'rule-based and AI assessments. Urgent clinician review required.'
            '</div>'
        ),
    }
    return banners.get(status, "")


def _plan_theater_html(record: HealthRecord, report: dict, result_data: dict) -> str:
    """Build Agentic Plan Theater audit section for PDF dossier."""
    plan = result_data.get("execution_plan") or {}
    summary_str = report.get("execution_plan_summary") or ""
    
    use_rag = plan.get("use_rag", False)
    is_fallback = plan.get("is_fallback", False)
    was_repaired = plan.get("was_repaired", False)
    reasoning = plan.get("reasoning") or "Standard clinical workflow execution."
    
    submitted = result_data.get("submitted") or {}
    mandatory_tools = [
        ("VoiceTranscriber", bool(submitted.get("audio_uploaded")) or "✓ VoiceTranscriber" in summary_str),
        ("SymptomExtractor", bool(submitted.get("symptoms_text")) or bool(record.symptoms_text) or "✓ SymptomExtractor" in summary_str),
        ("LabReportParser", bool(submitted.get("lab_pdf_uploaded")) or "✓ LabReportParser" in summary_str),
        ("XRayProcessor", bool(submitted.get("xray_image_uploaded")) or bool(submitted.get("xray_findings")) or "✓ XRayProcessor" in summary_str),
        ("DrugInteractionChecker", bool(submitted.get("medications")) or "✓ DrugInteractionChecker" in summary_str),
    ]
    
    chips_html = ""
    for name, active in mandatory_tools:
        bg_col = "#d1fae5" if active else "#f3f4f6"
        txt_col = "#065f46" if active else "#6b7280"
        mark = "✓" if active else "✗"
        chips_html += f'<span style="display:inline-block;padding:4px 10px;margin:3px;border-radius:12px;background:{bg_col};color:{txt_col};font-weight:600;font-size:11px;">{mark} {name}</span>'
        
    rag_bg = "#d1fae5" if use_rag else "#f3f4f6"
    rag_txt = "#065f46" if use_rag else "#6b7280"
    rag_mark = "✓" if use_rag else "✗"
    chips_html += f'<span style="display:inline-block;padding:4px 10px;margin:3px;border-radius:12px;background:{rag_bg};color:{rag_txt};font-weight:600;font-size:11px;">{rag_mark} MedicalRAGSearch</span>'
    
    if was_repaired or "[REPAIRED]" in summary_str:
        chips_html += '<span style="display:inline-block;padding:4px 10px;margin:3px;border-radius:12px;background:#fef3c7;color:#92400e;font-weight:700;font-size:11px;border:1px solid #f59e0b;">[REPAIRED] Forced RAG</span>'
    if is_fallback or "[FALLBACK]" in summary_str:
        chips_html += '<span style="display:inline-block;padding:4px 10px;margin:3px;border-radius:12px;background:#fee2e2;color:#991b1b;font-weight:700;font-size:11px;border:1px solid #ef4444;">[FALLBACK] Planner Fallback</span>'

    if " | " in summary_str:
        parts = summary_str.split(" | ")
        if len(parts) >= 3 and parts[-1].strip():
            reasoning = parts[-1].strip()

    return f"""
<div style="background:#f8fafc;padding:14px;border-radius:8px;border:1px solid #e2e8f0;margin:16px 0;">
  <div style="margin-bottom:8px;"><strong style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Execution Plan & Tool Audit</strong></div>
  <div>{chips_html}</div>
  <div style="margin-top:10px;padding-top:10px;border-top:1px dashed #cbd5e1;font-size:11px;color:#334155;font-style:italic;">
    <strong>Planner Reasoning:</strong> {_safe(reasoning, 'Standard clinical workflow execution.')}
  </div>
</div>
"""


def _heatmap_html(record: HealthRecord, result_data: dict) -> str:
    """Build Grad-CAM X-ray heatmap explainability section for PDF dossier."""
    xray_res = result_data.get("xray_result") or {}
    if not isinstance(xray_res, dict):
        return ""
    heatmap_path = xray_res.get("heatmap_path")

    # Never scan another session's upload directory as a fallback. If this
    # report has no own heatmap, the PDF must not display an unrelated X-ray.
    if heatmap_path and Path(heatmap_path).is_file():
        try:
            img_bytes = Path(heatmap_path).read_bytes()
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            findings = xray_res.get("findings") or ["X-Ray Finding"]
            top_finding = _safe(findings[0] if findings else "Detected Finding")
            return f"""
<h2>Grad-CAM X-ray Explainability Heatmap</h2>
<div style="margin: 16px 0; text-align: center; padding: 16px 0;">
  <div style="font-weight:700;font-size:13px;color:#0d2167;margin-bottom:10px;">Primary Detected Finding: {top_finding}</div>
  <img src="data:image/png;base64,{b64}" style="max-width: 90%; max-height: 320px; border-radius: 6px; border: 1px solid #d1d5db;" alt="Grad-CAM X-ray Heatmap" />
  <p style="font-size: 11px; color: #6b7280; margin-top: 10px; line-height: 1.4;">
    <strong>Saliency Overlay:</strong> Highlighting structural regions and contrast gradients most strongly associated with the AI classification of <em>{top_finding}</em>. Best-effort non-fatal visual artifact.
  </p>
</div>
"""
        except Exception:
            pass
    return ""


def render_dossier_html(*, record: HealthRecord, user: User) -> str:
    """
    Build a complete HTML clinical dossier from a HealthRecord.

    All styles are inline — no external CSS. Images embedded as
    base64 data URIs where applicable. This HTML is ready for
    direct PDF rendering without CDN or network access.

    The Clinical Assessment body is rendered directly from
    report.text, which the ReportGenerator produces as a complete
    11-section markdown document including RAW_HTML blocks for
    the Findings, Critical Observations, Personalized Recommendations,
    and Care Plan sections.

    Layout: clean A4 document look — no card backgrounds or borders,
    just plain text sections flowing on white paper. Pagination is
    tuned so Patient Information and Submitted Information blocks stay
    intact, and Care Plan buckets flow naturally without wasting space.
    """
    try:
        medications = json.loads(record.medications_json)
    except (json.JSONDecodeError, TypeError):
        medications = []

    try:
        xray = json.loads(record.xray_findings_json)
    except (json.JSONDecodeError, TypeError):
        xray = []

    try:
        report = json.loads(record.report_json)
    except (json.JSONDecodeError, TypeError):
        report = {}

    try:
        result_data = json.loads(record.result_json)
    except (json.JSONDecodeError, TypeError, AttributeError):
        result_data = {}

    patient = result_data.get("patient") or {}
    submitted = result_data.get("submitted") or {}
    if not isinstance(patient, dict):
        patient = {}
    if not isinstance(submitted, dict):
        submitted = {}

    submitted_medications = submitted.get("medications")
    if isinstance(submitted_medications, list):
        medications = submitted_medications
    submitted_xray = submitted.get("xray_findings")
    if isinstance(submitted_xray, list):
        xray = submitted_xray

    report_markdown = _report_markdown(report)
    summary_markdown = _extract_markdown_section(report_markdown, "Summary")
    clinical_markdown = _without_markdown_sections(
        report_markdown,
        {
            "Patient Information",
            "Submitted Information",
            "Submitted Clinical Inputs",
            "Summary",
        },
    )
    summary_html = _markdown_to_html(
        summary_markdown
        or str(report.get("summary") or "See the clinical assessment below.")
    )
    clinical_html = _markdown_to_html(clinical_markdown)

    patient_name = _safe(patient.get("name") or user.display_name)
    patient_dob = _safe(patient.get("dob"))
    patient_sex = _safe(patient.get("sex"))
    patient_blood_group = _safe(patient.get("blood_group"))
    patient_weight = _safe(
        f"{patient.get('weight_kg'):g} kg" if patient.get("weight_kg") is not None else None
    )
    patient_height = _safe(
        f"{patient.get('height_cm'):g} cm" if patient.get("height_cm") is not None else None
    )
    patient_allergies = _safe(patient.get("allergies"))
    patient_conditions = _display_list(patient.get("medical_conditions"), "None reported")
    symptoms = _safe(submitted.get("symptoms_text") or record.symptoms_text, "None reported")
    medication_text = _display_list(medications)
    xray_findings = _display_list(xray, "None")
    lab_uploaded = "Yes" if submitted.get("lab_pdf_uploaded") else "No"
    xray_uploaded = "Yes" if submitted.get("xray_image_uploaded") else "No"
    audio_uploaded = "Yes" if submitted.get("audio_uploaded") else "No"

    color = _severity_color(record.severity)
    bg = _severity_bg(record.severity)
    rhash = _record_hash(record.id)
    created = record.created_at.strftime("%Y-%m-%d %H:%M UTC") if record.created_at else "N/A"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Aegis Health — Clinical Dossier</title>
<style>
  @page {{ size: A4; margin: 25mm 20mm; }}
  body {{ font-family: 'Helvetica', 'Arial', sans-serif; font-size: 12px; color: #1f2937; line-height: 1.6; }}
  h1 {{ font-size: 22px; color: #0d2167; margin: 0 0 4px 0; }}
  h2 {{ font-size: 16px; color: #0d2167; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; margin: 28px 0 14px 0; page-break-after: avoid; }}
  h3 {{ font-size: 13px; color: #374151; margin: 0 0 4px 0; }}
  .header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; }}
  .logo {{ font-size: 18px; font-weight: 800; color: #0d2167; }}
  .logo span {{ color: #2563ff; }}
  .meta {{ text-align: right; font-size: 11px; color: #6b7280; }}
  .severity-badge {{ display: inline-block; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 13px; }}
  .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px 24px; }}
  .info-item {{ padding: 8px 0; border-bottom: 1px solid #f3f4f6; }}
  .info-label {{ font-size: 10px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; }}
  .info-value {{ font-size: 13px; color: #1f2937; font-weight: 500; overflow-wrap: anywhere; }}

  /* Clean A4 look — no card backgrounds, no borders, sections flow as plain text.
     preview-card is set to page-break-inside: avoid so Patient Information and
     Submitted Information each stay intact on one page instead of splitting. */
  .preview-card {{ background:transparent; border:none; border-radius:0; padding:0; margin:0 0 20px 0; page-break-inside:avoid; }}
  .preview-title {{ color:#0d2167; font-size:14px; font-weight:700; margin:0 0 10px 0; page-break-after:avoid; }}
  .summary-card {{ background:transparent; border-left:none; border-radius:0; padding:0; margin-top:16px; }}
  .summary-card p {{ margin:0 0 7px 0; }}

  /* Clinical assessment body — no grey background, just clean prose */
  .report-text {{ background: transparent; padding: 0; border-radius: 0; font-size: 12px; line-height: 1.7; overflow-wrap:anywhere; }}
  .report-text p {{ margin:0 0 10px 0; }}
  .report-text ul, .report-text ol {{ margin:6px 0 14px 20px; padding:0; }}
  .report-text li {{ margin:4px 0; }}
  .report-heading {{ color:#0d2167; font-size:14px; border-bottom:1px solid #e5e7eb; padding-bottom:5px; margin:18px 0 10px 0; page-break-after:avoid; }}
  .report-text .report-heading:first-child {{ margin-top:0; }}

  /* Findings & vital lines — pixel-identical to preview */
  .findings-heading {{ font-size:12.5px; font-weight:700; color:#0d2167; text-transform:uppercase; letter-spacing:0.4px; margin:14px 0 6px 0; padding-bottom:3px; border-bottom:1px solid #e2e8f0; page-break-after:avoid; }}
  ul.findings-list {{ list-style:none; margin:0 0 12px 0 !important; padding:0 !important; }}
  li.vital-line, li.vital-line-plain {{ list-style:none; padding:5px 0; margin:0 !important; border-bottom:1px dashed #eef2f7; font-size:12px; line-height:1.5; color:#2e4378; }}
  li.vital-line:last-child, li.vital-line-plain:last-child {{ border-bottom:none; }}
  .vital-name {{ font-weight:600; color:#0d2167; }}
  .vital-value {{ font-weight:600; color:#0d2167; font-variant-numeric: tabular-nums; }}
  .vital-unit {{ color:#6b7ba8; font-size:11.5px; font-weight:400; }}
  .vital-status {{ font-weight:500; color:#4a5b8c; }}
  .vital-range {{ color:#7a8bb5; font-size:11.5px; font-weight:400; }}
  .vital-sep {{ color:#8b9cc0; font-weight:400; padding:0 6px; }}
  .vital-status-word {{ font-weight:700; color:#0d2167; text-transform:uppercase; letter-spacing:0.3px; font-size:11px; }}

  /* Care plan buckets — flow naturally, but keep subheading with content */
  .care-plan-block {{ margin:10px 0 14px 0; page-break-inside:auto; }}
  .care-plan-subhead {{ font-weight:700; color:#0d2167; font-size:11.5px; letter-spacing:0.3px; text-transform:uppercase; margin-bottom:5px; page-break-after:avoid; }}
  ul.care-plan-list {{ margin:0 0 4px 0 !important; padding-left:20px !important; list-style:disc !important; }}
  ul.care-plan-list li {{ margin:3px 0; color:#2e4378; }}
  
  code {{ background:#e2e8f0; border-radius:4px; padding:1px 4px; font-family:monospace; }}
  .muted {{ color:#6b7280; font-style:italic; }}
  .tag {{ display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 11px; margin: 2px; }}
  .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #e5e7eb; font-size: 10px; color: #9ca3af; text-align: center; }}
  .footer strong {{ color: #6b7280; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
  th {{ text-align: left; padding: 8px 6px; background: #f9fafb; border-bottom: 2px solid #e5e7eb; font-size: 10px; color: #6b7280; text-transform: uppercase; }}
  td {{ padding: 8px 6px; border-bottom: 1px solid #f3f4f6; }}
  .hash {{ font-family: monospace; font-size: 10px; color: #9ca3af; word-break: break-all; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="logo">Aegis<span>Health</span></div>
    <h1>Clinical Dossier</h1>
    <div style="color:#6b7280;font-size:12px;">Generated: {_now_iso()}</div>
  </div>
  <div class="meta">
    <div>Job ID: {_safe(record.job_id, 'N/A')}</div>
    <div>Record: {_safe(record.id[:8], 'N/A')}...</div>
    <div class="hash">Hash: {rhash}</div>
  </div>
</div>

<!-- Severity -->
<div style="margin:20px 0;display:flex;align-items:center;gap:12px;">
  <span class="severity-badge" style="background:{bg};color:{color};">
    {_safe(record.severity, 'N/A')}
  </span>
  <span style="font-size:13px;color:#6b7280;">
    Confidence: {record.confidence:.0%}
  </span>
</div>

<!-- Validation banner -->
{_validation_banner_html(record.validation_status)}

<!-- Patient Preview -->
<h2>Patient Preview</h2>
<div class="preview-card">
  <div class="preview-title">Patient Information</div>
  <div class="info-grid">
    <div class="info-item">
      <div class="info-label">Name</div>
      <div class="info-value">{patient_name}</div>
    </div>
    <div class="info-item">
      <div class="info-label">Date of Birth</div>
      <div class="info-value">{patient_dob}</div>
    </div>
    <div class="info-item">
      <div class="info-label">Sex</div>
      <div class="info-value">{patient_sex}</div>
    </div>
    <div class="info-item">
      <div class="info-label">Blood Group</div>
      <div class="info-value">{patient_blood_group}</div>
    </div>
    <div class="info-item">
      <div class="info-label">Weight</div>
      <div class="info-value">{patient_weight}</div>
    </div>
    <div class="info-item">
      <div class="info-label">Height</div>
      <div class="info-value">{patient_height}</div>
    </div>
    <div class="info-item">
      <div class="info-label">Allergies</div>
      <div class="info-value">{patient_allergies}</div>
    </div>
    <div class="info-item">
      <div class="info-label">Existing Medical Conditions</div>
      <div class="info-value">{patient_conditions}</div>
    </div>
    <div class="info-item">
      <div class="info-label">Record Date</div>
      <div class="info-value">{created}</div>
    </div>
  </div>
</div>

<div class="preview-card">
  <div class="preview-title">Submitted Information</div>
  <div class="info-grid">
    <div class="info-item">
      <div class="info-label">Symptoms / Medical History</div>
      <div class="info-value">{symptoms}</div>
    </div>
    <div class="info-item">
      <div class="info-label">Medications</div>
      <div class="info-value">{medication_text}</div>
    </div>
    <div class="info-item">
      <div class="info-label">Lab Report / PDF Processed</div>
      <div class="info-value">{lab_uploaded}</div>
    </div>
    <div class="info-item">
      <div class="info-label">X-ray Image Processed</div>
      <div class="info-value">{xray_uploaded}</div>
    </div>
    <div class="info-item">
      <div class="info-label">X-ray Findings</div>
      <div class="info-value">{xray_findings}</div>
    </div>
    <div class="info-item">
      <div class="info-label">Voice Recording Processed</div>
      <div class="info-value">{audio_uploaded}</div>
    </div>
  </div>
  <div class="summary-card">
    <div class="preview-title">Summary</div>
    {summary_html}
  </div>
</div>

<!-- Formatted clinical report -->
<h2>Clinical Assessment</h2>
<div class="report-text">{clinical_html}</div>

<!-- Grad-CAM X-ray Heatmap -->
{_heatmap_html(record, result_data)}

<!-- Citations -->
<h2>Evidence Citations</h2>
{_citations_html(report)}

<!-- Medical Disclaimer -->
<div class="footer">
  <p><strong>MEDICAL DISCLAIMER:</strong> This is an AI-generated triage assessment and does not constitute a medical diagnosis. Always consult a qualified healthcare provider for medical advice, diagnosis, or treatment. In case of emergency, contact emergency services immediately.</p>
  <p style="margin-top:8px;">
    Aegis Health · Privacy-First Local AI Triage · Record Hash: {rhash} · HIPAA-aware
  </p>
</div>

</body>
</html>"""


def _extract_report_text(report: dict) -> str:
    """Extract and format the narrative instead of dumping report JSON."""
    return _markdown_to_html(_report_markdown(report))


def _citations_html(report: dict) -> str:
    """Build citations section from report JSON."""
    citations = report.get("citations") or report.get("references") or []
    if not citations:
        return '<p style="color:#9ca3af;font-style:italic;">No citations available.</p>'

    rows = ""
    for i, cite in enumerate(citations, 1):
        if isinstance(cite, str):
            rows += f"<tr><td>{i}</td><td>{_safe(cite, '')}</td></tr>"
        elif isinstance(cite, dict):
            title = _safe(cite.get("title", cite.get("text", "")), "")
            source = _safe(cite.get("source", cite.get("url", "")), "")
            rows += f"<tr><td>{i}</td><td><strong>{title}</strong><br><span style='color:#6b7280'>{source}</span></td></tr>"

    return f"<table><tr><th>#</th><th>Citation</th></tr>{rows}</table>"


def render_pdf_bytes(html: str) -> bytes:
    """Render HTML to PDF bytes using WeasyPrint (offline-safe)."""
    return HTML(string=html).write_pdf()


# ── Endpoint ────────────────────────────────────────────────────
@router.get("/pdf/{job_id}")
def export_pdf(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Download a clinical dossier PDF for a completed triage job.

    The PDF is generated server-side with zero CDN dependencies.
    All styles are embedded inline. The dossier includes patient
    info, severity, confidence, validation banner, submitted inputs,
    full report, citations, and medical disclaimer.
    """
    from app.db.models import PipelineJobRow

    job_row = (
        db.query(PipelineJobRow)
        .filter_by(job_id=job_id, user_id=user.id)
        .first()
    )
    if not job_row:
        raise HTTPException(status_code=404, detail="Job not found")

    record = (
        db.query(HealthRecord)
        .filter_by(job_id=job_id, user_id=user.id)
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    html = render_dossier_html(record=record, user=user)
    pdf_bytes = render_pdf_bytes(html)

    return Response(
        pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="aegis_dossier_{job_id[:8]}.pdf"'
            ),
        },
    )