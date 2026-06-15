from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from schemas.drugs import DrugInteractionResult
from schemas.errors import ToolError
from schemas.lab import LabReportResult
from schemas.rag import RAGSearchResult
from schemas.report import TriageReport
from schemas.severity import SeverityResult
from schemas.symptom import SymptomExtractionResult
from schemas.voice import VoiceTranscriptionResult
from schemas.xray import XRayResult


class AegisState(BaseModel):
    """
    The single state object that flows through AegisPipeline.

    One instance per pipeline run. Created at submission time, passed
    sequentially through Steps 0–7, and consumed by ReportGenerator.

    Layout invariants (enforced by AegisPipeline, not by this model):
        - tools_run ∩ tools_failed = ∅
          A tool name appears in exactly one list, never both.
        - pipeline_complete becomes True when the pipeline finishes,
          regardless of success or failure (set in AegisPipeline.run's finally).
        - step_durations_ms keys are tool names; values are wall-clock ms.

    Each *_result field has three possible states:
        None         — tool has not run yet, or input was absent
        <Result>     — tool ran successfully
        ToolError    — tool ran but failed (fatal flag controls pipeline continuation)

    Mutation is intentional and required throughout the pipeline.
    Do not add model_config frozen=True.
    """

    # ── Session ──────────────────────────────────────────────
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Server-generated UUID. Never client-supplied.",
    )
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ── Inputs ───────────────────────────────────────────────
    raw_symptoms_text: str | None = None
    audio_file_path: str | None = None
    lab_pdf_path: str | None = None
    xray_image_path: str | None = None
    medications_raw: list[str] = Field(default_factory=list)

    xray_findings_raw: list[str] = Field(
        default_factory=list,
        description=(
            "Selected checklist findings for the X-ray, collected in the UI. "
            "Valid items match the XRayResult.findings vocabulary."
        ),
    )
    xray_free_text_raw: str | None = Field(
        default=None,
        description="Optional clinician free-text findings for the X-ray.",
    )

    # ── Tool Outputs ─────────────────────────────────────────
    voice_result: VoiceTranscriptionResult | ToolError | None = None
    symptom_result: SymptomExtractionResult | ToolError | None = None
    lab_result: LabReportResult | ToolError | None = None
    xray_result: XRayResult | ToolError | None = None
    rag_result: RAGSearchResult | ToolError | None = None
    drug_result: DrugInteractionResult | ToolError | None = None
    severity_result: SeverityResult | ToolError | None = None

    # ── Pipeline Metadata ────────────────────────────────────
    tools_run: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names that completed successfully. "
            "Mutually exclusive with tools_failed."
        ),
    )
    tools_failed: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names that produced a ToolError. "
            "Mutually exclusive with tools_run."
        ),
    )

    pipeline_complete: bool = False

    # ── Timing ───────────────────────────────────────────────
    pipeline_start_ms: float | None = None
    pipeline_end_ms: float | None = None
    step_durations_ms: dict[str, float] = Field(default_factory=dict)
    current_tool: str | None = None

    # ── Truncation Tracking ──────────────────────────────────
    core_fields_truncated: bool = False
    enrichment_fields_truncated: bool = False

    # ── Final Output ─────────────────────────────────────────
    report: TriageReport | None = None