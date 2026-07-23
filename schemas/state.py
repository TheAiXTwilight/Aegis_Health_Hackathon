"""
schemas/state.py — AegisState: single pipeline state object.

Phase 2.5 additions:
    execution_plan        — set by _run_execution_planner (Step -1)
    rule_validator_result — set by _run_rule_validator (Step 9)

Both fields sit after xray_free_text_raw and before voice_result,
at the logical boundary between inputs (which inform the plan) and
tool outputs (which the plan governs).

Layout invariant added in Phase 2.5:
    After _run_execution_planner, execution_plan is not None.
    Guaranteed by the single normalisation point in pipeline.

All other fields, invariants, and docstrings are unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from schemas.drugs import DrugInteractionResult
from schemas.errors import ToolError
from schemas.lab import LabReportResult
from schemas.plan import ExecutionPlan
from schemas.rag import RAGSearchResult
from schemas.report import TriageReport
from schemas.severity import SeverityResult
from schemas.symptom import SymptomExtractionResult
from schemas.validation import RuleValidatorResult
from schemas.voice import VoiceTranscriptionResult
from schemas.xray import XRayResult


class AegisState(BaseModel):
    """
    The single state object that flows through AegisPipeline.

    One instance per pipeline run. Created at submission time, passed
    sequentially through Steps -1 through 9, and consumed by
    ReportGenerator.

    Layout invariants (enforced by AegisPipeline, not by this model):
        tools_run ∩ tools_failed = ∅
        pipeline_complete becomes True when the pipeline finishes.
        step_durations_ms keys are tool names; values are wall-clock ms.
        After _run_execution_planner: execution_plan is not None.

    Each *_result field has three possible states:
        None        — tool has not run yet, or input was absent
        <Result>    — tool ran successfully
        ToolError   — tool ran but failed

    Mutation is intentional and required. Do not add frozen=True.
    """

    # ── Session ───────────────────────────────────────────────────
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Server-generated UUID. Never client-supplied.",
    )
    user_id: str | None = None
    priority: int = 1
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ── Patient metadata / submitted profile ─────────────────────────
    patient_name: str | None = None
    patient_dob: str | None = None
    patient_sex: str | None = None
    patient_blood_group: str | None = None
    patient_weight_kg: float | None = None
    patient_height_cm: float | None = None
    patient_allergies: str | None = None
    patient_medical_conditions: list[str] = Field(default_factory=list)

    # ── Inputs ────────────────────────────────────────────────────
    submitted_symptoms_text: str | None = None
    raw_symptoms_text: str | None = None
    audio_file_path:   str | None = None
    lab_pdf_path:      str | list[str] | None = None
    xray_image_path:   str | list[str] | None = None
    medications_raw:   list[str]  = Field(default_factory=list)

    xray_findings_raw: list[str] = Field(
        default_factory=list,
        description=(
            "Selected checklist findings for the X-ray, collected in the UI."
        ),
    )
    xray_free_text_raw: str | None = Field(
        default=None,
        description="Optional clinician free-text findings for the X-ray.",
    )

    # ── Agentic plan ──────────────────────────────────────────────
    # Produced by ExecutionPlanner + PlanValidator.
    # Guaranteed non-None after run_execution_planner completes.
    # Contains only use_rag — mandatory tools are pipeline invariants.
    execution_plan: ExecutionPlan | None = None

    # ── Tool Outputs ──────────────────────────────────────────────
    voice_result:    VoiceTranscriptionResult | ToolError | None = None
    symptom_result:  SymptomExtractionResult  | ToolError | None = None
    lab_result:      LabReportResult          | ToolError | None = None
    xray_result:     XRayResult               | ToolError | None = None
    rag_result:      RAGSearchResult          | ToolError | None = None
    drug_result:     DrugInteractionResult    | ToolError | None = None
    severity_result: SeverityResult           | ToolError | None = None

    # ── Validation ────────────────────────────────────────────────
    # None when RuleValidator has not yet run or returned ToolError.
    rule_validator_result: RuleValidatorResult | None = None

    # ── Text Finding Analyzer Signals (FIX #3 / FIX #10) ──────────
    # Populated by tools.report_generator._build_deterministic_report()
    # from tools.text_finding_analyzer.analyze_text_findings() output.
    #
    # Consumed by tools.severity_scorer synthesized rules to escalate
    # severity when interpretive smear/impression patterns match.
    #
    # Both fields default to empty/zero so the pipeline behaves
    # identically to before whenever the text analyzer produces no
    # matches or is unavailable.
    text_finding_matched_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Pattern IDs matched by tools.text_finding_analyzer. "
            "Read by severity_scorer synthesized text-finding rules."
        ),
    )
    text_finding_severity_boost: int = Field(
        default=0,
        description=(
            "Aggregated severity boost from matched text-finding "
            "patterns (0..3, capped in the analyzer)."
        ),
    )

    # ── Clinical Picture Synthesis (Row 3 dashboard) ──────────────
    # Populated by tools.report_generator._build_deterministic_report()
    # from tools.clinical_picture_synthesizer.synthesize_clinical_picture()
    # output. Consumed by the dashboard's ClinicalPictureSummaryCard
    # (Row 3, right card) via result_json["clinical_picture"].
    #
    # Defaults to empty dict so downstream persistence and dashboard
    # rendering work identically when synthesis is unavailable or
    # produces no findings.
    clinical_picture: dict = Field(
        default_factory=dict,
        description=(
            "Structured clinical picture from clinical_picture_synthesizer. "
            "Shape: {confident_findings: [...], differential_findings: [...], "
            "clinical_picture_summary: str}."
        ),
    )    

    # ── Pipeline Metadata ─────────────────────────────────────────
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

    # ── Timing ────────────────────────────────────────────────────
    pipeline_start_ms: float | None        = None
    pipeline_end_ms:   float | None        = None
    step_durations_ms: dict[str, float]    = Field(default_factory=dict)
    current_tool:      str | None          = None

    # ── Truncation Tracking ───────────────────────────────────────
    core_fields_truncated:       bool = False
    enrichment_fields_truncated: bool = False

    # ── Final Output ──────────────────────────────────────────────
    report: TriageReport | None = None

