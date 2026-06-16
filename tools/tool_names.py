"""
tools/tool_names.py — Pipeline tool name string constants.

Single source of truth for all tool identifiers used in:
    - agents/pipeline.py        (_run_step name argument)
    - tools/severity_scorer.py  (Rule.contributing_tool field)
    - schemas/errors.py         (ToolError.tool field, set by each tool)
    - tools/*/TOOL_NAME         (class attribute on each tool class)

Scope: pipeline tools only (Steps 0–7).
'input_validation' used in backend/uploads.py is NOT listed here —
it is a pre-queue validation layer, not a pipeline tool.
"""

TOOL_VOICE_TRANSCRIBER        = "VoiceTranscriber"
TOOL_SYMPTOM_EXTRACTOR        = "SymptomExtractor"
TOOL_LAB_REPORT_PARSER        = "LabReportParser"
TOOL_XRAY_PROCESSOR           = "XRayProcessor"
TOOL_MEDICAL_RAG_SEARCH       = "MedicalRAGSearch"
TOOL_DRUG_INTERACTION_CHECKER = "DrugInteractionChecker"
TOOL_SEVERITY_SCORER          = "SeverityScorer"
TOOL_REPORT_GENERATOR         = "ReportGenerator"