"""
tools/tool_names.py — Pipeline tool name string constants.

Single source of truth for all tool identifiers used in:
    agents/pipeline.py          (_run_step name argument)
    tools/severity_scorer.py    (Rule.contributing_tool field)
    schemas/errors.py           (ToolError.tool field)
    tools/*/TOOL_NAME           (class attribute on each tool class)

Phase 2.5 additions:
    TOOL_EXECUTION_PLANNER  — Step -1
    TOOL_RULE_VALIDATOR     — Step 9

Phase 4 additions:
    TOOL_INPUT_VALIDATION   — backend/uploads.py pre-queue validation
    TOOL_QUEUE              — backend/queue.py submission layer

Scope: pipeline tools (Steps -1 through 9) plus infrastructure tools
that produce ToolError objects visible in the API response.
PlanValidator is NOT listed here — it is called synchronously inside
_run_execution_planner, not via _run_step, and has no TOOL_NAME.
"""

TOOL_EXECUTION_PLANNER        = "ExecutionPlanner"
TOOL_VOICE_TRANSCRIBER        = "VoiceTranscriber"
TOOL_SYMPTOM_EXTRACTOR        = "SymptomExtractor"
TOOL_LAB_REPORT_PARSER        = "LabReportParser"
TOOL_XRAY_PROCESSOR           = "XRayProcessor"
TOOL_MEDICAL_RAG_SEARCH       = "MedicalRAGSearch"
TOOL_DRUG_INTERACTION_CHECKER = "DrugInteractionChecker"
TOOL_SEVERITY_SCORER          = "SeverityScorer"
TOOL_REPORT_GENERATOR         = "ReportGenerator"
TOOL_RULE_VALIDATOR           = "RuleValidator"
TOOL_TEXT_FINDING_ANALYZER    = "text_finding_analyzer"

# Infrastructure tools — not pipeline steps, but produce ToolError
# objects visible in API responses. Centralised here so the tool=
# field is consistent and greppable across the entire codebase.
TOOL_INPUT_VALIDATION         = "input_validation"
TOOL_QUEUE                    = "queue"