"""
Public exports for the Aegis Health tools package.
"""

from .base import BaseTool
from .confidence import calculate_confidence
from .drug_checker import DrugInteractionChecker
from .lab_report_parser import LabReportParser
from .medical_rag_search import MedicalRAGSearch
from .report_generator import ReportGenerator
from .severity_scorer import (
    ALL_RULE_CONSTANTS,
    RULE_ABNORMAL_LAB_ANY,
    RULE_CHEST_PAIN_AND_SOB,
    RULE_CRITICAL_LAB_HAEMOGLOBIN,
    RULE_CRITICAL_LAB_POTASSIUM,
    RULE_CRITICAL_LAB_TROPONIN,
    RULE_DEFAULT_LOW,
    RULE_MODERATE_DRUG_INTERACTION,
    RULE_PROLONGED_SYMPTOMS,
    RULE_SEVERE_DRUG_INTERACTION,
    RULE_XRAY_CARDIOMEGALY,
    RULE_XRAY_CONSOLIDATION,
    RULE_XRAY_PLEURAL_EFFUSION,
    RULE_XRAY_PNEUMOTHORAX,
    RULE_XRAY_PULMONARY_EDEMA,
    SeverityScorer,
    score,
)
from .symptom_extractor import SymptomExtractor
from .tool_names import (
    TOOL_DRUG_INTERACTION_CHECKER,
    TOOL_LAB_REPORT_PARSER,
    TOOL_MEDICAL_RAG_SEARCH,
    TOOL_REPORT_GENERATOR,
    TOOL_SEVERITY_SCORER,
    TOOL_SYMPTOM_EXTRACTOR,
    TOOL_VOICE_TRANSCRIBER,
    TOOL_XRAY_PROCESSOR,
)
from .voice_transcriber import VoiceTranscriber

__all__ = [
    # Tool classes
    "BaseTool",
    "calculate_confidence",
    "DrugInteractionChecker",
    "LabReportParser",
    "MedicalRAGSearch",
    "ReportGenerator",
    "SeverityScorer",
    "SymptomExtractor",
    "VoiceTranscriber",
    # Severity scorer entrypoint and constants
    "score",
    "ALL_RULE_CONSTANTS",
    "RULE_CHEST_PAIN_AND_SOB",
    "RULE_CRITICAL_LAB_TROPONIN",
    "RULE_CRITICAL_LAB_HAEMOGLOBIN",
    "RULE_CRITICAL_LAB_POTASSIUM",
    "RULE_XRAY_PNEUMOTHORAX",
    "RULE_XRAY_PULMONARY_EDEMA",
    "RULE_SEVERE_DRUG_INTERACTION",
    "RULE_ABNORMAL_LAB_ANY",
    "RULE_XRAY_CARDIOMEGALY",
    "RULE_XRAY_PLEURAL_EFFUSION",
    "RULE_XRAY_CONSOLIDATION",
    "RULE_PROLONGED_SYMPTOMS",
    "RULE_MODERATE_DRUG_INTERACTION",
    "RULE_DEFAULT_LOW",
    # Tool name constants
    "TOOL_VOICE_TRANSCRIBER",
    "TOOL_SYMPTOM_EXTRACTOR",
    "TOOL_LAB_REPORT_PARSER",
    "TOOL_XRAY_PROCESSOR",
    "TOOL_MEDICAL_RAG_SEARCH",
    "TOOL_DRUG_INTERACTION_CHECKER",
    "TOOL_SEVERITY_SCORER",
    "TOOL_REPORT_GENERATOR",
]