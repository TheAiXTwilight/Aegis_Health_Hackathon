# Aegis Health — Architecture


## Pipeline Topology

    Step -1 — ExecutionPlanner        always — SLM routing decision (use_rag only)
    Step  0 — PlanValidator           always — RAG safety override (synchronous)
    Step  1 — VoiceTranscriber        if audio_file_path provided
    Step  2 — SymptomExtractor        if symptoms text or voice available
    Step  3 — LabReportParser         if lab_pdf_path provided
    Step  4 — XRayProcessor           if xray input provided
    Step  5 — MedicalRAGSearch        if plan.use_rag=True (planner-controlled)
    Step  6 — DrugInteractionChecker  if medications_raw non-empty
    Step  7 — SeverityScorer          always
    Step  8 — ReportGenerator         always — only step that yields tokens
    Step  9 — RuleValidator           always — compares deterministic vs narrative

All steps run sequentially. Only one tool holds memory at a time.
No parallel inference. Pipeline is wall-clock bounded by PIPELINE_TIMEOUT_S.


## Planner Authority Invariant

The ExecutionPlanner may choose ONLY optional enrichment capabilities.
Mandatory input-driven tools run whenever their input exists — the
planner has no authority over them.

This invariant is enforced structurally:
    ExecutionPlan contains exactly one tool boolean: use_rag.
    The schema contains no fields for mandatory tools.
    A planner that wanted to suppress SymptomExtractor cannot — the
    field does not exist in the schema it is asked to populate.

The pipeline reads plan.use_rag only for MedicalRAGSearch (Step 5).
All other tool gates check input presence only.


## Agentic Layer Components

### ExecutionPlanner (Step -1)

SLM call — non-streaming, temperature=0.0, num_predict=128.
Reads: input metadata + ≤200-char symptom preview.
Outputs: {use_rag: bool, reasoning: str}.

Retry policy:
    Attempt 1: full metadata prompt → json.loads() → ExecutionPlan
    Attempt 2: simplified prompt → same validation
    Attempt 3: ToolError(fatal=False) — pipeline uses fallback

reasoning is audit metadata only. It is never interpreted programmatically
and never affects pipeline execution. May contain hallucinated or inaccurate
content from the 1B planner.

Prompt design:
    Intentionally minimal — metadata + ≤200-char preview only.
    The planner is routing, not diagnosing.
    Short prompts are dramatically more reliable on a 1B model.
    Full symptom text, lab values, and medication names are excluded.

### PlanValidator (Step 0, synchronous)

Single responsibility: validate and optionally repair use_rag.
Mechanism: tools/plan_validator.py
Configuration: tools/planner_constants.py

Forces use_rag=True when ANY of three conditions hold:

    Condition 1: Critical symptom terms in raw_symptoms_text
        Case-insensitive substring matching.

        Term selection policy — each term must satisfy BOTH:
            1. Independently interpretable as a high-acuity clinical signal
               without a qualifying phrase or contextual interpretation.
            2. Safe to match via case-insensitive substring search.

        Generic modifiers excluded:
            "severe"  — depends on what it modifies ("severe acne" is not high-acuity)
            "acute"   — same problem class
            "sudden"  — same problem class
            "cardiac" — adjective; matches "cardiac history", "cardiac rehab",
                        "cardiac clinic" — not independently high-acuity

        Current terms:
            Cardiac:      chest pain, chest tightness, chest pressure,
                          heart attack, troponin
            Respiratory:  shortness of breath, dyspnoea, dyspnea,
                          breathlessness
            Neurological: stroke, seizure, unconscious

        Known gap: "cardiac arrest" does not currently trigger RAG.
        It satisfies the selection criteria. Add to planner_constants.py
        if required.

    Condition 2: Critical X-ray findings in xray_findings_raw
        Case-insensitive substring matching against each finding.
        Current findings: pneumothorax, pulmonary edema, cardiomegaly.

    Condition 3: Polypharmacy threshold
        len(medications_raw) > RAG_FORCE_POLYPHARMACY_THRESHOLD (3).
        Strictly greater-than — exactly 3 medications does not trigger.

Always returns ExecutionPlan. Never raises. Never returns ToolError.
Records each repair in validation_errors.
Sets was_repaired=True if use_rag was changed from False to True.
Preserves is_fallback unchanged.
Returns a fresh ExecutionPlan — never mutates the input object.

### _run_execution_planner — single normalisation point

    result = await _run_step(TOOL_EXECUTION_PLANNER, ...)

    raw_plan = (
        result
        if isinstance(result, ExecutionPlan)
        else _make_fallback_plan(state)
    )

    state.execution_plan = self._plan_validator.validate(raw_plan, state)

After this method returns, state.execution_plan is guaranteed non-None.
No downstream code branches on planner success vs fallback.
All downstream code reads state.execution_plan only.

### _make_fallback_plan

    Returns ExecutionPlan(
        use_rag      = True,    # safety-first, Decision 50
        reasoning    = "Fallback plan: planner failed after retries.",
        is_fallback  = True,
        was_repaired = False,
    )

    use_rag=True is unconditional. When planner reasoning is unavailable,
    evidence retrieval defaults to enabled. Retrieving unnecessary evidence
    is acceptable; omitting evidence in a planner-failure scenario is not.

    The fallback plan is always input-independent — it does not set flags
    for any specific tool. PlanValidator still runs after the fallback
    and applies safety overrides as normal.

### RuleValidator (Step 9)

Synchronous logic in async wrapper. No Ollama call.

Extraction algorithm:
    1. Find "### Severity" in report text.
    2. Extract text from that header to the next "###" header (or end).
    3. Apply \bHIGH\b, \bMEDIUM\b, \bLOW\b in priority order.
       Whole-word regex, case-sensitive (uppercase only).
    4. First match wins. Returns "HIGH", "MEDIUM", "LOW", or None.

Word boundary matching prevents false positives:
    "highest priority"   does not match HIGH
    "medium-term"        does not match MEDIUM
    "low-level"          does not match LOW
Case sensitivity prevents "low" in patient narrative from matching LOW.

Three-state classification:
    narrative=None              → WARNING
    narrative==deterministic    → AGREEMENT
    deterministic==HIGH
        and narrative!=HIGH     → OVERRIDE (overridden=True)
    all other mismatches        → WARNING

Override rationale:
    False reassurance (LOW or MEDIUM narrative when rules say HIGH)
    is a clinical safety risk. Overclaiming (HIGH narrative when rules
    say LOW) is not — it produces WARNING not OVERRIDE.

TriageReport.severity is already set from the deterministic result
by ReportGenerator. Override surfaces the conflict for the UI to
show a safety banner without rewriting the report text.

Writes: state.rule_validator_result, state.report.validation_status.
Non-fatal ToolError when state.report or state.severity_result unavailable.


## Streaming Protocol

GET /queue/stream/{job_id}
    Raw chunked transfer encoding.
    Plain string tokens only — no SSE framing, no JSON wrapping.
    Consumed directly by any HTTP client supporting chunked transfer.
    None sentinel from backend/queue.py terminates the stream.

GET /queue/status/{job_id}
    Live pipeline state for the performance sidebar.
    Recommended polling interval: 2 seconds while queued or running.
    Returns current_tool, tools_run, tools_failed, step_durations_ms.
    No stream parsing required.

These two endpoints serve distinct concerns and are never mixed.


## State Ownership

Pipeline owns all state.*_result assignments.
Tools return results — they do not write to state.

Single exception: VoiceTranscriber writes state.raw_symptoms_text
because it converts one input modality (audio) into another (text),
which SymptomExtractor (Step 2) then reads.

Phase 2.5 state assignments (pipeline, not tools):
    _run_execution_planner  → state.execution_plan
    _run_report_generator   → state.report.confidence (via calculate_confidence)
                              state.report.execution_plan_summary (via _build_plan_summary)
    _run_rule_validator     → state.rule_validator_result
                              state.report.validation_status


## Tool Interface Contract

All tools expose:
    async def run(self, state: AegisState) -> Result | ToolError

SeverityScorer exposes:
    async def score(self, state: AegisState) -> SeverityResult | ToolError

Both are async to satisfy _run_step, which unconditionally awaits tool_fn(state).
SeverityScorer.score is synchronous in logic but async in interface —
this is a pipeline contract requirement, not an intrinsic scorer requirement.

PlanValidator exposes:
    def validate(self, raw_plan: ExecutionPlan, state: AegisState) -> ExecutionPlan
    Synchronous. Not a pipeline tool. Not called via _run_step.
    Has no TOOL_NAME. Not listed in tools/tool_names.py.
    Not exported via tools/__init__.py's __all__.


## _run_step Generic Typing

    T = TypeVar("T")

    async def _run_step(
        self,
        name: str,
        tool_fn: Callable[[AegisState], Awaitable[T | ToolError]],
        state: AegisState,
    ) -> T | ToolError | None:

mypy infers the concrete return type for each tool invocation.
Eliminates type: ignore[assignment] comments throughout pipeline methods.


## ExecutionPlan Schema

    class ExecutionPlan(BaseModel):
        schema_version:    int       = 1
        use_rag:           bool
        reasoning:         str       # audit metadata only — never interpreted
        was_repaired:      bool      = False
        validation_errors: list[str] = Field(default_factory=list)
        is_fallback:       bool      = False

    @model_validator(mode="after")
    def _check_fallback_repair_invariant(self):
        if self.is_fallback and self.was_repaired:
            raise ValueError("is_fallback and was_repaired cannot both be True.")

    | Scenario                           | is_fallback | was_repaired |
    |------------------------------------|-------------|--------------|
    | Planner succeeded, plan conformed  | False       | False        |
    | Planner succeeded, plan repaired   | False       | True         |
    | Planner failed, fallback used      | True        | False        |

    is_fallback=True + was_repaired=True is impossible by schema invariant.


## RuleValidatorResult Schema

    class ValidationStatus(str, Enum):
        AGREEMENT = "agreement"
        WARNING   = "warning"
        OVERRIDE  = "override"

    class RuleValidatorResult(BaseModel):
        status:               ValidationStatus
        deterministic_level:  Literal["LOW", "MEDIUM", "HIGH"]
        slm_narrative_level:  str | None = None
        disagreement_reason:  str | None = None
        overridden:           bool = False
        schema_version:       str = "1.0"

    overridden=True only when status==OVERRIDE. Redundant with status
    but simplifies downstream checks: if result.overridden: show_banner().


## _build_plan_summary Format

    "Mandatory: ✓ VoiceTranscriber | ✗ SymptomExtractor | ✗ LabReportParser |
     ✗ XRayProcessor | ✓ DrugInteractionChecker | Optional: ✓ MedicalRAGSearch
     [REPAIRED] | Chest pain warrants evidence retrieval."

    Mandatory tools: derived from input presence at submission time (not plan)
    Optional tools:  derived from plan.use_rag (planner decision)

    Suffix accumulation (not elif — future-proof):
        suffixes = []
        if plan.is_fallback:  suffixes.append("[FALLBACK]")
        if plan.was_repaired: suffixes.append("[REPAIRED]")
        suffix = " " + " ".join(suffixes) if suffixes else ""

    Written to state.report.execution_plan_summary after ReportGenerator
    completes and state.report is assigned. Never set on None report.


## Confidence Separation

Two distinct confidence values — unchanged from Phase 2:

SeverityResult.confidence
    Rule-level certainty from highest-priority fired rule.
    From rule_confidence on the highest-priority fired rule.
    Defined in _RULES table in tools/severity_scorer.py.
    Set at scoring time. Never recomputed downstream.

TriageReport.confidence
    Pipeline-level confidence.
    From calculate_confidence(state) in tools/confidence.py.
    Called after ReportGenerator completes and truncation flags are set.
    Formula: 0.4 * coverage + 0.4 * success_rate + 0.2 * truncation

    Coverage signal is handled / submitted — only counts modalities
    the user actually provided. RAG is always-submitted.

    Phase 2.5 effect:
        ExecutionPlanner and RuleValidator affect success_rate.
        When plan.use_rag=False: MedicalRAGSearch skipped,
        state.rag_result=None, RAG coverage reduces.
        This is correct — confidence reflects planner decision.

These two values are independent and serve different purposes.


## Severity Rule Engine

13 real rules in _RULES, evaluated in descending priority order.
RULE_DEFAULT_LOW is evaluator fallback — not a table entry.
ALL_RULE_CONSTANTS auto-derived: [r.constant for r in _RULES] + [RULE_DEFAULT_LOW]

RuleContext passed to every check function:
    state: AegisState
    any_high_fired: bool = False

any_high_fired updated immediately when a HIGH rule fires.
RULE_PROLONGED_SYMPTOMS and RULE_MODERATE_DRUG_INTERACTION
check ctx.any_high_fired and return False if True.

Full rule reference: docs/severity_rules.md

RuleValidator uses the deterministic level produced by SeverityScorer
to compare against the LLM narrative. It does not re-evaluate rules.


## Drug Interaction Model

DrugInteractionSeverity enum: SEVERE, MODERATE, MINOR
DrugInteraction structured model: drugs, severity, description
SeverityScorer checks interaction.severity == DrugInteractionSeverity.SEVERE
for RULE_SEVERE_DRUG_INTERACTION and == DrugInteractionSeverity.MODERATE
for RULE_MODERATE_DRUG_INTERACTION.


## Lab Key Normalisation

Canonical lab key strings: tools/lab_constants.py
British spelling throughout: haemoglobin, troponin, potassium, etc.
Alias normalisation map in tools/lab_report_parser.py.
Both "hemoglobin" and "haemoglobin" in incoming documents
normalise to LAB_KEY_HAEMOGLOBIN = "haemoglobin" at parse time.
Downstream code never deals with aliases.

Unknown lab keys preserved in LabReportResult.extra_measurements.
Not used by scorer. Available for future rules without reparsing.


## Numeric Thresholds

All numeric lab thresholds: tools/lab_thresholds.py

ABNORMAL_* prefix — detection thresholds for LabReportParser
CRITICAL_* prefix — severity thresholds for SeverityScorer

Both categories in one file. One authoritative source.
Changing a threshold requires editing only that file.


## Ollama Configuration

OLLAMA_BASE_URL environment variable controls Ollama connectivity.
Default: http://localhost:11434 (local development)
Docker: http://ollama:11434 (set in docker-compose.yml)

ExecutionPlanner:
    OLLAMA_STREAM_URL = OLLAMA_BASE_URL + "/api/generate"
    stream=False, temperature=0.0, num_predict=128
    Constructed at module import time in tools/execution_planner.py.

ReportGenerator:
    OLLAMA_STREAM_URL = OLLAMA_BASE_URL + "/api/generate"
    stream=True, temperature=0.2, num_predict=1024
    Constructed at module import time in tools/report_generator.py.


## Tool Name Constants

All tool name strings: tools/tool_names.py
No magic string literals in pipeline, scorer, or error objects.
Each tool class TOOL_NAME attribute points to a constant from tool_names.py.

Phase 2.5 additions:
    TOOL_EXECUTION_PLANNER = "ExecutionPlanner"
    TOOL_RULE_VALIDATOR    = "RuleValidator"

PlanValidator is NOT in tool_names.py — it is not a pipeline tool
and is not called via _run_step.


## Plan Constants

tools/planner_constants.py — configuration module, not a tool.
Not exported from tools/__init__.py.
Import directly where needed:

    from tools.planner_constants import RAG_FORCE_SYMPTOM_TERMS

Contents:
    RAG_FORCE_SYMPTOM_TERMS          frozenset[str]
    RAG_FORCE_XRAY_FINDINGS          frozenset[str]
    RAG_FORCE_POLYPHARMACY_THRESHOLD int = 3

frozenset provides constant-time membership testing for the trigger set.
Overall substring scanning cost depends on input text length, not set size.


## Frontend

Current status: no frontend implemented.
Streamlit prototype deprecated. React + Vite planned (Phase 5).
Backend is frontend-agnostic. All endpoints work identically regardless
of which client consumes them.

When React lands, two backend additions required:
    CORS middleware in backend/main.py
    StaticFiles mount: app.mount("/", StaticFiles(directory="frontend/dist", html=True))
Neither is required until the React frontend is built.

Client behaviour contract (Phase 2.5 addition):
    If validation_status == "override", display a safety banner.
    TriageReport.severity holds the deterministic (authoritative) level.
    The banner contextualises the narrative for the clinician.


## Placeholder Status (current implementation)

Tool                    Status
ExecutionPlanner        Real — LLM call via Ollama, retry, fallback
PlanValidator           Real — synchronous safety validation
RuleValidator           Real — regex-based severity comparison
VoiceTranscriber        Placeholder — reads text fixtures, fails on real WAV
LabReportParser         Placeholder — reads text fixtures, fails on real PDF
MedicalRAGSearch        Placeholder — keyword matching
XRayProcessor           Stub — returns None
SymptomExtractor        Placeholder — regex-based
DrugInteractionChecker  Placeholder — in-memory interaction table
SeverityScorer          Complete — 13-rule priority engine
ReportGenerator         Complete — Ollama streaming via httpx

Placeholder Parser Limitations:
    Regex does not include "+" in lab key character class.
    Aliases like "K+" and "Na+" are present in the alias map
    but unreachable until the real PDF parser (Phase 3) lands.
    Tracked as locked decision #31.


## Memory Safety

Sequential pipeline execution ensures only one tool holds
resident memory at a time.

OLLAMA_NUM_PARALLEL=1         — no concurrent LLM inference
OLLAMA_MAX_LOADED_MODELS=1    — only one model loaded at a time
OLLAMA_KEEP_ALIVE=-1          — model stays resident

EasyOCR (when implemented): explicitly released after use.
VoiceTranscriber (when implemented): model released after transcription.

Phase 2.5 additions (negligible):
    ExecutionPlanner: one non-streaming HTTP call, no model loaded locally
    PlanValidator: pure Python dict operations
    RuleValidator: regex over a string

Stream buffers: asyncio.Queue(maxsize=256) per active job.
Duration tracking: deque(maxlen=100) — bounded, never a plain list.
Both contribute negligibly to peak memory.


## Queue Architecture

MAX_QUEUE_SIZE = 10
PIPELINE_TIMEOUT_S = 180
JOB_RETENTION_SECONDS = 3600
STREAM_QUEUE_MAXSIZE = 256
STREAM_PUT_TIMEOUT_S = 30.0

Single inference worker. FIFO queue. asyncio.Lock held during execution.
Per-token timeout: if consumer does not read within STREAM_PUT_TIMEOUT_S,
job is marked FAILED and lock is released for next job.

All module-level state in backend/queue.py.
Correct only with --workers 1 (mandatory per spec).