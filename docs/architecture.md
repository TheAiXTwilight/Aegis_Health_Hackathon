# Aegis Health — Architecture


## Pipeline Topology

    Step 0 — VoiceTranscriber         optional — skipped if no audio
    Step 1 — SymptomExtractor
    Step 2 — LabReportParser
    Step 3 — XRayProcessor
    Step 4 — MedicalRAGSearch
    Step 5 — DrugInteractionChecker
    Step 6 — SeverityScorer
    Step 7 — ReportGenerator          only step that yields tokens

All steps run sequentially. Only one tool holds memory at a time.
No parallel inference. Pipeline is wall-clock bounded by PIPELINE_TIMEOUT_S.


## Streaming Protocol

GET /queue/stream/{job_id}
    Raw chunked transfer encoding.
    Plain string tokens only — no SSE framing, no JSON wrapping.
    Consumed directly by any HTTP client supporting chunked transfer.
    None sentinel from backend/queue.py terminates the stream.

GET /queue/status/{job_id}
    Live pipeline state for the performance sidebar.
    Recommended polling interval: 2 seconds while status is queued or running.
    Returns current_tool, tools_run, tools_failed, step_durations_ms.
    No stream parsing required.

These two endpoints serve distinct concerns and are never mixed.


## State Ownership

Pipeline owns all state.*_result assignments.
Tools return results — they do not write to state.

Single exception: VoiceTranscriber writes state.raw_symptoms_text
because it converts one input modality (audio) into another (text),
which Step 1 (SymptomExtractor) then reads.


## Tool Interface Contract

All tools expose:
    async def run(self, state: AegisState) -> Result | ToolError

SeverityScorer exposes:
    async def score(self, state: AegisState) -> SeverityResult | ToolError

Both are async to satisfy _run_step, which unconditionally awaits tool_fn(state).
SeverityScorer.score is synchronous in logic but async in interface —
this is a pipeline contract requirement, not an intrinsic scorer requirement.


## _run_step Generic Typing

    T = TypeVar("T")

    async def _run_step(
        self,
        name: str,
        tool_fn: Callable[[AegisState], Awaitable[T | ToolError]],
        state: AegisState,
    ) -> T | ToolError | None:

mypy infers the concrete return type for each tool invocation.
Eliminates the need for type: ignore[assignment] comments throughout
the pipeline step methods.


## Confidence Separation

Two distinct confidence values exist in the system:

SeverityResult.confidence
    Rule-level certainty.
    Comes from rule_confidence on the highest-priority fired rule.
    Defined in the _RULES table in tools/severity_scorer.py.
    Set at scoring time. Never recomputed downstream.

TriageReport.confidence
    Pipeline-level confidence.
    Comes from calculate_confidence(state) in tools/confidence.py.
    Called after ReportGenerator completes and truncation flags are set.
    Formula: 0.4 * coverage + 0.4 * success_rate + 0.2 * truncation

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

OLLAMA_STREAM_URL = OLLAMA_BASE_URL + "/api/generate"
Constructed at module import time in tools/report_generator.py.


## Tool Name Constants

All tool name strings: tools/tool_names.py
No magic string literals in pipeline, scorer, or error objects.
Each tool class TOOL_NAME attribute points to a constant from tool_names.py.


## Frontend

Current status: no frontend implemented.

The original Streamlit prototype has been deprecated.
The planned replacement stack is React + Vite + TypeScript +
Tailwind + shadcn/ui.

The backend is frontend-agnostic. All endpoints work identically
regardless of which client consumes them. The streaming protocol
returns raw chunked text/plain — consumed identically by any HTTP
client supporting chunked transfer encoding.

When React lands, two backend additions will be required:

CORS middleware in backend/main.py
    Allows the Vite dev server (http://localhost:5173) to call
    the FastAPI backend (http://localhost:8000) during development.

Static file mount in backend/main.py
    Serves the React production build from frontend/dist.
    Mounted as:
        app.mount("/", StaticFiles(directory="frontend/dist", html=True))

Neither change is required until the React frontend is built.

Planned client behaviour:
    POST inputs to /queue/submit
    Poll /queue/status every 2 seconds while queued or running
    Open /queue/stream when status transitions to running
    Consume chunked text with fetch + ReadableStream + TextDecoderStream
    After stream ends, check status — if failed, display mid-stream
    failure message from tools/report_generator.py

No SSE library required. Cleaner than st.write_stream was.


## Placeholder Status (current implementation)

Tool                    Status
VoiceTranscriber        Placeholder — reads text fixtures, fails on real WAV
LabReportParser         Placeholder — reads text fixtures, fails on real PDF
MedicalRAGSearch        Placeholder — keyword matching
XRayProcessor           Stub — returns None
SymptomExtractor        Placeholder — regex-based
DrugInteractionChecker  Placeholder — in-memory interaction table
SeverityScorer          Complete — 13-rule priority engine
ReportGenerator         Complete — Ollama streaming via httpx


## Memory Safety

Sequential pipeline execution ensures only one tool holds
resident memory at a time.

OLLAMA_NUM_PARALLEL=1         — no concurrent LLM inference
OLLAMA_MAX_LOADED_MODELS=1    — only one model loaded at a time
OLLAMA_KEEP_ALIVE=-1          — model stays resident

EasyOCR (when implemented): explicitly released after use.
VoiceTranscriber (when implemented): model released after transcription.

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