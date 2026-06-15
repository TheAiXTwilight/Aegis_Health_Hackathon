# Aegis Health
### Local AI Triage Assistant · Edge Deployment · Privacy-First

---

## Overview

Aegis Health is a fully local AI triage assistant that accepts patient inputs across five clinical data modalities — symptom input (typed or voice), lab report PDFs, chest X-rays, medication lists, and retrieved medical evidence — and produces a structured, evidence-backed triage report. All inference runs on-device with no external API calls at any stage.

The system is pipeline-driven: a custom AegisPipeline orchestrator routes inputs through a suite of specialised tools in a deterministic sequence. llama3.2:1b via Ollama acts exclusively as the synthesis layer, receiving validated structured outputs from all tools and streaming a coherent plain-language report to the UI. The intelligence of the system comes from tool composition and deterministic rules — not from the language model alone.

Designed for deployment on an NVIDIA Jetson Orin Nano 8 GB.

---

## The Problem

AI-powered healthcare tools almost universally depend on cloud APIs — patient data is transmitted to and processed on remote infrastructure outside clinical control. This creates real privacy risks and makes deployment impractical in low-connectivity or resource-constrained settings.

Existing tools compound this by operating on single input types — a symptom checker, an image analyser, a lab report parser — with no system capable of reasoning across all modalities together, locally, without a cloud dependency.

Aegis Health addresses both. All inference runs locally on the Jetson with no external calls. The system reasons across five clinical data modalities, retrieves evidence from authoritative medical sources, and streams a structured triage report — patient data never leaves the device.

It does not diagnose. It triages — giving clinicians a structured, evidence-backed starting point while keeping patient data private and under clinical control.

---

## Core Capabilities

Input Modality          Input Method                            Processing
Symptom description     Typed text or voice recording           VoiceTranscriber (if audio) → SymptomExtractor
Lab report              PDF upload                              Three-stage extraction, abnormal value detection
Chest X-ray             Image upload                            DICOM metadata extraction, clinician findings form
Medications             Typed list                              SQLite FTS5 lookup, interaction checking
Medical evidence        Derived from all inputs                 RAG retrieval, MedlinePlus citations

Voice and typed text are alternative input paths for symptom input — not separate modalities. One is used per session. The system supports five distinct clinical input modalities total.

All inputs — including all clinician X-ray findings — are collected in the UI before submission. The pipeline runs uninterrupted once started; no step blocks on human input mid-run.

---

## Frontend Status (current)

No frontend is currently implemented.

The original Streamlit prototype has been deprecated. The replacement is React + Vite + TypeScript + Tailwind + shadcn/ui. Backend development and API stabilisation take priority.

Backend architecture, API contracts, queue mechanics, and all tool behaviours remain exactly as specified in this document. The frontend swap is a presentation-layer change only — every endpoint works identically regardless of which client consumes it.

When the React frontend lands, the only backend changes required are CORS middleware (for the Vite dev server) and a StaticFiles mount (for the production build). Neither change is required until then.

---

## Architecture

### Pipeline Execution Order

    AegisPipeline — Deterministic sequential orchestrator
                    Pure Python · No framework · Wall-clock bounded

    Step 0 — VoiceTranscriber       optional, skipped if no audio
                                    Faster-Whisper tiny.en INT8
                                    CPU-only · lazy-loaded · released immediately
                                    Output populates raw_symptoms_text in AegisState

    Step 1 — SymptomExtractor       llama3.2:1b structured prompt
                                    3-attempt retry/repair before ToolError

    Step 2 — LabReportParser        PyMuPDF → pdfminer.six → EasyOCR (opt-in)
                                    EasyOCR lazy-loaded and explicitly released

    Step 3 — XRayProcessor          PIL · DICOM · clinician findings pre-collected
                                    No GPU · No ML model · No mid-pipeline human wait

    Step 4 — MedicalRAGSearch       ONNX MiniLM · ChromaDB primary · FAISS fallback
                                    Zero results → RAGSearchResult(passages=[])

    Step 5 — DrugInteractionChecker SQLite FTS5 · OpenFDA + RxNorm
                                    Zero model load · pure stdlib

    Step 6 — SeverityScorer         Rule-based · fully deterministic
                                    triggered_rules · highest_priority_rule
                                    reasons · contributing_tools

    Step 7 — ReportGenerator        llama3.2:1b · num_ctx 4096
                                    Six sections · buffer-validate-yield

    Final Triage Assessment streamed via FastAPI StreamingResponse.
    Live performance state polled via GET /queue/status.

### Architecture Principles

Separation of concerns. The language model acts exclusively as the synthesis layer. SeverityScorer handles triage level through fully specified rule-based logic. The LLM explains severity but cannot change it.

Sequential order is a memory-safety choice. Steps run sequentially so only one tool holds resident memory at a time.

No mid-pipeline human wait. All human input collected before submission. Pipeline runs to completion once dequeued.

Contracts before code. All schemas, severity rules, rule constants, queue schema, and retry policy must exist before implementation that depends on them.

Honest failure mode. Every failure produces an explicit ToolError. Zero evidence is valid output.

Single state object. AegisState defined once in schemas/state.py.

Pipeline owns state mutation. Tools return results — they do not write state. Single exception: VoiceTranscriber writes state.raw_symptoms_text because it converts one input modality (audio) into another (text).

### Why AegisPipeline over LangGraph

Concern                 LangGraph                                  AegisPipeline
ARM64 compatibility     Known ARM64 build failures                 Zero external dependencies
Problem fit             Non-deterministic graphs                   Sequential deterministic pipelines
Streaming               Complex version-sensitive API              Simple async generator
Debuggability           40+ transitive dependencies                Fully readable, internally owned
Upstream risk           API changed across major versions          No upstream breakage possible

---

## Data Contracts

### Schema Version Policy

schema_version increments only on breaking contract changes. Additive changes do not increment. Current version: "1.0".
Schemas carrying schema_version: SeverityResult, DrugInteractionResult, RAGSearchResult, TriageReport, PipelineJob, LabReportResult, SymptomExtractionResult, VoiceTranscriptionResult, XRayResult.

The change from DrugInteractionResult.interactions: list[str] to list[DrugInteraction] is a breaking schema change accepted as intentional. The project is pre-release, no serialized data exists on disk, schema_version stays "1.0" as a conscious policy decision.

### PipelineJob

Defined in schemas/queue.py.

    class JobStatus(str, Enum):
        QUEUED    = "queued"
        RUNNING   = "running"
        COMPLETED = "completed"
        FAILED    = "failed"

    class PipelineJob(BaseModel):
        job_id:       str       = Field(default_factory=lambda: str(uuid4()))
        session_id:   str
        status:       JobStatus = JobStatus.QUEUED
        submitted_at: datetime  = Field(default_factory=lambda: datetime.now(timezone.utc))
        started_at:   datetime | None = None
        completed_at: datetime | None = None
        error:        str | None      = None
        schema_version: str = "1.0"

queue_position is computed dynamically via get_queue_position(job_id) — never stored, because stored values go stale.

### Job Store and Stream Buffer

backend/queue.py constants and state:

    MAX_QUEUE_SIZE        = 10
    PIPELINE_TIMEOUT_S    = 180
    JOB_RETENTION_SECONDS = 3600
    STREAM_QUEUE_MAXSIZE  = 256
    STREAM_PUT_TIMEOUT_S  = 30.0

All in-memory state. Correct only with --workers 1 (mandatory per spec). Counters reset on container restart.

### AegisState

Defined once in schemas/state.py. Mutation is intentional and required throughout the pipeline. Do not add model_config frozen=True.

Layout invariants enforced by AegisPipeline:

    tools_run ∩ tools_failed = ∅
    A tool name appears in exactly one list, never both.
    pipeline_complete becomes True when the pipeline finishes,
    regardless of success or failure.

Each *_result field has three possible states:
    None       — tool has not run yet, or input was absent
    Result     — tool ran successfully
    ToolError  — tool ran but failed (fatal flag controls pipeline continuation)

### ToolError

    class ToolError(BaseModel):
        tool:      str
        reason:    str
        timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
        fatal:     bool     = False
        model_config = {"frozen": True}

Fatal vs non-fatal policy:

Scenario                                    fatal
VoiceTranscriber fails, typed text present  False
VoiceTranscriber fails, no typed text       True
LabReportParser fails                       False
Oversized / over-duration upload            True — rejected before queue
Queue full                                  True — rejected before queue
Pipeline wall-clock timeout                 True
SeverityScorer fails                        True
ReportGenerator mid-stream failure          True
SymptomExtractor fails after all retries    True
RAG mechanism fails                         False
RAG zero results                            Not a ToolError — RAGSearchResult(passages=[])
Stream consumer disconnected                True — job FAILED, lock released

### SeverityResult

Single authoritative definition. Invariants enforced by Pydantic model_validator:

    highest_priority_rule == triggered_rules[0]
    len(reasons) == len(triggered_rules)
    triggered_rules contains at least one entry

A malformed SeverityResult cannot exist as a constructed object.

### DrugInteractionResult

    class DrugInteractionSeverity(str, Enum):
        SEVERE   = "severe"
        MODERATE = "moderate"
        MINOR    = "minor"

    class DrugInteraction(BaseModel):
        drugs:       list[str]
        severity:    DrugInteractionSeverity
        description: str

    class DrugInteractionResult(BaseModel):
        resolved:       list[str]
        unresolved:     list[str]
        interactions:   list[DrugInteraction]
        warnings:       list[str]
        confidence:     float
        schema_version: str = "1.0"

confidence = len(resolved) / (len(resolved) + len(unresolved))
Zero resolved is valid data (0.0), not a ToolError.

### RAGSearchResult

    class RAGPassage(BaseModel):
        text:     str
        source:   str
        citation: str

    class RAGSearchResult(BaseModel):
        passages:             list[RAGPassage]
        citations:            list[str]
        query_used:           str
        retrieval_successful: bool
        schema_version:       str = "1.0"

retrieval_successful=True even if passages=[] — ran correctly, found nothing.
retrieval_successful=False only if mechanism failed — ToolError(fatal=False) used instead.

### LabReportResult

    class LabReportResult(BaseModel):
        abnormal_values:    list[str]
        measurements:       dict[str, float]
        extra_measurements: dict[str, float]
        schema_version:     str = "1.0"

measurements uses canonical keys only (tools/lab_constants.py).
extra_measurements preserves unrecognised lab keys for future use without reparsing.

### TriageReport

    class TriageReport(BaseModel):
        severity:               Literal["LOW", "MEDIUM", "HIGH"]
        confidence:             float
        text:                   str
        citations:              list[str]
        disclaimer:             str
        knowledge_base_version: str | None
        knowledge_base_date:    str | None
        schema_version:         str = "1.0"

### Confidence Formula

Two distinct confidence values exist:

SeverityResult.confidence — rule-level certainty
    From rule_confidence on the highest-priority fired rule.
    Defined in _RULES table in tools/severity_scorer.py.
    Set at scoring time. Never recomputed.

TriageReport.confidence — pipeline-level confidence
    From calculate_confidence(state) in tools/confidence.py.
    Formula:
        confidence = 0.4 * coverage + 0.4 * success_rate + 0.2 * truncation
        clamped to [0.0, 1.0]

    Signal 1 — Modality coverage (weight 0.4)
        active_modalities / 5
        symptoms · lab · xray · meds · rag (counted if RAGSearchResult)

    Signal 2 — Tool success rate (weight 0.4)
        len(tools_run) / max(len(tools_run) + len(tools_failed), 1)

    Signal 3 — Truncation score (weight 0.2)
        1.0 — none
        0.7 — enrichment truncated
        0.5 — core truncated (ERROR state)

Severity level does not reduce confidence. LOW severity on clean data yields high confidence.

### Report Sections

Six required sections — authoritative and complete:

    ### Summary
    ### Findings        (Radiological subsection if X-ray submitted)
    ### Evidence        (explicit absence message if passages=[])
    ### Severity        (Confidence line · highest_priority_rule displayed)
    ### Recommendations
    ### Disclaimer

Missing section → ToolError(fatal=True). Partial report never rendered.

Mid-stream failure message — hardcoded in tools/report_generator.py, never LLM-generated:

    "Report generation was interrupted. The output above is incomplete
    and must not be used for clinical decisions. Please resubmit or
    contact support."

### Test Scenarios

Scenario A — Full modality (clean run)
    audio:       tests/fixtures/symptoms_clear.wav
    pdf:         tests/fixtures/lab_normal.pdf
    xray:        tests/fixtures/chest_normal.jpg
    medications: ["Metformin 500mg", "Lisinopril 10mg"]

    expected:
        severity                = LOW
        triggered_rules         = [RULE_DEFAULT_LOW]
        highest_priority_rule   = RULE_DEFAULT_LOW
        confidence              > 0.9
        six sections present
        EasyOCR never loaded
        no ToolError
        pipeline_duration_ms    < 60000
        job.status              = COMPLETED

Scenario B — Partial modality (degraded run)
    audio:       None
    text:        "Chest pain for 3 days, shortness of breath"
    pdf:         tests/fixtures/lab_scanned.pdf
    xray:        None
    medications: ["AEGIS_TEST_UNRESOLVABLE_DRUG_XYZ"]

    expected:
        severity                = HIGH
        triggered_rules ⊇       {RULE_CHEST_PAIN_AND_SOB}
        highest_priority_rule   = RULE_CHEST_PAIN_AND_SOB
        drug_result.unresolved  = ["AEGIS_TEST_UNRESOLVABLE_DRUG_XYZ"]
        drug_result.confidence  = 0.0
        rag_result              = RAGSearchResult (not ToolError)
        confidence              < 1.0
        six sections present
        pipeline_duration_ms    < 90000

Scenario C — Token budget stress
    audio:       None
    text:        [800-token symptom description]
    pdf:         tests/fixtures/lab_verbose.pdf
    xray:        tests/fixtures/chest_findings.jpg
    medications: [20 medications]

    expected:
        context EXCEEDS num_ctx before generation
        truncation fired and logged
        core_fields_truncated  = False
        six sections present
        no OOM

Tests must assert on rule constants from tools.severity_scorer.ALL_RULE_CONSTANTS, never on reasons strings.

---

## Concurrency and Performance

### Multi-User, Single-Inference-Worker Architecture

Multiple users may submit sessions concurrently. Inference executes one pipeline at a time. Sessions are queued FIFO and processed sequentially.

llama3.2:1b plus OCR-active peaks at approximately 3.4–4.1 GB. Parallel inference on 8 GB would risk OOM. OLLAMA_NUM_PARALLEL=1 and OLLAMA_MAX_LOADED_MODELS=1 remain mandatory.

### API Endpoints

Endpoint                            Method  Description
POST /queue/submit                  POST    Validate → create job → enqueue. Returns PipelineJob. Rejects if full.
GET /queue/status/{job_id}          GET     Returns PipelineJob + live queue_position + estimated_wait_seconds + pipeline state.
GET /queue/stream/{job_id}          GET     Drains asyncio.Queue via None sentinel. Returns 425 if QUEUED.
GET /health                         GET     System status + queue metrics. Never blocks.

Full endpoint reference: docs/api.md.

### Upload Bounds — Enforced Before Queue Entry

Input            Limit                              On violation
Lab PDF          ≤ 25 MB                            ToolError(fatal=True) before queue
X-ray image      ≤ 25 MB                            ToolError(fatal=True) before queue
Audio            ≤ 15 MB and ≤ 120 s                ToolError(fatal=True) before queue
Medications      ≤ 50 entries                       ToolError(fatal=True) before queue
Queue            ≤ 10 jobs                          ToolError(fatal=True) before queue

### Ollama Configuration

OLLAMA_BASE_URL environment variable controls Ollama connectivity.
Default: http://localhost:11434
Docker:  http://ollama:11434 (set in docker-compose.yml)

OLLAMA_STREAM_URL = OLLAMA_BASE_URL + "/api/generate" constructed at import time in tools/report_generator.py.

Ollama concurrency guards (mandatory):

    OLLAMA_NUM_PARALLEL=1
    OLLAMA_MAX_LOADED_MODELS=1
    OLLAMA_KEEP_ALIVE=-1

### Frontend — Client Behaviour Contract

The frontend is not yet implemented (see Frontend Status section above).

Whichever client consumes the API must:

    1. POST inputs to /queue/submit
    2. Poll GET /queue/status/{job_id} every 2 seconds
    3. When status transitions to "running", open GET /queue/stream/{job_id}
    4. Consume the chunked text response token by token
    5. After stream ends, check status again — if "failed", display
       the mid-stream failure message defined in tools/report_generator.py

This contract is satisfied identically by Streamlit, React, curl, or any other client.

---

## Model Stack

### Language Model — llama3.2:1b

num_ctx 4096 via Modelfile. Two LLM calls per run.

    FROM llama3.2:1b
    PARAMETER num_ctx    4096
    PARAMETER num_predict 1024
    PARAMETER temperature 0.2

### SymptomExtractor Retry Policy

    Attempt 1: Standard structured prompt → Pydantic validate → proceed if valid
    Attempt 2: JSON-only with schema verbatim → Pydantic validate → proceed if valid
    Attempt 3: Local JSON repair → proceed if valid · ToolError(fatal=True) if not

Max added latency: 2–6 seconds. Acceptable — symptom extraction is non-negotiable.

### Embeddings

all-MiniLM-L6-v2 exported to ONNX. Committed as data/knowledge/minilm.onnx. No torch.

### Voice Transcription

Faster-Whisper tiny.en INT8. CPU-only. ~150 MB. Released after transcription.

### OCR

Stage       Tool                Memory
Primary     PyMuPDF             negligible
Fallback 1  pdfminer.six        negligible
Fallback 2  EasyOCR (opt-in)    ~1.2–1.5 GB

EasyOCR release:

    del self.ocr_reader
    self.ocr_reader = None
    gc.collect()

Validate EasyOCR ARM64 in Week 1. If it fails: switch to pytesseract, redefine Scenario B, document in docs/setup_jetson.md.

### Drug Data

SQLite FTS5. Built on Mac, committed. Returns DrugInteractionResult with structured DrugInteraction objects.

---

## Tools

Steps 0–7 unchanged. AegisPipeline is unaware of the queue — it receives AegisState and runs to completion.

### Step 0 — VoiceTranscriber (optional)

Bounded ≤ 15 MB / ≤ 120 s. CPU-only. Lazy-load, transcribe, release.
Writes raw_symptoms_text. Skipped if no audio.

### Step 1 — SymptomExtractor

llama3.2:1b. Three-attempt retry/repair. ToolError(fatal=True) after all fail.

### Step 2 — LabReportParser

Bounded ≤ 25 MB. PyMuPDF → pdfminer.six → EasyOCR (opt-in). Explicit EasyOCR release.
Alias normalisation map normalises all variants to canonical keys.
Unknown keys preserved in extra_measurements.

### Step 3 — XRayProcessor

Bounded ≤ 25 MB. PIL + DICOM. Pre-collected findings. No GPU.

Findings checklist:
    Cardiomegaly · Pleural Effusion · Pneumonia · Pneumothorax
    Consolidation · Atelectasis · Infiltrates · Pulmonary Edema
    Nodule/Mass · Fracture · Normal/No significant findings
    Plus free text field

### Step 4 — MedicalRAGSearch

Query from Steps 1–3. Zero results → RAGSearchResult(passages=[], retrieval_successful=True).
ToolError(fatal=False) only if mechanism fails.

### Step 5 — DrugInteractionChecker

Bounded ≤ 50. FTS5. Returns DrugInteractionResult with structured DrugInteraction objects. Zero model load.

### Step 6 — SeverityScorer

Deterministic. Priority-driven rule engine.
13 real rules + RULE_DEFAULT_LOW fallback.
Full reference: docs/severity_rules.md.

### Step 7 — ReportGenerator

Six-section validated streaming. One LLM call. Buffer-validate-yield.

Token budget priority:
    1. SeverityResult           never truncated
    2. SymptomExtractionResult  symptoms + duration + indicators never truncated
                                medical_entities + negations may be truncated
    3. LabReportResult          abnormal values protected; measurements may be truncated
    4. DrugInteractionResult    flagged interactions if space allows
    5. XRayResult               checklist summary if space allows
    6. RAGSearchResult          top-2 citations if space allows
    7. Pipeline metadata        dropped first

---

## Severity Rules

Full specification in docs/severity_rules.md.
Every rule has a fixture. test_all_rules_have_fixtures, test_highest_priority_rule_invariant, test_reasons_length_invariant enforce correctness.

Rule constants and thresholds defined in:
    tools/severity_scorer.py    rule constants and check functions
    tools/lab_thresholds.py     all numeric thresholds (ABNORMAL_* and CRITICAL_*)
    tools/lab_constants.py      canonical lab measurement keys

ALL_RULE_CONSTANTS is auto-derived in tools/severity_scorer.py:

    ALL_RULE_CONSTANTS = [r.constant for r in _RULES] + [RULE_DEFAULT_LOW]

---

## Medical Knowledge

Source                          Status
MedlinePlus (NIH)               Planned v1.0 — not yet built
PubMed Abstracts                Planned v2.0
NIH Clinical Guidelines         Planned v2.0
WHO Public Health Guidance      Planned v2.0

knowledge_base_version and knowledge_base_date loaded from docs/corpus_version.md at startup, injected into every TriageReport. Currently null until corpus is built.

---

## Memory Management

State                                       Estimated Peak
Idle                                        ~2.2–2.6 GB
Voice active                                ~2.4–2.8 GB
Digital-PDF                                 ~2.3–2.7 GB
OCR active (EasyOCR, scanned only)          ~3.4–4.1 GB
X-Ray active                                ~2.2–2.6 GB

Stream buffers (asyncio.Queue maxsize=256) and _completed_durations (deque maxlen=100) contribute negligibly to peak. Measure on Jetson in Week 1. Update docs/memory_profile.md.

If OCR-active exceeds 5.5 GB: EasyOCR staged-loading in Week 2.

---

## Tech Stack

Layer               Technology
LLM Runtime         Ollama (digest-pinned · NUM_PARALLEL=1 · MAX_LOADED_MODELS=1 · KEEP_ALIVE=-1)
LLM Model           llama3.2:1b · num_ctx 4096 · temperature 0.2
Pipeline            AegisPipeline (pure Python, async, wall-clock bounded)
Queue               backend/queue.py · asyncio FIFO · single worker · MAX_QUEUE_SIZE=10
Stream buffer       asyncio.Queue(maxsize=256) per job · put_nowait(None) in finally
Duration tracking   deque(float, maxlen=100) · rolling average of last 10
Validation          Pydantic v2
Embeddings          all-MiniLM-L6-v2 via ONNX Runtime (ARM64)
Vector Store        ChromaDB primary + FAISS fallback
Voice               Faster-Whisper tiny.en INT8 (CPU-only, optional)
PDF                 PyMuPDF → pdfminer.six → EasyOCR (opt-in, explicitly released)
X-Ray               XRayProcessor (PIL + DICOM, clinician-assisted)
Drugs               SQLite FTS5 · OpenFDA + RxNorm
Upload Guards       backend/uploads.py — before queue entry
Logging             Loguru (structured JSON)
API                 FastAPI (single Uvicorn worker)
GPU Detection       tegrastats / nvidia-smi — no torch
Frontend            React + Vite + TypeScript + Tailwind + shadcn/ui (planned — not yet implemented)
                    Streamlit prototype deprecated
Containers          Docker ARM64 + Compose
Hardware            NVIDIA Jetson Orin Nano 8 GB
Python              3.11+ required (asyncio.timeout requires 3.11)

---

## Repository Structure

    aegis-health/
    ├── backend/
    │   ├── __init__.py
    │   ├── main.py
    │   ├── health.py
    │   ├── queue.py
    │   ├── streaming.py
    │   └── uploads.py
    ├── frontend/                       deprecated Streamlit prototype
    │                                   will be replaced by React + Vite project
    ├── agents/
    │   ├── __init__.py
    │   └── pipeline.py
    ├── tools/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── tool_names.py               pipeline tool name constants
    │   ├── lab_constants.py            canonical lab measurement keys
    │   ├── lab_thresholds.py           all numeric lab thresholds
    │   ├── voice_transcriber.py
    │   ├── symptom_extractor.py
    │   ├── lab_report_parser.py
    │   ├── medical_rag_search.py
    │   ├── drug_checker.py
    │   ├── severity_scorer.py
    │   ├── confidence.py
    │   └── report_generator.py
    ├── vision/
    │   ├── __init__.py
    │   ├── xray_processor.py
    │   └── dicom_reader.py
    ├── rag/
    │   ├── __init__.py
    │   ├── download_corpus.py
    │   ├── ingest.py
    │   ├── chunk.py
    │   ├── export_minilm_onnx.py
    │   ├── embed.py
    │   ├── build_chroma.py
    │   ├── build_faiss.py
    │   └── retriever.py
    ├── schemas/
    │   ├── __init__.py
    │   ├── queue.py
    │   ├── state.py
    │   ├── errors.py
    │   ├── symptom.py
    │   ├── voice.py
    │   ├── rag.py
    │   ├── lab.py
    │   ├── drugs.py
    │   ├── severity.py
    │   ├── xray.py
    │   └── report.py
    ├── config/
    │   └── Modelfile
    ├── data/
    │   ├── knowledge/
    │   │   ├── raw/                    gitignored
    │   │   ├── minilm.onnx
    │   │   ├── chroma/
    │   │   ├── faiss.index
    │   │   └── faiss.docs
    │   └── drugs/
    │       ├── build_drug_db.py
    │       ├── aegis_drugs.db
    │       └── rxnorm_map.json
    ├── docker/
    │   ├── Dockerfile
    │   ├── docker-compose.yml
    │   └── entrypoint.sh
    ├── tests/
    │   ├── __init__.py
    │   ├── tools/
    │   ├── integration/
    │   ├── benchmarks/
    │   └── scenarios/
    └── docs/
        ├── Technical_Project_Spec.md
        ├── Thesis.md
        ├── architecture.md
        ├── api.md
        ├── severity_rules.md
        ├── corpus_version.md
        ├── memory_profile.md
        ├── setup_jetson.md
        └── requirements.txt

---

## Deployment

Mac-only setup:

    python rag/download_corpus.py
    python rag/export_minilm_onnx.py
    python rag/build_chroma.py && python rag/build_faiss.py

    echo "snapshot_date: $(date -u +%Y-%m-%d)" >> docs/corpus_version.md
    echo "source_url: https://medlineplus.gov/xml.html" >> docs/corpus_version.md
    echo "git_commit: $(git rev-parse HEAD)" >> docs/corpus_version.md

    python data/drugs/build_drug_db.py

    ollama pull llama3.2:1b
    ollama create aegis-llama -f config/Modelfile

    docker inspect ollama/ollama:latest --format '{{index .RepoDigests 0}}'
    git add data/ config/ && git commit -m "build: add committed assets"

Jetson:

    git clone <repo> && cd aegis-health && docker compose up

---

## 6-Week Development Plan

Week 1 — Contracts First, Then Foundation

Mandatory pre-implementation gate:

    pytest --collect-only

Implementation order:

    schemas/errors.py
    schemas/queue.py
    schemas/severity.py
    schemas/drugs.py · rag.py · symptom.py · voice.py · lab.py · xray.py
    schemas/report.py
    schemas/state.py
    backend/uploads.py
    backend/queue.py
    agents/pipeline.py
    backend/main.py · health.py · streaming.py
    tools/tool_names.py · lab_constants.py · lab_thresholds.py
    tools/severity_scorer.py
    tests/tools/test_severity_scorer.py
    tools/symptom_extractor.py · drug_checker.py
    tests/integration/test_queue.py
    tests/benchmarks/test_queue_load.py

Week 1 primary exit criterion:
    Scenario A passes end-to-end on Jetson — submitted via queue, position
    accurate, highest_priority_rule=RULE_DEFAULT_LOW, report streams fully,
    pipeline_duration_ms recorded.

Required gates:
    pytest --collect-only zero errors
    All schema imports clean
    test_all_rules_have_fixtures green
    test_highest_priority_rule_invariant green
    test_reasons_length_invariant green
    test_disconnected_client green
    test_queue_load green
    deque(maxlen=100) confirmed in _completed_durations
    put_nowait(None) confirmed in finally block
    Jetson baseline memory in docs/memory_profile.md
    EasyOCR ARM64 decision documented

Week 2 — Retrieval + OCR + Voice + X-Ray

Build and commit ONNX indexes. Implement LabReportParser, VoiceTranscriber, XRayProcessor. Validate full tool suite on Jetson. Measure OCR-active memory peak.

Done when:
    minilm.onnx committed, docs/corpus_version.md complete
    RAG zero results → RAGSearchResult(passages=[]) confirmed in practice
    FAISS fallback tested
    EasyOCR never loads on digital PDF; explicit release confirmed
    VoiceTranscriber Step 0 → Step 1 chaining confirmed
    OCR-active peak measured and documented
    docs/memory_profile.md updated

If OCR-active peak > 5.5 GB: EasyOCR staged-loading implemented here.

Week 3 — Full Integration = Minimum Viable Demo

Done when:
    All severity rules, all fixtures green
    calculate_confidence implemented and correct
    ReportGenerator: six-section streaming, validation, missing section → ToolError
    Zero-RAG and RAG ToolError handled in Evidence section
    Golden-output tests written for A/B/C
    Scenario A passes on Jetson
    Scenario B passes on Jetson
    Scenario C passes on Jetson
    No Pydantic validation errors across A/B/C

End of Week 3: Minimum Viable Demo
    Five modalities · Eight tools · Streamed report · Six validated sections
    Deterministic severity · Defined confidence · Golden tests · Three validated scenarios
    Jetson · Fully local.

Week 4 — Optimisation and Stability

Real Loguru data. No new features. No schema changes.

Done when:
    10 consecutive Scenario A runs on Jetson without failure
    Golden-output tests pass across all 10
    Memory within measured limits
    Scenario C stable across 5 consecutive runs

End of Week 4: Feature Freeze.

Week 5 — Frontend Implementation + Demo Preparation

React + Vite + TypeScript + Tailwind + shadcn/ui project scaffolded.
CORS middleware added to backend/main.py.
Components implemented against the stable API contract.
Tier 2 Mac confirmed. Tier 3 pre-recorded. All fixtures tested. Demo script rehearsed.

Week 6 — Demo and Submission

3 Scenario A run-throughs. Tier 2 switch under 2 minutes. Submitted.

---

## Demo Scenario

Multiple clinicians submit sessions. Each receives a job_id immediately. UI shows queue position and estimated wait. When job reaches RUNNING, report streams section-by-section. Severity deterministic — triggered_rules, highest_priority_rule, and reasons surfaced. Confidence calculated. Knowledge base provenance in every report. Entirely local on Jetson.

---

## Demo Backup Strategy

Tier        Setup                                       When
Tier 1      Live Jetson Orin Nano 8 GB                  Primary
Tier 2      Live Mac (identical Docker image)           Jetson unstable
Tier 3      Pre-recorded Scenario A on Jetson           Both fail

---

## Non-Functional Requirements

Requirement                         Detail
Multi-user queued                   FIFO queue · single inference worker · MAX_QUEUE_SIZE=10
Queue position                      Computed dynamically — never stored
Estimated wait                      Rolling average of last 10 · null until 3+ completions
Queue full                          ToolError(fatal=True) before queue entry
Job lifecycle                       QUEUED → RUNNING → COMPLETED / FAILED
Job retention                       JOB_RETENTION_SECONDS=3600 · stream queue deleted on purge
Stream buffer                       asyncio.Queue(maxsize=256) · bounded · backpressure
Sentinel                            put_nowait(None) in finally · unconditional · non-blocking
Per-token timeout                   asyncio.wait_for(..., STREAM_PUT_TIMEOUT_S=30.0)
Disconnected client                 Job → FAILED · sentinel emitted · lock released
Duration tracking                   deque(float, maxlen=100) · never a plain list
No parallel inference               OLLAMA_NUM_PARALLEL=1 · OLLAMA_MAX_LOADED_MODELS=1
Ollama image pinned                 Digest in docs/setup_jetson.md
Warmup                              Failure logged via Loguru
Single worker                       One Uvicorn worker
Confidence formula                  Non-overlapping signals · documented · tunable
Severity rules                      In docs/severity_rules.md before implementation
Rule constants                      ALL_RULE_CONSTANTS · auto-derived · imported by tests
highest_priority_rule               Always triggered_rules[0] · set at scoring time
len(reasons) == len(triggered_rules) Enforced by SeverityScorer schema validator
Six report sections                 Validated · missing → ToolError(fatal=True)
Zero-RAG                            RAGSearchResult(passages=[]) · explicit message
Knowledge base provenance           knowledge_base_version + knowledge_base_date in every report
Golden-output tests                 Week 3 · assert constants not strings
pytest --collect-only gate          Zero errors before any tool logic
Contracts before code               All schemas + rules before implementation
Timezone-aware                      datetime.now(timezone.utc) throughout
Mutable defaults                    Field(default_factory=...) everywhere
Single schema source                schemas/ only
Session isolation                   uuid4 — never client-supplied
gitignored raw data                 data/knowledge/raw/
Week 1 exit criterion               Scenario A passes end-to-end on Jetson
Stable                              10 consecutive Scenario A runs — Week 4
Observable                          Per-tool JSON events via Loguru
Frontend-agnostic backend           Same endpoints serve Streamlit, React, curl, anything

---

## Post-Demo Enhancements

SQLite job persistence              Cross-restart durability for PipelineJob metadata
SSE push notifications              Replace 2s polling
Confidence formula tuning           Denominator=4 if RAG-in-coverage misleads clinicians
Priority queue                      FIFO override for clinical urgency
Auto-generated TS types             From FastAPI OpenAPI schema

---

## Scope Boundaries

In scope:
    Multi-user queued sessions · job lifecycle · live queue position
    Wait estimates · queue metrics · all clinical modalities
    Six-section streaming report

Out of scope (until post-demo):
    True concurrent inference · SSE · persistent jobs
    User authentication · FIFO priority overrides
    Production-grade frontend (React frontend lands in Week 5+)

---

## Safety and Limitations

Does not diagnose. Assists triage — does not replace clinical judgment.
All outputs require review by a qualified healthcare professional.
Not for emergencies. Contact emergency services immediately.
Severity rule-based and fully auditable — triggered_rules maps to docs/severity_rules.md.
Queue capacity limited — full queue returns explicit rejection.
Queue not persistent across restarts.
FIFO does not account for clinical urgency.
Disconnected stream clients cause job failure — resubmit required.
Estimated wait approximate — unavailable until 3+ completions.
Knowledge base limited to indexed MedlinePlus corpus (when built).
Partial results always flagged — never presented as complete.

---

## AI Domains Covered

Agentic AI · Small Language Models · Retrieval-Augmented Generation
Vector Databases · Semantic Search · Document AI · OCR
Speech-to-Text · Edge AI Deployment · Prompt Engineering · MLOps
Privacy-Preserving AI · Model Quantisation · ONNX Runtime Optimisation

---

Aegis Health — built for privacy, designed for the edge.