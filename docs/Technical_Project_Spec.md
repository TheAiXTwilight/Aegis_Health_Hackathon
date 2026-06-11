# Aegis Health
### Local AI Triage Assistant · Edge Deployment · Privacy-First

---

## Overview

Aegis Health is a fully local AI triage assistant that accepts patient inputs across five clinical data modalities — symptom input (typed or voice), lab report PDFs, chest X-rays, medication lists, and retrieved medical evidence — and produces a structured, evidence-backed triage report. All inference runs on-device with no external API calls at any stage.

The system is pipeline-driven: a custom **AegisPipeline** orchestrator routes inputs through a suite of specialised tools in a deterministic sequence. **llama3.2:1b** via Ollama acts exclusively as the synthesis layer, receiving validated structured outputs from all tools and streaming a coherent plain-language report to the UI. The intelligence of the system comes from tool composition and deterministic rules — not from the language model alone.

Designed for deployment on an **NVIDIA Jetson Orin Nano (8 GB)**.

---

## The Problem

AI-powered healthcare tools almost universally depend on cloud APIs — patient data is transmitted to and processed on remote infrastructure outside clinical control. This creates real privacy risks and makes deployment impractical in low-connectivity or resource-constrained settings.

Existing tools compound this by operating on single input types — a symptom checker, an image analyser, a lab report parser — with no system capable of reasoning across all modalities together, locally, without a cloud dependency.

**Aegis Health addresses both.** All inference runs locally on the Jetson with no external calls. The system reasons across five clinical data modalities, retrieves evidence from authoritative medical sources, and streams a structured triage report — patient data never leaves the device.

It does not diagnose. It triages — giving clinicians a structured, evidence-backed starting point while keeping patient data private and under clinical control.

---

## Core Capabilities

| Input Modality | Input Method | Processing |
|---|---|---|
| Symptom description | Typed text **or** voice recording (one per session) | VoiceTranscriber (if audio) → SymptomExtractor |
| Lab report | PDF upload | Three-stage extraction, abnormal value detection |
| Chest X-ray | Image upload | DICOM metadata extraction, clinician findings form |
| Medications | Typed list | SQLite FTS5 lookup, interaction checking |
| Medical evidence | Derived from all inputs | RAG retrieval, MedlinePlus citations |

> Voice and typed text are **alternative input paths** for symptom input — not separate modalities. One is used per session. The system supports five distinct clinical input modalities total.

> **All inputs — including all clinician X-ray findings — are collected in the UI *before* submission.** The pipeline runs uninterrupted once started; no step blocks on human input mid-run.

---

## Architecture

### Pipeline Execution Order

```
┌─────────────────────────────────────────────────────────────┐
│                        AegisPipeline                        │
│              Deterministic sequential orchestrator          │
│              Pure Python · No framework · Wall-clock bounded │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ▼  [STEP 0 — optional, skipped if no audio submitted]
    ┌─────────────────┐
    │ VoiceTranscriber│  Faster-Whisper tiny.en INT8
    │                 │  CPU-only · lazy-loaded · released immediately
    │                 │  Output → populates raw_symptoms_text in AegisState
    └────────┬────────┘
             │
             ▼  [STEP 1]
    ┌─────────────────┐
    │ SymptomExtractor│  llama3.2:1b structured prompt
    │                 │  3-attempt retry/repair before ToolError
    └────────┬────────┘
             │
             ▼  [STEP 2]
    ┌─────────────────┐
    │  LabReportParser│  PyMuPDF → pdfminer.six → EasyOCR (opt-in)
    │                 │  EasyOCR lazy-loaded · explicitly released
    └────────┬────────┘
             │
             ▼  [STEP 3]
    ┌─────────────────┐
    │  XRayProcessor  │  PIL · DICOM · clinician findings pre-collected
    │                 │  No GPU · No ML model · No mid-pipeline human wait
    └────────┬────────┘
             │
             ▼  [STEP 4]
    ┌─────────────────┐
    │ MedicalRAGSearch│  ONNX MiniLM · ChromaDB (primary) / FAISS (fallback)
    │                 │  Zero results → RAGSearchResult(passages=[])
    └────────┬────────┘
             │
             ▼  [STEP 5]
    ┌─────────────────┐
    │DrugInteraction  │  SQLite FTS5 · OpenFDA + RxNorm
    │    Checker      │  Zero model load · pure stdlib
    └────────┬────────┘
             │
             ▼  [STEP 6]
    ┌─────────────────┐
    │ SeverityScorer  │  Rule-based · fully deterministic
    │                 │  triggered_rules · highest_priority_rule
    │                 │  reasons · contributing_tools
    └────────┬────────┘
             │
             ▼  [STEP 7]
    ┌─────────────────┐
    │ ReportGenerator │  llama3.2:1b · num_ctx 4096
    │                 │  Six sections · buffer-validate-yield
    └────────┬────────┘
             │
             ▼
    Final Triage Assessment
    Streamed → FastAPI StreamingResponse → Streamlit st.write_stream
    Live performance sidebar via Loguru JSON events
```

### Architecture Principles

**Separation of concerns.** The language model acts exclusively as the synthesis layer. SeverityScorer handles triage level through fully specified rule-based logic. The LLM explains severity but cannot change it.

**Sequential order is a memory-safety choice.** Steps run sequentially so only one tool holds resident memory at a time.

**No mid-pipeline human wait.** All human input collected before submission. Pipeline runs to completion once dequeued.

**Contracts before code.** All schemas, severity rules, rule constants, queue schema, and retry policy must exist before implementation that depends on them.

**Honest failure mode.** Every failure produces an explicit `ToolError`. Zero evidence is valid output.

**Single state object.** `AegisState` defined once in `schemas/state.py`.

### Why AegisPipeline over LangGraph

| Concern | LangGraph | AegisPipeline |
|---|---|---|
| ARM64 compatibility | Known ARM64 build failures | Zero external dependencies |
| Problem fit | Non-deterministic graphs | Sequential deterministic pipelines |
| Streaming | Complex version-sensitive API | Simple async generator |
| Debuggability | 40+ transitive dependencies | Fully readable, internally owned |
| Upstream risk | API changed across major versions | No upstream breakage possible |

---

## Data Contracts

### Schema Version Policy

> `schema_version` increments only on **breaking contract changes**. Additive changes do not increment. Current version: `"1.0"`. Schemas carrying `schema_version`: `SeverityResult`, `DrugInteractionResult`, `RAGSearchResult`, `TriageReport`, `PipelineJob`.

### PipelineJob

Defined in `schemas/queue.py`.

```python
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field
from uuid import uuid4


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

    # queue_position is NOT stored here.
    # Computed dynamically via get_queue_position(job_id).
    # Stored values go stale as jobs ahead complete.

    schema_version: str = "1.0"
```

### Job Store and Stream Buffer

```python
# backend/queue.py — constants and state

from collections import deque

MAX_QUEUE_SIZE        = 10
PIPELINE_TIMEOUT_S    = 180
JOB_RETENTION_SECONDS = 3600
STREAM_QUEUE_MAXSIZE  = 256
STREAM_PUT_TIMEOUT_S  = 30.0
# Per-token timeout on stream_q.put(). If a token cannot be placed within
# this window, the client is considered disconnected. Job is marked FAILED,
# sentinel emitted in finally, inference lock released, next job runs.

_job_store:    dict[str, PipelineJob]                = {}
_job_queue:    deque[str]                            = deque()
_job_streams:  dict[str, asyncio.Queue[str | None]]  = {}
# One asyncio.Queue(maxsize=STREAM_QUEUE_MAXSIZE) per active job.
# Created when job transitions to RUNNING.
# Deleted when job is purged from _job_store after JOB_RETENTION_SECONDS.

_inference_lock: asyncio.Lock = asyncio.Lock()

_completed_durations: deque[float] = deque(maxlen=100)
# Bounded rolling window. Oldest entries dropped automatically.
# maxlen=100 bounds memory regardless of service uptime.
# Only last 10 used for averaging — extra history available for future analytics.

_jobs_completed_today: int = 0
_jobs_failed_today:    int = 0
# In-memory only. Reset on container restart.
# Post-demo enhancement: persist to SQLite for cross-restart accuracy.

# On container startup, all state above is empty.
# No jobs, streams, or counters survive a restart. This is by design.
```

### `_execute_job()` — Definitive Implementation

```python
async def _execute_job(job: PipelineJob) -> None:
    """
    Executes one pipeline job under the inference lock.
    Called exclusively by run_inference_worker() while _inference_lock is held.

    Sentinel guarantee: stream_q sentinel (None) always emitted in finally —
    exactly once, unconditionally, regardless of exit path.
    put_nowait used so the cleanup path itself cannot block.

    Disconnected client protection: each stream_q.put(token) wrapped with
    asyncio.wait_for(..., timeout=STREAM_PUT_TIMEOUT_S). If consumer has not
    read within that window, job is marked FAILED and finally emits sentinel.
    """
    global _jobs_completed_today, _jobs_failed_today

    stream_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=STREAM_QUEUE_MAXSIZE)
    _job_streams[job.job_id] = stream_q

    job.status     = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)

    try:
        state = _get_state(job.session_id)
        async for token in _bounded(pipeline.run(state), PIPELINE_TIMEOUT_S):
            try:
                await asyncio.wait_for(
                    stream_q.put(token),
                    timeout=STREAM_PUT_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                job.status       = JobStatus.FAILED
                job.completed_at = datetime.now(timezone.utc)
                job.error        = (
                    f"Stream consumer did not read within {STREAM_PUT_TIMEOUT_S}s. "
                    "Client may have disconnected."
                )
                _jobs_failed_today += 1
                return  # finally will emit sentinel

        job.status       = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        duration = (job.completed_at - job.started_at).total_seconds()
        _completed_durations.append(duration)
        _jobs_completed_today += 1

    except asyncio.TimeoutError:
        job.status       = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.error        = f"Pipeline exceeded {PIPELINE_TIMEOUT_S}s wall-clock limit."
        _jobs_failed_today += 1

    except Exception as e:
        job.status       = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.error        = str(e)
        _jobs_failed_today += 1

    finally:
        # Non-blocking sentinel. put_nowait ensures cleanup cannot block
        # even if queue is full (consumer gone). Sentinel is best-effort
        # for disconnected consumers — they are not reading it anyway.
        try:
            stream_q.put_nowait(None)
        except asyncio.QueueFull:
            pass
```

### AegisState

Defined once in `schemas/state.py`.

> **`tools_run` and `tools_failed` are mutually exclusive.** A tool name appears in exactly one list, never both. Enforced by AegisPipeline.

```python
from datetime import datetime, timezone
from uuid import uuid4
from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AegisState(BaseModel):
    # ── Session ──────────────────────────────────────────────
    session_id:   str      = Field(default_factory=lambda: str(uuid4()))
    submitted_at: datetime = Field(default_factory=_utcnow)

    # ── Inputs ───────────────────────────────────────────────
    raw_symptoms_text: str | None = None
    audio_file_path:   str | None = None
    lab_pdf_path:      str | None = None
    xray_image_path:   str | None = None
    medications_raw:   list[str]  = Field(default_factory=list)

    # ── Tool Outputs ─────────────────────────────────────────
    voice_result:    VoiceTranscriptionResult | ToolError | None = None
    symptom_result:  SymptomExtractionResult  | ToolError | None = None
    lab_result:      LabReportResult          | ToolError | None = None
    xray_result:     XRayResult               | ToolError | None = None
    rag_result:      RAGSearchResult          | ToolError | None = None
    drug_result:     DrugInteractionResult    | ToolError | None = None
    severity_result: SeverityResult           | ToolError | None = None

    # ── Pipeline Metadata ────────────────────────────────────
    tools_run:    list[str] = Field(default_factory=list)
    # Mutually exclusive with tools_failed — enforced by AegisPipeline

    tools_failed: list[str] = Field(default_factory=list)
    # Mutually exclusive with tools_run — enforced by AegisPipeline

    pipeline_complete: bool = False

    # ── Timing ───────────────────────────────────────────────
    pipeline_start_ms: float | None = None
    pipeline_end_ms:   float | None = None
    step_durations_ms: dict[str, float] = Field(default_factory=dict)

    # ── Truncation Tracking ──────────────────────────────────
    core_fields_truncated:       bool = False
    enrichment_fields_truncated: bool = False

    # ── Final Output ─────────────────────────────────────────
    report: TriageReport | None = None
```

### ToolError

```python
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ToolError(BaseModel):
    tool:      str
    reason:    str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    fatal:     bool     = False
```

**Fatal vs non-fatal policy:**

| Scenario | fatal |
|---|---|
| VoiceTranscriber fails, typed text present | `False` |
| VoiceTranscriber fails, no typed text | `True` |
| LabReportParser fails | `False` |
| Oversized / over-duration upload | `True` — rejected before queue |
| Queue full | `True` — rejected before queue |
| Pipeline wall-clock timeout | `True` |
| SeverityScorer fails | `True` |
| ReportGenerator mid-stream failure | `True` |
| SymptomExtractor fails after all retries | `True` |
| RAG mechanism fails | `False` |
| RAG zero results | **Not a ToolError** — `RAGSearchResult(passages=[])` |
| Stream consumer disconnected (per-token timeout) | `True` — job FAILED, lock released |

### SeverityResult

Defined in `schemas/severity.py`. **Single authoritative definition.**

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class SeverityResult(BaseModel):
    level: Literal["LOW", "MEDIUM", "HIGH"]

    confidence: float = Field(ge=0.0, le=1.0)

    triggered_rules: list[str] = Field(
        default_factory=list,
        description=(
            "All rule constants that fired, in descending priority order. "
            "Machine-readable. Maps to docs/severity_rules.md. "
            "Assert on these in tests — never on reasons strings."
        ),
    )

    highest_priority_rule: str | None = Field(
        default=None,
        description=(
            "The single rule that determined the severity level. "
            "Always equals triggered_rules[0] when triggered_rules is non-empty. "
            "Set by SeverityScorer at scoring time — never recomputed downstream. "
            "Used by UI for primary reason display and analytics."
        ),
    )

    reasons: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable explanations, one per triggered rule, in priority order. "
            "len(reasons) == len(triggered_rules) — enforced by SeverityScorer. "
            "Wording may change without breaking tests."
        ),
    )

    contributing_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names whose outputs contributed to at least one triggered rule."
        ),
    )

    schema_version: str = "1.0"
```

> **Invariants enforced by SeverityScorer:**
> - `highest_priority_rule == triggered_rules[0]` when `triggered_rules` is non-empty
> - `len(reasons) == len(triggered_rules)`
> - Tests assert on constants, never on `reasons` strings

### DrugInteractionResult

```python
from pydantic import BaseModel, Field


class DrugInteractionResult(BaseModel):
    resolved:       list[str] = Field(default_factory=list)
    unresolved:     list[str] = Field(default_factory=list)
    interactions:   list[str] = Field(default_factory=list)
    warnings:       list[str] = Field(default_factory=list)
    confidence:     float     = Field(ge=0.0, le=1.0)
    # len(resolved) / (len(resolved) + len(unresolved))
    # Zero resolved = 0.0 — valid data, not ToolError
    schema_version: str       = "1.0"
```

### RAGSearchResult

```python
from pydantic import BaseModel, Field


class RAGPassage(BaseModel):
    text:     str
    source:   str
    citation: str


class RAGSearchResult(BaseModel):
    passages:             list[RAGPassage] = Field(default_factory=list)
    citations:            list[str]        = Field(default_factory=list)
    query_used:           str
    retrieval_successful: bool
    # True even if passages=[] — ran correctly, found nothing
    # False only if mechanism failed — ToolError(fatal=False) used instead
    schema_version:       str = "1.0"
```

### TriageReport

```python
from pydantic import BaseModel, Field


class TriageReport(BaseModel):
    severity:               str
    confidence:             float
    text:                   str
    citations:              list[str] = Field(default_factory=list)
    disclaimer:             str
    knowledge_base_version: str | None = None
    knowledge_base_date:    str | None = None
    schema_version:         str = "1.0"
```

### Confidence Formula

```python
def calculate_confidence(state: "AegisState") -> float:
    """
    Three non-overlapping signals.
    Precondition: tools_run ∩ tools_failed = ∅ (enforced by AegisPipeline).

    Signal 1 — Modality coverage (weight 0.4)
        coverage = active_modalities / 5
        symptoms · lab · xray · meds · rag (derived — counted if ran)
        NOTE: denominator=5 includes RAG. Post-Week-3 tuning candidate
        if clinicians find RAG inclusion in coverage misleading.

    Signal 2 — Tool success rate (weight 0.4)
        success_rate = len(tools_run) / max(len(tools_run) + len(tools_failed), 1)

    Signal 3 — Truncation penalty (weight 0.2)
        1.0 — none · 0.7 — enrichment truncated · 0.5 — core (ERROR)

    confidence = 0.4*coverage + 0.4*success_rate + 0.2*truncation
    clamped to [0.0, 1.0]. Weights tunable after Week 3 measurement.
    LOW severity does not reduce confidence.
    """
    coverage     = _modality_coverage(state)
    success_rate = len(state.tools_run) / max(
                       len(state.tools_run) + len(state.tools_failed), 1)
    truncation   = _truncation_score(state)
    return max(0.0, min(1.0,
        0.4 * coverage + 0.4 * success_rate + 0.2 * truncation
    ))


def _modality_coverage(state: "AegisState") -> float:
    return sum([
        bool(state.raw_symptoms_text),
        bool(state.lab_pdf_path),
        bool(state.xray_image_path),
        bool(state.medications_raw),
        state.rag_result is not None,
    ]) / 5


def _truncation_score(state: "AegisState") -> float:
    if state.core_fields_truncated:        return 0.5
    if state.enrichment_fields_truncated:  return 0.7
    return 1.0
```

### Report Sections

Six sections — authoritative and complete:

```
### Summary
### Findings        (Radiological subsection if X-ray submitted)
### Evidence        (explicit absence message if passages=[])
### Severity        (Confidence line · highest_priority_rule displayed)
### Recommendations
### Disclaimer
```

Missing section → `ToolError(fatal=True)`. Partial report never rendered.

**Mid-stream failure message** — hardcoded in `stream_render.py`, never LLM-generated:

```
⚠️ Report generation was interrupted. The output above is incomplete
and must not be used for clinical decisions. Please resubmit or
contact support.
```

### Test Scenario Definitions

**Scenario A — Full modality (clean run)**
```
audio:       tests/fixtures/symptoms_clear.wav
pdf:         tests/fixtures/lab_normal.pdf
xray:        tests/fixtures/chest_normal.jpg
medications: ["Metformin 500mg", "Lisinopril 10mg"]

expected:
  severity               = LOW
  triggered_rules        = [RULE_DEFAULT_LOW]
  highest_priority_rule  = RULE_DEFAULT_LOW
  confidence             > 0.9
  six sections           present
  EasyOCR                never loaded
  no ToolError
  pipeline_duration_ms   < 60_000
  job.status             = COMPLETED
```

**Scenario B — Partial modality (degraded run)**
```
audio:       None
text:        "Chest pain for 3 days, shortness of breath"
pdf:         tests/fixtures/lab_scanned.pdf
xray:        None
medications: ["AEGIS_TEST_UNRESOLVABLE_DRUG_XYZ"]

expected:
  severity               = HIGH
  triggered_rules ⊇      {RULE_CHEST_PAIN_AND_SOB}
  highest_priority_rule  = RULE_CHEST_PAIN_AND_SOB
  drug_result.unresolved = ["AEGIS_TEST_UNRESOLVABLE_DRUG_XYZ"]
  drug_result.confidence = 0.0
  rag_result             = RAGSearchResult (not ToolError)
  confidence             < 1.0
  six sections           present
  pipeline_duration_ms   < 90_000
```

**Scenario C — Token budget stress**
```
audio:       None
text:        [800-token symptom description]
pdf:         tests/fixtures/lab_verbose.pdf
xray:        tests/fixtures/chest_findings.jpg
medications: [20 medications]

expected:
  context EXCEEDS num_ctx before generation
  truncation fired and logged
  core_fields_truncated  = False
  six sections           present
  no OOM
```

**Golden-output assertions:**
```python
from tools.severity_scorer import RULE_DEFAULT_LOW, RULE_CHEST_PAIN_AND_SOB
from schemas.queue import JobStatus

assert severity_result.highest_priority_rule == expected_highest_rule
assert severity_result.highest_priority_rule == severity_result.triggered_rules[0]
assert len(severity_result.reasons) == len(severity_result.triggered_rules)
assert all(s in report.text for s in [
    "### Summary", "### Findings", "### Evidence",
    "### Severity", "### Recommendations", "### Disclaimer"
])
assert report.confidence == pytest.approx(expected_confidence, abs=0.05)
assert report.knowledge_base_version is not None
assert state.pipeline_duration_ms < expected_max_ms
assert job.status == JobStatus.COMPLETED
```

**Rule coverage and invariant tests:**
```python
def test_all_rules_have_fixtures():
    missing = [r for r in ALL_RULE_CONSTANTS if r not in RULES_WITH_FIXTURES]
    assert not missing, f"Rules missing fixtures: {missing}"

def test_highest_priority_rule_invariant():
    for rule_constant in ALL_RULE_CONSTANTS:
        result = score_with_single_rule(rule_constant)
        assert result.highest_priority_rule == result.triggered_rules[0]
        assert result.highest_priority_rule == rule_constant

def test_reasons_length_invariant():
    for rule_constant in ALL_RULE_CONSTANTS:
        result = score_with_single_rule(rule_constant)
        assert len(result.reasons) == len(result.triggered_rules)
```

**Disconnected client test:**
```python
async def test_disconnected_client():
    job = await submit_job(PipelineJob(session_id="test-disconnect"))
    # Do NOT consume from stream — let queue fill to maxsize
    await asyncio.sleep(STREAM_PUT_TIMEOUT_S + 5)
    assert job.status == JobStatus.FAILED
    assert "disconnected" in job.error.lower() or "slow" in job.error.lower()
    # Verify lock released — next job can proceed
    next_job = await submit_job(PipelineJob(session_id="test-next"))
    await asyncio.sleep(1)
    assert next_job.status in (JobStatus.RUNNING, JobStatus.COMPLETED)
```

**Field-level truncation guarantees:**

| Field | Truncation |
|---|---|
| `SeverityResult.level` | Never |
| `SeverityResult.confidence` | Never |
| `SeverityResult.triggered_rules` | Never |
| `SeverityResult.highest_priority_rule` | Never |
| `SeverityResult.reasons` | Never |
| `SymptomExtractionResult.symptoms` | Never |
| `SymptomExtractionResult.duration` | Never |
| `SymptomExtractionResult.severity_indicators` | Never |
| `SymptomExtractionResult.medical_entities` | May be truncated (enrichment) |
| `SymptomExtractionResult.negations` | May be truncated last |

---

## Concurrency and Performance

### Multi-User, Single-Inference-Worker Architecture

Multiple users may submit sessions concurrently. Inference executes one pipeline at a time. Sessions are queued FIFO and processed sequentially.

```
User A ──┐
User B ──┤──→ Job Queue (FIFO · max 10) ──→ Inference Worker ──→ Ollama
User C ──┘                                    (asyncio.Lock)
```

**Why not parallel inference:** llama3.2:1b plus OCR-active peaks at ~3.4–4.1 GB. Parallel inference on 8 GB would risk OOM. `OLLAMA_NUM_PARALLEL=1` and `OLLAMA_MAX_LOADED_MODELS=1` remain mandatory.

### Queue Functions

```python
def get_queue_position(job_id: str) -> int | None:
    """
    Returns current 1-based position, or None if not queued.
    Computed dynamically — never stored. Always accurate.
    O(n) over MAX_QUEUE_SIZE=10 — effectively free.
    """
    try:
        return list(_job_queue).index(job_id) + 1
    except ValueError:
        return None


def get_estimated_wait_seconds(job_id: str) -> float | None:
    """
    Returns estimated wait in seconds, or None if not queued or < 3 completions.
    Uses rolling average of last 10 entries from _completed_durations.
    """
    position = get_queue_position(job_id)
    if position is None or len(_completed_durations) < 3:
        return None
    recent = list(_completed_durations)[-10:]
    avg = sum(recent) / len(recent)
    return position * avg


def get_average_pipeline_duration_s() -> float | None:
    if len(_completed_durations) < 3:
        return None
    recent = list(_completed_durations)[-10:]
    return sum(recent) / len(recent)


async def submit_job(job: PipelineJob) -> PipelineJob | ToolError:
    if len(_job_queue) >= MAX_QUEUE_SIZE:
        return ToolError(
            tool="queue",
            reason=f"Queue full ({MAX_QUEUE_SIZE} jobs). Try again shortly.",
            fatal=True,
        )
    _job_store[job.job_id] = job
    _job_queue.append(job.job_id)
    return job


async def run_inference_worker():
    """Single worker — runs for application lifetime."""
    while True:
        _purge_expired_jobs()
        if not _job_queue:
            await asyncio.sleep(0.1)
            continue
        job_id = _job_queue.popleft()
        job    = _job_store[job_id]
        async with _inference_lock:
            await _execute_job(job)


def _purge_expired_jobs() -> None:
    now = datetime.now(timezone.utc)
    expired = [
        jid for jid, job in _job_store.items()
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
        and job.completed_at is not None
        and (now - job.completed_at).total_seconds() > JOB_RETENTION_SECONDS
    ]
    for jid in expired:
        del _job_store[jid]
        _job_streams.pop(jid, None)
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `POST /queue/submit` | POST | Validate → create job → enqueue. Returns `PipelineJob`. Rejects if full. |
| `GET /queue/status/{job_id}` | GET | Returns `PipelineJob` + live `queue_position` + `estimated_wait_seconds`. |
| `GET /queue/stream/{job_id}` | GET | Drains `asyncio.Queue` via `None` sentinel. Returns 425 if QUEUED. |
| `GET /health` | GET | System status + queue metrics. Never blocks. |

**Health response:**
```json
{
  "system_status":               "ok",
  "inference_active":            false,
  "model_loaded":                true,
  "gpu_available":               true,
  "memory_used_mb":              2340,
  "memory_total_mb":             8192,
  "rag_index_ready":             true,
  "queue_depth":                 2,
  "queue_max":                   10,
  "average_pipeline_duration_s": 47.3,
  "jobs_completed_today":        12,
  "jobs_failed_today":           0
}
```

**Status response:**
```json
{
  "job_id":                  "abc-123",
  "status":                  "queued",
  "submitted_at":            "2024-11-15T10:00:00Z",
  "queue_position":          2,
  "estimated_wait_seconds":  94.6
}
```

### Upload Bounds — Enforced Before Queue Entry

| Input | Limit | On violation |
|---|---|---|
| Lab PDF | ≤ 25 MB | `ToolError(fatal=True)` before queue |
| X-ray image | ≤ 25 MB | `ToolError(fatal=True)` before queue |
| Audio | ≤ 15 MB and ≤ 120 s | `ToolError(fatal=True)` before queue |
| Medications | ≤ 50 entries | `ToolError(fatal=True)` before queue |
| Queue | ≤ 10 jobs | `ToolError(fatal=True)` before queue |

### Ollama Concurrency Guards

```yaml
environment:
  - OLLAMA_NUM_PARALLEL=1
  - OLLAMA_MAX_LOADED_MODELS=1
  - OLLAMA_KEEP_ALIVE=-1
```

### Model Warmup

```bash
#!/bin/bash
set -e
until ollama list > /dev/null 2>&1; do sleep 2; done
ollama rm aegis-llama 2>/dev/null || true
ollama create aegis-llama -f /app/config/Modelfile
if ! ollama run aegis-llama "ok" > /dev/null 2>&1; then
  python -c "from loguru import logger; logger.warning('warmup failed')"
fi
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log
```

### Docker Compose

```yaml
services:
  ollama:
    image: ollama/ollama@sha256:<pinned-digest>
    runtime: nvidia
    environment:
      - OLLAMA_NUM_PARALLEL=1
      - OLLAMA_MAX_LOADED_MODELS=1
      - OLLAMA_KEEP_ALIVE=-1
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 5s
      timeout: 10s
      retries: 6
      start_period: 15s
  app:
    build: .
    depends_on:
      ollama:
        condition: service_healthy
    ports:
      - "8000:8000"
      - "8501:8501"
    runtime: nvidia
```

### Frontend — Queue-Aware Polling

```python
job = requests.post("/queue/submit", data=inputs).json()
job_id = job["job_id"]

while True:
    status = requests.get(f"/queue/status/{job_id}").json()
    if status["status"] == "queued":
        pos  = status["queue_position"]
        wait = status.get("estimated_wait_seconds")
        msg  = f"Position {pos} in queue"
        if wait:
            msg += f" · ~{int(wait)}s estimated wait"
        st.info(msg)
        time.sleep(2)
    elif status["status"] == "running":
        stream_report(job_id)
        break
    elif status["status"] == "failed":
        st.error(status["error"])
        break
```

> Polling every 2 seconds is appropriate for ≤10 concurrent users. SSE deferred — polling is simpler and more robust at demo scale.

### Health Check

`/health` never blocks. `average_pipeline_duration_s` returns `null` until 3+ completions. `jobs_completed_today` and `jobs_failed_today` reset on container restart. `rag_index_ready` cached, re-probed at most once per minute. GPU via `tegrastats`/`nvidia-smi` — torch never imported.

---

## Model Stack

### Language Model — llama3.2:1b

`num_ctx 4096` via Modelfile. Benchmark 2048 in Week 1. Two LLM calls per run. GQA — measure actual KV cache size Week 1.

```
FROM llama3.2:1b
PARAMETER num_ctx    4096
PARAMETER num_predict 1024
PARAMETER temperature 0.2
```

### SymptomExtractor Retry Policy

```
Attempt 1: Standard structured prompt → Pydantic validate → proceed if valid
Attempt 2: JSON-only with schema verbatim → Pydantic validate → proceed if valid
Attempt 3: Local JSON repair → proceed if valid · ToolError(fatal=True) if not

Max added latency: ~2–6s. Acceptable — symptom extraction is non-negotiable.
```

### Embeddings

`all-MiniLM-L6-v2` → ONNX. Committed as `data/knowledge/minilm.onnx`. No torch.

### Voice Transcription

Faster-Whisper tiny.en INT8. CPU-only. ~150 MB. Released after transcription.

### OCR

| Stage | Tool | Memory |
|---|---|---|
| Primary | PyMuPDF | negligible |
| Fallback 1 | pdfminer.six | negligible |
| Fallback 2 | EasyOCR (opt-in) | ~1.2–1.5 GB |

EasyOCR release: `del self.ocr_reader; self.ocr_reader = None; gc.collect()`

> ⚠️ Validate EasyOCR ARM64 in Week 1. If fails: switch to pytesseract, redefine Scenario B, document in `docs/setup_jetson.md`.

### Drug Data

SQLite FTS5. Built on Mac, committed. Returns `DrugInteractionResult`.

### Observability

```json
{"tool": "LabReportParser", "event": "model_loaded",       "duration_ms": 843, "memory_mb": 780}
{"tool": "LabReportParser", "event": "inference_complete", "duration_ms": 1204}
{"tool": "LabReportParser", "event": "model_released",                          "memory_mb": 0}
```

---

## Tools

Steps 0–7 unchanged. AegisPipeline is unaware of the queue — it receives `AegisState` and runs to completion.

### Step 0 · VoiceTranscriber *(optional)*

Bounded ≤ 15 MB / ≤ 120 s. CPU-only. Lazy-load, transcribe, release. Writes `raw_symptoms_text`. Skipped if no audio.

### Step 1 · SymptomExtractor

llama3.2:1b. Three-attempt retry/repair. `ToolError(fatal=True)` after all fail.

### Step 2 · LabReportParser

Bounded ≤ 25 MB. PyMuPDF → pdfminer.six → EasyOCR (opt-in). Explicit EasyOCR release.

### Step 3 · XRayProcessor

Bounded ≤ 25 MB. PIL + DICOM. Pre-collected findings. No GPU.

**Findings checklist:**
```
☐ Cardiomegaly    ☐ Pleural Effusion   ☐ Pneumonia      ☐ Pneumothorax
☐ Consolidation   ☐ Atelectasis        ☐ Infiltrates    ☐ Pulmonary Edema
☐ Nodule / Mass   ☐ Fracture           ☐ Normal / No significant findings
+ Free text
```

### Step 4 · MedicalRAGSearch

Query from Steps 1–3. Zero results → `RAGSearchResult(passages=[], retrieval_successful=True)`. `ToolError(fatal=False)` only if mechanism fails.

### Step 5 · DrugInteractionChecker

Bounded ≤ 50. FTS5. Returns `DrugInteractionResult`. Zero model load.

### Step 6 · SeverityScorer

Deterministic. Rules from `docs/severity_rules.md`. Sets `highest_priority_rule = triggered_rules[0]`. `len(reasons) == len(triggered_rules)` enforced. `ToolError(fatal=True)` on failure.

**Rule constants in `tools/severity_scorer.py`:**

```python
RULE_CHEST_PAIN_AND_SOB       = "chest_pain_and_sob"
RULE_CRITICAL_LAB_TROPONIN    = "critical_lab_troponin"
RULE_CRITICAL_LAB_HAEMOGLOBIN = "critical_lab_haemoglobin"
RULE_CRITICAL_LAB_POTASSIUM   = "critical_lab_potassium"
RULE_XRAY_PNEUMOTHORAX        = "xray_pneumothorax"
RULE_XRAY_PULMONARY_EDEMA     = "xray_pulmonary_edema"
RULE_SEVERE_DRUG_INTERACTION  = "severe_drug_interaction"
RULE_ABNORMAL_LAB_ANY          = "abnormal_lab_any"
RULE_XRAY_CARDIOMEGALY         = "xray_cardiomegaly"
RULE_XRAY_PLEURAL_EFFUSION     = "xray_pleural_effusion"
RULE_XRAY_CONSOLIDATION        = "xray_consolidation"
RULE_PROLONGED_SYMPTOMS        = "prolonged_symptoms"
RULE_MODERATE_DRUG_INTERACTION = "moderate_drug_interaction"
RULE_DEFAULT_LOW               = "default_low"

ALL_RULE_CONSTANTS = [
    RULE_CHEST_PAIN_AND_SOB, RULE_CRITICAL_LAB_TROPONIN,
    RULE_CRITICAL_LAB_HAEMOGLOBIN, RULE_CRITICAL_LAB_POTASSIUM,
    RULE_XRAY_PNEUMOTHORAX, RULE_XRAY_PULMONARY_EDEMA,
    RULE_SEVERE_DRUG_INTERACTION, RULE_ABNORMAL_LAB_ANY,
    RULE_XRAY_CARDIOMEGALY, RULE_XRAY_PLEURAL_EFFUSION,
    RULE_XRAY_CONSOLIDATION, RULE_PROLONGED_SYMPTOMS,
    RULE_MODERATE_DRUG_INTERACTION, RULE_DEFAULT_LOW,
]
```

**Severity levels:**

| Level | Condition |
|---|---|
| 🔴 HIGH | Any HIGH rule fires |
| 🟡 MEDIUM | Any MEDIUM rule fires, no HIGH fired |
| 🟢 LOW | No rules fire — `triggered_rules=[RULE_DEFAULT_LOW]` |

**Precedence example:**
```
Patient: chest pain + SOB + abnormal HbA1c

  RULE_CHEST_PAIN_AND_SOB (190) → fires → level=HIGH
  RULE_ABNORMAL_LAB_ANY   (90)  → fires → added to triggered_rules

Result:
  level                  = HIGH
  triggered_rules        = ["chest_pain_and_sob", "abnormal_lab_any"]
  highest_priority_rule  = "chest_pain_and_sob"
  reasons                = [
    "Chest pain with shortness of breath detected",
    "Abnormal lab values detected"
  ]
```

### Step 7 · ReportGenerator

Six-section validated streaming. One LLM call. Buffer-validate-yield.

**Token budget priority:**
```
1. SeverityResult          level + confidence + triggered_rules + highest_priority_rule
                           + reasons — never truncated
2. SymptomExtractionResult symptoms + duration + indicators — never truncated
                           medical_entities + negations — may be truncated
3. LabReportResult         key abnormal values if space tight
4. DrugInteractionResult   flagged interactions if space tight
5. XRayResult              checklist summary if space tight
6. RAGSearchResult         top-2 citations if space tight
7. Pipeline metadata       dropped first
```

---

## Severity Rules

Full specification in `docs/severity_rules.md`. Every rule has a fixture. `test_all_rules_have_fixtures()`, `test_highest_priority_rule_invariant()`, `test_reasons_length_invariant()` enforce correctness.

**HIGH rules (priority 100–199):**

| Constant | Trigger | Priority |
|---|---|---|
| `RULE_CHEST_PAIN_AND_SOB` | Chest pain + shortness of breath | 190 |
| `RULE_CRITICAL_LAB_TROPONIN` | Troponin abnormal | 180 |
| `RULE_CRITICAL_LAB_HAEMOGLOBIN` | Haemoglobin < 7 | 170 |
| `RULE_CRITICAL_LAB_POTASSIUM` | Potassium > 6.5 | 160 |
| `RULE_XRAY_PNEUMOTHORAX` | Pneumothorax | 150 |
| `RULE_XRAY_PULMONARY_EDEMA` | Pulmonary Edema | 140 |
| `RULE_SEVERE_DRUG_INTERACTION` | severity="severe" in aegis_drugs.db | 130 |

**MEDIUM rules (priority 50–99):**

| Constant | Trigger | Priority |
|---|---|---|
| `RULE_ABNORMAL_LAB_ANY` | Any abnormal lab, no HIGH fired | 90 |
| `RULE_XRAY_CARDIOMEGALY` | Cardiomegaly | 80 |
| `RULE_XRAY_PLEURAL_EFFUSION` | Pleural Effusion | 75 |
| `RULE_XRAY_CONSOLIDATION` | Consolidation | 70 |
| `RULE_PROLONGED_SYMPTOMS` | Duration > 7 days + indicators, no HIGH | 60 |
| `RULE_MODERATE_DRUG_INTERACTION` | severity="moderate", no HIGH | 50 |

**LOW:** `RULE_DEFAULT_LOW`. `highest_priority_rule = RULE_DEFAULT_LOW`.

---

## Medical Knowledge

| Source | Status |
|---|---|
| MedlinePlus (NIH) | ✅ v1.0 |
| PubMed Abstracts | 🔲 v2.0 |
| NIH Clinical Guidelines | 🔲 v2.0 |
| WHO Public Health Guidance | 🔲 v2.0 |

`knowledge_base_version` and `knowledge_base_date` loaded from `docs/corpus_version.md` at startup, injected into every `TriageReport`.

---

## Memory Management

| State | Estimated Peak |
|---|---|
| Idle | ~2.2–2.6 GB |
| Voice active | ~2.4–2.8 GB |
| Digital-PDF | ~2.3–2.7 GB |
| OCR active (EasyOCR, scanned only) | **~3.4–4.1 GB** |
| X-Ray active | ~2.2–2.6 GB |

Stream buffers (`asyncio.Queue(maxsize=256)`) and `_completed_durations` (`deque(maxlen=100)`) contribute negligibly to peak. Measure Week 1. Update `docs/memory_profile.md`.

> ⚠️ OCR-active > ~5.5 GB → EasyOCR staged-loading in Week 2.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Runtime | Ollama (digest-pinned · `NUM_PARALLEL=1` · `MAX_LOADED_MODELS=1` · `KEEP_ALIVE=-1`) |
| LLM Model | `llama3.2:1b` · `num_ctx 4096` · `temperature 0.2` |
| Pipeline | AegisPipeline (pure Python, async, wall-clock bounded) |
| Queue | `backend/queue.py` · asyncio FIFO · single worker · `MAX_QUEUE_SIZE=10` |
| Stream buffer | `asyncio.Queue(maxsize=256)` per job · `put_nowait(None)` in `finally` |
| Duration tracking | `deque(float, maxlen=100)` · bounded · rolling average of last 10 |
| Validation | Pydantic v2 |
| Embeddings | all-MiniLM-L6-v2 via ONNX Runtime (ARM64) |
| Vector Store | ChromaDB + FAISS fallback |
| Voice | Faster-Whisper tiny.en INT8 (CPU-only, optional) |
| PDF | PyMuPDF → pdfminer.six → EasyOCR (opt-in, explicitly released) |
| X-Ray | XRayProcessor (PIL + DICOM, clinician-assisted) |
| Drugs | SQLite FTS5 · OpenFDA + RxNorm |
| Upload Guards | `backend/uploads.py` — before queue entry |
| Logging | Loguru (structured JSON) |
| API | FastAPI (single Uvicorn worker) |
| GPU Detection | `tegrastats` / `nvidia-smi` — no torch |
| Frontend | Streamlit (queue position · wait estimate · live stream · sidebar) |
| Containers | Docker ARM64 + Compose |
| Hardware | NVIDIA Jetson Orin Nano 8 GB |

---

## Repository Structure

```
aegis-health/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── health.py
│   ├── queue.py                 # deque(maxlen=100) · put_nowait sentinel · per-token timeout
│   │                            # MAX_QUEUE_SIZE=10 · STREAM_QUEUE_MAXSIZE=256
│   │                            # STREAM_PUT_TIMEOUT_S=30.0 · JOB_RETENTION_SECONDS=3600
│   ├── streaming.py
│   └── uploads.py
├── frontend/
│   ├── __init__.py
│   ├── app.py
│   ├── health_check.py
│   ├── stream_render.py         # hardcoded failure message
│   ├── performance_sidebar.py
│   └── xray_component.py
├── agents/
│   ├── __init__.py
│   └── pipeline.py              # step_durations_ms · tools_run ∩ tools_failed = ∅
├── tools/
│   ├── __init__.py
│   ├── base.py
│   ├── voice_transcriber.py
│   ├── symptom_extractor.py     # 3-attempt retry/repair
│   ├── lab_report_parser.py     # explicit EasyOCR release
│   ├── rag_search.py
│   ├── drug_checker.py
│   ├── severity_scorer.py       # ALL_RULE_CONSTANTS · highest_priority_rule
│   └── report_generator.py      # six sections · buffer-validate-yield
├── vision/
│   ├── __init__.py
│   ├── xray_processor.py
│   └── dicom_reader.py
├── rag/
│   ├── __init__.py
│   ├── download_corpus.py       # Mac-only
│   ├── ingest.py
│   ├── chunk.py
│   ├── export_minilm_onnx.py    # Mac-only
│   ├── embed.py
│   ├── build_chroma.py
│   ├── build_faiss.py
│   └── retriever.py
├── schemas/
│   ├── __init__.py
│   ├── queue.py                 # PipelineJob · JobStatus · no stored queue_position
│   ├── state.py                 # tools_run ∩ tools_failed = ∅ · timing · truncation
│   ├── errors.py                # ToolError
│   ├── symptom.py
│   ├── voice.py
│   ├── rag.py
│   ├── lab.py
│   ├── drugs.py
│   ├── severity.py              # single definition · highest_priority_rule · invariants
│   ├── xray.py
│   └── report.py                # TriageReport · calculate_confidence()
├── config/
│   └── Modelfile
├── data/
│   ├── knowledge/
│   │   ├── raw/                 # gitignored
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
│   │   ├── __init__.py
│   │   └── test_severity_scorer.py  # all_rules_have_fixtures · invariants · reasons_length
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_queue.py        # FIFO · full · lifecycle · position · purge
│   │                            # test_disconnected_client()
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   └── test_queue_load.py   # 10 simultaneous · FIFO · no lost · lock released
│   └── scenarios/
│       ├── __init__.py
│       ├── scenario_a.py
│       ├── scenario_b.py
│       └── scenario_c.py
├── tests/fixtures/
│   ├── symptoms_clear.wav           lab_normal.pdf      chest_normal.jpg
│   ├── demo_symptoms_audio.wav      lab_scanned.pdf     chest_findings.jpg
│   ├── demo_symptoms.txt            lab_verbose.pdf     demo_xray.jpg
│   └── demo_medications.txt         demo_lab_report.pdf
└── docs/
    ├── architecture.md
    ├── setup_jetson.md
    ├── memory_profile.md
    ├── corpus_version.md
    ├── severity_rules.md
    └── api.md
.gitignore
```

---

## Deployment

**Mac-only setup:**

```bash
python rag/download_corpus.py
python rag/export_minilm_onnx.py
python rag/build_chroma.py && python rag/build_faiss.py

echo "snapshot_date: $(date -u +%Y-%m-%d)" >> docs/corpus_version.md
echo "source_url: https://medlineplus.gov/xml.html" >> docs/corpus_version.md
echo "git_commit: $(git rev-parse HEAD)" >> docs/corpus_version.md

python data/drugs/build_drug_db.py

ollama pull llama3.2:1b
ollama create aegis-llama -f config/Modelfile
ollama list | grep aegis-llama

docker inspect ollama/ollama:latest --format '{{index .RepoDigests 0}}'
git add data/ config/ && git commit -m "build: add committed assets"
```

**Jetson:**

```bash
git clone <repo> && cd aegis-health && docker compose up
```

---

## 6-Week Development Plan

### Week 1 — Contracts First, Then Foundation

**Mandatory pre-implementation gate:**

```bash
pytest --collect-only  # zero errors before any tool logic
```

**Must exist before implementation:**

| Document | Blocks |
|---|---|
| `schemas/queue.py` | Queue, job lifecycle |
| `schemas/severity.py` with `highest_priority_rule` | SeverityScorer, all tests |
| `docs/severity_rules.md` | SeverityScorer, fixtures |
| All other schemas | See implementation order |

**Implementation order:**

```
schemas/errors.py
→ schemas/queue.py
→ schemas/severity.py
→ schemas/drugs.py · rag.py · symptom.py · voice.py · lab.py · xray.py
→ schemas/report.py
→ schemas/state.py
→ backend/uploads.py
→ backend/queue.py              # deque(maxlen=100) · put_nowait sentinel · per-token timeout
→ agents/pipeline.py
→ backend/main.py · health.py · streaming.py
→ tools/severity_scorer.py
→ tests/tools/test_severity_scorer.py
→ tools/symptom_extractor.py · drug_checker.py
→ frontend/app.py
→ tests/integration/test_queue.py  # includes test_disconnected_client()
→ tests/benchmarks/test_queue_load.py
```

**Week 1 primary exit criterion:**

> **Scenario A passes end-to-end on Jetson** — submitted via queue, position accurate, `highest_priority_rule=RULE_DEFAULT_LOW`, report streams fully, `pipeline_duration_ms` recorded.

**Required gates:**
- `pytest --collect-only` zero errors
- All schema imports clean
- `test_all_rules_have_fixtures()` green
- `test_highest_priority_rule_invariant()` green
- `test_reasons_length_invariant()` green
- `test_disconnected_client()` green
- `test_queue_load.py` green
- `deque(maxlen=100)` confirmed in `_completed_durations`
- `put_nowait(None)` confirmed in `finally` block
- Jetson baseline memory in `docs/memory_profile.md`
- EasyOCR ARM64 decision documented

---

### Week 2 — Retrieval + OCR + Voice + X-Ray

Build and commit ONNX indexes. Implement LabReportParser, VoiceTranscriber, XRayProcessor. Validate full tool suite on Jetson. Measure OCR-active memory peak.

**Done when:**
- `minilm.onnx` committed, `docs/corpus_version.md` complete
- RAG zero results → `RAGSearchResult(passages=[])` confirmed in practice
- FAISS fallback tested
- EasyOCR never loads on digital PDF; explicit release confirmed
- VoiceTranscriber Step 0 → Step 1 chaining confirmed
- OCR-active peak measured and documented
- `docs/memory_profile.md` updated

> ⚠️ If OCR-active peak > ~5.5 GB: EasyOCR staged-loading implemented here — not Week 4.

---

### Week 3 — Full Integration = Minimum Viable Demo

**Done when:**
- All severity rules, all fixtures green
- `calculate_confidence()` implemented and correct
- ReportGenerator: six-section streaming, validation, missing section → `ToolError`
- Zero-RAG and RAG ToolError handled in Evidence section
- Golden-output tests written for A/B/C
- **Scenario A passes** on Jetson
- **Scenario B passes** on Jetson
- **Scenario C passes** on Jetson
- No Pydantic validation errors across A/B/C

---

> ### 🚨 Minimum Viable Demo — End of Week 3
>
> Five modalities · Eight tools · Streamed report · Six validated sections · Deterministic severity · Defined confidence · Golden tests · Live sidebar · Three validated scenarios · Jetson · Fully local.

---

### Week 4 — Optimisation and Stability

Real Loguru data. No new features. No schema changes.

**Done when:**
- 10 consecutive Scenario A runs on Jetson without failure
- Golden-output tests pass across all 10
- Memory within measured limits
- Scenario C stable across 5 consecutive runs

---

> ### 🚨 Feature Freeze — End of Week 4

---

### Week 5 — Demo Preparation

UI polish · Tier 2 Mac confirmed · Tier 3 pre-recorded · All fixtures tested · Demo script rehearsed.

### Week 6 — Demo and Submission

3 Scenario A run-throughs · Tier 2 switch under 2 minutes · Submitted.

---

## Demo Scenario

Multiple clinicians submit sessions. Each receives a `job_id` immediately. UI shows queue position and estimated wait. When job reaches RUNNING, report streams section-by-section. Severity deterministic — `triggered_rules`, `highest_priority_rule`, and `reasons` surfaced. Confidence calculated. Knowledge base provenance in every report. Entirely local on Jetson.

---

## Demo Backup Strategy

| Tier | Setup | When |
|---|---|---|
| **Tier 1** | Live Jetson Orin Nano 8 GB | Primary |
| **Tier 2** | Live Mac (identical Docker image) | Jetson unstable |
| **Tier 3** | Pre-recorded Scenario A on Jetson | Both fail |

---

## Non-Functional Requirements

| Requirement | Detail |
|---|---|
| Multi-user queued | FIFO queue · single inference worker · `MAX_QUEUE_SIZE=10` |
| Queue position | Computed dynamically — never stored |
| Estimated wait | Rolling average of last 10 · `null` until 3+ completions |
| Queue full | `ToolError(fatal=True)` before queue entry |
| Job lifecycle | QUEUED → RUNNING → COMPLETED / FAILED |
| Job retention | `JOB_RETENTION_SECONDS=3600` · stream queue deleted on purge |
| Stream buffer | `asyncio.Queue(maxsize=256)` · bounded · backpressure |
| Sentinel | `put_nowait(None)` in `finally` · unconditional · non-blocking |
| Per-token timeout | `asyncio.wait_for(..., STREAM_PUT_TIMEOUT_S=30.0)` |
| Disconnected client | Job → FAILED · sentinel emitted · lock released |
| Duration tracking | `deque(float, maxlen=100)` · bounded · never a plain list |
| No parallel inference | `OLLAMA_NUM_PARALLEL=1` · `OLLAMA_MAX_LOADED_MODELS=1` |
| Ollama image pinned | Digest in `docs/setup_jetson.md` |
| Warmup | Failure via Loguru |
| Single worker | One Uvicorn worker |
| Confidence formula | Non-overlapping signals · documented · tunable |
| Severity rules | In `docs/severity_rules.md` before implementation |
| Rule constants | `ALL_RULE_CONSTANTS` · imported by tests |
| `highest_priority_rule` | Always `triggered_rules[0]` · set at scoring time |
| `len(reasons) == len(triggered_rules)` | Enforced by SeverityScorer |
| Six report sections | Validated · missing → `ToolError(fatal=True)` |
| Zero-RAG | `RAGSearchResult(passages=[])` · explicit message |
| Knowledge base provenance | `knowledge_base_version` + `knowledge_base_date` in every report |
| Golden-output tests | Week 3 · assert constants not strings |
| `pytest --collect-only` gate | Zero errors before any tool logic |
| Contracts before code | All schemas + rules before implementation |
| Timezone-aware | `datetime.now(timezone.utc)` throughout |
| Mutable defaults | `Field(default_factory=...)` everywhere |
| Single schema source | `schemas/` only |
| Session isolation | `uuid4` — never client-supplied |
| gitignored raw data | `data/knowledge/raw/` |
| Week 1 exit criterion | Scenario A passes end-to-end on Jetson |
| Stable | 10 consecutive Scenario A runs — Week 4 |
| Observable | Per-tool JSON events — Streamlit sidebar |

---

## Post-Demo Enhancements

> These enhancements are intentionally excluded from MVP scope and must not block Week 1–6 deliverables.

| Enhancement | Description |
|---|---|
| SQLite job persistence | Cross-restart durability for `PipelineJob` metadata |
| SSE push notifications | Replace 2s polling |
| Confidence formula tuning | Denominator=4 if RAG-in-coverage misleads clinicians |
| Priority queue | FIFO override for clinical urgency |

---

## Scope Boundaries

**In scope:** Multi-user queued sessions · job lifecycle · live queue position · wait estimates · queue metrics · all clinical modalities · six-section streaming report

**Out of scope:** True concurrent inference · SSE · persistent jobs · user authentication · FIFO priority overrides

---

## Safety and Limitations

- Does not diagnose. Assists triage — does not replace clinical judgment.
- All outputs require review by a qualified healthcare professional.
- Not for emergencies. Contact emergency services immediately.
- Severity rule-based and fully auditable — `triggered_rules` maps to `docs/severity_rules.md`.
- Queue capacity limited — full queue returns explicit rejection.
- Queue not persistent across restarts.
- FIFO does not account for clinical urgency.
- Disconnected stream clients cause job failure — resubmit required.
- Estimated wait approximate — unavailable until 3+ completions.
- Knowledge base limited to indexed MedlinePlus corpus.
- Partial results always flagged — never presented as complete.

---

## AI Domains Covered

`Agentic AI` · `Small Language Models` · `Retrieval-Augmented Generation` · `Vector Databases` · `Semantic Search` · `Document AI` · `OCR` · `Speech-to-Text` · `Edge AI Deployment` · `Prompt Engineering` · `MLOps` · `Privacy-Preserving AI` · `Model Quantisation` · `ONNX Runtime Optimisation`

---

*Aegis Health — built for privacy, designed for the edge.*