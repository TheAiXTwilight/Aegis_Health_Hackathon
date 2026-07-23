# Aegis Health — API Reference

Base URL: http://localhost:8000 (development)
          http://app:8000        (Docker internal)

The API is frontend-agnostic. Any HTTP client can drive it:
curl, Postman, React (planned), or any other.


## POST /queue/submit

Submit a new pipeline job.

Request: multipart/form-data

Field           Type            Required    Description
symptoms_text   string          No          Plain text symptom description
medications     string (JSON)   No          JSON-encoded list[str], default "[]"
xray_findings   string (JSON)   No          JSON-encoded list[str], default "[]"
xray_free_text  string          No          Clinician free-text X-ray note
audio           file            No          Audio file (WAV preferred)
lab_pdf         file            No          Lab report PDF
xray_image      file            No          X-ray image

At least one input must be provided.
All clinician X-ray findings must be collected before submission.
The pipeline runs uninterrupted once started.

Upload limits:

Input           Limit
Lab PDF         25 MB
X-ray image     25 MB
Audio           15 MB and 120 seconds
Medications     50 entries
Queue           10 jobs

Status codes:

Code    Meaning
200     Accepted — returns PipelineJob JSON
400     Invalid input (size, duration, count, malformed JSON)
409     Duplicate job or active session for this session_id
503     Queue full (10 jobs waiting)

Response (200):

    {
        "job_id": "abc-123",
        "session_id": "xyz-456",
        "status": "queued",
        "submitted_at": "2024-11-15T10:00:00Z",
        "started_at": null,
        "completed_at": null,
        "error": null,
        "schema_version": "1.0"
    }

Error response (400 / 409 / 503):

    {
        "tool": "input_validation",
        "reason": "Audio exceeds size limit: 16777216 bytes (max 15728640 bytes / 15 MB)",
        "timestamp": "2024-11-15T10:00:00Z",
        "fatal": true
    }


## GET /queue/status/{job_id}

Returns live job status including pipeline progress.

Recommended polling interval: 2 seconds while status is queued or running.
Stop polling once status reaches "completed" or "failed".

Status codes:

Code    Meaning
200     Job found
404     Unknown or purged job_id

Response (200 — while queued):

    {
        "job_id": "abc-123",
        "session_id": "xyz-456",
        "status": "queued",
        "submitted_at": "2024-11-15T10:00:00Z",
        "started_at": null,
        "completed_at": null,
        "error": null,
        "schema_version": "1.0",
        "queue_position": 2,
        "estimated_wait_seconds": 94.6,
        "current_tool": null,
        "tools_run": [],
        "tools_failed": [],
        "step_durations_ms": {}
    }

Response (200 — while running):

    {
        "job_id": "abc-123",
        "session_id": "xyz-456",
        "status": "running",
        "submitted_at": "2024-11-15T10:00:00Z",
        "started_at": "2024-11-15T10:00:02Z",
        "completed_at": null,
        "error": null,
        "schema_version": "1.0",
        "queue_position": null,
        "estimated_wait_seconds": null,
        "current_tool": "MedicalRAGSearch",
        "tools_run": ["ExecutionPlanner", "SymptomExtractor"],
        "tools_failed": [],
        "step_durations_ms": {
            "ExecutionPlanner": 1240.5,
            "SymptomExtractor": 45.2
        }
    }

Response (200 — completed, Phase 2.5):

    {
        "job_id": "abc-123",
        "session_id": "xyz-456",
        "status": "completed",
        "submitted_at": "2024-11-15T10:00:00Z",
        "started_at": "2024-11-15T10:00:02Z",
        "completed_at": "2024-11-15T10:00:49Z",
        "error": null,
        "schema_version": "1.0",
        "queue_position": null,
        "estimated_wait_seconds": null,
        "current_tool": null,
        "tools_run": [
            "ExecutionPlanner",
            "SymptomExtractor",
            "MedicalRAGSearch",
            "DrugInteractionChecker",
            "SeverityScorer",
            "ReportGenerator",
            "RuleValidator"
        ],
        "tools_failed": [],
        "step_durations_ms": {
            "ExecutionPlanner": 1240.5,
            "SymptomExtractor": 45.2,
            "MedicalRAGSearch": 340.2,
            "DrugInteractionChecker": 8.1,
            "SeverityScorer": 3.2,
            "ReportGenerator": 41203.5,
            "RuleValidator": 2.1
        }
    }

Notes:
    ExecutionPlanner appears in tools_run when it returns ExecutionPlan.
    When the planner fails and the fallback is used, ExecutionPlanner
    still appears in tools_run — a fallback plan is a valid result.
    PlanValidator is not a pipeline tool and does not appear in either list.
    VoiceTranscriber appears only when audio_file_path was submitted.
    RuleValidator appears in tools_run on success, tools_failed on ToolError.

    queue_position and estimated_wait_seconds populated only when
    status == "queued".
    estimated_wait_seconds is null until 3+ pipeline completions recorded.

    tools_run and tools_failed are mutually exclusive.
    A tool name appears in exactly one list, never both.

Confidence note:

    report.confidence follows the formula in tools/confidence.py:
        confidence = 0.4 * coverage + 0.4 * success_rate + 0.2 * truncation

    Phase 2.5 effect: when plan.use_rag=False, MedicalRAGSearch is skipped.
    RAG is always-submitted in the coverage formula. state.rag_result stays
    None. Coverage reduces accordingly — the planner's decision to skip RAG
    is reflected in the confidence score. The clinician sees why via
    execution_plan_summary in the report.


## GET /queue/stream/{job_id}

Stream raw report tokens for a running or completed job.

Status codes:

Code    Meaning
200     Stream begins
404     Unknown or purged job_id
425     Job still queued — no stream available yet

Response (200):

    Content-Type: text/plain; charset=utf-8
    Transfer-Encoding: chunked

    Plain text markdown tokens streamed as they are generated.
    No SSE framing. No JSON wrapping. No protocol markers.
    Stream ends when the pipeline emits the None sentinel.

Client implementation patterns:

    React clients use fetch + ReadableStream + TextDecoderStream.
    Python clients use httpx.AsyncClient with stream=True.
    curl --no-buffer works directly.

    No SSE library required.

After-stream behaviour:

    Clients should check job status via GET /queue/status/{job_id}
    after the stream ends. If status is "failed", display the
    mid-stream failure message defined in tools/report_generator.py.

    If validation_status == "override" in the completed report,
    display a safety banner: the deterministic severity (HIGH) is
    authoritative and the SLM narrative expressed a lower severity.
    TriageReport.severity already holds the correct deterministic level.


## GET /health

Returns system status.
Never blocks. Never acquires the inference lock.

Response (200):

    {
        "system_status": "ok",
        "inference_active": false,
        "model_loaded": false,
        "gpu_available": true,
        "memory_used_mb": 2340,
        "memory_total_mb": 8192,
        "rag_index_ready": false,
        "queue_depth": 2,
        "queue_max": 10,
        "average_pipeline_duration_s": 47.3,
        "jobs_completed_today": 12,
        "jobs_failed_today": 0
    }

Field notes:

    system_status               Always "ok" until system-level checks land
    inference_active            True when worker holds the inference lock
    model_loaded                Real probe against Ollama /api/tags, TTL-cached 60s
                                Model name matched by name.split(":")[0] == "aegis-llama"
    gpu_available               Via Jetson device nodes / nvidia-smi / tegrastats
    memory_used_mb              nvidia-smi VRAM or /proc/meminfo used (null if both fail)
    memory_total_mb             nvidia-smi VRAM total or /proc/meminfo total (null if both fail)
    rag_index_ready             False placeholder until Phase 3 ChromaDB/FAISS probe
    queue_depth                 Current number of jobs waiting in FIFO queue
    queue_max                   Always 10
    average_pipeline_duration_s Rolling avg of last 10 completions (null until 3+ recorded)
    jobs_completed_today        In-memory counter, resets on container restart
    jobs_failed_today           In-memory counter, resets on container restart


## TriageReport fields (Phase 2.5 additions)

Two new fields on TriageReport, accessible after a completed job:

execution_plan_summary: str | None
    Human-readable execution plan summary.
    Format: "Mandatory: ✓ Tool ✗ Tool ... | Optional: ✓ Tool [FALLBACK] | reasoning"

    Mandatory tools reflect input presence at submission time.
    Not all mandatory tools will show ✓ — only those whose
    corresponding input was provided.

    Optional tools reflect plan.use_rag (the planner's decision).

    Suffix tags:
        [FALLBACK]  — ExecutionPlanner failed; deterministic fallback used
        [REPAIRED]  — PlanValidator overrode use_rag=False to True
        (no tag)    — planner succeeded, plan conformed to safety policy

    Example (chest pain session, RAG forced by PlanValidator):
        "Mandatory: ✓ VoiceTranscriber | ✗ SymptomExtractor | ✗ LabReportParser |
         ✗ XRayProcessor | ✓ DrugInteractionChecker | Optional: ✓ MedicalRAGSearch
         [REPAIRED] | Planner chose no RAG but safety override applied."

    Note: VoiceTranscriber shows ✓ when audio was submitted, regardless of
    whether transcription succeeded. The ✓/✗ reflects input presence, not
    tool success — tool success is in tools_run/tools_failed.

    None only if execution_plan was unavailable. Cannot occur in normal
    pipeline flow — execution_plan is guaranteed non-None after
    _run_execution_planner.

validation_status: str | None
    RuleValidator outcome: "agreement", "warning", or "override".
    None when RuleValidator did not run or returned ToolError.

    "agreement" — SLM narrative and deterministic severity concur.
                  No action required.

    "warning"   — Minor discrepancy or unextractable narrative level.
                  Flagged for clinician review. No automatic correction.

    "override"  — Deterministic rules require HIGH but narrative
                  expresses a lower severity. TriageReport.severity
                  is already set to the deterministic level (HIGH).
                  Clients SHOULD display a safety banner.

    Client behaviour on "override":
        Display a prominent safety banner stating that the deterministic
        severity assessment (HIGH) is authoritative and that the SLM
        narrative expressed a lower severity level. The discrepancy does
        not affect the structured severity field — it is informational,
        surfacing a quality signal about the LLM output for the clinician.


## CORS Status

CORS middleware is not currently configured.

When the React frontend lands, CORS middleware will be added to
backend/main.py to allow the Vite dev server (http://localhost:5173)
to call the FastAPI backend (http://localhost:8000) during development.

Production deployment serves the React build from FastAPI as static
files, eliminating CORS entirely.