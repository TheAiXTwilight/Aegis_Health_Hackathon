# Aegis Health — Memory Profile

Target hardware: NVIDIA Jetson Orin Nano 8 GB

Actual measurements to be taken on Jetson during Week 1 and Week 2.
All figures below are estimates from the spec — not yet verified.


## Estimated Peak Memory by Pipeline State

State                                   Estimated Peak
Idle                                    2.2 – 2.6 GB
VoiceTranscriber active                 2.3 – 2.7 GB
LabReportParser (digital PDF)           2.3 – 2.7 GB
LabReportParser (EasyOCR active)        3.4 – 4.1 GB
XRayProcessor active                    TBD — measure after Commit 6
ReportGenerator (LLM active)            TBD — measure Week 1

Phase 2.5 additions (all negligible):
    ExecutionPlanner: one non-streaming HTTP call — no model loaded
                      locally, no significant memory allocation
    PlanValidator:    pure Python dict operations — negligible
    RuleValidator:    regex over a string — negligible

Phase 3 Commit 4 — LabReportParser:
    PyMuPDF extraction:   negligible additional allocation
    pdfminer fallback:    negligible additional allocation
    EasyOCR (opt-in):     peak estimated at 3.4–4.1 GB total
                          EasyOCR is explicitly released after use
                          (see Memory Safety Measures below)

Phase 3 Commit 5 — VoiceTranscriber:
    Model:            Faster-Whisper tiny.en INT8
    Model size:       ~40 MB on disk
    Runtime RSS:      ~150–200 MB additional (CTranslate2 INT8 engine
                      + audio buffer for typical <60s clip)
    Singleton:        Loaded once on first call, resident for process
                      lifetime — no reload between requests
    Device:           CPU only (Phase 3) — no CUDA allocation
    Note:             VoiceTranscriber active peak estimated at
                      2.3–2.7 GB total (idle + ~200 MB model overhead)
                      Verify on Jetson during Week 1 profiling.

All Phase 2 memory estimates remain valid.

Warning: if EasyOCR peak exceeds 5.5 GB, implement staged loading in Week 2.


## Memory Safety Measures

Sequential pipeline execution ensures only one tool holds
resident memory at a time.

OLLAMA_NUM_PARALLEL=1
    No concurrent LLM inference.

OLLAMA_MAX_LOADED_MODELS=1
    Only one model loaded at a time.

OLLAMA_KEEP_ALIVE=-1
    Model stays resident. Avoids reload latency between pipeline runs.

EasyOCR (Phase 3 Commit 4 — LabReportParser):
    The EasyOCR Reader is created inside _extract_via_easyocr() and
    is not retained after the function exits. Under normal conditions
    it becomes eligible for garbage collection. If memory pressure is
    observed on Jetson during profiling, explicit cleanup may be enabled:
        del reader
        gc.collect()

Faster-Whisper (Phase 3 Commit 5 — VoiceTranscriber):
    _MODEL singleton is retained for the process lifetime (lazy singleton
    pattern). This is intentional — reload cost (~1–2s on CPU) is
    unacceptable for a real-time triage tool. If memory pressure is
    observed on Jetson, consider unloading between pipeline runs:
        import tools.voice_transcriber as vt
        vt._MODEL = None
    This is not done by default.


## Stream Buffer Memory

asyncio.Queue(maxsize=256) per active job.
One queue exists per running job. Deleted when job is purged.
At 256 tokens of roughly 4 bytes each this is approximately 1 KB.
Negligible contribution to peak.


## Duration Tracking Memory

deque(maxlen=100) for pipeline duration floats.
100 floats at 8 bytes each is 800 bytes.
Negligible contribution to peak.


## Frontend Memory Considerations

The frontend runs in the user's browser — not on the Jetson.
React + Vite + TypeScript + Tailwind + shadcn/ui contributes zero
runtime memory to the Jetson when used as a web frontend.

The Streamlit prototype (deprecated) was a server-side Python process
and contributed runtime memory. It is being removed entirely.


## Action Items

Week 1:
    Measure idle baseline on Jetson
    Measure ReportGenerator (LLM active) peak on Jetson
    Measure ExecutionPlanner Ollama call peak (expected negligible)
    Update this file with real measurements

Week 2:
    Measure VoiceTranscriber (Faster-Whisper) peak
    Measure LabReportParser digital PDF peak (PyMuPDF path)
    Measure LabReportParser digital PDF peak (pdfminer fallback path)
    Measure LabReportParser EasyOCR active peak on scanned PDF
    If EasyOCR peak exceeds 5.5 GB, implement staged loading immediately
    Validate EasyOCR ARM64 — see docs/setup_jetson.md
    Update this file with real measurements

Week 3:
    Measure full pipeline peak across Scenario A, B, C
    Verify no OOM under Scenario C (token budget stress)
    Update this file with final measurements