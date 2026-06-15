# Aegis Health — Memory Profile

Target hardware: NVIDIA Jetson Orin Nano 8 GB

Actual measurements to be taken on Jetson during Week 1 and Week 2.
All figures below are estimates from the spec — not yet verified.


## Estimated Peak Memory by Pipeline State

State                                   Estimated Peak
Idle                                    2.2 – 2.6 GB
VoiceTranscriber active                 2.4 – 2.8 GB
LabReportParser (digital PDF)           2.3 – 2.7 GB
LabReportParser (EasyOCR active)        3.4 – 4.1 GB
XRayProcessor active                    2.2 – 2.6 GB
ReportGenerator (LLM active)            TBD — measure Week 1

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

EasyOCR (when implemented)
    Explicitly released after use:
        del self.ocr_reader
        self.ocr_reader = None
        gc.collect()

VoiceTranscriber (when implemented)
    Faster-Whisper model released immediately after transcription.


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
and would have contributed runtime memory if retained. It is being
removed entirely.


## Action Items

Week 1:
    Measure idle baseline on Jetson
    Measure ReportGenerator (LLM active) peak on Jetson
    Update this file with real measurements

Week 2:
    Measure VoiceTranscriber (Faster-Whisper) peak
    Measure LabReportParser digital PDF peak
    Measure LabReportParser EasyOCR active peak on scanned PDF
    If EasyOCR peak exceeds 5.5 GB, implement staged loading immediately
    Update this file with real measurements

Week 3:
    Measure full pipeline peak across Scenario A, B, C
    Verify no OOM under Scenario C (token budget stress)
    Update this file with final measurements