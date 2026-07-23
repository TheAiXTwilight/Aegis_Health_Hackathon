# Aegis Health — Jetson Setup

Target hardware: NVIDIA Jetson Orin Nano 8 GB
JetPack: 6.x


## Python Version Requirement

Aegis Health requires Python 3.11 or higher.

asyncio.timeout() used in backend/queue.py was introduced in Python 3.11.
JetPack 6 ships Python 3.10 as default. Python 3.11 must be installed explicitly.

The Dockerfile pins python:3.11-slim.

Verify the Python version inside your container or environment:

    python --version

If below 3.11, install via deadsnakes PPA:

    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.11 python3.11-venv

Or build from source if deadsnakes is unavailable on your JetPack version.


## Prerequisites

    JetPack 6.x installed
    Docker and Docker Compose installed
    NVIDIA Container Runtime configured
    Git installed


## Deployment (Jetson)

Clone the repository:

    git clone <repo-url>
    cd aegis-health

Start all services:

    docker compose up

The entrypoint.sh script handles:
    Waiting for Ollama to become healthy
    Pulling and creating the aegis-llama model from config/Modelfile
    Running model warmup
    Starting uvicorn on port 8000


## Ollama Image Pinning

The docker-compose.yml uses a digest-pinned Ollama image.
To get the current digest:

    docker inspect ollama/ollama:latest --format '{{index .RepoDigests 0}}'

Update the sha256 digest in docker-compose.yml before deployment.


## Ollama Concurrency Guards

These environment variables are mandatory and must not be changed:

    OLLAMA_NUM_PARALLEL=1
    OLLAMA_MAX_LOADED_MODELS=1
    OLLAMA_KEEP_ALIVE=-1

Increasing NUM_PARALLEL or MAX_LOADED_MODELS risks OOM on 8 GB.


## OLLAMA_BASE_URL

The app service communicates with the Ollama service via:

    OLLAMA_BASE_URL=http://ollama:11434

This is set in docker-compose.yml. Do not change it unless the
Ollama service name changes.

Both ExecutionPlanner (non-streaming, temperature=0.0) and
ReportGenerator (streaming, temperature=0.2) use this base URL.


## Ports

Port    Service             Status
8000    FastAPI backend     Active
8501    Streamlit           Deprecated — Streamlit prototype is being removed
5173    Vite dev server     Planned — when React frontend lands

The React frontend will be served by FastAPI in production
(via StaticFiles mount on /). Port 5173 is only used during
local development with `npm run dev`.


## Frontend Status

No frontend is currently implemented.

The original Streamlit prototype has been deprecated. The replacement
is React + Vite + TypeScript + Tailwind + shadcn/ui — not yet built.

This does not affect Jetson deployment. The backend runs and serves
the API regardless of whether a frontend exists.

When the React frontend lands, the production Dockerfile gains a
Node build stage. The output (frontend/dist) is served by FastAPI
as static files in the same container — no separate frontend container
required.


## EasyOCR ARM64 Validation (Phase 3 — Required before Commit 4 production use)

EasyOCR is used by LabReportParser as the third-tier fallback for
scanned PDFs. It is only loaded when AEGIS_OCR=1 (or "true" / "yes")
is set in the environment. Digital PDFs do not trigger EasyOCR.

Validate EasyOCR on ARM64 before enabling AEGIS_OCR in production:

    AEGIS_OCR=1 python -c "
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=True)
    print('EasyOCR ARM64: OK')
    "

If EasyOCR fails on ARM64:
    1. Switch to pytesseract
    2. Replace _extract_via_easyocr() in tools/lab_report_parser.py
       with a pytesseract implementation (same signature: Path → str | None)
    3. Update the pytesseract entry in docs/requirements.md
    4. Update this file with the outcome and date

EasyOCR ARM64 status: NOT YET VALIDATED


## GPU Detection

The health endpoint detects GPU presence via:
    1. Jetson device nodes (/dev/nvhost-gpu, /dev/nvidiactl)
    2. nvidia-smi query
    3. tegrastats

torch is never imported for GPU detection.


## Phase 2.5 Notes

ExecutionPlanner, PlanValidator, and RuleValidator add no new
system dependencies. They use httpx (already present), stdlib
modules (json, re), and pydantic (already present).

ExecutionPlanner makes one additional non-streaming Ollama call
per pipeline run. This call uses temperature=0.0 and num_predict=128,
making it fast and low-latency relative to the main ReportGenerator call.

Memory impact of Phase 2.5 components is negligible — no new models
are loaded, no significant memory allocations.


## Phase 3 Notes

Commit 4 (LabReportParser) introduces no new system packages beyond
what Phase 3 Commit 1 already required. PyMuPDF, pdfminer.six, and
EasyOCR are all listed in docs/requirements.md.

LabReportParser extraction waterfall on Jetson:
    PyMuPDF and pdfminer.six run on CPU — no GPU required.
    EasyOCR (when enabled) runs with gpu=False — CPU-only on Jetson.
    Memory peak estimate for EasyOCR active: 3.4–4.1 GB (see docs/memory_profile.md).

AEGIS_OCR should remain unset in standard Jetson deployments unless
the input includes scanned (image-only) PDFs. Digital lab reports
extracted by PyMuPDF or pdfminer.six do not need OCR.


## Known Issues

Python 3.10 default on JetPack 6
    asyncio.timeout() requires 3.11.
    Install Python 3.11 explicitly as described above.

EasyOCR ARM64 compatibility
    Not yet validated. Validate before enabling AEGIS_OCR=1 in production.
    See EasyOCR ARM64 Validation section above.

Knowledge base not yet built
    TriageReport.knowledge_base_version and knowledge_base_date
    will be null until corpus build is completed on Mac and committed.

Frontend not yet built
    Backend API is stable and fully functional via curl, Postman,
    or any HTTP client. React frontend lands in Phase 5.