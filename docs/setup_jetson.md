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


## EasyOCR ARM64 Validation

EasyOCR must be validated on ARM64 before Week 2 implementation.

If EasyOCR fails on ARM64:
    Switch to pytesseract
    Redefine Scenario B fixture accordingly
    Document the decision in this file


## GPU Detection

The health endpoint detects GPU presence via:
    1. Jetson device nodes (/dev/nvhost-gpu, /dev/nvidiactl)
    2. nvidia-smi query
    3. tegrastats

torch is never imported for GPU detection.


## Known Issues

Python 3.10 default on JetPack 6
    asyncio.timeout() requires 3.11.
    Install Python 3.11 explicitly as described above.

EasyOCR ARM64 compatibility
    Not yet validated. Validate in Week 1.

Knowledge base not yet built
    TriageReport.knowledge_base_version and knowledge_base_date
    will be null until corpus build is completed on Mac and committed.

Frontend not yet built
    Backend API is stable and fully functional via curl, Postman,
    or any HTTP client. React frontend lands in Week 5+.