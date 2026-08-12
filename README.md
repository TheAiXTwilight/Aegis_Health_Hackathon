# Aegis Health

**Offline-first AI health triage assistant.** Aegis Health runs entirely on-device — on a laptop for development, or on an NVIDIA Jetson board at the edge — combining a local LLM, speech, computer vision, and retrieval-augmented generation to turn symptoms, lab reports, chest X-rays, and voice input into a structured, clinician-style triage report. No cloud API calls, no third-party inference providers, no patient data leaving the device.

> ⚠️ **Not a medical device.** Aegis Health is a research/demo project for educational and triage-assistance purposes only. It does not diagnose disease and is not a substitute for professional medical evaluation.

---

## Features

- Multimodal intake — free-text or voice symptoms, lab PDFs, chest X-rays, and medication lists, in any combination
- Deterministic, auditable pipeline instead of a freeform agent loop (see [Architecture](#architecture))
- On-device speech-to-text (faster-whisper) and text-to-speech (Piper), the latter run as a prewarmed, idle-evicted worker process
- Chest X-ray classification (torchxrayvision DenseNet121) with Grad-CAM explanations and DICOM support
- RAG-grounded medical knowledge via a local FAISS/ChromaDB index over a versioned corpus
- Local drug interaction checking (RxNorm-mapped), no external API calls
- Transparent, rule-based severity scoring (see [`docs/severity_rules.md`](docs/severity_rules.md)), cross-checked against the LLM narrative by a `RuleValidator` step
- Streamed report generation over SSE, PDF and FHIR exports, JWT auth, and per-user health/vitals history

## Architecture

A **sequential, planner-gated agent pipeline** (full detail in [`docs/architecture.md`](docs/architecture.md)):

```
Step -1  ExecutionPlanner        SLM routing decision (use_rag only)
Step  0  PlanValidator           Safety override — forces RAG on for high-acuity input
Step  1  VoiceTranscriber        if audio provided
Step  2  SymptomExtractor        if symptoms/voice available
Step  3  LabReportParser         if lab PDF provided
Step  4  XRayProcessor           if X-ray provided
Step  5  MedicalRAGSearch        if planner requests it
Step  6  DrugInteractionChecker  if medications provided
Step  7  SeverityScorer          always
Step  8  ReportGenerator         always — streams the narrative report
Step  9  RuleValidator           always — checks narrative against deterministic rules
```

**Design invariant:** the LLM planner can only opt in to *optional* enrichment (RAG search) — it has no authority to skip mandatory, input-driven tools like severity scoring or drug checks. This is enforced structurally: the schema it populates (`ExecutionPlan`) has exactly one field, `use_rag`, so there's no field through which it could suppress a safety-relevant tool. Steps run strictly sequentially and the whole pipeline is wall-clock bounded by `PIPELINE_TIMEOUT_S`.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn, SQLAlchemy, Pydantic |
| Local LLM | Ollama (e.g. Llama 3.2 1B) — planning + streamed report generation |
| Speech | faster-whisper (STT), Piper (TTS) — both on-device |
| Vision | torchxrayvision DenseNet121 + Grad-CAM |
| RAG | FAISS + ChromaDB, ONNX MiniLM embeddings |
| Exports | WeasyPrint/ReportLab (PDF), FHIR bundle export |
| Frontend | React 19, React Router, Vite |
| Auth | JWT (access + refresh cookies), bcrypt |
| Database | SQLite |

## Project structure

```
backend/   FastAPI routes: auth, queue, streaming, chat, exports, TTS, accounts
app/       Settings, auth utilities, DB session/models
agents/    Pipeline orchestration
tools/     Individual pipeline tools — one per pipeline step
vision/    X-ray processing, DICOM reading, Grad-CAM
rag/       Corpus ingestion, chunking, embedding, index building
schemas/   Pydantic schemas shared across the pipeline
frontend/  React + Vite SPA
data/      SQLite DB, drug DB, audio models, knowledge base, X-ray weights (large — see below)
docs/      Architecture, API, severity rules, and setup documentation
tests/     Unit, integration, scenario, and benchmark suites
```

### Data & models

`data/` ships pretrained weights and indices rather than requiring a training step:

| Path | Contents |
|---|---|
| `data/aegis.db` | SQLite application database |
| `data/drugs/` | RxNorm-mapped local drug interaction database |
| `data/audio/whisper-tiny-en/` | faster-whisper STT model |
| `data/audio/piper-en-gb-jenny-medium/` | Piper TTS voice model |
| `data/knowledge/` | FAISS index, ChromaDB store, MiniLM ONNX embeddings, versioned corpus (~150 MB) |
| `data/xray/*.pt` | torchxrayvision DenseNet121 weights (~28 MB) |

Expect a large checkout (several hundred MB) due to bundled weights — consider Git LFS if forking for your own deployment.

## Getting started

### Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- [Ollama](https://ollama.com) reachable from the backend, with a model pulled:
  ```bash
  ollama pull llama3.2:1b
  ```

---

### Part 1 — Local development

**Backend**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` in the project root (see `app/settings.py` for the full list of variables, and [Configuration](#configuration) below):

```bash
AEGIS_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
AEGIS_ENV=development
COOKIE_SECURE=false

AEGIS_OLLAMA_BASE_URL=http://localhost:11434
AEGIS_OLLAMA_MODEL=llama3.2:1b

AEGIS_CORS_ORIGINS=["http://localhost:5173"]
```

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

API available at `http://localhost:8000` (health check at `/health`, readiness at `/readyz`).

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

App available at `http://localhost:5173`, pointed at the backend URL configured in `frontend/src/services/api.js`.

---

### Part 2 — Jetson / edge deployment

Designed to run fully offline on an NVIDIA Jetson board, with Ollama on the host and the app exposed via an ngrok tunnel for remote/demo access.

Production `.env` differs from local dev in a few key ways:

```bash
AEGIS_SECRET_KEY=<generate-a-unique-value-per-device>
AEGIS_ENV=production
COOKIE_SECURE=true

# Ollama runs on the host — reach it via the Docker host gateway, not localhost
AEGIS_OLLAMA_BASE_URL=http://172.17.0.1:11434
AEGIS_OLLAMA_MODEL=llama3.2:1b

# Update these if the ngrok tunnel URL changes between sessions
AEGIS_CORS_ORIGINS=["https://<your-tunnel>.ngrok-free.dev"]
AEGIS_PUBLIC_URL=https://<your-tunnel>.ngrok-free.dev

# Piper TTS worker — prewarmed at boot, killed on idle to reclaim RAM
AEGIS_TTS_IDLE_EVICT_SECS=300
```

> 🔒 Never commit a real `.env` — `AEGIS_SECRET_KEY` signs auth tokens. Generate a fresh one per device and rotate it if exposed.

Bootstrap a fresh session with the included script:

```bash
bash setup_aegis.sh
```

It installs missing `tzdata`, clones/updates the repo, verifies required patches are present, checks Ollama reachability, generates `.env` if missing, kills stray processes, and starts the backend, waiting up to 40s for `/health` to report ready.

See [`docs/setup_jetson.md`](docs/setup_jetson.md) for the full guide and [`docs/memory_profile.md`](docs/memory_profile.md) for hardware/thread tuning on constrained devices.

---

## Configuration

Selected settings from `app/settings.py` (full list there; all overridable via `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `AEGIS_SECRET_KEY` | dev placeholder | Signs JWT access/refresh tokens — override in production |
| `AEGIS_ACCESS_TOKEN_EXPIRE_MINUTES` / `AEGIS_REFRESH_TOKEN_EXPIRE_HOURS` | `30` / `72` | Token lifetimes |
| `AEGIS_CORS_ORIGINS` | `["http://localhost:5173"]` | Allowed frontend origins |
| `COOKIE_SECURE` | `false` | Set `true` behind HTTPS in production |
| `AEGIS_QUEUE_MIN_SIZE` / `AEGIS_QUEUE_MAX_SIZE` | `5` / `20` | Pipeline job queue bounds |
| `AEGIS_OLLAMA_BASE_URL` / `AEGIS_OLLAMA_MODEL` | `http://localhost:11434` / `llama3.2` | LLM endpoint and model |
| `AEGIS_TTS_IDLE_EVICT_SECS` | `300` | Idle time before the TTS worker is killed to reclaim RAM |

## API

FastAPI serves interactive docs at `/docs` when running. Key endpoint groups:

| Group | Examples | Purpose |
|---|---|---|
| Auth | `POST /auth/register`, `/auth/login`, `/auth/refresh`, `GET /auth/me` | Account creation, login, session management |
| Pipeline | `POST /queue/submit`, `GET /queue/status/{job_id}`, `GET /queue/stream/{job_id}` | Submit a multimodal job, poll/stream progress |
| Chat | `POST /chat`, `GET /chat/{job_id}/init` | Follow-up Q&A grounded in a completed report |
| Records | `GET /records`, `POST /checkin`, `GET /trends` | Saved reports, vitals check-ins, trend charts |
| Exports | `GET /pdf/{job_id}`, `GET /fhir/{id}`, `GET /zip` | PDF, FHIR bundle, and zipped exports |
| Voice | `POST /speak`, `/speak/stream` | Text-to-speech synthesis |
| Health | `GET /health`, `/readyz`, `/metrics` | Liveness, readiness, metrics |

Full request/response payloads, status codes, and upload limits (file size/type/duration, queue depth) are documented in [`docs/api.md`](docs/api.md).

## Testing

```bash
pytest
```

- `tests/tools/` — unit tests per pipeline tool (severity scoring, drug checking, RAG search, PDF/FHIR export, X-ray processing, etc.)
- `tests/integration/` — API endpoint tests (auth, records, queue, streaming, TTS, vitals)
- `tests/scenarios/` — end-to-end regressions against realistic clinical cases
- `tests/benchmarks/` — load/performance tests

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — pipeline design and invariants
- [`docs/api.md`](docs/api.md) — full API reference
- [`docs/severity_rules.md`](docs/severity_rules.md) — severity scoring rules
- [`docs/setup_jetson.md`](docs/setup_jetson.md) — edge deployment guide
- [`docs/memory_profile.md`](docs/memory_profile.md) — memory/thread tuning for constrained hardware

## Known limitations

- Single-node pipeline — no horizontal scaling across multiple GPUs/devices
- In-process FIFO queue, not backed by an external broker
- SQLite fits single-device/single-clinic deployments, not multi-tenant scale
- The 1B-parameter planning LLM is intentionally lightweight for edge latency; robustness is backstopped by the deterministic `PlanValidator`/`RuleValidator` safety checks rather than the model itself
