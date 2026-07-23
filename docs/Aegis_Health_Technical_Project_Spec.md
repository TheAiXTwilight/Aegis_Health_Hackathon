# Aegis Health — Technical Project Spec
### Local AI Triage Assistant · Edge Deployment · Privacy-First · Multi-User Demo-Production System

**Spec date:** 2026-07-01  
**Target hardware:** NVIDIA Jetson Orin Nano 8 GB  
**Build structure:** Two phases only  

---

## 1. Executive Summary

Aegis Health is a privacy-first local AI triage assistant that accepts patient inputs across multiple clinical modalities and produces a structured, evidence-backed triage report without sending patient data to external inference APIs.

The current project already has a strong working baseline:

- FastAPI backend.
- React + Vite frontend.
- Local Ollama-based Small Language Model orchestration.
- Sequential multimodal triage pipeline.
- In-memory queue with report streaming.
- Health/status endpoint.
- Medical form UI.
- Report UI.
- Dashboard UI.
- Voice, symptom, lab, medication, X-ray, RAG, severity, and validation components.

This spec divides the remaining project into exactly two phases:

1. **Phase 1 — Development**  
   Implement and integrate all remaining technical work: the locked 22 enhancements, modifications/updations of existing backend/frontend/pipeline work, tests, and local development validation.

2. **Phase 2 — Deployment**  
   Package, deploy, harden, validate, and demo the completed system on Jetson with Docker, model prewarm, persistent volumes, smoke tests, reset scripts, and final pitch/demo assets.

The locked 22 enhancements remain the core scope. Additional accepted refinements from the extended technical context are integrated directly where they strengthen security, traffic control, observability, deployment, or demo stability.

---

## 2. Problem Statement

AI-powered healthcare tools almost universally depend on cloud APIs. Patient data is transmitted to and processed on remote infrastructure outside clinical control. This creates real privacy risks and makes deployment impractical in low-connectivity or resource-constrained settings.

Existing tools compound this by operating on single input types: a symptom checker, an image analyser, a lab report parser, or a medication lookup tool. They rarely reason across symptoms, labs, chest X-rays, medications, vitals, and retrieved medical evidence together in one local workflow.

Current triage tools also usually lack continuity. Each visit is treated as a one-off event, with no authenticated user history, no longitudinal health vault, no current-vs-last comparison, and no personalized baseline trends. That makes them feel like isolated AI demos instead of a practical clinical companion.

Multi-user operation is another gap. A real clinic, classroom demo, or hackathon booth needs multiple users submitting cases at the same time, with user-level data isolation, queue visibility, rate limits, priority handling for urgent cases, and safe backpressure instead of a hard failure when the queue fills.

**Aegis Health addresses these problems.** All inference runs locally on the Jetson with no external inference API calls. The system reasons across submitted clinical modalities, retrieves evidence from local medical knowledge sources, streams a structured triage report, persists user-owned health history, and supports multi-user traffic control on a single edge device.

Aegis Health does **not** diagnose. It triages: it gives clinicians and users a structured, evidence-backed starting point while keeping patient data private, local, auditable, and under user or clinical control.

---

## 3. Technology Stack

The project stack to be documented and completed is:

```text
FastAPI + React + Vite + TypeScript + Tailwind
```

`shadcn/ui` is **not** part of the final stack. It was considered earlier, but it has been removed from the project scope.

### 3.1 Backend Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Runtime | Python |
| Validation | Pydantic |
| Queue | In-memory async queue with single inference worker |
| Database | SQLite with WAL mode |
| ORM | SQLAlchemy |
| Auth | JWT access token + httpOnly refresh cookie |
| Local LLM runtime | Ollama |
| Deployment | Docker Compose on Jetson Orin Nano |

### 3.2 Frontend Stack

| Layer | Technology |
|---|---|
| UI framework | React |
| Build tool | Vite |
| Language target | TypeScript |
| Styling | Tailwind CSS |
| Charting | Recharts for dashboard timelines and vitals sparklines |
| UI components | Custom project components |
| Removed from scope | shadcn/ui |

The current attached frontend is already React + Vite, but it uses JavaScript/JSX and custom CSS. During Phase 1 development, the frontend should be standardized to the target stack where needed: React + Vite + TypeScript + Tailwind, without adding shadcn/ui.

---

##### Implementation snippet — Tailwind Setup Without shadcn/ui

`frontend/tailwind.config.ts`

```ts
import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        aegis: {
          navy: "#0d2167",
          blue: "#2563ff",
          text: "#425894",
        },
      },
      borderRadius: {
        glass: "2rem",
      },
    },
  },
  plugins: [],
} satisfies Config;
```

`frontend/src/styles/globals.css`

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer components {
  .glass-card {
    @apply rounded-glass border border-white/30 bg-white/15 shadow-xl backdrop-blur-2xl;
  }
  .aegis-button {
    @apply rounded-2xl bg-blue-600 px-5 py-3 font-bold text-white transition hover:bg-blue-700;
  }
}
```

## 4. Current Existing Work

This section records the existing completed baseline. These parts are not discarded. They are modified and upgraded during Phase 1.

### 4.1 Existing Backend Work

| Area | Current Status |
|---|---|
| FastAPI app | Implemented |
| Lifespan startup/shutdown | Implemented |
| Background inference worker | Implemented |
| In-memory job store | Implemented |
| In-memory queue | Implemented |
| Stream queue per job | Implemented |
| Upload saving/validation | Implemented |
| Patient metadata in state | Implemented |
| Multiple lab PDF upload | Implemented |
| Multiple X-ray image upload | Implemented |
| Structured result endpoint | Implemented |
| Health endpoint | Implemented |
| ToolError error codes | Implemented |
| Upload cleanup after purge | Implemented |

Current backend endpoints:

| Endpoint | Method | Current Status | Purpose |
|---|---:|---|---|
| `/queue/submit` | POST | Implemented | Submit medical form and enqueue job |
| `/queue/status/{job_id}` | GET | Implemented | Return queue and pipeline status |
| `/queue/stream/{job_id}` | GET | Implemented | Stream report text |
| `/queue/result/{job_id}` | GET | Implemented | Return structured final result |
| `/health` | GET | Implemented | Return system and queue status |

### 4.2 Existing Pipeline Work

| Step | Component | Current Responsibility |
|---:|---|---|
| -1 | ExecutionPlanner | Local SLM decides optional RAG use |
| 0 | PlanValidator | Repairs unsafe planner decisions |
| 1 | VoiceTranscriber | Transcribes uploaded audio |
| 2 | SymptomExtractor | Extracts structured symptoms |
| 3 | LabReportParser | Parses lab PDFs |
| 4 | XRayProcessor | Processes chest X-ray images |
| 5 | MedicalRAGSearch | Retrieves local evidence when selected |
| 6 | DrugInteractionChecker | Checks medication interactions |
| 7 | SeverityScorer | Produces deterministic triage severity |
| 8 | ReportGenerator | Generates structured report; current export uses deterministic report engine with validated sections |
| 9 | RuleValidator | Compares deterministic severity against narrative severity |

The existing pipeline already enforces the key safety invariant:

> If the user submits a clinical modality, the corresponding mandatory processing tool must run. The planner may control only optional enrichment.

### 4.3 Existing Frontend Work

| Area | Current Status |
|---|---|
| React + Vite app | Implemented |
| TypeScript + Tailwind target | Required update in Phase 1 |
| shadcn/ui | Removed from scope |
| Home page | Implemented |
| Health Scan page | Implemented |
| Medical Form page | Implemented |
| Report page | Implemented |
| Dashboard page | Implemented |
| About page | Implemented |
| Navbar | Implemented |
| SystemIndicator | Implemented |
| Voice recorder | Implemented |
| Multiple lab file badges | Implemented |
| Multiple X-ray file badges | Implemented |
| X-ray checklist | Implemented |
| Report streaming UI | Implemented |
| Pipeline tracker | Implemented |
| Result fetch after completion | Implemented |
| PDF preview/download workflow | Implemented |
| API client service | Implemented |

### 4.4 Existing Deployment Work

| Area | Current Status |
|---|---|
| Backend Dockerfile | Basic version exists |
| Docker Compose | Basic backend + Ollama compose exists |
| Entrypoint script | Basic version exists |
| Vite dev proxy | Exists for `/queue` and `/health` |
| Ollama environment guards | Partially present |
| Upload volume | Basic mount exists |

Deployment is completed in Phase 2 after Phase 1 development is integrated.

### 4.5 Existing Data / Model Asset Directory

The current codebase includes a populated `data/` directory with local model and knowledge assets already present. These are part of the existing baseline and must be preserved during deployment.

| Data Path | Current Contents | Spec / Deployment Meaning |
|---|---|---|
| `data/audio/whisper-tiny-en/` | `config.json`, `model.bin`, `tokenizer.json`, `vocabulary.txt` | Faster-Whisper tiny English assets are already local. Deployment must mount/copy this directory and set `AEGIS_WHISPER_DIR=/app/data/audio/whisper-tiny-en`. |
| `data/drugs/` | `aegis_drugs.db`, `build_drug_db.py`, `rxnorm_map.json` | DrugInteractionChecker already has a local SQLite FTS/RxNorm-backed drug data source. Deployment must preserve `aegis_drugs.db`. |
| `data/knowledge/chroma/` | `chroma.sqlite3` | ChromaDB persistent local vector store exists. `/health.rag_index_ready` should check this path. |
| `data/knowledge/` | `faiss.index`, `faiss.docs`, `minilm.onnx`, `tokenizer.json`, `raw/` | FAISS fallback, ONNX MiniLM embedding model, tokenizer, and raw corpus directory exist locally. Deployment must not require runtime downloads. |
| `data/xray/` | Chest X-ray model asset file(s) visible in current data tree | X-ray model weights/assets exist under `data/xray`. Deployment must mount/copy this directory and set `AEGIS_XRAY_MODEL_DIR=/app/data/xray`. |

Deployment implication:

- `data/` is not disposable cache; it contains required local inference assets.
- Phase 2 Docker/Jetson deployment must mount `../data:/app/data` or copy these assets into the image/volume.
- Smoke tests must verify these asset paths exist before running the demo.
- The app must not download Whisper, MiniLM, FAISS/Chroma, drug DB, or X-ray assets at demo time.

### 4.6 Current Codebase Review Updates

After reviewing the attached current backend and frontend codebase exports, these spec-level clarifications are required:

| Finding from Current Codebase | Required Spec / Implementation Update |
|---|---|
| `vision/xray_processor.py` is referenced by `agents/pipeline.py`, but it is not present in the latest backend export file list | Phase 1 must verify, restore, or commit `vision/xray_processor.py` before any Grad-CAM work. If the file exists locally but was omitted from export, ensure it is included in the repository package. |
| Current frontend is React + Vite with JavaScript/JSX and custom CSS | Phase 1 keeps the current UI but standardizes touched/new code to TypeScript + Tailwind. No shadcn/ui is added. |
| Current `frontend/vite.config.js` proxies only `/queue` and `/health` | Phase 1 must update proxy/API handling for `/auth`, `/dashboard`, `/records`, `/vitals`, `/export`, `/metrics`, and `/readyz`, or use `VITE_API_BASE_URL`. |
| Current frontend PDF path loads `html2pdf` from CDN | Phase 2 must remove CDN dependency for the final deployed PDF path by vendoring the library or generating PDFs backend-side. |
| Current Dockerfile uses `python:3.11-slim` and basic backend-only startup | Phase 2 must finalize Docker for the production deployment; Python version can remain if dependency-compatible, but DB init, frontend static serving, seed, reset, prewarm, and smoke scripts must be added. |
| Current `config/Modelfile` still uses `num_ctx 4096`, `num_predict 1024`, `temperature 0.2` | Phase 1/2 must tune the Modelfile or runtime Ollama options to the target demo profile: `num_ctx 3072`, `num_predict 768`, lower report temperature where applicable. |
| Current backend has robust endpoint tests and tool tests in the export | Phase 1 must preserve the existing test suite and add auth, persistence, priority queue, cache, metrics, export, and deployment tests without breaking current coverage. |

---

## 5. Locked 22 Enhancements

The full technical completion scope is the following 22 enhancements plus the modification/updation of existing work required to integrate them.

| # | Enhancement | Phase 1 Development Responsibility | Phase 2 Deployment Responsibility |
|---:|---|---|---|
| 1 | Auth JWT — 3 demo users | Implement auth, users, tokens, frontend login | Seed users and verify auth in deployed system |
| 2 | Health persistence — SQLite HealthRecord + VitalSnapshot | Implement DB models and persistence hooks | Mount DB volume and verify restart persistence |
| 3 | Personalized dashboard — vitals, trends, comparisons | Implement user-scoped dashboard data and UI | Validate seeded dashboards in deployed demo |
| 4 | Queue — priority lanes, rate limit, adaptive max, poll backoff | Implement queue/rate/polling changes | Load/smoke test queue behavior on Jetson |
| 5 | Model Registry + Ollama speed tuning | Implement registry/prewarm hooks and settings | Validate memory/latency under deployment |
| 6 | RuleValidator safety banner | Implement UI banner and data mapping | Verify safety banner in demo cases |
| 7 | Agentic plan theater | Implement chips, repaired/fallback markers | Verify plan theater in deployed report |
| 8 | Grad-CAM X-ray heatmap | Implement heatmap artifact generation/display | Verify heatmap file paths and serving |
| 9 | FHIR R4 export | Implement FHIR bundle export | Verify export works from deployed app |
| 10 | Personal-baseline risk z-score | Implement baseline calculation and UI | Verify seeded user history produces correct display |
| 11 | Voice TTS report readout | Implement readout controls | Verify browser/device behavior in deployed UI |
| 12 | Conversational follow-up `/queue/chat` | Implement chat endpoint and UI | Verify ownership/rate behavior in deployment |
| 13 | Daily vitals check-in | Implement vitals form and persistence | Verify persistence and dashboard update |
| 14 | PDF Clinical Dossier | Implement offline-safe dossier generation | Verify PDF works without CDN/network dependency |
| 15 | Result cache | Implement LRU cache and UI cache badge | Verify cache metrics and no user-data leakage |
| 16 | Observability | Implement `/metrics` and system card | Verify metrics and health in deployment |
| 17 | Data export ZIP + delete account | Implement export/delete logic | Verify exported files and account deletion in deployment |
| 18 | Suggested questions | Implement `suggested_questions[]` | Verify display in report/chat demo |
| 19 | Security headers + CSRF + CORS lock | Implement middleware/security behavior | Verify headers/CORS/cookies in production deployment |
| 20 | Demo hardening — checkpoint, prewarm, reset button | Implement checkpoint/recover/prewarm/reset APIs | Verify reset/prewarm/recover on Jetson |
| 21 | Medical Form polish | Implement validation, badges, retry UX | Verify upload/error UX in deployed browser |
| 22 | Pitch assets | Prepare assets from completed system | Present final demo video/poster/slides |

### 5.1 Accepted Additional Technical Refinements

These are not a separate roadmap. They are accepted refinements merged into the two-phase plan because they improve security, stability, observability, or demo reliability without changing the core product direction.

| Refinement | Where It Is Integrated | Reason |
|---|---|---|
| Timing-safe login | Auth implementation | Prevents email/user enumeration through response timing |
| Refresh-token rotation and replay rejection | Auth implementation | Makes stolen refresh tokens one-use at most |
| Queue starvation guard | Queue implementation | Critical cases jump ahead without freezing normal users forever |
| Stream reconnect/resume support | Report stream endpoint and Report page | Prevents white screen if a judge refreshes mid-stream |
| `/readyz` readiness endpoint | Deployment health checks | Lets smoke tests confirm model, RAG, and DB readiness separately from `/health` |
| Lightweight `audit_log` table | Security and compliance layer | Records login, submit, export, delete, and report-view events |
| Lightweight `consent_ledger` table | Registration and account settings | Records user consent snapshot without adding a large compliance subsystem |
| Encryption-ready PHI fields | Database and security layer | Keeps the schema ready for field encryption without blocking hackathon debugging |
| Recharts dashboard timelines | Dashboard UI | Makes vitals and severity history visually clear |
| Static frontend serving from FastAPI | Deployment | Avoids production CORS complexity and keeps the app local |
| Memory guard before heavy tools | Model registry / tool wrappers | Avoids Jetson OOM during OCR, X-ray, or prewarm paths |

---

## 6. Core Architecture

### 6.1 System Architecture

```text
React Frontend
    │
    ▼
FastAPI Backend
    │
    ├── Auth / User Context
    ├── Upload Validation
    ├── Queue Submit / Status / Stream / Result
    ├── SQLite Persistence
    ├── Metrics / Health
    ├── Export APIs
    └── Demo Admin APIs
          │
          ▼
Single Inference Worker
          │
          ▼
AegisPipeline
          │
          ├── ExecutionPlanner
          ├── PlanValidator
          ├── Mandatory Modality Tools
          ├── Optional MedicalRAGSearch
          ├── SeverityScorer
          ├── ReportGenerator
          └── RuleValidator
          │
          ▼
Persisted User Health Record + Report + Exports
```

### 6.2 Single-Inference-Worker Rule

The target hardware is Jetson Orin Nano 8 GB. The system must not run multiple full inference pipelines in parallel.

Rules:

- One FastAPI app instance.
- One Uvicorn worker in deployment.
- One inference worker.
- One Ollama model active at a time.
- Queue accepts multiple users.
- Pipeline execution remains sequential.

Throughput is improved through:

- priority queueing,
- rate limiting,
- adaptive queue max,
- result cache,
- model prewarm,
- tuned prompts and Ollama options,
- frontend polling backoff.

### 6.3 Planner Authority Invariant

The planner controls only optional enrichment.

| Tool | Trigger | Planner Control |
|---|---|---:|
| VoiceTranscriber | Audio file exists | No |
| SymptomExtractor | Symptom text or voice transcript exists | No |
| LabReportParser | Lab PDF exists | No |
| XRayProcessor | X-ray image/findings exist | No |
| DrugInteractionChecker | Medication list exists | No |
| SeverityScorer | Always | No |
| ReportGenerator | Always | No |
| RuleValidator | Always after report | No |
| MedicalRAGSearch | `execution_plan.use_rag=True` | Yes |

---

## 7. Data Model Required for Completion

SQLite with WAL mode is the required persistence layer for the completed system.

### 7.1 `users`

| Column | Type | Notes |
|---|---|---|
| id | string/UUID | Primary key |
| email | string | Unique |
| username | string | Optional unique login/display handle |
| display_name | string | Shown in UI |
| password_hash | string | bcrypt |
| role | string | `user` or `admin` |
| created_at | datetime | UTC |
| last_login_at | datetime/null | UTC |
| is_active | bool | Account status |

### 7.2 `refresh_tokens`

| Column | Type | Notes |
|---|---|---|
| id | string/UUID | Primary key |
| user_id | FK users.id | Required |
| token_hash | string | Store hash only |
| expires_at | datetime | 7 days |
| revoked_at | datetime/null | Set on logout/reset |
| created_at | datetime | UTC |

### 7.3 `health_records`

| Column | Type | Notes |
|---|---|---|
| id | string/UUID | Primary key |
| user_id | FK users.id | Required and indexed |
| job_id | string | Queue job id |
| severity | string | LOW/MEDIUM/HIGH |
| confidence | float | Report confidence |
| validation_status | string/null | agreement/warning/override |
| symptoms_text | text/null | Submitted symptoms |
| medications_json | text | Submitted medications |
| xray_findings_json | text | Submitted X-ray findings |
| report_json | text | Serialized report |
| result_json | text | Serialized structured outputs |
| created_at | datetime | UTC |

### 7.4 `vital_snapshots`

| Column | Type | Notes |
|---|---|---|
| id | string/UUID | Primary key |
| user_id | FK users.id | Required and indexed |
| systolic_bp | int/null | Optional |
| diastolic_bp | int/null | Optional |
| heart_rate | int/null | Optional |
| spo2 | float/null | Optional |
| temperature_c | float/null | Optional |
| glucose_mg_dl | float/null | Optional |
| weight_kg | float/null | Optional |
| notes | text/null | Optional |
| created_at | datetime | UTC |

### 7.5 `pipeline_jobs`

| Column | Type | Notes |
|---|---|---|
| job_id | string | Primary key |
| user_id | FK users.id | Required and indexed |
| session_id | string | Runtime state link |
| status | string | queued/running/completed/failed |
| priority | int | 0 critical, 5 normal |
| submitted_at | datetime | UTC |
| started_at | datetime/null | UTC |
| completed_at | datetime/null | UTC |
| error | text/null | Failure message |

### 7.6 `audit_log`

| Column | Type | Notes |
|---|---|---|
| id | string/UUID | Primary key |
| user_id | FK users.id/null | Nullable after account deletion/anonymization |
| action | string | login_success, login_failed, pipeline_submit, report_viewed, export_zip, delete_account, etc. |
| resource_type | string/null | job, record, account, export |
| resource_id | string/null | Related id when safe |
| ip_hash | string/null | Hash only, never raw IP |
| user_agent_hash | string/null | Hash only |
| metadata_json | text | JSON metadata without PHI |
| created_at | datetime | UTC |

### 7.7 `consent_ledger`

| Column | Type | Notes |
|---|---|---|
| id | string/UUID | Primary key |
| user_id | FK users.id | Required and indexed |
| consent_version | string | Version of consent text shown to user |
| consent_flags_json | text | JSON snapshot of user consent flags |
| created_at | datetime | UTC |

### 7.8 User Isolation Requirement

Every user-owned query must filter by authenticated `user_id`.

Required pattern:

```sql
WHERE user_id = :current_user_id
```

This applies to:

- health records,
- vitals,
- pipeline jobs,
- chat history,
- exports,
- delete account,
- dashboard trends,
- admin reset operations when scoped to demo users.

User isolation must be enforced in backend query logic, not by frontend hiding.

---

##### Implementation snippet — Backend Database Session

`app/db/session.py`

```python
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.settings import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.AEGIS_DB_URL,
    connect_args={"check_same_thread": False} if settings.AEGIS_DB_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    from app.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

##### Implementation snippet — Backend SQLAlchemy Models

`app/db/models.py`

```python
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    username: Mapped[str | None] = mapped_column(String, unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    records = relationship("HealthRecord", back_populates="user", cascade="all, delete-orphan")
    vitals = relationship("VitalSnapshot", back_populates="user", cascade="all, delete-orphan")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class HealthRecord(Base):
    __tablename__ = "health_records"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    validation_status: Mapped[str | None] = mapped_column(String)
    symptoms_text: Mapped[str | None] = mapped_column(Text)
    medications_json: Mapped[str] = mapped_column(Text, default="[]")
    xray_findings_json: Mapped[str] = mapped_column(Text, default="[]")
    report_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    user = relationship("User", back_populates="records")


class VitalSnapshot(Base):
    __tablename__ = "vital_snapshots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    systolic_bp: Mapped[int | None] = mapped_column(Integer)
    diastolic_bp: Mapped[int | None] = mapped_column(Integer)
    heart_rate: Mapped[int | None] = mapped_column(Integer)
    spo2: Mapped[float | None] = mapped_column(Float)
    temperature_c: Mapped[float | None] = mapped_column(Float)
    glucose_mg_dl: Mapped[float | None] = mapped_column(Float)
    weight_kg: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    user = relationship("User", back_populates="vitals")


class PipelineJobRow(Base):
    __tablename__ = "pipeline_jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=5)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
```

## 8. API Contract

### 8.1 Existing Endpoints to Keep and Upgrade

| Endpoint | Method | Required Upgrade in Phase 1 |
|---|---:|---|
| `/queue/submit` | POST | Add auth, user ownership, priority, rate limit, cache, persistence metadata |
| `/queue/status/{job_id}` | GET | Add user ownership checks, improved wait estimates |
| `/queue/stream/{job_id}` | GET | Add user ownership checks and reconnect-safe behavior, including optional `?offset=` resume |
| `/queue/result/{job_id}` | GET | Add user ownership checks and richer result fields |
| `/health` | GET | Keep lightweight; add fields needed by Navbar system card |

### 8.2 New Endpoints to Add in Phase 1 Development

| Endpoint | Method | Purpose |
|---|---:|---|
| `/auth/register` | POST | Create account |
| `/auth/login` | POST | Login and issue tokens |
| `/auth/refresh` | POST | Rotate refresh/access token |
| `/auth/logout` | POST | Revoke refresh token |
| `/auth/me` | GET | Return current user |
| `/dashboard` | GET | User-specific dashboard summary |
| `/records` | GET | List current user health records |
| `/records/{record_id}` | GET | Return one user-owned record |
| `/metrics` | GET | Prometheus-style metrics |
| `/admin/demo/reset` | POST | Reset seeded demo data |
| `/admin/demo/prewarm` | POST | Trigger model prewarm |
| `/queue/recover/{job_id}` | POST | Recover from checkpoint |
| `/vitals/checkin` | POST | Save vitals snapshot |
| `/vitals/trends` | GET | Return trend and baseline data |
| `/queue/chat` | POST | Ask follow-up question |
| `/queue/rerun/{job_id}` | POST | Optional fast re-score for a recently completed owned job |
| `/export/fhir/{record_id}` | GET | Export FHIR R4 bundle |
| `/export/pdf/{job_id}` | GET | Export clinical dossier PDF |
| `/export/zip` | GET | Export current user data |
| `/account` | DELETE | Delete account and user data |
| `/readyz` | GET | Deployment readiness: DB, model, and RAG ready |

Phase 2 validates and exposes these endpoints in the deployed environment.

---

##### Implementation snippet — Frontend API Client

`frontend/src/services/api.ts`

```ts
const BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function url(path: string) {
  return `${BASE_URL.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
}

let accessToken: string | null = null;

export function setAccessToken(token: string | null) {
  accessToken = token;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);
  const resp = await fetch(url(path), { ...init, headers, credentials: "include" });
  const data = await resp.json().catch(() => null);
  if (!resp.ok) {
    const err = new Error(data?.reason || data?.detail || `HTTP ${resp.status}`);
    (err as any).status = resp.status;
    (err as any).retryAfter = resp.headers.get("Retry-After");
    throw err;
  }
  return data as T;
}

export const api = {
  login: (email: string, password: string) => request<{ access_token: string }>("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  }),
  me: () => request<{ id: string; email: string; display_name: string }>("/auth/me"),
  dashboard: () => request<any>("/dashboard"),
  health: () => request<any>("/health"),
  metrics: () => fetch(url("/metrics")).then((r) => r.text()),
};
```


### 8.3 Frontend Client Behavior Contract

The React client must follow this flow:

1. Login with `/auth/login`; store the access token in React memory only.
2. Let the browser keep the refresh token in the httpOnly cookie.
3. Call `/auth/me` on app boot to hydrate the current user.
4. Submit scans with `Authorization: Bearer <access_token>`.
5. On `429`, read `Retry-After`, show a countdown, and retry only after the server window.
6. Poll `/queue/status/{job_id}` with adaptive backoff.
7. Open `/queue/stream/{job_id}` only after status becomes `running`.
8. After the stream ends, fetch `/queue/result/{job_id}` and replace streamed text with canonical report text.
9. If `validation_status === "override"`, render the red safety banner.
10. Use `/dashboard`, `/records`, `/vitals`, and exports only through authenticated requests.

---

## 9. Phase 1 — Development

### 9.1 Phase 1 Goal

Phase 1 implements all code-level work required to complete Aegis Health.

This includes:

- all 22 enhancements,
- modification/updation of existing backend work,
- modification/updation of existing frontend work,
- modification/updation of the existing pipeline,
- modification/updation of existing queue/status/stream/result behavior,
- local tests,
- local development validation before deployment.

Phase 1 ends when the completed system works locally in development mode and is ready to be packaged and deployed in Phase 2.


#### 9.1.1 Phase 1 Implementation Order

Development should proceed in this order so each later feature has the required foundation already available:

| Order | Work Item | Why It Comes Here |
|---:|---|---|
| 1 | Verify current codebase integrity, data assets, and `vision/xray_processor.py` | Pipeline imports and local model/data paths must resolve before feature work starts |
| 2 | Convert/standardize frontend target to TypeScript + Tailwind where touched | Prevents reworking newly added UI components later |
| 3 | Add DB session, SQLAlchemy models, and SQLite WAL setup | Auth, records, dashboard, vitals, and exports depend on persistence |
| 4 | Add auth backend: register/login/refresh/logout/me | Required before user-scoped records and protected routes |
| 5 | Add frontend AuthContext, LoginPage, and ProtectedRoute | Makes user flows testable early |
| 6 | Add demo seed users: Priya, Arjun, Judge | Enables dashboard and isolation testing |
| 7 | Add user_id into queue submit, state, job metadata, and result ownership checks | Prevents building features on session-only isolation |
| 8 | Add HealthRecord persistence hook after completed pipeline jobs | Creates source data for dashboard, records, exports, and demo history |
| 9 | Upgrade dashboard to user-specific history | Makes seeded users visibly different |
| 10 | Add priority queue, rate limits, adaptive max, and polling backoff | Stabilizes multi-user traffic before UX polish |
| 11 | Add result cache and cache metrics | Improves throughput and feeds observability |
| 12 | Add `/metrics` and expanded SystemIndicator | Makes congestion and performance visible |
| 13 | Add security headers, CSRF, and CORS lock | Protects auth/session work before final deployment |
| 14 | Add checkpoint, recover, prewarm, and reset APIs | Makes demo flow recoverable |
| 15 | Add clinical UI: safety banner, plan theater, Grad-CAM panel | Makes backend safety and explainability visible |
| 16 | Add vitals, z-score, suggested questions, and chat | Completes personalized copilot behavior |
| 17 | Add FHIR, PDF dossier, ZIP export, and delete account | Completes data ownership and export features |
| 18 | Polish Medical Form and report error/retry states | Removes demo-friction after core behavior works |
| 19 | Run full local tests and fix regressions | Phase 1 must end with a deployable codebase |

---

### 9.2 Existing Work to Modify and Update

| Existing Area | Current State | Required Phase 1 Development Update |
|---|---|---|
| FastAPI app | Queue, health, stream, result endpoints exist | Add auth dependencies, user scoping, DB session wiring, security middleware |
| Queue | FIFO with hard cap | Add priority, adaptive max, user-aware submission, rate-limit responses |
| Pipeline | Sequential working pipeline | Add persistence hook, checkpoint hook, cache hook, user/job metadata |
| AegisState | Patient metadata and tool outputs exist | Add authenticated `user_id`, persisted record id, cache metadata, checkpoint-safe serialization |
| `/health` | Basic system metrics exist | Expand fields for UI system card and connect to metrics counters |
| React routes | Public routes exist | Add login route, protected routes, auth-aware navigation |
| Vite proxy / API client | Current proxy covers only `/queue` and `/health`; current API client has no auth helpers | Add auth-aware API client and proxy/API base handling for all new endpoints |
| Medical form | Functional with uploads and voice | Add user-aware submit, better validation, retry/rate-limit messages, draft preservation |
| Report page | Streams report and shows pipeline | Add safety banner, plan theater, heatmap, cache badge, exports, TTS, chat |
| Dashboard | Uses result data | Add user history, vitals check-in, trends, baseline risk z-score |
| PDF flow | Basic frontend workflow exists | Upgrade to offline-safe clinical dossier generation |
| XRayProcessor | Pipeline imports `vision.xray_processor`; latest export must verify file is present | Restore/commit missing module if needed, then add optional Grad-CAM heatmap artifact |
| ReportGenerator | Produces clinical report | Add dossier-ready sections and suggested questions |
| SystemIndicator | Shows RAM/CPU/backend status | Add queue depth, avg duration, completed jobs, metrics display |

---

### 9.3 Enhancement Development Requirements

#### 9.3.1 Auth JWT and Demo Users

Backend:

- Add `users` table.
- Add `refresh_tokens` table.
- Add bcrypt password hashing.
- Add JWT access tokens with 15-minute expiry.
- Add httpOnly refresh token cookie with 7-day expiry.
- Add `/auth/register`, `/auth/login`, `/auth/refresh`, `/auth/logout`, `/auth/me`.
- Use timing-safe login behavior to avoid email enumeration.
- Rotate refresh tokens and reject replayed refresh tokens.
- Add login rate limit: 5 attempts/minute/IP.
- Add auth dependency for protected endpoints.

Frontend:

- Add `AuthContext`.
- Add `LoginPage`.
- Add protected route wrapper.
- Add user greeting.
- Add logout button.
- Persist login state through refresh flow.

Demo users:

| User | Purpose | Seed State |
|---|---|---|
| Priya | Rich history demo | Multiple visits and dashboard history |
| Arjun | Safety override demo | At least one high-risk previous case |
| Judge | Clean live demo | Empty account |

Additional security records:

- Write `audit_log` rows for login success/failure, submit, result view, export, account delete, and demo reset.
- Write `consent_ledger` rows at registration and whenever consent flags change.
- Store IP and user-agent as hashes only.

##### Implementation snippet — Auth Utilities and Router

`backend/auth.py`

```python
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from uuid import uuid4
import hmac
import secrets
import time

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.db.models import RefreshToken, User
from app.db.session import get_db
from app.settings import settings

router = APIRouter(prefix="/auth", tags=["auth"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
DUMMY_PASSWORD_HASH = pwd_context.hash("aegis-dummy-password")


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class RegisterIn(LoginIn):
    display_name: str
    username: str | None = None


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password_timing_safe(password: str, stored_hash: str | None) -> bool:
    # Always verify against a bcrypt hash, even when the email does not exist.
    # This prevents user enumeration through timing differences.
    candidate_hash = stored_hash or DUMMY_PASSWORD_HASH
    ok = pwd_context.verify(password, candidate_hash)
    return bool(stored_hash and ok)


def create_access_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.id,
        "email": user.email,
        "role": user.role,
        "jti": str(uuid4()),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=15)).timestamp()),
    }
    return jwt.encode(payload, settings.AEGIS_JWT_SECRET, algorithm="HS256")


def hash_refresh_token(raw: str) -> str:
    return sha256(raw.encode()).hexdigest()


def issue_refresh_token(db: Session, user: User, response: Response) -> None:
    raw = secrets.token_urlsafe(32)
    row = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token(raw),
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
    )
    db.add(row)
    db.commit()
    response.set_cookie(
        settings.AEGIS_REFRESH_COOKIE_NAME,
        raw,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite="lax",
        max_age=7 * 24 * 3600,
        path="/auth",
    )


@router.post("/login")
def login(payload: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)):
    started = time.perf_counter()
    user = db.query(User).filter(User.email == payload.email.lower(), User.is_active.is_(True)).first()
    password_ok = verify_password_timing_safe(payload.password, user.password_hash if user else None)

    # Keep response timing close for both valid and invalid emails.
    elapsed = time.perf_counter() - started
    if elapsed < 0.110:
        time.sleep(0.110 - elapsed)

    if not user or not password_ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    issue_refresh_token(db, user, response)
    return {"access_token": create_access_token(user), "token_type": "bearer"}


@router.post("/refresh")
def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    raw = request.cookies.get(settings.AEGIS_REFRESH_COOKIE_NAME)
    if not raw:
        raise HTTPException(status_code=401, detail="Missing refresh token")

    token_hash = hash_refresh_token(raw)
    row = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
    now = datetime.now(timezone.utc)

    if not row or row.revoked_at is not None or row.expires_at <= now:
        # Replay or invalid refresh token: fail closed.
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    row.revoked_at = now
    user = db.get(User, row.user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User unavailable")

    db.commit()
    issue_refresh_token(db, user, response)
    return {"access_token": create_access_token(user), "token_type": "bearer"}
```

##### Implementation snippet — Current User Dependency

`backend/dependencies.py`

```python
import jwt
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.db.models import User
from app.db.session import get_db
from app.settings import settings


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing access token")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, settings.AEGIS_JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid access token")
    user = db.get(User, payload.get("sub"))
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User inactive or not found")
    return user
```

##### Implementation snippet — AuthContext, ProtectedRoute, and LoginPage

`frontend/src/contexts/AuthContext.tsx`

```tsx
import { createContext, useContext, useEffect, useState } from "react";
import { api, setAccessToken } from "../services/api";

type User = { id: string; email: string; display_name: string };

type AuthContextValue = {
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);

  async function login(email: string, password: string) {
    const result = await api.login(email, password);
    setAccessToken(result.access_token);
    setUser(await api.me());
  }

  function logout() {
    setAccessToken(null);
    setUser(null);
  }

  useEffect(() => {
    api.me().then(setUser).catch(() => setUser(null));
  }, []);

  return <AuthContext.Provider value={{ user, login, logout }}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside AuthProvider");
  return value;
}
```

`frontend/src/routes/ProtectedRoute.tsx`

```tsx
import { Navigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

export function ProtectedRoute({ children }: { children: JSX.Element }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  return children;
}
```

`frontend/src/features/auth/LoginPage.tsx`

```tsx
export function LoginPage() {
  const { login } = useAuth();
  const [email, setEmail] = useState("priya@example.com");
  const [password, setPassword] = useState("demo123");
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    try {
      await login(email, password);
      window.location.href = "/dashboard";
    } catch (err: any) {
      setError(err.retryAfter ? `Try again in ${err.retryAfter}s` : err.message);
    }
  }

  return (
    <form onSubmit={onSubmit} className="mx-auto mt-24 max-w-md rounded-3xl bg-white/20 p-8 shadow-xl backdrop-blur">
      <h1 className="text-2xl font-bold text-blue-950">Aegis Health Login</h1>
      {error && <p className="mt-3 rounded-xl bg-red-100 p-3 text-red-700">{error}</p>}
      <input className="mt-6 w-full rounded-xl border p-3" value={email} onChange={(e) => setEmail(e.target.value)} />
      <input className="mt-3 w-full rounded-xl border p-3" type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
      <button className="mt-6 w-full rounded-xl bg-blue-600 p-3 font-bold text-white">Login</button>
    </form>
  );
}
```

#### 9.3.2 Health Persistence

Requirements:

- Add SQLite setup with WAL mode.
- Add SQLAlchemy models.
- Persist completed pipeline runs to `health_records`.
- Persist job metadata to `pipeline_jobs`.
- Persist vitals to `vital_snapshots`.
- Save records under authenticated `user_id`.
- Update result/record/dashboard endpoints to enforce ownership.

##### Implementation snippet — Pipeline Persistence and Checkpoint Hooks

`agents/pipeline_hooks.py`

```python
import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import HealthRecord
from schemas.state import AegisState


def write_checkpoint(job_id: str, state: AegisState, checkpoint_dir: str) -> None:
    target = Path(checkpoint_dir) / f"{job_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "user_id": getattr(state, "user_id", None),
        "session_id": state.session_id,
        "current_tool": state.current_tool,
        "tools_run": list(state.tools_run),
        "tools_failed": list(state.tools_failed),
        "step_durations_ms": dict(state.step_durations_ms),
        "execution_plan": state.execution_plan.model_dump(mode="json") if state.execution_plan else None,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def persist_health_record(db: Session, user_id: str, job_id: str, state: AegisState) -> HealthRecord:
    report = state.report.model_dump(mode="json") if state.report else {}
    result = {
        "severity_result": state.severity_result.model_dump(mode="json") if hasattr(state.severity_result, "model_dump") else None,
        "rule_validator_result": state.rule_validator_result.model_dump(mode="json") if state.rule_validator_result else None,
        "execution_plan": state.execution_plan.model_dump(mode="json") if state.execution_plan else None,
    }
    row = HealthRecord(
        user_id=user_id,
        job_id=job_id,
        severity=report.get("severity", "LOW"),
        confidence=float(report.get("confidence", 0.0)),
        validation_status=report.get("validation_status"),
        symptoms_text=state.submitted_symptoms_text or state.raw_symptoms_text,
        medications_json=json.dumps(state.medications_raw),
        xray_findings_json=json.dumps(state.xray_findings_raw),
        report_json=json.dumps(report),
        result_json=json.dumps(result),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
```

#### 9.3.3 Personalized Dashboard

Requirements:

- Use user history, not only current job.
- Show recent reports.
- Show last severity.
- Show validation warnings.
- Show current vs previous record comparison.
- Show vitals trend cards.
- Show baseline risk z-score where enough history exists.
- Show “insufficient history” when not enough data exists.

##### Implementation snippet — Dashboard, Vitals, and Baseline Z-Score

`backend/dashboard.py`

```python
from statistics import mean, stdev

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.models import HealthRecord, User, VitalSnapshot
from app.db.session import get_db
from backend.dependencies import get_current_user

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def z_score(current: float, history: list[float]) -> dict:
    if len(history) < 3:
        return {"status": "insufficient_history", "z": None}
    sigma = stdev(history)
    if sigma == 0:
        return {"status": "flat_history", "z": 0.0}
    return {"status": "ok", "z": (current - mean(history)) / sigma}


@router.get("")
def dashboard(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    records = (
        db.query(HealthRecord)
        .filter(HealthRecord.user_id == current_user.id)
        .order_by(HealthRecord.created_at.desc())
        .limit(10)
        .all()
    )
    vitals = (
        db.query(VitalSnapshot)
        .filter(VitalSnapshot.user_id == current_user.id)
        .order_by(VitalSnapshot.created_at.desc())
        .limit(30)
        .all()
    )
    return {
        "user": {"display_name": current_user.display_name},
        "recent_records": [{"id": r.id, "severity": r.severity, "created_at": r.created_at} for r in records],
        "latest_severity": records[0].severity if records else None,
        "vitals_count": len(vitals),
    }
```

`backend/vitals.py`

```python
@router.post("/vitals/checkin")
def checkin(payload: VitalIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    row = VitalSnapshot(user_id=user.id, **payload.model_dump())
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"id": row.id, "created_at": row.created_at}
```

#### 9.3.4 Queue Priority, Rate Limit, Adaptive Max, Poll Backoff

Backend:

- Add `priority` to job metadata.
- Priority `0` for critical terms.
- Priority `5` for normal jobs.
- Stable FIFO order inside each priority level.
- Add submit rate limit: 10 submissions/hour/user.
- Add 429 responses with `Retry-After`.
- Add adaptive queue max:

```python
queue_max = max(8, int(180 / avg_duration_s * 1.5))
```

Critical priority triggers:

- chest pain
- chest tightness
- chest pressure
- heart attack
- cardiac arrest
- troponin
- shortness of breath
- dyspnoea
- dyspnea
- breathlessness
- stroke
- seizure
- unconscious
- pneumothorax
- pulmonary edema
- cardiomegaly

Frontend polling:

| Job State | Poll Interval |
|---|---:|
| First 2 seconds after submit | 200 ms |
| Queued after first 2 seconds | 700 ms |
| Running | 500–700 ms |
| Completed/failed | Stop |

##### Implementation snippet — Queue Priority, Adaptive Max, and Rate Limit Hook

`backend/queue_priority.py`

```python
from collections import deque

CRITICAL_TERMS = {
    "chest pain", "chest tightness", "chest pressure", "heart attack", "cardiac arrest",
    "troponin", "shortness of breath", "dyspnoea", "dyspnea", "breathlessness",
    "stroke", "seizure", "unconscious", "pneumothorax", "pulmonary edema", "cardiomegaly",
}

HIGH_QUEUE: deque[str] = deque()
NORMAL_QUEUE: deque[str] = deque()
CONSECUTIVE_HIGH = 0
STARVATION_GUARD = 5


def compute_priority(symptoms: str | None, xray_findings: list[str]) -> int:
    haystack = " ".join([symptoms or "", *xray_findings]).lower()
    return 0 if any(term in haystack for term in CRITICAL_TERMS) else 5


def enqueue_job(job_id: str, priority: int) -> None:
    if priority == 0:
        HIGH_QUEUE.append(job_id)
    else:
        NORMAL_QUEUE.append(job_id)


def dequeue_next_job() -> str | None:
    global CONSECUTIVE_HIGH

    # Starvation guard: after 5 critical cases, allow one normal job through
    # if normal jobs are waiting. This keeps triage priority without freezing
    # low-acuity users forever during a busy demo.
    if CONSECUTIVE_HIGH >= STARVATION_GUARD and NORMAL_QUEUE:
        CONSECUTIVE_HIGH = 0
        return NORMAL_QUEUE.popleft()

    if HIGH_QUEUE:
        CONSECUTIVE_HIGH += 1
        return HIGH_QUEUE.popleft()

    if NORMAL_QUEUE:
        CONSECUTIVE_HIGH = 0
        return NORMAL_QUEUE.popleft()

    return None


def adaptive_queue_max(avg_duration_s: float | None) -> int:
    if not avg_duration_s or avg_duration_s <= 0:
        return 10
    return max(8, int(180 / avg_duration_s * 1.5))


def get_queue_position(job_id: str) -> int | None:
    if job_id in HIGH_QUEUE:
        return list(HIGH_QUEUE).index(job_id) + 1
    if job_id in NORMAL_QUEUE:
        return len(HIGH_QUEUE) + list(NORMAL_QUEUE).index(job_id) + 1
    return None
```

`backend/main.py` submit integration sketch:

```python
@app.post("/queue/submit")
@limiter.limit("10/hour")
async def submit(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    # existing form fields...
):
    priority = compute_priority(symptoms_text, xray_findings_list)
    state = AegisState(user_id=current_user.id, raw_symptoms_text=symptoms_text, ...)
    job = PipelineJob(session_id=state.session_id, priority=priority)
    result = await submit_job(job, state, user_id=current_user.id, priority=priority)
    return JSONResponse(content=result.model_dump(mode="json"))
```

#### 9.3.5 Model Registry and Ollama Speed Tuning

Requirements:

- Centralize model loading/prewarm logic.
- Prewarm Ollama.
- Prewarm RAG embedding path.
- Optionally prewarm X-ray model if weights are available.
- Keep only one heavy inference path active at a time.
- Tune Ollama parameters.

Recommended environment:

```text
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_KEEP_ALIVE=-1
AEGIS_UI_STEP_SECONDS=0
```

Recommended generation options:

```text
num_ctx=3072
num_predict=768
```

##### Implementation snippet — `backend/model_registry.py`

```python
from pathlib import Path

class ModelRegistry:
    def __init__(self, memory_floor_mb: int = 900) -> None:
        self.loaded: set[str] = set()
        self.memory_floor_mb = memory_floor_mb

    def memory_available_mb(self) -> int | None:
        try:
            data = Path("/proc/meminfo").read_text()
            values = dict(line.split(":", 1) for line in data.splitlines() if ":" in line)
            kb = int(values["MemAvailable"].strip().split()[0])
            return kb // 1024
        except Exception:
            return None

    def assert_memory_headroom(self, component: str) -> None:
        available = self.memory_available_mb()
        if available is not None and available < self.memory_floor_mb:
            raise RuntimeError(f"memory_pressure: {component} delayed, only {available} MB available")

    async def prewarm(self) -> None:
        self.assert_memory_headroom("prewarm")
        await self._ollama_generate("health check", num_predict=1)
        await self._rag_embed("prewarm")

    async def before_heavy_tool(self, component: str) -> None:
        # Call before OCR, X-ray, or any large model path.
        self.assert_memory_headroom(component)

    async def _ollama_generate(self, prompt: str, num_predict: int = 1) -> None:
        # call OLLAMA_BASE_URL/api/generate with NUM_PARALLEL=1
        ...

    async def _rag_embed(self, text: str) -> None:
        # run a tiny embedding call so the first user is not cold
        ...

model_registry = ModelRegistry()
```

#### 9.3.6 RuleValidator Safety Banner

Requirements:

- Show validation status clearly on Report page.
- Show it in Dashboard when viewing last report.
- Include it in PDF dossier.

Banner behavior:

| Status | UI Treatment |
|---|---|
| `agreement` | Green confirmation |
| `warning` | Amber review banner |
| `override` | Red safety banner |

Override wording:

> Deterministic safety rules require HIGH severity. The generated narrative expressed a lower level. The structured HIGH severity is authoritative and should be reviewed urgently by a clinician.

##### Implementation snippet — Report Safety, Plan, TTS, and Chat Components

`frontend/src/features/report/SafetyBanner.tsx`

```tsx
export function SafetyBanner({ status }: { status?: "agreement" | "warning" | "override" | null }) {
  if (!status) return null;
  const map = {
    agreement: ["bg-emerald-100 text-emerald-800", "Safety agreement", "Narrative and deterministic severity agree."],
    warning: ["bg-amber-100 text-amber-800", "Review warning", "Narrative severity should be reviewed."],
    override: ["bg-red-100 text-red-800", "Safety override", "Deterministic rules require HIGH severity. Structured severity is authoritative."],
  } as const;
  const [klass, title, body] = map[status];
  return <div className={`rounded-2xl p-4 font-semibold ${klass}`}><b>{title}:</b> {body}</div>;
}
```

`frontend/src/features/report/PlanTheater.tsx`

```tsx
export function PlanTheater({ summary }: { summary?: string | null }) {
  if (!summary) return null;
  const repaired = summary.includes("[REPAIRED]");
  const fallback = summary.includes("[FALLBACK]");
  return (
    <section className="glass-card p-5">
      <h3 className="text-lg font-bold text-aegis-navy">Agentic Execution Plan</h3>
      <p className="mt-2 text-sm text-aegis-text">{summary}</p>
      <div className="mt-4 flex gap-2">
        {repaired && <span className="rounded-full bg-amber-100 px-3 py-1 text-amber-800">REPAIRED</span>}
        {fallback && <span className="rounded-full bg-slate-100 px-3 py-1 text-slate-800">FALLBACK</span>}
      </div>
    </section>
  );
}
```

`frontend/src/features/report/ReportReadout.tsx`

```tsx
export function ReportReadout({ text }: { text: string }) {
  function speak() {
    const utterance = new SpeechSynthesisUtterance(text.slice(0, 1500));
    speechSynthesis.cancel();
    speechSynthesis.speak(utterance);
  }
  return (
    <div className="flex gap-2">
      <button className="aegis-button" onClick={speak}>Read aloud</button>
      <button className="rounded-2xl border px-5 py-3" onClick={() => speechSynthesis.pause()}>Pause</button>
      <button className="rounded-2xl border px-5 py-3" onClick={() => speechSynthesis.cancel()}>Stop</button>
    </div>
  );
}
```

#### 9.3.7 Agentic Plan Theater

Requirements:

- Convert `execution_plan_summary` into visible chips.
- Show mandatory tools.
- Show optional RAG decision.
- Show `[REPAIRED]` when PlanValidator forced RAG.
- Show `[FALLBACK]` when planner fallback was used.
- Show planner reasoning.
- Include same audit trail in PDF dossier.

#### 9.3.8 Grad-CAM X-ray Heatmap

Requirements:

- Generate Grad-CAM heatmap for top positive X-ray finding.
- Save heatmap artifact locally.
- Return heatmap reference in job result.
- Show heatmap on Report page.
- Include heatmap in PDF dossier when available.
- Treat failure as non-fatal.

##### Implementation snippet — Grad-CAM Heatmap Service

`vision/gradcam.py`

```python
from pathlib import Path


def generate_xray_heatmap(image_path: str, output_dir: str, finding: str) -> str | None:
    """Generate a Grad-CAM heatmap artifact for the top X-ray finding.

    Return the saved PNG path. Return None on non-fatal failure.
    """
    try:
        output = Path(output_dir) / (Path(image_path).stem + f"_{finding}_gradcam.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Hook torchxrayvision model gradients here.
        # Save overlay PNG to output.
        return str(output)
    except Exception:
        return None
```

#### 9.3.9 FHIR R4 Export

Requirements:

- Add FHIR JSON export endpoint.
- User must own the record.
- Export a local FHIR Bundle.

Minimum resources:

- `Patient`
- `Encounter`
- `Observation`
- `DiagnosticReport`
- `MedicationStatement`
- `DocumentReference`

##### Implementation snippet — Clinical Export and Copilot Endpoints

`backend/exports.py`

```python
@router.get("/export/fhir/{record_id}")
def fhir_export(record_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    record = db.query(HealthRecord).filter_by(id=record_id, user_id=user.id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": user.id, "name": [{"text": user.display_name}]}},
            {"resource": {"resourceType": "DiagnosticReport", "id": record.id, "status": "final", "conclusion": record.severity}},
        ],
    }


@router.get("/export/zip")
def export_zip(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Build an in-memory ZIP containing profile.json, health_records.json, vital_snapshots.json.
    ...


@router.delete("/account")
def delete_account(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.delete(user)
    db.commit()
    return {"deleted": True}
```

`backend/chat.py`

```python
@router.post("/queue/chat")
def queue_chat(payload: ChatIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.query(PipelineJobRow).filter_by(job_id=payload.job_id, user_id=user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": payload.job_id,
        "turn": payload.turn,
        "answer": "Based on the completed triage record, please review the highlighted safety findings with a clinician.",
        "severity_delta": "unchanged",
        "suggested_questions": [
            "When did the symptoms start?",
            "Are symptoms worsening?",
            "Do you have recent vital signs?",
        ],
    }
```

#### 9.3.10 Personal Baseline Risk Z-Score

Requirements:

- Compute baseline using current user history only.
- Compare current value with historical mean/std.
- Require at least 3 historical values.
- Show “insufficient history” when not enough data exists.

Formula:

```text
z = (current_value - user_baseline_mean) / user_baseline_std
```

##### Implementation snippet — baseline helper

```python
def z_score(current: float, history: list[float]) -> float | None:
    if len(history) < 3:
        return None
    mu = sum(history) / len(history)
    var = sum((x - mu) ** 2 for x in history) / (len(history) - 1)
    return 0.0 if var == 0 else (current - mu) / (var ** 0.5)
```

#### 9.3.11 Voice TTS Report Readout

Requirements:

- Add readout controls on Report page.
- Read Summary, Severity, and Recommendations.
- Add pause/resume/stop.
- Keep feature local/browser-side.

##### Implementation snippet — `frontend/src/features/report/ReportReadout.tsx`

```tsx
export function ReportReadout({ text }: { text: string }) {
  const speak = () => {
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(text.slice(0, 1500)));
  };
  return <button className="aegis-button" onClick={speak}>Read aloud</button>;
}
```

#### 9.3.12 Conversational Follow-Up `/queue/chat`

Backend:

- Add `/queue/chat` endpoint.
- Require authenticated user.
- Ensure user owns the job.
- Keep 3-turn memory per job.
- Use existing structured outputs as context.
- Do not rerun the full heavy pipeline.
- For new symptoms, rerun only lightweight symptom extraction/severity scoring when needed.
- Allow `/queue/rerun/{job_id}` for a recently completed owned job when the user adds clinically relevant follow-up information.

Response shape:

```json
{
  "job_id": "...",
  "turn": 1,
  "answer": "...",
  "severity_delta": "unchanged",
  "suggested_questions": ["...", "...", "..."]
}
```

Frontend:

- Add chat panel on Report page.
- Show turn limit.
- Show suggested questions.

##### Implementation snippet — `backend/chat.py`

```python
@router.post("/queue/chat")
def queue_chat(payload: ChatIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.query(PipelineJobRow).filter_by(job_id=payload.job_id, user_id=user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": payload.job_id, "answer": "Review the highlighted safety findings with a clinician.", "suggested_questions": ["Are symptoms worsening?"]}
```

#### 9.3.13 Daily Vitals Check-In

Requirements:

- Add vitals form to Dashboard.
- Save vitals to `vital_snapshots`.
- Show vitals trend cards.
- Feed baseline comparison where enough data exists.

Fields:

- systolic BP
- diastolic BP
- heart rate
- SpO2
- temperature
- glucose
- weight
- notes

##### Implementation snippet — Dashboard and Vitals Components

`frontend/src/features/dashboard/VitalsCheckIn.tsx`

```tsx
export function VitalsCheckIn() {
  const [heartRate, setHeartRate] = useState("");
  const [spo2, setSpo2] = useState("");

  async function submit() {
    await api.vitalsCheckin({ heart_rate: Number(heartRate), spo2: Number(spo2) });
  }

  return (
    <section className="glass-card p-5">
      <h3 className="text-lg font-bold text-aegis-navy">Daily Vitals Check-In</h3>
      <div className="mt-4 grid grid-cols-2 gap-3">
        <input className="rounded-xl border p-3" placeholder="Heart rate" value={heartRate} onChange={(e) => setHeartRate(e.target.value)} />
        <input className="rounded-xl border p-3" placeholder="SpO2" value={spo2} onChange={(e) => setSpo2(e.target.value)} />
      </div>
      <button className="aegis-button mt-4" onClick={submit}>Save vitals</button>
    </section>
  );
}
```

#### 9.3.14 PDF Clinical Dossier

Upgrade the existing PDF workflow into an offline-safe clinical dossier.

Required content:

- Patient header.
- Date/time.
- Job id.
- Severity.
- Confidence.
- Validation banner.
- Submitted inputs.
- Tool audit trail.
- Execution plan.
- Full report text.
- Citations.
- Grad-CAM heatmap if available.
- QR code or local record hash.
- Medical disclaimer.

Important rule:

- The deployed PDF path must not depend on a CDN.
- Either vendor the PDF library into the frontend bundle or generate the PDF in the backend.

##### Implementation snippet — `backend/pdf_export.py`

```python
@router.get("/export/pdf/{job_id}")
def export_pdf(job_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.query(PipelineJobRow).filter_by(job_id=job_id, user_id=user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    record = db.query(HealthRecord).filter_by(job_id=job_id, user_id=user.id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    html = render_dossier_html(record=record, user=user)
    pdf_bytes = render_pdf_offline(html)  # WeasyPrint/ReportLab or vendored renderer
    return Response(pdf_bytes, media_type="application/pdf")
```

`frontend/src/features/report/ExportButtons.tsx`

```tsx
export function ExportButtons({ jobId }: { jobId: string }) {
  return <a className="aegis-button" href={`/export/pdf/${jobId}`}>Download Clinical Dossier</a>;
}
```

#### 9.3.15 Result Cache

Requirements:

- Add in-memory LRU cache.
- Capacity: 128 entries.
- Key from normalized symptoms, medications, X-ray findings, and major lab signals.
- Do not store patient identity in cache values.
- Rehydrate current patient header on cache hit.
- Add cache hit/miss metrics.
- Add UI cache badge.

Key pattern:

```python
sha256(symptoms + medications + xray_findings + lab_signal).hexdigest()
```

##### Implementation snippet — Result Cache

`backend/result_cache.py`

```python
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    value: dict[str, Any]
    hits: int = 0


class ResultCache:
    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max_entries
        self._data: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def file_signature(self, paths: list[str] | None) -> str:
        if not paths:
            return ""
        digest = hashlib.sha256()
        for raw in sorted(paths):
            path = Path(raw)
            if not path.exists():
                continue
            digest.update(path.name.encode())
            with path.open("rb") as f:
                digest.update(f.read(65536))  # first 64 KB is enough for cache separation
        return digest.hexdigest()

    def make_key(
        self,
        *,
        symptoms: str,
        medications: list[str],
        xray_findings: list[str],
        lab_paths: list[str] | None = None,
        xray_paths: list[str] | None = None,
    ) -> str:
        payload = {
            "symptoms": " ".join(symptoms.lower().split()),
            "medications": sorted(m.lower().strip() for m in medications),
            "xray_findings": sorted(f.lower().strip() for f in xray_findings),
            "lab_sig": self.file_signature(lab_paths),
            "xray_sig": self.file_signature(xray_paths),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self._data.get(key)
        if not entry:
            self.misses += 1
            return None
        self._data.move_to_end(key)
        entry.hits += 1
        self.hits += 1
        return entry.value.copy()

    def set(self, key: str, value: dict[str, Any]) -> None:
        # Cache only clinical content. Never cache patient identity.
        blocked = {"patient", "user", "user_id", "patient_name", "patient_dob", "email"}
        sanitized = {k: v for k, v in value.items() if k not in blocked}
        self._data[key] = CacheEntry(value=sanitized)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)


result_cache = ResultCache(max_entries=128)
```

#### 9.3.16 Observability

Requirements:

- Add `/metrics` endpoint in Prometheus text format.
- Add `/readyz` readiness endpoint for deployment checks.
- Expand SystemIndicator into a system card.

Minimum metrics:

```text
aegis_queue_depth
aegis_queue_max
aegis_inference_active
aegis_avg_duration_seconds
aegis_jobs_completed_today
aegis_jobs_failed_today
aegis_cache_hits_total
aegis_cache_misses_total
aegis_model_loaded
aegis_rag_index_ready
aegis_memory_used_mb
aegis_memory_total_mb
aegis_jobs_in_flight
aegis_pipeline_cache_hits_total
aegis_pipeline_cache_misses_total
aegis_rule_validator_total{status}
aegis_auth_failures_total
```

##### Implementation snippet — Metrics Endpoint

`backend/metrics.py`

```python
from fastapi import APIRouter, Response

from backend.queue import get_average_pipeline_duration_s, get_queue_depth, get_queue_max, is_inference_active
from backend.result_cache import result_cache

router = APIRouter()


def line(name: str, value: int | float, labels: dict[str, str] | None = None) -> str:
    if labels:
        label_text = ",".join(f'{k}="{v}"' for k, v in labels.items())
        return f"{name}{{{label_text}}} {value}"
    return f"{name} {value}"


@router.get("/metrics")
def metrics() -> Response:
    avg = get_average_pipeline_duration_s() or 0
    lines = [
        line("aegis_queue_depth", get_queue_depth()),
        line("aegis_queue_max", get_queue_max()),
        line("aegis_inference_active", 1 if is_inference_active() else 0),
        line("aegis_avg_duration_seconds", avg),
        line("aegis_pipeline_cache_hits_total", result_cache.hits),
        line("aegis_pipeline_cache_misses_total", result_cache.misses),
        line("aegis_jobs_in_flight", 1 if is_inference_active() else 0),
        line("aegis_auth_failures_total", 0),
        line("aegis_rule_validator_total", 0, {"status": "agreement"}),
        line("aegis_rule_validator_total", 0, {"status": "warning"}),
        line("aegis_rule_validator_total", 0, {"status": "override"}),
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
```

#### 9.3.17 Data Export ZIP and Delete Account

Export ZIP contents:

- `profile.json`
- `health_records.json`
- `vital_snapshots.json`
- FHIR bundles
- PDF dossiers where generated
- audit metadata

Delete account:

- Authenticated user only.
- Prefer password confirmation.
- Delete only that user’s data.
- Delete refresh tokens.
- Delete uploads/checkpoints linked to that user.

##### Implementation snippet — `backend/exports.py`

```python
@router.get("/export/zip")
def export_zip(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Build ZIP with only this user's profile, records, vitals, FHIR, PDFs.
    ...

@router.delete("/account")
def delete_account(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.delete(user)
    db.commit()
    return {"deleted": True}
```

#### 9.3.18 Suggested Questions

Requirements:

- Add `suggested_questions[]` to result output.
- Generate from structured report context.
- Display suggested questions in Report page and chat panel.

Examples:

- “When did the chest pain start?”
- “Do you have shortness of breath at rest?”
- “What was your most recent oxygen saturation?”

##### Implementation snippet — `tools/suggested_questions.py`

```python
def build_suggested_questions(severity: str, symptoms: list[str]) -> list[str]:
    questions = ["When did the symptoms start?", "Are symptoms worsening?"]
    text = " ".join(symptoms).lower()
    if "chest pain" in text:
        questions.append("Do you have shortness of breath at rest?")
    if severity == "HIGH":
        questions.append("Do you have recent oxygen saturation or blood pressure readings?")
    return questions[:4]
```

#### 9.3.19 Security Headers, CSRF, and CORS Lock

Requirements:

- Add production CORS allowlist.
- Add CSRF protection for cookie-bearing mutation endpoints.
- Add security headers:
  - `X-Content-Type-Options: nosniff`
  - `Referrer-Policy: no-referrer`
  - `X-Frame-Options: DENY` or CSP equivalent
  - `Permissions-Policy`
  - `Content-Security-Policy`
- Ensure refresh token cookie uses secure settings in production.


Security posture note:

- Passwords are bcrypt-hashed.
- Access tokens are short-lived.
- Refresh tokens are httpOnly, rotated, and stored in the database only as hashes.
- User-owned data must be filtered by `user_id` at query level.
- The first completed build may keep SQLite clinical rows readable for debugging, but the schema includes encryption-ready fields and the service boundaries should avoid blocking later field-level encryption.
- Audit and consent rows should avoid PHI and use hashed IP/user-agent values.
- The privacy claim relies on local inference, authenticated access, user isolation, auditability, export/delete controls, and no external inference API calls.

##### Implementation snippet — Security Middleware

`backend/security.py`

```python
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), geolocation=(), payment=()"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; media-src 'self' blob:;"
        return response


def install_security(app: FastAPI) -> None:
    app.add_middleware(SecurityHeadersMiddleware)
```

#### 9.3.20 Demo Hardening

Requirements:

- Add checkpoint after every pipeline tool.
- Add recover endpoint.
- Add prewarm endpoint.
- Add demo reset endpoint.
- Add upload/checkpoint cleanup.

Checkpoint path:

```text
/tmp/aegis_checkpoint/{job_id}.json
```

Checkpoint contents:

- job id
- user id
- session id
- current tool
- completed tools
- failed tools
- execution plan
- step durations
- completed tool outputs
- upload paths

##### Implementation snippet — checkpoint/recover

```python
def write_checkpoint(job_id: str, state: AegisState) -> None:
    path = Path(settings.AEGIS_CHECKPOINT_DIR) / f"{job_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json())

@router.post("/queue/recover/{job_id}")
def recover(job_id: str, user: User = Depends(get_current_user)):
    # Verify ownership, load checkpoint, enqueue resume job.
    return {"job_id": job_id, "recovered": True}
```

#### 9.3.21 Medical Form Polish

Requirements:

- Improve client-side file validation.
- Show per-file size/type errors.
- Preserve draft on submit failure.
- Show clear queue/rate-limit messages.
- Keep voice recording state safe before submit.
- Ensure frontend fields exactly match backend contract.
- Improve retry UX.

##### Implementation snippet — Medical Form Validation Helpers

`frontend/src/features/medical-form/validation.ts`

```ts
export function validateFile(file: File, kind: "pdf" | "xray" | "audio"): string | null {
  const limits = { pdf: 25, xray: 25, audio: 15 };
  const maxMb = limits[kind];
  if (file.size > maxMb * 1024 * 1024) return `${file.name} exceeds ${maxMb} MB`;
  if (kind === "pdf" && file.type !== "application/pdf") return `${file.name} must be a PDF`;
  if (kind === "xray" && !file.type.startsWith("image/") && !file.name.endsWith(".dcm")) return `${file.name} must be image or DICOM`;
  return null;
}

export function nextPollInterval(status: string, submittedAtMs: number): number {
  const ageMs = Date.now() - submittedAtMs;
  if (status === "completed" || status === "failed") return 0;
  if (ageMs < 2000) return 200;
  if (status === "queued") return 700;
  return 600;
}
```

#### 9.3.22 Pitch Assets Preparation

Phase 1 prepares the content source for pitch assets:

- architecture diagram draft,
- metrics list,
- demo flow outline,
- screen list for recording,
- privacy and safety talking points.

Final exported pitch assets are produced in Phase 2 after deployment is validated.

---

##### Implementation snippet — `docs/demo_runbook.md`

```md
# Demo Runbook
1. Run prewarm.
2. Login as Priya and show historical dashboard.
3. Submit Judge case and show live queue/pipeline.
4. Open report: safety banner, plan theater, exports.
5. Show /metrics and reset demo before the next judge group.
```

### 9.4 Phase 1 Testing

Add or update tests for:

- Current codebase import integrity, especially `vision.xray_processor`.
- Auth register/login/refresh/logout/me.
- Timing-safe login behavior.
- Refresh-token rotation/replay rejection.
- Password hashing.
- User-scoped records.
- SQLite persistence.
- Queue priority.
- Queue starvation guard.
- Adaptive queue max.
- Rate limit 429 + `Retry-After`.
- Cache hit/miss.
- Metrics endpoint.
- Memory guard before heavy tools.
- Security headers.
- Demo reset.
- Checkpoint writing.
- RuleValidator banner data mapping.
- Plan theater data mapping.
- Grad-CAM artifact fallback.
- FHIR export ownership and resource shape.
- Baseline z-score calculation.
- Vitals check-in and trends.
- Chat endpoint ownership and turn limit.
- PDF dossier content.
- ZIP export.
- Delete account.
- Suggested questions.
- Frontend auth context.
- Protected routes.
- Vite proxy / `VITE_API_BASE_URL` coverage for new endpoints.
- Medical form error states.

Existing tests must remain green.

---

### 9.5 Phase 1 Acceptance Criteria

Phase 1 is complete when:

- All 22 enhancements are implemented in code.
- Existing backend, frontend, queue, pipeline, report, dashboard, and form code are updated in place.
- Users can log in locally.
- Demo users exist locally.
- User-scoped records work locally.
- Queue priority and rate limiting work locally.
- Dashboard uses user history locally.
- Report page shows safety banner and plan theater locally.
- Exports, chat, vitals, PDF, Grad-CAM, cache, and metrics work locally.
- Local test suite is green.
- The app is ready to be packaged and deployed in Phase 2.

---

## 10. Phase 2 — Deployment

### 10.1 Phase 2 Goal

Phase 2 packages and deploys the completed Phase 1 system.

This phase focuses on:

- production Docker setup,
- Jetson deployment,
- persistent volumes,
- static frontend serving,
- model prewarm,
- demo reset,
- smoke tests,
- load tests,
- offline PDF validation,
- metrics validation,
- final pitch/demo assets.

No new product enhancements are added in Phase 2. Phase 2 validates and operationalizes the completed Phase 1 development work.

---

### 10.2 Deployment Files to Finalize

| File/Area | Required Phase 2 Deployment Work |
|---|---|
| `docker/docker-compose.yml` | Final production compose with backend, Ollama, `data/` model asset volume, upload/checkpoint volumes, static frontend serving |
| `data/` volume | Mount or package Whisper, drug DB, Chroma, FAISS, MiniLM ONNX, tokenizer, and X-ray assets so demo has zero runtime downloads |
| `docker/Dockerfile` | Include production frontend build or static copy strategy |
| `docker/entrypoint.sh` | Final DB init, optional seed/reset flags, prewarm hooks |
| `config/Modelfile` | Tune or override current `num_ctx 4096`, `num_predict 1024`, `temperature 0.2` for the final demo profile |
| FastAPI StaticFiles mount | Serve `frontend/dist` locally in production |
| `.env.example` | Complete final env file with auth, DB, CORS, checkpoint, metrics options |
| `scripts/seed_demo_users.py` | Seed Priya, Arjun, Judge in deployed DB |
| `scripts/prewarm_models.py` | Final prewarm for Ollama/RAG/X-ray where available |
| `scripts/reset_demo.py` | Final safe reset for demo users and artifacts |
| `scripts/smoke_test_final.sh` | Full end-to-end smoke test |
| `docs/setup_jetson.md` | Final Jetson deployment instructions |
| `docs/demo_runbook.md` | Judge/demo operating procedure |

---

##### Implementation snippet — Deployment Compose and Entrypoint

`docker/docker-compose.yml`

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: aegis_ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_storage:/root/.ollama
    environment:
      OLLAMA_NUM_PARALLEL: "1"
      OLLAMA_MAX_LOADED_MODELS: "1"
      OLLAMA_KEEP_ALIVE: "-1"

  backend:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: aegis_backend
    depends_on:
      - ollama
    ports:
      - "8000:8000"
    volumes:
      - ../data:/app/data
      - ../uploads:/tmp/aegis_uploads
      - ../checkpoints:/tmp/aegis_checkpoint
    env_file:
      - ../.env

volumes:
  ollama_storage:
```

`docker/entrypoint.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/data /tmp/aegis_uploads /tmp/aegis_checkpoint
python -m scripts.init_db
python -m scripts.seed_demo_users --if-empty
python -m scripts.prewarm_models || true
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### 10.3 Required Deployment Environment

```text
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_KEEP_ALIVE=-1
AEGIS_DB_URL=sqlite:////app/data/aegis.db
AEGIS_WHISPER_DIR=/app/data/audio/whisper-tiny-en
AEGIS_XRAY_MODEL_DIR=/app/data/xray
AEGIS_KNOWLEDGE_DIR=/app/data/knowledge
AEGIS_DRUG_DB_PATH=/app/data/drugs/aegis_drugs.db
AEGIS_OCR=0
AEGIS_JWT_SECRET=change-me
AEGIS_REFRESH_COOKIE_NAME=aegis_refresh
AEGIS_CORS_ORIGINS=http://localhost:5173
AEGIS_UI_STEP_SECONDS=0
AEGIS_CHECKPOINT_DIR=/tmp/aegis_checkpoint
```

Production deployment should serve the frontend from the backend or from the same local Compose stack to avoid unnecessary CORS complexity.

---

### 10.4 Deployment Requirements

The final deployment must:

- Serve the production frontend locally.
- Run FastAPI with one worker.
- Run Ollama locally.
- Persist SQLite database under a mounted volume.
- Mount or package the existing `data/` directory containing Whisper, drug DB, Chroma, FAISS, MiniLM, and X-ray assets.
- Persist required upload/checkpoint artifacts.
- Support reset between demo groups.
- Support prewarm before live judging.
- Avoid external inference APIs.
- Avoid CDN dependency for critical demo features such as PDF export.
- Preserve user records across backend restart.
- Keep user data isolated.
- Keep Jetson memory inside safe headroom.

---

### 10.5 Deployment Smoke Test

The final deployment must pass:

1. `docker compose up` starts the system.
2. Frontend loads locally.
3. Login works.
4. Demo users exist.
5. `/health` works.
6. `/metrics` works.
7. `/readyz` works.
8. Queue submit works.
9. Report streams.
10. Result persists after backend restart.
11. Dashboard shows historical data.
12. Safety banner displays when applicable.
13. Plan theater displays.
14. Grad-CAM displays when available.
15. PDF export works offline.
16. FHIR export works.
17. ZIP export works.
18. Chat follow-up works.
19. Vitals check-in works.
20. Delete account works on a non-demo test user.
21. Demo reset works.
22. Prewarm works.
23. Recover endpoint handles a checkpointed job safely.
24. Required local data/model assets exist under `/app/data`.
25. Jetson memory remains within safe headroom.

---

##### Implementation snippet — Demo Reset, Prewarm, and Smoke Test Scripts

`scripts/reset_demo.py`

```python
from app.db.session import SessionLocal
from scripts.seed_demo_users import seed_demo_users


def main() -> None:
    db = SessionLocal()
    try:
        # Delete demo records/uploads/checkpoints, then reseed.
        seed_demo_users(db, reset=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()
```

`scripts/smoke_test_final.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-http://localhost:8000}

curl -fsS "$BASE/health" >/dev/null
curl -fsS "$BASE/metrics" >/dev/null
curl -fsS "$BASE/readyz" >/dev/null

# Local model/data assets must already be present; demo must not download them at runtime.
test -f /app/data/audio/whisper-tiny-en/model.bin
test -f /app/data/drugs/aegis_drugs.db
test -f /app/data/knowledge/minilm.onnx
test -f /app/data/knowledge/faiss.index
test -f /app/data/knowledge/faiss.docs
test -d /app/data/knowledge/chroma

TOKEN=$(curl -fsS -X POST "$BASE/auth/login" \
  -H 'Content-Type: application/json' \
  -d '{"email":"priya@example.com","password":"demo123"}' | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

curl -fsS "$BASE/dashboard" -H "Authorization: Bearer $TOKEN" >/dev/null

echo "Aegis smoke test passed"
```

### 10.6 Deployment Traffic Test

Minimum traffic validation:

- Multiple browser sessions logged in as different users.
- Priya and Arjun data remain isolated.
- Judge can submit a fresh case.
- Queue positions update correctly.
- Critical case receives priority.
- Rate-limited user receives 429 and `Retry-After`.
- Cache hit displays cache badge.
- Metrics reflect queue depth, completed jobs, and cache counters.

---


### 10.7 Final Demo Flow

The final deployed demo should follow this exact flow so the multi-user, traffic-control, safety, and deployment claims are all visible in under five minutes.

| Step | Demo Action | What It Proves |
|---:|---|---|
| 1 | Open deployed local app on Jetson | Product is locally deployed, not just running in dev mode |
| 2 | Show `/health` or SystemIndicator | Backend, model, queue, and memory status are live |
| 3 | Login as Priya | Auth works |
| 4 | Open Priya dashboard | User-specific persisted history works |
| 5 | Show Priya trends / previous records | SQLite persistence and dashboard comparisons work |
| 6 | Logout and login as Judge | Multi-user flow works |
| 7 | Submit a fresh medical form as Judge | End-to-end queue submit works |
| 8 | Show queue position and live pipeline | Traffic control and streaming progress work |
| 9 | Submit or show a critical Arjun-style case | Priority triage and safety handling are visible |
| 10 | Open completed report | Report generation and structured result work |
| 11 | Show safety banner and plan theater | RuleValidator and agentic auditability are visible |
| 12 | Show cache badge if repeating similar case | Result cache and throughput optimization are visible |
| 13 | Show Grad-CAM panel if X-ray exists | Imaging explainability is visible |
| 14 | Ask one follow-up chat question | Copilot path works without rerunning full pipeline |
| 15 | Save vitals and return to dashboard | Vitals persistence and dashboard update work |
| 16 | Export PDF/FHIR/ZIP | Data ownership and exports work |
| 17 | Show `/metrics` | Observability is real |
| 18 | Run demo reset | System can reset between judge groups |

---

### 10.8 Final Pitch and Demo Assets

Phase 2 produces final assets from the deployed system:

- 90-second demo video.
- Architecture poster.
- Metrics slide.
- Privacy slide.
- Safety slide.
- Multi-user traffic demo script.
- Jetson deployment proof screenshot or live checklist.

---

### 10.9 Phase 2 Acceptance Criteria

Phase 2 is complete when:

- The completed Phase 1 system is deployed locally on Jetson.
- Docker Compose starts the full system reliably.
- Frontend, backend, Ollama, SQLite, uploads, checkpoints, and metrics work together.
- Demo users are seeded.
- Prewarm and reset are operational.
- Smoke tests pass.
- Multi-user traffic test passes.
- PDF/FHIR/ZIP exports work in deployment.
- No critical demo feature depends on external inference APIs.
- The final demo can be run repeatedly without data leakage, cold-start embarrassment, or unrecoverable UI failure.

---

## 11. Testing Strategy for the Whole Project

Testing must cover both phases and preserve existing passing tests.

| Category | Coverage |
|---|---|
| Unit tests | schemas, validators, queue logic, cache, z-score, metrics |
| Backend API tests | auth, queue, result, dashboard, records, vitals, exports |
| Security tests | user scoping, CSRF, CORS, headers, token handling |
| Integration tests | full queue run, persistence, report result, dashboard update |
| Frontend tests | auth context, protected route, form validation, report states |
| Deployment tests | Docker startup, health, metrics, login, submit, stream, persist |
| Jetson smoke tests | memory, cold start, prewarm, one full demo run |

No test should depend on external inference APIs.

---

## 12. Performance Targets

| Metric | Target After Completion |
|---|---:|
| Waiting users | 30–40 smooth with adaptive queue/backpressure |
| p50 pipeline | 28–31 seconds where models are warm and cache is active |
| p95 pipeline | 52–58 seconds under demo load |
| Effective throughput | 75–85 jobs/hour with cache contribution |
| API workers | 1 |
| Inference workers | 1 |
| Peak memory | Approximately 5.1 GB or lower on Jetson |
| Headroom | Approximately 2.9 GB |
| Queue failure behavior | 429/Retry-After or graceful queue message, not white screen |
| Restart behavior | User records persist |
| Demo reset | One-click/admin endpoint reset |

---


## 13. Project Risk Table

| Risk | Why It Matters | Mitigation |
|---|---|---|
| Jetson memory pressure | OCR, X-ray, Ollama, and RAG can compete for memory | Keep one inference worker, prewarm carefully, time-slice models, run memory guard before heavy tools |
| Cold-start latency | First judge interaction can look slow | Run prewarm script before demo and expose model status in SystemIndicator |
| PDF export dependency | CDN-based PDF generation can fail offline | Vendor frontend PDF library or generate PDF in backend |
| Auth/user isolation bug | Multi-user demo fails if records leak | Enforce `WHERE user_id = current_user.id` in every query and test cross-user access |
| Queue overload | Busy booth can flood `/queue/submit` or `/queue/status` | Use rate limits, adaptive queue max, Retry-After, and frontend polling backoff |
| Normal-priority starvation | A long burst of critical cases can freeze normal users | Use the 5-critical-then-1-normal starvation guard |
| Grad-CAM latency or failure | Heatmap should not break core report | Treat Grad-CAM as non-fatal artifact generation |
| Browser TTS differences | Speech synthesis behavior varies by browser/OS | Keep TTS optional and do not depend on it for core clinical output |
| Local model unavailable | Ollama/model files may not be ready | Health check, `/readyz`, prewarm endpoint, deterministic fallback report, deployment smoke test |
| Required `data/` assets missing | Whisper, drug DB, RAG index, MiniLM, or X-ray assets missing would break local/offline demo | Mount/copy `data/` into `/app/data` and verify assets in smoke test |
| Demo reset deleting wrong data | Reset must not harm non-demo records | Scope reset to seeded demo users only and test with a non-demo account |
| External network assumptions | The privacy claim requires local operation | Avoid external inference APIs and remove CDN dependence from critical demo paths |
| Audit/consent overreach | A large compliance subsystem can distract from demo delivery | Keep audit_log and consent_ledger lightweight and append-only |

---

## 14. Safety and Limitations

- Aegis Health does not diagnose.
- Aegis Health does not replace a qualified clinician.
- It is not for emergencies.
- Deterministic severity is authoritative over generated narrative text.
- SLM planner reasoning is audit metadata and may be inaccurate.
- RAG evidence is limited to the local indexed corpus.
- X-ray outputs and heatmaps are screening/explainability aids, not radiology proof.
- User history comparisons are meaningful only when enough prior data exists.
- Missing inputs reduce confidence and must be shown honestly.
- Browser convenience features must not weaken the local privacy claim.
- All clinical outputs require professional review.

---

## 15. Final Definition of Done

Aegis Health is complete when both phases are finished:

- **Phase 1 Development** has implemented all 22 enhancements and updated existing backend, frontend, queue, pipeline, dashboard, report, form, and export flows.
- **Phase 2 Deployment** has packaged, deployed, tested, and validated the completed system on Jetson.
- Every user’s clinical data is isolated at backend query level.
- Jobs are prioritized, rate-limited, cached where safe, and observable.
- Reports persist and can be exported.
- The UI shows safety validation and agentic execution clearly.
- The system includes demo reset, prewarm, checkpoint, and recover behavior.
- The full app deploys locally on Jetson Orin Nano 8 GB.
- The demo can be run repeatedly without data leakage, cold-start embarrassment, or unrecoverable UI failure.

---

Aegis Health — built for privacy, designed for the edge.
