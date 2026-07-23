# Aegis Health — Comprehensive Pitch Assets & Technical Reference

This document compiles the foundational technical assets, architectural diagrams, observability metrics, video shot lists, and defensible clinical talking points required for pitching and evaluating **Aegis Health**.

---

## 1. Architecture Diagram Draft

Aegis Health is engineered as a **local-first, multimodal, multi-agent clinical triage copilot** running on edge hardware (Nvidia Jetson / Edge Server). The architecture enforces a strict separation between **probabilistic small language models (SLMs)** and **deterministic clinical safety guardrails**.

```text
====================================================================================================
                                      AEGIS HEALTH — EDGE ARCHITECTURE
====================================================================================================

  +----------------------------------------------------------------------------------------------+
  |                                 FRONTEND EXPERIENCE (SPA)                                    |
  |  React + Vite + Tailwind CSS + Glassmorphism UI (100% Local Browser Execution)               |
  |                                                                                              |
  |  [Auth / Login]  <--->  [Personalized Dashboard]  <--->  [Multi-Modal Medical Form]          |
  |  (JWT + Cookies)        (SQLite History / Vitals)        (Voice WAV + Lab PDFs + X-rays)     |
  +----------------------------------------------------------------------------------------------+
                                                  |
                                                  |  HTTP POST /queue/submit (multipart/form-data)
                                                  v
  +----------------------------------------------------------------------------------------------+
  |                              TRAFFIC CONTROL & ORCHESTRATION LAYER                           |
  |  FastAPI + Uvicorn (Single Worker Invariant for Edge Hardware Memory Headroom)               |
  |                                                                                              |
  |  +--------------------+    +-----------------------+    +---------------------------------+  |
  |  | Input Validation   | -> | Priority Triage Queue | -> | Adaptive Capacity & Rate Limits |  |
  |  | (Size/Format/JSON) |    | (5 Lanes: Routine->   |    | (Rolling Latency Max Sizing,    |  |
  |  | [TOOL_INPUT_VALID] |    |  Critical Chest/Trauma|    |  5 attempts / 60s per user)     |  |
  +--------------------+    +-----------------------+    +---------------------------------+  |
  +----------------------------------------------------------------------------------------------+
                                                  |
                                                  |  FIFO Pick (Under Async Inference Lock)
                                                  v
  +----------------------------------------------------------------------------------------------+
  |                           THE 11-TOOL MULTI-AGENT CLINICAL PIPELINE                          |
  |  (Sequential Execution inside AegisPipeline — Owned & Mutated via AegisState)                |
  |                                                                                              |
  |  STAGE 1: PLANNER & SAFETY AUDIT                                                             |
  |  +-------------------------------------+          +---------------------------------------+  |
  |  | Step -1: ExecutionPlanner (1B SLM)  | -------> | Step 0: PlanValidator (Deterministic) |  |
  |  | Decides optional RAG retrieval      |          | Audits plan against clinical rules;   |  |
  |  | [TOOL_EXECUTION_PLANNER]            |          | forces RAG if red-flag symptoms exist |  |
  |  +-------------------------------------+          +---------------------------------------+  |
  |                                                                   |                          |
  |  STAGE 2: MULTI-MODAL INPUT EXTRACTION                            v                          |
  |  +----------------------------------------------------------------------------------------+  |
  |  | Step 1: VoiceTranscriber   — CPU faster-whisper speech-to-text [TOOL_VOICE_TRANSCRIB]  |  |
  |  | Step 2: SymptomExtractor   — LLM unstructured-to-JSON parsing  [TOOL_SYMPTOM_EXTRACT]  |  |
  |  | Step 3: LabReportParser    — 3-Tier PDF waterfall (PyMuPDF/OCR)[TOOL_LAB_PARSER]       |  |
  |  | Step 4: XRayProcessor      — torchxrayvision DenseNet-121      [TOOL_XRAY_PROCESSOR]   |  |
  |  |                              + Grad-CAM Saliency Heatmap PNG Generator                 |  |
  |  +----------------------------------------------------------------------------------------+  |
  |                                                                   |                          |
  |  STAGE 3: CLINICAL EVIDENCE & PHARMACOLOGY                        v                          |
  |  +----------------------------------------------------------------------------------------+  |
  |  | Step 5: MedicalRAGSearch   — ChromaDB / FAISS + MiniLM ONNX    [TOOL_MEDICAL_RAG]      |  |
  |  | Step 6: DrugChecker        — SQLite pharmacology database      [TOOL_DRUG_CHECKER]     |  |
  |  |                              (Flags contraindicated drug-drug combinations)            |  |
  |  +----------------------------------------------------------------------------------------+  |
  |                                                                   |                          |
  |  STAGE 4: TRIAGE SYNTHESIS & SAFETY VALIDATION                    v                          |
  |  +----------------------------------------------------------------------------------------+  |
  |  | Step 7: SeverityScorer     — Hybrid rule-based + AI evaluation [TOOL_SEVERITY_SCORER]  |  |
  |  |                              Assigns tier: LOW | MODERATE | HIGH | CRITICAL            |  |
  |  | Step 8: ReportGenerator    — Llama 3.2 streaming synthesis     [TOOL_REPORT_GENERATOR] |  |
  |  |                              Yields 6-section structured Markdown report               |  |
  |  | Step 9: RuleValidator      — Post-generation clinical auditor  [TOOL_RULE_VALIDATOR]   |  |
  |  |                              Enforces agreement, warning, or safety override banners   |  |
  |  +----------------------------------------------------------------------------------------+  |
  +----------------------------------------------------------------------------------------------+
                                                  |
                     +----------------------------+----------------------------+
                     | (Streamed via text/plain)                               | (Persisted on Complete)
                     v                                                         v
  +---------------------------------------+                  +-----------------------------------+
  |       LIVE PIPELINE THEATER (UI)      |                  |      LONGITUDINAL HEALTH VAULT    |
  |  - Live Animated Step Tracker         |                  |  SQLite (WAL Mode + Foreign Keys) |
  |  - Agentic Plan Theater Chips         |                  |  SQLAlchemy ORM Schemas           |
  |  - [REPAIRED] / [FALLBACK] Badges     |                  |                                   |
  |  - Grad-CAM Heatmap Image Card        |                  |  - users (Accounts & Roles)       |
  |  - RuleValidator Safety Banner        |                  |  - health_records (Triage Runs)   |
  |  - Key Insights & Suggested Questions |                  |  - vital_snapshots (Daily Vitals) |
  +---------------------------------------+                  |  - pipeline_jobs (Job Metadata)   |
                     |                                       |  - audit_log (Security Trail)     |
                     | (Offline Rendered)                    +-----------------------------------+
                     v
  +---------------------------------------+
  |      CLINICAL DOSSIER EXPORTS         |
  |  - PDF Dossier (WeasyPrint Server-Side|
  |    with SHA-256 Hash & Base64 Images) |
  |  - FHIR R4 JSON Bundle Export         |
  |  - Data Vault ZIP Archive Export      |
  +---------------------------------------+
====================================================================================================
```

---

## 2. Observability & Prometheus Metrics List (`/metrics`)

In deployment, Aegis Health exposes a dedicated Prometheus text-format endpoint (`GET /metrics`) and a JSON system health summary (`GET /health`). These metrics provide continuous visibility into edge hardware headroom, traffic congestion, cache efficiency, and clinical safety agreements.

### Core Metrics Table

| Metric Name | Type | Labels / Dimensions | Description & Clinical Value |
| :--- | :--- | :--- | :--- |
| `aegis_queue_depth` | Gauge | `None` | Current number of jobs waiting in the priority queue. Useful for detecting traffic congestion on the edge node. |
| `aegis_queue_max_size` | Gauge | `None` | Dynamic adaptive queue capacity ($5 \le \text{max} \le 20$), throttled automatically based on rolling inference duration. |
| `aegis_queue_avg_wait_seconds` | Gauge | `None` | Rolling average wait time per job, calculated over the last 10 completions. Used to feed UI wait estimates. |
| `aegis_jobs_in_flight` | Gauge | `None` | `1` when the single inference worker is actively executing a pipeline run; `0` when idle. |
| `aegis_jobs_completed_today` | Counter | `None` | Cumulative number of successfully processed triage assessments since process startup. |
| `aegis_jobs_failed_today` | Counter | `None` | Cumulative number of jobs terminating in `FatalPipelineError` or wall-clock timeouts (`180s`). |
| `aegis_pipeline_step_duration_seconds` | Summary | `tool="VoiceTranscriber|..."` | Wall-clock execution time per tool. Essential for profiling edge hardware bottlenecks (e.g., Whisper vs. DenseNet vs. Ollama). |
| `aegis_pipeline_cache_hits_total` | Counter | `None` | Number of times a repeated/similar encounter hit the 128-entry LRU result cache, bypassing heavy SLM re-inference. |
| `aegis_pipeline_cache_misses_total` | Counter | `None` | Number of cache misses requiring full pipeline evaluation. |
| `aegis_rule_validator_total` | Counter | `status="agreement|warning|override"` | **Clinical Safety Metric:** Tracks how often the LLM narrative agreed with (`agreement`), slightly diverged from (`warning`), or was overridden by (`override`) deterministic clinical rules. |
| `aegis_auth_failures_total` | Counter | `None` | Total failed login attempts or invalid JWT token presentations. |
| `aegis_system_ram_free_mb` | Gauge | `None` | Available system RAM in megabytes. Gated against `OLLAMA_MEMORY_FLOOR_MB` (`900MB`) to prevent out-of-memory kernel panics. |
| `aegis_system_cpu_percent` | Gauge | `None` | Current CPU utilization percentage across edge cores. |

---

## 3. Video Recording Shot List & Screen Outline

When producing video demonstrations or pitch b-roll, capture the following 7 standardized screens to tell a coherent, professional product story:

### Shot 1: The Glassmorphic Login & User Isolation
* **Visual:** The login screen (`/login`). Enter credentials for `priya@aegis.health`.
* **Focus:** Clean aesthetic, smooth transition into the protected application.
* **Narrative:** Highlight that every session is authenticated via JWT with httpOnly refresh cookies, guaranteeing multi-user privacy on shared edge hardware.

### Shot 2: The Longitudinal Personalized Dashboard
* **Visual:** The user dashboard (`/dashboard`).
* **Focus:** Show the **Overall Health Score gauge**, the **Critical Clinical Signals cards**, and historical vitals check-ins.
* **Narrative:** Show how SQLite persistence turns AI triage into a long-term health companion, tracking 90-day trends and comparing current readings against personal baselines.

### Shot 3: Multi-Modal Input Capture (Medical Form)
* **Visual:** The medical form (`/medical-form`).
* **Focus:** 
  1. Type or speak symptoms into the microphone recorder.
  2. Enter medications (`Warfarin, Lisinopril`).
  3. Drag and drop a Lab Report PDF and a chest X-ray image into the multi-file dropzone.
* **Narrative:** Demonstrate true multi-modal edge capture—handling speech, text, structured blood work, and radiography in a single encounter.

### Shot 4: Live Priority Queueing & Pipeline Tracker
* **Visual:** Click **Submit** and show the Report page (`/report`) during streaming.
* **Focus:** Show the status badge (`RUNNING`), queue position indicator, and the left-hand **Live Pipeline Tracker** pulsing as nodes complete sequentially.
* **Narrative:** Explain that our single-node architecture uses priority triage lanes so critical chest pain jumps ahead of routine cases, while streaming plain-text tokens live without white-screen freezes.

### Shot 5: Explainability Deep Dive (Plan Theater & Grad-CAM)
* **Visual:** Scroll to the completed **Agentic Plan Theater** card and **Grad-CAM Heatmap** card.
* **Focus:**
  1. Zoom in on the mandatory tool checkmarks (`✓`) and the orange **`[REPAIRED] Forced RAG`** badge.
  2. Zoom in on the chest X-ray image with the glowing red/orange Grad-CAM saliency overlay over the lung opacity.
* **Narrative:** Prove that our AI is not a black box. Clinicians can verify why literature was retrieved and visually confirm that the neural network identified genuine lung consolidation rather than image artifacts.

### Shot 6: Clinical Safety Guardrails (RuleValidator Banner)
* **Visual:** The top of the Report output area, showing the **RuleValidator Safety Banner** (Green Agreement or Yellow Warning / Red Override).
* **Focus:** The clear distinction between the structured triage severity badge (`HIGH` / `MODERATE`) and the narrative recommendations.
* **Narrative:** Explain our core thesis: deterministic pharmacology databases and hardcoded severity rules supervise the LLM, guaranteeing that safety rules always override generative hallucination.

### Shot 7: Offline Dossier Download & Observability
* **Visual:** Click **"Download PDF"** to open the generated dossier, then show a terminal query to `/metrics`.
* **Focus:** The clean WeasyPrint PDF layout (with embedded base64 Grad-CAM image and SHA-256 hash) alongside live Prometheus metrics.
* **Narrative:** Conclude with data ownership and enterprise reliability—generating offline-safe clinical dossiers while monitoring edge system headroom in real time.

---

## 4. Privacy, Safety, and Clinical Talking Points

When answering questions from judges, physicians, or thesis examiners, rely on these four defensible architectural pillars:

### 1. "100% On-Device Edge Intelligence — Zero Data Leakage"
> *"Unlike commercial AI wrappers that transmit sensitive Protected Health Information (PHI) to cloud APIs like OpenAI, Anthropic, or Google, Aegis Health is engineered from the silicon up to run entirely on local edge hardware (like an Nvidia Jetson Orin or local hospital server). All speech transcription (`faster-whisper`), computer vision (`torchxrayvision`), vector retrieval (`ChromaDB`), and language synthesis (`Llama 3.2`) execute inside local memory. Patient data never touches the internet, ensuring out-of-the-box HIPAA and GDPR privacy compliance without relying on third-party cloud trust."*

### 2. "Deterministic Safety Guardrails Override LLM Hallucination"
> *"A fundamental limitation of using Small Language Models in healthcare is their susceptibility to hallucination and omission. Our thesis solves this by establishing a strict **Planner Authority Invariant**: an LLM is never permitted to make unsupervised clinical decisions. In Aegis Health:
> * Mandatory input tools (`LabReportParser`, `XRayProcessor`, `DrugInteractionChecker`) execute deterministically based on input presence—they cannot be skipped by an LLM.
> * If a patient reports red-flag cardiac symptoms, our deterministic `PlanValidator` automatically intercepts and repairs the LLM's execution plan (`[REPAIRED]`) to mandate medical literature retrieval.
> * Finally, our post-generation `RuleValidator` independently calculates triage severity against hardcoded clinical guidelines. If the LLM narrative downplays a severe drug interaction or lab abnormality, deterministic rules trigger an authoritative **Safety Override banner**, ensuring clinical safety always trumps generative text."*

### 3. "Auditable Explainability — Opening the Black Box"
> *"Clinicians rightfully reject black-box AI recommendations. Aegis Health bridges the trust gap through dual-layer explainability:
> * **Algorithmic Transparency:** Our **Agentic Plan Theater** exposes the exact execution trail, showing physicians which tools ran, whether fallback algorithms were invoked, and why the planner retrieved specific medical evidence.
> * **Visual Anatomical Attribution:** When our computer vision model detects chest pathology like Pneumonia or Pleural Effusion, our **Grad-CAM Heatmap** generates an anatomical saliency overlay. Doctors do not have to guess what the AI saw—they can visually verify that the neural network keyed in on real lung opacities rather than background imaging noise."*

### 4. "Longitudinal Care Vault vs. Single-Use Chatbots"
> *"Standard AI triage tools treat every patient encounter as an amnesiac, single-use chat session. Aegis Health upgrades triage into a **longitudinal clinical companion**. By combining SQLite Write-Ahead Logging with SQLAlchemy ORM schemas, every triage assessment, biomarker abnormality, and daily vital sign check-in is permanently stored under an authenticated user vault. This enables our system to compute 90-day rolling averages, track worsening or improving severity trends, and calculate personal-baseline Z-scores—alerting patients when a vital sign deviates from their personal normal even if it falls within broad population averages."*
