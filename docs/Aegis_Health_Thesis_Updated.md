# Aegis Health
### Built for Privacy, Designed for the Edge

## Updated Thesis Statement

Aegis Health is a privacy-first, edge-deployed clinical triage assistant that runs locally on an NVIDIA Jetson Orin Nano. It combines a local Small Language Model, deterministic clinical tools, multimodal input processing, rule-based severity scoring, and a multi-user longitudinal health vault to produce structured, evidence-backed triage reports without sending patient data to external inference APIs.

The system does not diagnose. It triages. Its purpose is to give clinicians and users a structured, auditable, evidence-backed starting point while keeping patient data private, local, and under user or clinical control.

---

## Problem Statement

AI-powered healthcare tools almost universally depend on cloud APIs. Patient data is transmitted to and processed on remote infrastructure outside clinical control. This creates real privacy risks, introduces internet dependency, and makes deployment impractical in low-connectivity or resource-constrained settings.

Existing healthcare AI tools also tend to operate on isolated input types. A symptom checker may process text symptoms, a separate parser may process lab reports, a separate imaging model may classify X-rays, and a medication lookup tool may check drug interactions. These systems rarely combine symptoms, labs, chest X-rays, medications, vitals, and retrieved evidence into one coherent local triage workflow.

A second limitation is lack of continuity. Most triage tools treat every encounter as a one-off event. They do not know whether the user has had repeated similar complaints, whether a lab value is changing over time, whether a previous high-risk rule fired, or whether the current result is unusual for that user’s own baseline. This makes the tool feel like a single-use AI form instead of a longitudinal clinical companion.

A third limitation is poor multi-user readiness. A real clinic, classroom demonstration, or hackathon booth requires multiple users to submit cases at the same time. Without authenticated accounts, user-level data isolation, queue visibility, rate limits, urgent-case priority handling, and safe backpressure, a local triage assistant can easily leak data or fail under load.

Aegis Health addresses these problems by running inference locally on edge hardware, processing multiple clinical modalities together, enforcing deterministic safety rules, storing user-owned longitudinal health records, and controlling multi-user traffic through a single-inference-worker queue. Patient data remains local unless the user explicitly exports or shares it.

---

## Challenges and Considerations

### 1. Local inference on constrained hardware

Running a multimodal AI system on a Jetson Orin Nano introduces strict memory and latency constraints. Language model inference, audio transcription, document parsing, X-ray processing, embedding search, and API serving must coexist within a limited memory envelope. The architecture therefore uses a single inference worker, one active heavy model path at a time, model prewarming, memory guards before heavy tools, and deployment smoke tests that verify local model assets are present before demo execution.

### 2. Reliable reasoning with a Small Language Model

Aegis Health uses a local Small Language Model through Ollama. The SLM is valuable for structured symptom extraction, optional evidence-planning, and final report synthesis, but it must not become an uncontrolled clinical authority. The system therefore uses a planner-executor architecture: the SLM may decide optional enrichment such as whether to retrieve evidence, while mandatory clinical processing is controlled by deterministic pipeline rules.

### 3. Planner Authority Invariant

A central safety property is that submitted clinical evidence must not be skipped by the planner. If the user submits audio, symptom text, lab reports, X-ray images, medication lists, or clinician X-ray findings, the corresponding processing path must execute based on input presence. The planner controls only optional enrichment. This makes it structurally impossible for the SLM to suppress mandatory clinical tools.

### 4. Deterministic clinical severity

Clinical triage severity must be explainable and reproducible. Aegis Health uses a deterministic `SeverityScorer` with documented rule constants, priority ordering, triggered rules, highest-priority rule, reasons, confidence, and contributing tools. The language model may synthesize the narrative, but deterministic severity remains authoritative.

### 5. Safety validation of generated reports

Generated text can be incomplete or inconsistent. Aegis Health therefore includes a `RuleValidator` that compares deterministic severity against the severity expressed in the generated report. It produces three outcomes: agreement, warning, or override. Override is used when deterministic rules require HIGH severity but the narrative expresses a lower level. In that case, the structured deterministic severity remains authoritative and the UI must show a prominent safety banner.

### 6. Multi-user isolation

The final system requires authenticated users, JWT access tokens, httpOnly refresh cookies, and SQL-level user isolation. Every persisted clinical object must be linked to `user_id`, including health records, vitals, jobs, exports, chat history, and account deletion. User isolation must be enforced in backend queries, not by hiding data in the frontend.

### 7. Traffic control on a single node

The system must serve multiple users while running only one inference pipeline at a time. To achieve this, Aegis Health uses a queue with priority handling for urgent clinical cases, adaptive queue capacity, rate limits, retry-after responses, frontend polling backoff, and a starvation guard so lower-priority users are not blocked indefinitely.

### 8. Longitudinal health vault

A core improvement is persistence. Completed pipeline results are stored as user-owned health records, and vitals are stored as time-series snapshots. This enables personalized dashboards, recent-report history, current-vs-last comparisons, 90-day trends, and personal-baseline z-scores where enough historical data exists.

### 9. Offline deployment and reproducible assets

The current codebase already includes local model/data assets under the `data/` directory, including Faster-Whisper files, drug database files, Chroma/FAISS knowledge indexes, MiniLM ONNX, tokenizer assets, and X-ray model assets. Deployment must preserve these assets and must not rely on runtime downloads during the demo.

### 10. Frontend modernization without unnecessary scope expansion

The current frontend is React + Vite with JavaScript/JSX and custom CSS. The final target stack is FastAPI + React + Vite + TypeScript + Tailwind. `shadcn/ui` is explicitly removed from scope. New and modified frontend components should be standardized to TypeScript + Tailwind where practical, while preserving the existing glassmorphism visual design.

---

## Proposed Solution

Aegis Health is implemented as a local, multimodal, queue-driven clinical triage system using FastAPI on the backend and React + Vite on the frontend.

### Backend and pipeline

The backend exposes authenticated APIs for queue submission, queue status, report streaming, final structured results, dashboard data, vitals, exports, metrics, demo reset, model prewarm, and readiness checks. The pipeline is sequential and memory-safe, with one inference worker running at a time.

The pipeline processes:

- typed symptoms or voice symptoms,
- lab report PDFs,
- chest X-ray images and clinician findings,
- medication lists,
- local medical evidence retrieval,
- deterministic severity scoring,
- report generation,
- rule validation.

Each submitted modality triggers its corresponding mandatory tool. Optional RAG enrichment is planned by the SLM but safety-validated by deterministic rules.

### User accounts and health vault

The system adds authenticated users with bcrypt password hashing, short-lived access tokens, rotating refresh tokens, and user-scoped database records. Completed reports are persisted into `health_records`, while vitals are stored in `vital_snapshots`. This creates a personalized longitudinal health vault that powers dashboard history, current-vs-last comparison, 90-day averages, and personal baseline scoring.

### Multi-user traffic control

To support multiple users on a single edge node, Aegis Health keeps inference sequential but makes the queue intelligent. Urgent clinical terms such as chest pain, troponin, shortness of breath, unconsciousness, pneumothorax, pulmonary edema, and cardiomegaly receive priority. The queue uses adaptive capacity, rate limits, retry-after responses, and frontend polling backoff. A starvation guard prevents normal-priority cases from waiting forever during bursts of urgent submissions.

### Clinical credibility and safety

The system surfaces deterministic severity, triggered rules, plan summary, validation status, warnings, and overrides. The report page shows a live pipeline theater, safety banners, cache state, and export options. Grad-CAM heatmap support is added for X-ray explainability as a non-fatal artifact. FHIR export and PDF Clinical Dossier export make the output usable outside the web UI.

### Frontend experience

The frontend is upgraded into an authenticated SPA with:

- login and protected routes,
- personalized dashboard,
- medical form with voice and multi-file upload,
- live report streaming,
- pipeline theater,
- safety validation banner,
- chat follow-up,
- vitals check-in,
- PDF/FHIR/ZIP export,
- metrics/system status display.

The UI remains local-first and privacy-first. The access token is held in memory, the refresh token is stored in an httpOnly cookie, and all clinical API calls are authenticated.

### Deployment

Deployment is performed in a second phase after development is complete. Docker Compose runs the backend and Ollama, mounts persistent volumes for SQLite, uploads, checkpoints, and local model/data assets, serves the production frontend locally, initializes the database, seeds demo users, prewarms models, and runs smoke tests. The deployment must pass health, metrics, readiness, login, queue, streaming, persistence, export, reset, and memory-headroom checks.

---

## Expected Outcome

Aegis Health demonstrates that a local Small Language Model can participate in a safe, multimodal clinical triage workflow without replacing deterministic clinical logic. The SLM helps extract structured information and synthesize readable reports, while deterministic tools handle mandatory clinical processing, severity scoring, validation, user isolation, persistence, and traffic control.

The completed system is expected to provide:

- private local inference on Jetson,
- multimodal clinical input processing,
- authenticated multi-user access,
- SQL-level user data isolation,
- persistent health records and vitals,
- personalized dashboard trends,
- priority queueing and rate limits,
- result caching for repeated similar cases,
- rule-based deterministic severity,
- safety validation banners,
- Grad-CAM X-ray explainability,
- FHIR and PDF exports,
- data export and account deletion,
- deployment smoke tests and demo reset.

The final result is not a diagnosis engine. It is a privacy-preserving triage assistant that gives clinicians and users a structured, auditable, evidence-backed starting point while keeping patient data local and under control.

---

## Final Thesis Summary

Aegis Health shows that practical healthcare AI does not need to depend on cloud APIs or large remote models. A carefully designed edge system can combine Small Language Models, deterministic clinical tools, local RAG, document AI, speech processing, X-ray analysis, queue management, and longitudinal health records into a coherent, privacy-first product.

The key contribution is the separation of responsibilities: deterministic tools own clinical computation and safety, while the SLM assists with extraction, optional planning, and narrative synthesis. The result is a transparent, auditable, local-first clinical assistant that supports multi-user operation, personalized health history, and edge deployment without sacrificing privacy or safety.
