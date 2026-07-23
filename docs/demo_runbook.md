# Aegis Health — Demo Runbook & Execution Guide

This document is the authoritative step-by-step execution guide for presenting Aegis Health before thesis evaluators, hackathon judges, or clinical review committees.

---

## 1. Pre-Demo Checklist & Prewarm (Minute -5:00)

Before the presentation begins, verify that the edge environment (Nvidia Jetson / Local Server) is healthy, models are loaded into RAM, and the database is initialized.

1. **Verify Edge Environment & Offline Operation:**
   ```bash
   # Ensure local backend is responsive
   curl -s http://localhost:8000/health | jq .
   ```
   *Confirm `status: "ok"`, `backend_online: true`, and check free RAM (`memory_free_mb`).*

2. **Trigger Model Prewarm:**
   Prewarm Ollama (`llama3.2`), ONNX MiniLM embeddings, and `torchxrayvision` DenseNet weights so the first judge interaction has zero cold-start latency:
   ```bash
   curl -X POST http://localhost:8000/admin/prewarm -H "Authorization: Bearer <admin_token>"
   ```

3. **Verify Seeded Demo Accounts:**
   Confirm that the standard clinical demo accounts exist and are accessible:
   * **Routine Patient:** `priya@aegis.health` (password: `demo123`)
   * **Urgent Trauma Case:** `arjun@aegis.health` (password: `demo123`)
   * **Clinical Evaluator:** `judge@aegis.health` (password: `demo123`)

---

## 2. Step-by-Step Live Demo Flow (7-Minute Pitch)

### Act I: Longitudinal Health Vault & User Isolation (Minute 0:00 – 1:30)
* **Action:** Open the browser to `http://localhost:5173` and log in as `priya@aegis.health`.
* **Visual:** Show the clean glassmorphic **Personalized Dashboard**.
* **Talking Script:**
  > *"Most medical AI tools today are one-off chatbots—they treat every visit as an isolated event with no memory of your medical history. Aegis Health is engineered as a **longitudinal edge health vault**. Here on Priya's dashboard, notice that her previous triage records, severity scores, and vital sign trends are permanently persisted in a local SQLite database with Write-Ahead Logging. Every query is scoped to her authenticated JWT session—enforcing SQL-level patient data isolation without any cloud servers."*

### Act II: Multi-Modal Triage & Intelligent Priority Queueing (Minute 1:30 – 3:00)
* **Action:** Click **"Health Scan"** → **"Fill the Form"** to open the Multi-Modal Medical Form.
* **Input Demonstration:**
  1. **Symptoms:** Type or use microphone voice recording: *"Severe chest pain radiating to left arm, shortness of breath, and dizziness since this morning."*
  2. **Medications:** Enter: *"Warfarin, Lisinopril, Aspirin"*.
  3. **Attachments:** Upload a sample Lab Report PDF and a chest X-ray image (e.g., Pneumonia/Pleural Effusion DICOM or PNG).
* **Action:** Click **"Submit"**. Point out the **Live Pipeline Tracker**.
* **Talking Script:**
  > *"When we submit this case, notice what happens in the backend. We are running on a single edge node with limited RAM, so our backend uses an intelligent **priority triage queue**. Because our natural language engine detected red-flag terms—chest pain and shortness of breath—this encounter is automatically elevated to **Priority 5 (Critical)**. If a routine checkup was waiting in line, this urgent case immediately jumps ahead, protected by rate limits and anti-starvation guards."*

### Act III: The 11-Tool Pipeline & Agentic Plan Theater (Minute 3:00 – 4:30)
* **Action:** As the report streams live into the UI, draw attention to the left-hand **Live Pipeline Tracker** lighting up across all 5 stages, and then examine the completed **Agentic Plan Theater** card.
* **Talking Script:**
  > *"Under the hood, an **11-tool multi-agent pipeline** orchestrates this analysis. But notice: we never let an LLM make unsupervised decisions. Look at our **Agentic Plan Theater**:
  > 1. **Mandatory Invariants:** You can see green checkmarks (`✓`) on `VoiceTranscriber`, `SymptomExtractor`, `LabReportParser`, `XRayProcessor`, and `DrugInteractionChecker`. These tools are triggered deterministically by patient inputs—not by LLM whim.
  > 2. **Forced Safety Repair:** Notice the orange badge: **`[REPAIRED] Forced RAG`**. Our edge 1B planner initially evaluated this encounter, but our deterministic `PlanValidator` detected critical cardiac symptoms and automatically overridden the planner to mandate clinical literature retrieval from our local vector database."*

### Act IV: Explainability & Safety Guardrails (Minute 4:30 – 5:45)
* **Action:** Scroll down to show the **Grad-CAM X-ray Explainability Heatmap** card and the **RuleValidator Safety Banner**.
* **Talking Script:**
  > *"When an AI diagnoses a chest X-ray with Pneumonia or Pleural Effusion, a doctor cannot blindly trust a text label. Here in our **Grad-CAM Heatmap card**, our local computer vision model (`torchxrayvision` DenseNet) generates an anatomical saliency overlay. The red glow shows the exact lung opacity that drove the classification—proving the model is identifying real pathology, not image noise.
  > 
  > Finally, look at the **Safety Validation Banner** above the report. Our post-generation `RuleValidator` compares the LLM's narrative against our hardcoded clinical severity score. If the LLM ever attempts to downplay a critical biomarker or drug interaction, our deterministic rules trigger an authoritative **Safety Override**, alerting the clinician immediately."*

### Act V: Data Ownership, Offline Dossier & Observability (Minute 5:45 – 7:00)
* **Action:** Click **"Download PDF"** to open the offline clinical dossier, and then show the terminal with `curl http://localhost:8000/metrics`.
* **Talking Script:**
  > *"Because the patient owns their data, clicking **Download PDF** generates an offline-safe clinical dossier rendered entirely server-side using pure Python WeasyPrint—with zero CDN or cloud dependencies. It embeds SHA-256 cryptographic integrity hashes, all input evidence, the tool audit trail, and the Grad-CAM heatmap.
  > 
  > In production, our system exposes full **Prometheus observability** (`/metrics`), monitoring queue depth, adaptive capacity, LRU cache hit rates, and rule validation agreements—proving that local edge AI can deliver enterprise-grade reliability without compromising patient privacy."*

---

## 3. Post-Demo Reset & Cleanup (Transition Between Judge Groups)

To prepare the system for the next group of evaluators without rebooting the server or restarting Docker containers, run the automated demo reset endpoint:

```bash
# Reset seeded demo users and purge temporary job files
curl -X POST http://localhost:8000/admin/demo-reset -H "Authorization: Bearer <admin_token>"
```
* **What it does:** Purges `/tmp/aegis_uploads/`, removes temporary pipeline run states, clears LRU cache entries, and resets `health_records` and `vital_snapshots` for the 3 demo accounts back to their clean baseline state.
