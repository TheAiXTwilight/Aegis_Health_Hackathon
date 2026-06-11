# Aegis Health
*Built for Privacy, Designed for the Edge.*

## Problem Statement

Modern healthcare AI systems largely depend on cloud-hosted language models and external APIs,
requiring sensitive patient information to be processed outside clinical control. This creates
privacy concerns, internet dependency, and deployment challenges in low-connectivity or
resource-constrained environments. Existing solutions are also fragmented, typically addressing
a single task — symptom checking, laboratory analysis, medication lookup, or imaging
interpretation — forcing clinicians to manually consolidate information before making triage
decisions.

Aegis Health addresses this through a fully local, privacy-first clinical triage assistant
designed for deployment on an NVIDIA Jetson Orin Nano. The system accepts five clinical input
modalities — symptom descriptions (typed or voice), laboratory report PDFs, chest X-rays,
medication lists, and locally retrieved medical evidence — and produces a unified, structured
triage report entirely on-device. All processing occurs with no external API calls.
`llama3.2:1b` via Ollama serves as the synthesis component, generating the final clinical
report from validated structured outputs produced by specialised tools; the system's reasoning
capability is derived from the orchestration of specialised tools and deterministic clinical
rules rather than from the language model in isolation. Multiple clinicians may submit sessions
concurrently through a managed job queue; inference executes one pipeline at a time, preserving
memory safety without sacrificing multi-user accessibility.

## Challenges & Considerations

- Ensuring reliable structured output generation despite the reasoning and context-window
  limitations of Small Language Models, requiring validation, correction, and explicit failure
  handling rather than silent acceptance.
- Latency and context-window constraints introduced by multiple language model invocations
  within a single pipeline execution, requiring careful orchestration to maintain acceptable
  clinical response times.
- Memory management challenges associated with deploying a multimodal AI system on
  resource-constrained edge hardware, where the language model, embedding inference, vector
  stores, API serving, and document processing components must coexist within a fixed physical
  memory envelope.
- Preservation of clinically critical information when combining multiple input modalities
  within a bounded context window, requiring a principled prioritisation system that
  unconditionally retains severity assessments and core clinical findings regardless of total
  input volume.
- Risk of hallucinations and unsupported medical claims, mitigated by grounding all evidence
  citations in a local medical knowledge base and delegating triage severity to a fully
  deterministic rule-based system that operates independently of the language model.
- Reliable multimodal extraction from audio recordings, PDF laboratory reports, and
  clinician-provided radiological findings on ARM64 edge hardware, where third-party library
  compatibility cannot be assumed and must be validated before integration.
- Multi-user session management on a single-inference device, requiring a queuing architecture
  that provides clinicians with submission acknowledgement, real-time queue visibility, and
  predictable wait estimates while guaranteeing that only one inference pipeline executes at
  any moment.
- Complete offline operation with zero external API calls, reproducible knowledge base
  versioning, and transparent failure handling across all input conditions through explicit and
  auditable error contracts.

## Proposed Solution

- Specialised tools handle all clinical extraction and analysis — transcribing voice input,
  parsing laboratory reports, processing radiological findings, retrieving medical evidence,
  and checking drug interactions — while `llama3.2:1b` makes exactly two targeted inference
  calls per session: structured symptom extraction and final report synthesis.
- A multi-user FIFO job queue accepts concurrent clinician submissions, assigns each session a
  job identifier with live queue position and estimated wait time, and feeds a single inference
  worker that holds an exclusive execution lock — preserving the single-inference memory model
  on resource-constrained hardware while giving every clinician immediate submission
  acknowledgement and transparent queue visibility.
- A deterministic pipeline orchestrator sequences all processing stages under a single bounded
  execution, preventing concurrent inference, guaranteeing reproducibility, and preserving
  memory safety across edge deployments.
- A fully specified rule-based SeverityScorer produces triage level deterministically from a
  documented clinical rule set, returning the complete set of triggered rules, the primary
  determining rule, and human-readable explanations — the language model cannot influence
  severity, and every triage decision is traceable to a named, auditable rule.
- Local medical evidence retrieval via ONNX-optimised embeddings and a vector store grounded
  in the MedlinePlus corpus provides citation-backed evidence passages with zero internet
  dependency at runtime; absence of relevant evidence is explicitly stated rather than
  fabricated.
- Structured data contracts with explicit fatal and non-fatal failure classifications, a
  documented confidence scoring formula, and section-level report validation ensure that silent
  wrong answers are structurally impossible and all outputs are auditable.
- Field-level prioritisation unconditionally preserves severity assessments and core clinical
  findings regardless of total input size; upload bounds, pipeline timeouts, and stream
  disconnection handling guarantee honest, explicit failure under all input conditions rather
  than silent degradation.
- Containerised deployment ensures reproducibility across development and edge environments
  while minimising startup latency and simplifying system maintenance.

## Expected Outcome

Aegis Health demonstrates that a Small Language Model can serve as the synthesis component of
a five-modality clinical triage workflow running entirely on edge hardware, within the memory
constraints of a resource-limited device — leaving practical headroom for all pipeline
components under realistic clinical load. Multiple clinicians can submit concurrent sessions
through a managed queue with live position visibility and estimated wait times, while inference
always executes one pipeline at a time to preserve memory safety. All routing, severity
scoring, and clinical reasoning are handled by deterministic tools with fully auditable rule
sets; the language model synthesises, it does not decide. The system triages — it does not
diagnose — providing clinicians with a structured, evidence-backed starting point where every
severity assessment is explainable through named rule triggers, every confidence score is
derived from a documented formula, and patient data never leaves the device. The project
integrates Small Language Models, Retrieval-Augmented Generation, Document AI, Speech
Processing, Edge AI Deployment, and multi-user queue management into a privacy-preserving
clinical assistant that operates without cloud infrastructure while maintaining complete
transparency, explicit failure handling, and local data ownership.