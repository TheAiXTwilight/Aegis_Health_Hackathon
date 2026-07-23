# Aegis Health — Python Dependencies

Python 3.11 or higher required (asyncio.timeout was added in 3.11).
See docs/setup_jetson.md for Jetson-specific setup notes.


## Core

pydantic>=2.0,<3.0


## API

fastapi>=0.111,<1.0
uvicorn[standard]>=0.29,<1.0
aiofiles>=23.0,<25.0


## LLM

httpx>=0.27,<1.0

Ollama is called via its HTTP API at OLLAMA_BASE_URL/api/generate.
No Ollama Python client is required.

Phase 2.5 note: ExecutionPlanner uses the same httpx dependency as
ReportGenerator. No additional LLM dependencies introduced.
ExecutionPlanner call: non-streaming (stream=False), temperature=0.0.
ReportGenerator call:  streaming (stream=True), temperature=0.2.


## PDF Parsing

pymupdf>=1.24,<2.0
pdfminer.six>=20221105

LabReportParser uses a three-tier extraction waterfall:
    PyMuPDF (primary) → pdfminer.six (fallback) → EasyOCR (opt-in)

PyMuPDF is also used in tests to generate real one-page PDF fixtures
(via fitz.open() + new_page() + insert_text()). No additional test
dependency is required — fitz is already a runtime dependency.


## OCR (optional — ARM64 validation required)

easyocr>=1.7,<2.0

Enabled at runtime via AEGIS_OCR env var.
Accepted values (case-insensitive): 1, true, yes.
Any other value (including unset) leaves OCR disabled.

If EasyOCR fails on ARM64, replace with:

pytesseract>=0.3,<1.0

Document the decision in docs/setup_jetson.md.


## Voice Transcription

faster-whisper>=1.0,<2.0

CPU-only. Lazy-loaded. Released immediately after transcription.
Approximately 150 MB on disk.


## Embeddings

onnxruntime>=1.18,<2.0
numpy>=1.26,<3.0

ONNX Runtime used for all-MiniLM-L6-v2 inference.
No torch dependency for embeddings.


## Vector Store

chromadb>=0.5,<1.0
faiss-cpu>=1.8,<2.0

ChromaDB is primary retriever.
FAISS is fallback retriever.
Both use the committed ONNX MiniLM model.

Phase 2.5 note: MedicalRAGSearch now runs conditionally (plan.use_rag).
No changes to vector store dependencies.


## Image and DICOM

pillow>=10.0,<12.0
pydicom>=2.4,<3.0


## Logging

loguru>=0.7,<1.0


## Testing

pytest>=8.0,<9.0
pytest-asyncio>=0.23,<1.0
pytest-cov>=5.0,<8.0


## Static Analysis (development only)

mypy>=1.0,<2.0
ruff>=0.4,<1.0

Install in development environment:

    pip install mypy ruff


## Installation

    pip install -r docs/requirements.txt

Or using uv:

    uv pip install -r docs/requirements.txt


## Frontend Dependencies

The frontend is being migrated from Streamlit to React + Vite +
TypeScript + Tailwind + shadcn/ui.

Python frontend dependencies (streamlit, requests) have been removed
from this file because the Streamlit prototype is deprecated and not
maintained alongside the planned React replacement.

Node dependencies will be managed via frontend/package.json once the
React project is scaffolded. There is no Python frontend dependency
on this list.


## Phase 2.5 Dependencies

Phase 2.5 introduced ExecutionPlanner, PlanValidator, and RuleValidator.

No new Python packages are required:
    ExecutionPlanner: uses httpx (already listed), json (stdlib)
    PlanValidator:    uses pydantic (already listed), stdlib only
    RuleValidator:    uses re (stdlib), pydantic (already listed)

No changes to this file are required for Phase 2.5.


## Phase 3 Dependencies

Phase 3 replaces placeholder implementations with real ones.
No new packages are introduced beyond what was already listed.

    Commit 1 — MedicalRAGSearch:     onnxruntime, chromadb, faiss-cpu (all listed)
    Commit 2 — DrugInteractionChecker: sqlite3 (stdlib), no new packages
    Commit 3 — SymptomExtractor:     httpx (already listed)
    Commit 4 — LabReportParser:      pymupdf, pdfminer.six, easyocr (all listed)

The AEGIS_OCR=1 flag gates EasyOCR loading. When unset, EasyOCR is
never imported — the package can be omitted from minimal deployments
that only process digital (not scanned) PDFs.


## Notes

torch is never imported at runtime. GPU detection uses nvidia-smi
and Jetson device nodes only.

EasyOCR is opt-in. It is not loaded unless a scanned PDF requires OCR.
It is explicitly released after use. Controlled by _ocr_enabled() in
tools/lab_report_parser.py — the single gate for AEGIS_OCR.

faster-whisper is lazy-loaded by VoiceTranscriber and released
immediately after transcription completes.

All inference runs locally. No external API calls at runtime.