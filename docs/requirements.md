# Aegis Health — Python Dependencies


## Core

pydantic>=2.0,<3.0


## API

fastapi>=0.111,<1.0
uvicorn[standard]>=0.29,<1.0
aiofiles>=23.0,<25.0


## LLM

ollama>=0.2,<1.0
httpx>=0.27,<1.0


## PDF Parsing

pymupdf>=1.24,<2.0
pdfminer.six>=20221105


## OCR (optional — ARM64 validation required in Week 1)

easyocr>=1.7,<2.0

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


## Image and DICOM

pillow>=10.0,<12.0
pydicom>=2.4,<3.0


## Logging

loguru>=0.7,<1.0


## Testing

pytest>=8.0,<9.0
pytest-asyncio>=0.23,<1.0


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


## Notes

torch is never imported at runtime. GPU detection uses nvidia-smi
and Jetson device nodes only.

EasyOCR is opt-in. It is not loaded unless a scanned PDF requires OCR.
It is explicitly released after use.

faster-whisper is lazy-loaded by VoiceTranscriber and released
immediately after transcription completes.

All inference runs locally. No external API calls at runtime.

Python 3.11 or higher required. See docs/setup_jetson.md.