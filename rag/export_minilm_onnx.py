"""
rag/export_minilm_onnx.py — Step 2 of the knowledge base build (see
docs/corpus_version.md).

Mac-only, one-time step. Exports sentence-transformers/all-MiniLM-L6-v2
to ONNX format so the Jetson/ARM64 runtime can embed text with
onnxruntime only — no torch/transformers dependency at inference time.

This step requires network access to HuggingFace to download the
source PyTorch weights, and is NOT run as part of rag/ingest.py or in
network-restricted build environments. The exported artifacts
(data/knowledge/minilm.onnx, data/knowledge/tokenizer.json) are
committed to the repository per docs/corpus_version.md so downstream
builds and the Jetson deployment never need to run this script or
reach HuggingFace at all.

Run this script only when the embedding model needs to be re-exported
(e.g. upgrading to a newer sentence-transformers checkpoint).

Requires (Mac dev machine only — not part of requirements.txt):
    pip install torch transformers optimum[onnxruntime] onnx

Usage:
    python rag/export_minilm_onnx.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from loguru import logger

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_OUTPUT_DIR = Path("data/knowledge")
_ONNX_OUTPUT = _OUTPUT_DIR / "minilm.onnx"
_TOKENIZER_OUTPUT = _OUTPUT_DIR / "tokenizer.json"


def export() -> int:
    if _ONNX_OUTPUT.exists() and _TOKENIZER_OUTPUT.exists():
        logger.info(
            "export_minilm_onnx · outputs already exist, skipping re-export "
            "(delete data/knowledge/minilm.onnx to force)",
            onnx=str(_ONNX_OUTPUT),
            tokenizer=str(_TOKENIZER_OUTPUT),
        )
        return 0

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
    except ImportError:
        logger.error(
            "export_minilm_onnx · optimum/transformers not installed. "
            "This is a Mac-only dev dependency, not part of requirements.txt. "
            "Install with: pip install torch transformers optimum[onnxruntime] onnx"
        )
        return 1

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = _OUTPUT_DIR / "_onnx_export_tmp"
    tmp_dir.mkdir(exist_ok=True)

    logger.info("export_minilm_onnx · exporting", model=_MODEL_NAME)

    model = ORTModelForFeatureExtraction.from_pretrained(_MODEL_NAME, export=True)
    model.save_pretrained(tmp_dir)

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    tokenizer.save_pretrained(tmp_dir)

    # optimum names the exported graph model.onnx — rename to our
    # conventional filename and flatten into data/knowledge/.
    exported_onnx = tmp_dir / "model.onnx"
    exported_tokenizer_json = tmp_dir / "tokenizer.json"

    shutil.move(str(exported_onnx), str(_ONNX_OUTPUT))
    shutil.move(str(exported_tokenizer_json), str(_TOKENIZER_OUTPUT))
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(
        "export_minilm_onnx · export complete",
        onnx=str(_ONNX_OUTPUT),
        tokenizer=str(_TOKENIZER_OUTPUT),
    )
    return 0


if __name__ == "__main__":
    sys.exit(export())
