"""
rag/embed.py — Shared ONNX MiniLM embedding helpers.

Single source of truth for turning text into the 384-dim MiniLM vector
used across the RAG pipeline. Two consumers:

    1. Build-time:  rag/build_chroma.py, rag/build_faiss.py
       (embed the whole corpus once, offline)

    2. Runtime prewarm:  backend/model_registry.py::_rag_embed()
       (embed a throwaway string at startup so the first real user
       request isn't cold)

    tools/medical_rag_search.py keeps its OWN copy of this logic
    (by design — see that file's module docstring) so the runtime
    query path has zero import-time dependency on the rag/ build
    package. This module intentionally mirrors that implementation
    exactly so build-time and runtime embeddings are identical
    vectors for the same input text.

Paths (relative to CWD — run from project root):
    data/knowledge/minilm.onnx     ONNX MiniLM model
    data/knowledge/tokenizer.json  HuggingFace tokenizers file
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

_ONNX_MODEL = Path("data/knowledge/minilm.onnx")
_TOKENIZER_PATH = Path("data/knowledge/tokenizer.json")
_MAX_SEQ_LEN = 128
_EMBED_DIM = 384

_session: Any | None = None
_session_attempted = False
_tokenizer: Any | None = None
_tokenizer_attempted = False


def _get_session() -> Any | None:
    global _session, _session_attempted
    if _session_attempted:
        return _session
    _session_attempted = True

    if not _ONNX_MODEL.exists():
        logger.warning("rag.embed · ONNX model not found", path=str(_ONNX_MODEL))
        return None

    try:
        import onnxruntime as ort  # type: ignore[import]

        _session = ort.InferenceSession(str(_ONNX_MODEL), providers=["CPUExecutionProvider"])
        logger.info("rag.embed · ONNX session loaded", path=str(_ONNX_MODEL))
        return _session
    except Exception:
        logger.exception("rag.embed · ONNX session load failed")
        return None


def _get_tokenizer() -> Any | None:
    global _tokenizer, _tokenizer_attempted
    if _tokenizer_attempted:
        return _tokenizer
    _tokenizer_attempted = True

    if not _TOKENIZER_PATH.exists():
        logger.warning("rag.embed · tokenizer.json not found", path=str(_TOKENIZER_PATH))
        return None

    try:
        from tokenizers import Tokenizer  # type: ignore[import]

        tok = Tokenizer.from_file(str(_TOKENIZER_PATH))
        tok.enable_truncation(max_length=_MAX_SEQ_LEN)
        tok.enable_padding(length=_MAX_SEQ_LEN)
        _tokenizer = tok
        return tok
    except Exception:
        logger.exception("rag.embed · tokenizer load failed")
        return None


def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[:, :, np.newaxis].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    pooled = (summed / counts)[0]
    norm = np.linalg.norm(pooled)
    return pooled / norm if norm > 1e-9 else pooled


def embed_text(text: str) -> np.ndarray | None:
    """
    Embed a single string. Returns a (384,) float32 unit-normalised
    vector, or None if the model/tokenizer are unavailable or
    inference fails.
    """
    session = _get_session()
    tokenizer = _get_tokenizer()
    if session is None or tokenizer is None:
        return None

    try:
        enc = tokenizer.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        token_type_ids = np.array([enc.type_ids], dtype=np.int64)

        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        return _mean_pool(outputs[0], attention_mask)
    except Exception:
        logger.exception("rag.embed · embedding failed")
        return None


_DEFAULT_BATCH_SIZE = 8  # measured optimal on this build hardware — see rag/ingest.py benchmark note


def _embed_batch_raw(texts: list[str]) -> list[np.ndarray | None]:
    """Run one real batched forward pass for a list of texts (all-or-nothing)."""
    session = _get_session()
    tokenizer = _get_tokenizer()
    if session is None or tokenizer is None:
        return [None] * len(texts)

    try:
        encs = tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encs], dtype=np.int64)

        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        last_hidden = outputs[0]  # (batch, seq_len, hidden)
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = (last_hidden * mask).sum(axis=1)
        counts = mask.sum(axis=1).clip(min=1e-9)
        pooled = summed / counts
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        pooled = pooled / norms
        return [pooled[i] for i in range(len(texts))]
    except Exception:
        logger.exception("rag.embed · batched embedding failed")
        return [None] * len(texts)


def embed_batch(texts: list[str], batch_size: int = _DEFAULT_BATCH_SIZE) -> list[np.ndarray | None]:
    """
    Embed multiple strings using real batched ONNX forward passes
    (batch_size sequences per call) rather than one call per text.

    Batching amortizes per-call session overhead — significant at
    corpus-build scale (thousands of chunks), even on a single CPU
    core where there's no parallelism to exploit, because it's the
    fixed per-invocation overhead (not raw FLOPs) that dominates at
    small per-call sizes.

    Falls back to per-item embed_text() for any batch chunk that
    fails outright, so a single bad input doesn't drop the whole
    batch's embeddings.
    """
    if not texts:
        return []

    results: list[np.ndarray | None] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        chunk_results = _embed_batch_raw(chunk)
        if all(r is None for r in chunk_results) and is_available():
            # Batched call itself failed for an unexpected reason;
            # retry item-by-item rather than losing the whole chunk.
            chunk_results = [embed_text(t) for t in chunk]
        results.extend(chunk_results)
    return results


def is_available() -> bool:
    """True if both the ONNX model and tokenizer loaded successfully."""
    return _get_session() is not None and _get_tokenizer() is not None
