"""

tools/medical_rag_search.py — Medical RAG search (Step 5).


Phase 3 replacement: ChromaDB primary retriever with ONNX MiniLM
embeddings. FAISS used as fallback when ChromaDB query fails.


Architecture decisions:
    - ChromaDB as primary  (simpler API, metadata built-in, collection
      already populated by rag/build_chroma.py)
    - FAISS as documented fallback (ANN control, offline export)
    - ONNX Runtime for inference — no torch dependency, ARM64 safe
    - Index objects initialised lazily at first call, held for
      application lifetime (module-level singletons)
    - Zero results is valid: RAGSearchResult(passages=[], retrieval_successful=True)
    - Any mechanism failure → ToolError(fatal=False), pipeline continues


Public interface unchanged:
    async def run(self, state: AegisState) -> RAGSearchResult | ToolError


Paths (relative to CWD — run from project root):
    data/knowledge/chroma/       ChromaDB persistent directory
    data/knowledge/faiss.index   FAISS index file
    data/knowledge/faiss.docs    FAISS doc store (JSON lines)
    data/knowledge/minilm.onnx   ONNX MiniLM model


ChromaDB collection name: "aegis_knowledge"
    Set by rag/build_chroma.py — must match here.


Top-k retrieval: TOP_K = 4
    Returns up to 4 passages. ReportGenerator uses top-2 of these.
    Changing TOP_K here does not require pipeline changes.


Query construction:
    Same logic as the placeholder — join raw_symptoms_text +
    extracted symptom strings + abnormal lab values. Consistent
    query across placeholder and real implementation ensures the
    confidence score is comparable.


ONNX session:
    Loaded once at module level on first call to _get_onnx_session().
    tokenizers library used for MiniLM tokenization (huggingface/tokenizers).
    tokenizer.json must be present at data/knowledge/tokenizer.json.
    If missing, _embed() returns None and the tool returns ToolError(fatal=False).
    No whitespace fallback — it silently produces garbage embeddings.

"""


from __future__ import annotations


import json
from pathlib import Path
from typing import Any


import numpy as np
from loguru import logger


from schemas.errors import ToolError
from schemas.rag import RAGPassage, RAGSearchResult
from schemas.state import AegisState
from tools.tool_names import TOOL_MEDICAL_RAG_SEARCH


# ── Paths ──────────────────────────────────────────────────────────


_CHROMA_DIR   = Path("data/knowledge/chroma")
_FAISS_INDEX  = Path("data/knowledge/faiss.index")
_FAISS_DOCS   = Path("data/knowledge/faiss.docs")
_ONNX_MODEL   = Path("data/knowledge/minilm.onnx")


_COLLECTION_NAME = "aegis_knowledge"
TOP_K            = 4
_EMBED_DIM       = 384   # MiniLM all-MiniLM-L6-v2 output dimension
_MAX_SEQ_LEN     = 128   # MiniLM maximum sequence length

# ── Relevance thresholds ────────────────────────────────────────────
# Without these, a query with no good match in the corpus still
# returns the "least bad" TOP_K passages as if they were confident
# matches, which can mislead a report. Values below were calibrated
# empirically against this build's actual corpus and index (not
# guessed): in-corpus medical queries ("chest pain and shortness of
# breath", "type 2 diabetes symptoms", "vague fatigue") consistently
# scored within the accepted range; clearly unrelated queries
# ("best pizza toppings", "recommend a good laptop") consistently
# fell outside it with a wide margin. Re-calibrate if the corpus is
# rebuilt at a different size/composition (see docs/corpus_version.md).
#
# ChromaDB (default metric = L2 distance on normalized vectors,
# LOWER is more similar): observed real matches ~0.6-1.1,
# unrelated queries ~1.35-1.8. Cutoff set at 1.2 — above all
# observed real matches, below all observed unrelated queries.
_CHROMA_MAX_DISTANCE = 1.2

# FAISS (IndexFlatIP = inner product / cosine similarity on
# normalized vectors, HIGHER is more similar): observed real matches
# ~0.53-0.69, unrelated queries ~0.11-0.19. Cutoff set at 0.35 —
# below all observed real matches, above all observed unrelated
# queries.
_FAISS_MIN_SCORE = 0.35


# ── Module-level singletons (lazy init) ───────────────────────────
# Two variables per singleton — same pattern as drug_checker.py
# and corpus_version.py (Phase 4 sentinel refactor, Decision 94).
#
# *attempted = False  → load not yet tried
# *attempted = True   → load was tried; *value holds result or None
#
# This replaces the T | None | bool sentinel anti-pattern where None
# means both "not tried" and "tried and returned None". Keeping them
# separate makes the state machine explicit and avoids the fixture
# reset bug where resetting to None confused "not tried" with
# "tried and unavailable".


_onnx_attempted:    bool       = False
_onnx_session:      Any | None = None


_chroma_attempted:  bool       = False
_chroma_collection: Any | None = None


_faiss_attempted:   bool       = False
_faiss_index:       Any | None = None
_faiss_docs:        list[dict] | None = None


def _get_onnx_session() -> Any | None:
    """
    Load ONNX Runtime session once. Returns None if unavailable.

    Cached at module level — model load is expensive on Jetson.
    """
    global _onnx_attempted, _onnx_session
    if _onnx_attempted:
        return _onnx_session

    _onnx_attempted = True

    if not _ONNX_MODEL.exists():
        logger.warning(
            "medical_rag_search · ONNX model not found",
            path=str(_ONNX_MODEL),
        )
        return None

    try:
        import onnxruntime as ort  # type: ignore[import]
        sess = ort.InferenceSession(
            str(_ONNX_MODEL),
            providers=["CPUExecutionProvider"],
        )
        _onnx_session = sess
        logger.info(
            "medical_rag_search · ONNX session loaded",
            path=str(_ONNX_MODEL),
        )
        return sess
    except Exception as exc:
        logger.warning(
            "medical_rag_search · ONNX session load failed",
            error=str(exc),
        )
        return None



def _get_faiss() -> tuple[Any, list[dict]] | tuple[None, None]:
    """
    Load FAISS index and doc store once. Returns (index, docs) or (None, None).
    """
    global _faiss_attempted, _faiss_index, _faiss_docs
    if _faiss_attempted:
        if _faiss_index is None:
            return None, None
        return _faiss_index, _faiss_docs  # type: ignore[return-value]

    _faiss_attempted = True

    if not _FAISS_INDEX.exists() or not _FAISS_DOCS.exists():
        logger.warning(
            "medical_rag_search · FAISS files not found",
            index=str(_FAISS_INDEX),
            docs=str(_FAISS_DOCS),
        )
        return None, None

    try:
        import faiss  # type: ignore[import]
        index = faiss.read_index(str(_FAISS_INDEX))
        docs  = [
            json.loads(line)
            for line in _FAISS_DOCS.read_text().splitlines()
            if line.strip()
        ]
        _faiss_index = index
        _faiss_docs  = docs
        logger.info(
            "medical_rag_search · FAISS index loaded",
            total_vectors=index.ntotal,
        )
        return index, docs
    except Exception as exc:
        logger.warning(
            "medical_rag_search · FAISS load failed",
            error=str(exc),
        )
        return None, None


# ── ONNX embedding ─────────────────────────────────────────────────


def _get_onnx_session() -> Any | None:
    """
    Load ONNX Runtime session once. Returns None if unavailable.

    Cached at module level — model load is expensive on Jetson.
    """
    global _onnx_session
    if _onnx_session is not None:
        return _onnx_session if _onnx_session is not False else None

    if not _ONNX_MODEL.exists():
        logger.warning(
            "medical_rag_search · ONNX model not found",
            path=str(_ONNX_MODEL),
        )
        _onnx_session = False
        return None

    try:
        import onnxruntime as ort  # type: ignore[import]
        sess = ort.InferenceSession(
            str(_ONNX_MODEL),
            providers=["CPUExecutionProvider"],
        )
        _onnx_session = sess
        logger.info(
            "medical_rag_search · ONNX session loaded",
            path=str(_ONNX_MODEL),
        )
        return sess
    except Exception as exc:
        logger.exception("medical_rag_search · ONNX session load failed")  # Changed to .exception
        return None


def _tokenize(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tokenize text for MiniLM using huggingface tokenizers + tokenizer.json.

    Requires tokenizer.json alongside minilm.onnx at data/knowledge/.
    Obtain from the original sentence-transformers/all-MiniLM-L6-v2
    HuggingFace repository and copy to data/knowledge/tokenizer.json.

    NO whitespace fallback. MiniLM embeddings depend on WordPiece
    tokenization. A whitespace fallback silently produces garbage
    embeddings that degrade retrieval quality without any visible
    error signal. If the tokenizer is unavailable, the caller raises
    and _embed() returns None, surfacing a ToolError instead.

    Raises:
        FileNotFoundError  — tokenizer.json missing
        ImportError        — tokenizers library not installed
        Exception          — tokenizer encode failed

    Returns (input_ids, attention_mask, token_type_ids) each shape (1, seq_len).
    """
    from tokenizers import Tokenizer  # type: ignore[import]

    tok_path = _ONNX_MODEL.parent / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"tokenizer.json not found at {tok_path}. "
            "Copy from sentence-transformers/all-MiniLM-L6-v2 on HuggingFace."
        )

    tokenizer = Tokenizer.from_file(str(tok_path))
    tokenizer.enable_truncation(max_length=_MAX_SEQ_LEN)
    tokenizer.enable_padding(length=_MAX_SEQ_LEN)

    enc = tokenizer.encode(text)
    input_ids      = np.array([enc.ids],           dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)
    token_type_ids = np.array([enc.type_ids],       dtype=np.int64)
    return input_ids, attention_mask, token_type_ids


def _mean_pool(
    last_hidden: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """
    Mean-pool token embeddings weighted by attention mask.

    last_hidden shape: (1, seq_len, hidden_dim)
    attention_mask:    (1, seq_len)
    Returns:           (hidden_dim,) normalised vector
    """
    mask = attention_mask[:, :, np.newaxis].astype(np.float32)  # (1, seq, 1)
    summed = (last_hidden * mask).sum(axis=1)                   # (1, hidden)
    counts = mask.sum(axis=1).clip(min=1e-9)                    # (1, 1)
    pooled = (summed / counts)[0]                               # (hidden,)
    norm   = np.linalg.norm(pooled)
    return pooled / norm if norm > 1e-9 else pooled


def _embed(text: str) -> np.ndarray | None:
    """
    Embed a query string with MiniLM ONNX model.

    Returns (384,) float32 array, or None on any failure.

    Failure cases that return None (all surface as ToolError upstream):
        - ONNX session unavailable (missing model file or ort not installed)
        - tokenizer.json missing (no fallback — see _tokenize docstring)
        - tokenizers library not installed
        - ONNX inference error
    """
    sess = _get_onnx_session()
    if sess is None:
        return None

    try:
        input_ids, attention_mask, token_type_ids = _tokenize(text)
        outputs = sess.run(
            None,
            {
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        # outputs[0] = last_hidden_state (1, seq_len, 384)
        return _mean_pool(outputs[0], attention_mask)
    except Exception as exc:
        logger.warning(
            "medical_rag_search · embedding failed",
            error=str(exc),
        )
        return None


# ── ChromaDB (primary) ─────────────────────────────────────────────


def _get_chroma_collection() -> Any | None:
    """
    Load ChromaDB collection once. Returns None if unavailable.
    """
    global _chroma_attempted, _chroma_collection
    if _chroma_attempted:
        return _chroma_collection

    _chroma_attempted = True

    if not _CHROMA_DIR.exists():
        logger.warning(
            "medical_rag_search · ChromaDB directory not found",
            path=str(_CHROMA_DIR),
        )
        return None

    try:
        import chromadb  # type: ignore[import]
        client     = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        collection = client.get_collection(_COLLECTION_NAME)
        _chroma_collection = collection
        logger.info(
            "medical_rag_search · ChromaDB collection loaded",
            name=_COLLECTION_NAME,
            count=collection.count(),
        )
        return collection
    except KeyError as exc:
        if str(exc) == "'_type'":
            # Known failure mode: the installed chromadb version doesn't
            # match the version that wrote data/knowledge/chroma/'s
            # collection metadata. See the chromadb pin + comment in
            # requirements.txt for the full explanation and fix.
            logger.warning(
                f"medical_rag_search · ChromaDB load failed: likely a "
                f"chromadb version mismatch against the committed index "
                f"(KeyError: {exc}). Run `pip show chromadb` and compare "
                f"against the pinned version in requirements.txt — see "
                f"the comment there for details.",
                error=str(exc),
            )
        else:
            logger.warning(
                f"medical_rag_search · ChromaDB load failed: "
                f"KeyError: {exc}",
                error=str(exc),
            )
        return None
    except Exception as exc:
        logger.warning(
            f"medical_rag_search · ChromaDB load failed: "
            f"{type(exc).__name__}: {exc}",
            error=str(exc),
        )
        return None


def _chroma_query(
    query_embedding: np.ndarray,
    n_results: int,
) -> list[RAGPassage]:
    """
    Query ChromaDB collection with pre-computed embedding.
    Returns list of RAGPassage, possibly empty.
    """
    collection = _get_chroma_collection()
    if collection is None:
        raise RuntimeError("ChromaDB collection unavailable")

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    passages: list[RAGPassage] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        if not doc:
            continue
        if distance > _CHROMA_MAX_DISTANCE:
            # Below relevance threshold — a weak/unrelated match.
            # Skip rather than return it as if it were a confident hit.
            continue
        passages.append(
            RAGPassage(
                text     = doc,
                source   = meta.get("source", "Unknown"),
                citation = meta.get("citation", ""),
            )
        )

    return passages


# ── FAISS (fallback) ───────────────────────────────────────────────


def _get_faiss() -> tuple[Any, list[dict]] | tuple[None, None]:
    """
    Load FAISS index and doc store once. Returns (index, docs) or (None, None).
    """
    global _faiss_index, _faiss_docs
    if _faiss_index is not None:
        if _faiss_index is False:
            return None, None
        return _faiss_index, _faiss_docs  # type: ignore[return-value]

    if not _FAISS_INDEX.exists() or not _FAISS_DOCS.exists():
        logger.warning(
            "medical_rag_search · FAISS files not found",
            index=str(_FAISS_INDEX),
            docs=str(_FAISS_DOCS),
        )
        _faiss_index = False
        return None, None

    try:
        import faiss  # type: ignore[import]
        index = faiss.read_index(str(_FAISS_INDEX))
        docs  = [
            json.loads(line)
            for line in _FAISS_DOCS.read_text().splitlines()
            if line.strip()
        ]
        _faiss_index = index
        _faiss_docs  = docs
        logger.info(
            "medical_rag_search · FAISS index loaded",
            total_vectors=index.ntotal,
        )
        return index, docs
    except Exception as exc:
        logger.warning(
            "medical_rag_search · FAISS load failed",
            error=str(exc),
        )
        _faiss_index = False
        return None, None


def _faiss_query(
    query_embedding: np.ndarray,
    n_results: int,
) -> list[RAGPassage]:
    """
    Query FAISS index with pre-computed embedding.
    Returns list of RAGPassage, possibly empty.
    """
    index, docs = _get_faiss()
    if index is None or not docs:
        raise RuntimeError("FAISS index unavailable")

    vec = query_embedding.reshape(1, -1).astype(np.float32)
    k   = min(n_results, index.ntotal)
    scores, indices = index.search(vec, k)

    passages: list[RAGPassage] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(docs):
            continue
        if score < _FAISS_MIN_SCORE:
            # Below relevance threshold — a weak/unrelated match.
            # Skip rather than return it as if it were a confident hit.
            continue
        doc = docs[idx]
        passages.append(
            RAGPassage(
                text     = doc.get("text", ""),
                source   = doc.get("source", "Unknown"),
                citation = doc.get("citation", ""),
            )
        )

    return passages


# ── Query builder ──────────────────────────────────────────────────


def _build_query(state: AegisState) -> str:
    """
    Build retrieval query from available state fields.

    Matches placeholder logic exactly — consistent with the query
    used during index build (symptom text + extracted symptoms +
    abnormal lab values). Consistent query ensures the confidence
    score is comparable across placeholder and real implementations.
    """
    parts: list[str] = []

    if state.raw_symptoms_text:
        parts.append(state.raw_symptoms_text)

    if (
        state.symptom_result
        and not isinstance(state.symptom_result, ToolError)
        and state.symptom_result.symptoms
    ):
        parts.extend(state.symptom_result.symptoms)

    if (
        state.lab_result
        and not isinstance(state.lab_result, ToolError)
        and state.lab_result.abnormal_values
    ):
        parts.extend(state.lab_result.abnormal_values)

    return " ".join(parts).strip()


# ── Tool ──────────────────────────────────────────────────────────


class MedicalRAGSearch:
    """
    Semantic RAG search using ChromaDB + ONNX MiniLM.

    Retrieval chain:
        1. Embed query with MiniLM ONNX model
        2. Query ChromaDB collection (primary)
        3. On ChromaDB failure: query FAISS index (fallback)
        4. On both failures: return ToolError(fatal=False)

    Does not write to state — pipeline owns state mutation.
    """

    TOOL_NAME = TOOL_MEDICAL_RAG_SEARCH

    async def run(
        self,
        state: AegisState,
    ) -> RAGSearchResult | ToolError:

        try:
            query = _build_query(state)

            if not query:
                # No query material — return empty successful result.
                return RAGSearchResult(
                    passages=[],
                    citations=[],
                    query_used="",
                    retrieval_successful=True,
                )

            # ── Embed query ──────────────────────────────────────
            embedding = _embed(query)
            if embedding is None:
                return ToolError(
                    tool=TOOL_MEDICAL_RAG_SEARCH,
                    reason=(
                        "Query embedding failed — ONNX model unavailable "
                        "or inference error."
                    ),
                    fatal=False,
                )

            # ── ChromaDB (primary) ────────────────────────────────
            passages: list[RAGPassage] = []
            retrieval_successful       = False

            try:
                passages             = _chroma_query(embedding, TOP_K)
                retrieval_successful = True
                logger.info(
                    "medical_rag_search · ChromaDB retrieved",
                    n_passages=len(passages),
                    session_id=state.session_id,
                )
            except Exception as chroma_exc:
                logger.warning(
                    "medical_rag_search · ChromaDB failed, trying FAISS",
                    error=str(chroma_exc),
                    session_id=state.session_id,
                )
                # ── FAISS (fallback) ──────────────────────────────
                try:
                    passages             = _faiss_query(embedding, TOP_K)
                    retrieval_successful = True
                    logger.info(
                        "medical_rag_search · FAISS retrieved",
                        n_passages=len(passages),
                        session_id=state.session_id,
                    )
                except Exception as faiss_exc:
                    logger.error(
                        "medical_rag_search · both retrievers failed",
                        chroma_error=str(chroma_exc),
                        faiss_error=str(faiss_exc),
                        session_id=state.session_id,
                    )
                    return ToolError(
                        tool=TOOL_MEDICAL_RAG_SEARCH,
                        reason=(
                            f"Both ChromaDB and FAISS retrieval failed. "
                            f"ChromaDB: {chroma_exc}. "
                            f"FAISS: {faiss_exc}."
                        ),
                        fatal=False,
                    )

            citations = [p.citation for p in passages if p.citation]

            return RAGSearchResult(
                passages             = passages,
                citations            = citations,
                query_used           = query,
                retrieval_successful = retrieval_successful,
            )

        except Exception as exc:
            return ToolError(
                tool=TOOL_MEDICAL_RAG_SEARCH,
                reason=str(exc),
                fatal=False,
            )


async def search(state: AegisState) -> RAGSearchResult | ToolError:
    """Canonical functional entrypoint."""
    return await MedicalRAGSearch().run(state)


# ── Index readiness probe (used by backend/health.py) ─────────────


def is_index_ready() -> bool:
    """
    Return True if at least one retriever is available and loaded.

    Called by _probe_rag_index_ready() in backend/health.py.
    Checks ChromaDB first (primary), then FAISS (fallback).
    Does not force-load — probes existing singleton state.

    Criteria:
        ChromaDB: collection exists AND collection.count() > 0
        FAISS:    index file exists AND index.ntotal > 0
    """
    # Try ChromaDB
    try:
        collection = _get_chroma_collection()
        if collection is not None and collection.count() > 0:
            return True
    except Exception:
        pass

    # Try FAISS
    try:
        index, docs = _get_faiss()
        if index is not None and index.ntotal > 0 and docs:
            return True
    except Exception:
        pass

    return False