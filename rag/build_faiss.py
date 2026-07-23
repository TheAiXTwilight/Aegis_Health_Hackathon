"""
rag/build_faiss.py — Step 4 of the knowledge base build (see
docs/corpus_version.md).

Builds the FAISS fallback index used by tools/medical_rag_search.py
when ChromaDB is unavailable or a query fails. Reads the same
corpus/chunk/embed path as rag/build_chroma.py so both retrievers are
built from identical vectors.

Writes:
    data/knowledge/faiss.index   FAISS IndexFlatIP (cosine via
                                  pre-normalised vectors)
    data/knowledge/faiss.docs    JSON-lines doc store, one line per
                                  vector, row order matches the index

Usage:
    python rag/build_faiss.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from rag.chunk import chunk_corpus
from rag.embed import embed_batch

_RAW_CORPUS_PATH = Path("data/knowledge/raw/corpus.json")
_FAISS_INDEX_PATH = Path("data/knowledge/faiss.index")
_FAISS_DOCS_PATH = Path("data/knowledge/faiss.docs")
_EMBED_DIM = 384


def load_corpus() -> list[dict]:
    if not _RAW_CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"{_RAW_CORPUS_PATH} not found. Run rag/download_corpus.py first."
        )
    return json.loads(_RAW_CORPUS_PATH.read_text(encoding="utf-8"))


def build() -> int:
    import faiss

    documents = load_corpus()
    chunks = chunk_corpus(documents)
    if not chunks:
        logger.error("build_faiss · no chunks produced from corpus, aborting")
        return 0

    logger.info("build_faiss · chunked corpus", n_documents=len(documents), n_chunks=len(chunks))

    embeddings = embed_batch([c["text"] for c in chunks])

    valid_chunks = []
    valid_vectors = []
    for chunk, emb in zip(chunks, embeddings):
        if emb is None:
            logger.warning("build_faiss · skipping chunk with failed embedding", chunk_id=chunk["id"])
            continue
        valid_chunks.append(chunk)
        valid_vectors.append(emb.astype(np.float32))

    if not valid_chunks:
        logger.error("build_faiss · all embeddings failed, aborting (check ONNX model/tokenizer)")
        return 0

    matrix = np.vstack(valid_vectors).astype(np.float32)

    # Vectors are already L2-normalised by rag/embed.py's mean-pool step,
    # so inner product is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(_EMBED_DIM)
    index.add(matrix)

    _FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(_FAISS_INDEX_PATH))

    with _FAISS_DOCS_PATH.open("w", encoding="utf-8") as f:
        for chunk in valid_chunks:
            f.write(
                json.dumps(
                    {
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "citation": chunk["citation"],
                        "topic": chunk["topic"],
                    }
                )
                + "\n"
            )

    logger.info(
        "build_faiss · index built",
        n_vectors=index.ntotal,
        index_path=str(_FAISS_INDEX_PATH),
        docs_path=str(_FAISS_DOCS_PATH),
    )
    return index.ntotal


if __name__ == "__main__":
    n = build()
    sys.exit(0 if n > 0 else 1)
