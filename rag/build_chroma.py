"""
rag/build_chroma.py — Step 3 of the knowledge base build (see
docs/corpus_version.md).

Reads data/knowledge/raw/corpus.json, chunks it (rag/chunk.py), embeds
each chunk with the ONNX MiniLM model (rag/embed.py), and writes the
result into a persistent ChromaDB collection at data/knowledge/chroma/.

Collection name: "aegis_knowledge" — MUST match
tools/medical_rag_search.py's _COLLECTION_NAME.

Usage:
    python rag/build_chroma.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

from rag.chunk import chunk_corpus
from rag.embed import embed_batch

_RAW_CORPUS_PATH = Path("data/knowledge/raw/corpus.json")
_CHROMA_DIR = Path("data/knowledge/chroma")
_COLLECTION_NAME = "aegis_knowledge"


def load_corpus() -> list[dict]:
    if not _RAW_CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"{_RAW_CORPUS_PATH} not found. Run rag/download_corpus.py first."
        )
    return json.loads(_RAW_CORPUS_PATH.read_text(encoding="utf-8"))


def build() -> int:
    import chromadb

    documents = load_corpus()
    chunks = chunk_corpus(documents)
    if not chunks:
        logger.error("build_chroma · no chunks produced from corpus, aborting")
        return 0

    logger.info("build_chroma · chunked corpus", n_documents=len(documents), n_chunks=len(chunks))

    embeddings = embed_batch([c["text"] for c in chunks])

    valid_chunks = []
    valid_embeddings = []
    for chunk, emb in zip(chunks, embeddings):
        if emb is None:
            logger.warning("build_chroma · skipping chunk with failed embedding", chunk_id=chunk["id"])
            continue
        valid_chunks.append(chunk)
        valid_embeddings.append(emb.tolist())

    if not valid_chunks:
        logger.error("build_chroma · all embeddings failed, aborting (check ONNX model/tokenizer)")
        return 0

    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(_CHROMA_DIR))

    # Fresh build — drop any existing collection with the same name so
    # re-running this script is idempotent rather than appending duplicates.
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(_COLLECTION_NAME)

    # Chroma enforces a max batch size per add() call (raises
    # InternalError above it) — chunk the insert to stay under that
    # limit rather than assuming the whole corpus fits in one call.
    ids = [c["id"] for c in valid_chunks]
    documents_text = [c["text"] for c in valid_chunks]
    metadatas = [
        {"source": c["source"], "citation": c["citation"], "topic": c["topic"]}
        for c in valid_chunks
    ]

    try:
        add_batch_size = client.get_max_batch_size()
    except Exception:
        add_batch_size = 5000  # conservative fallback if the API isn't available

    for start in range(0, len(valid_chunks), add_batch_size):
        end = start + add_batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=valid_embeddings[start:end],
            documents=documents_text[start:end],
            metadatas=metadatas[start:end],
        )
        logger.info(
            "build_chroma · inserted batch",
            start=start,
            end=min(end, len(valid_chunks)),
            total=len(valid_chunks),
        )

    logger.info(
        "build_chroma · collection built",
        name=_COLLECTION_NAME,
        n_vectors=collection.count(),
        path=str(_CHROMA_DIR),
    )
    return collection.count()


if __name__ == "__main__":
    n = build()
    sys.exit(0 if n > 0 else 1)
