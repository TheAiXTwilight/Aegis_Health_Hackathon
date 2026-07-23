"""
rag/chunk.py — Splits raw corpus documents into retrieval-sized chunks.

Input:
    A list of raw documents, each a dict:
        {
            "topic":    "Chest Pain",              # MedlinePlus-style topic title
            "source":   "MedlinePlus",              # display source
            "citation": "MEDLINE-CHESTPAIN-001",    # stable citation id
            "text":     "<full topic body text>",
        }

Output:
    A list of chunk dicts, one per retrieval unit:
        {
            "id":       "MEDLINE-CHESTPAIN-001::0",
            "text":     "<chunk text>",
            "source":   "MedlinePlus",
            "citation": "MEDLINE-CHESTPAIN-001",
            "topic":    "Chest Pain",
        }

Chunking strategy:
    Sentence-aware sliding window. Splits on sentence boundaries first,
    then packs sentences into chunks up to CHUNK_MAX_CHARS, with
    CHUNK_OVERLAP_SENTENCES sentences repeated between consecutive
    chunks so retrieval doesn't lose context at a boundary.

    This is intentionally simple (no NLP dependency) — regex sentence
    splitting is sufficient for the structured, short-paragraph style
    of MedlinePlus topic summaries used in this corpus.
"""

from __future__ import annotations

import re
from typing import Any

CHUNK_MAX_CHARS = 600
CHUNK_OVERLAP_SENTENCES = 1
CHUNK_MIN_CHARS = 40  # drop trailing fragments shorter than this

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> list[str]:
    """Split a text block into sentences using a lightweight regex."""
    text = " ".join(text.split())  # normalise whitespace
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Chunk a single raw document into overlapping sentence-packed windows.

    Returns [] if the document has no usable text.
    """
    text = (doc.get("text") or "").strip()
    if not text:
        return []

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_len = 0
    idx = 0

    def _flush() -> None:
        nonlocal current, current_len, idx
        if not current:
            return
        chunk_text = " ".join(current).strip()
        if len(chunk_text) >= CHUNK_MIN_CHARS:
            chunks.append(
                {
                    "id": f"{doc['citation']}::{idx}",
                    "text": chunk_text,
                    "source": doc.get("source", "Unknown"),
                    "citation": doc["citation"],
                    "topic": doc.get("topic", ""),
                }
            )
            idx += 1

    for sentence in sentences:
        if current_len + len(sentence) + 1 > CHUNK_MAX_CHARS and current:
            _flush()
            # start next window with overlap from the tail of the previous one
            current = current[-CHUNK_OVERLAP_SENTENCES:] if CHUNK_OVERLAP_SENTENCES else []
            current_len = sum(len(s) + 1 for s in current)

        current.append(sentence)
        current_len += len(sentence) + 1

    _flush()
    return chunks


def chunk_corpus(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chunk every document in the corpus. Order is preserved."""
    all_chunks: list[dict[str, Any]] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


if __name__ == "__main__":
    # Quick manual smoke test
    sample = {
        "topic": "Chest Pain",
        "source": "MedlinePlus",
        "citation": "MEDLINE-CHESTPAIN-001",
        "text": (
            "Chest pain can be a symptom of many conditions, some of which "
            "can be serious. Not all chest pain is a heart attack. However, "
            "you should seek emergency care if chest pain is severe, comes "
            "with shortness of breath, or radiates to the arm or jaw."
        ),
    }
    for c in chunk_document(sample):
        print(c)
