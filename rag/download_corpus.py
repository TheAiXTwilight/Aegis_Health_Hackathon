"""
rag/download_corpus.py — Step 1 of the knowledge base build (see
docs/corpus_version.md).

Attempts to pull the MedlinePlus "Health Topics" XML export
(https://medlineplus.gov/xml.html) and convert it into the corpus JSON
format consumed by rag/chunk.py. If that source is unreachable — for
example in a network-restricted build environment — falls back to the
bundled offline seed corpus in rag/seed_corpus.py so the rest of the
pipeline (chunking, embedding, index build) can still run end-to-end.

Output:
    data/knowledge/raw/corpus.json
        [
          {"topic": ..., "source": ..., "citation": ..., "text": ...},
          ...
        ]

Usage:
    python rag/download_corpus.py
    python rag/download_corpus.py --offline     # skip network attempt

Environment:
    AEGIS_RAG_OFFLINE=1   also forces offline mode
    MEDLINEPLUS_SOURCE_URL  override the source URL (default below)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from loguru import logger

_RAW_DIR = Path("data/knowledge/raw")
_OUTPUT_PATH = _RAW_DIR / "corpus.json"
_DEFAULT_SOURCE_URL = "https://medlineplus.gov/xml.html"


def _is_offline_forced(cli_offline: bool) -> bool:
    if cli_offline:
        return True
    return os.getenv("AEGIS_RAG_OFFLINE", "").lower() in ("1", "true", "yes")


def _try_fetch_medlineplus(source_url: str) -> list[dict[str, str]] | None:
    """
    Attempt to fetch and parse the MedlinePlus health topics XML feed.

    Returns None on any failure (network error, unexpected schema, etc.)
    so the caller can fall back to the offline seed corpus. This keeps
    the build reproducible in restricted network environments while
    still supporting a real pull where MedlinePlus is reachable.
    """
    try:
        import httpx
    except ImportError:
        logger.warning("download_corpus · httpx not installed, cannot fetch live corpus")
        return None

    try:
        logger.info("download_corpus · attempting live MedlinePlus fetch", url=source_url)
        resp = httpx.get(source_url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        documents: list[dict[str, str]] = []
        for i, entry in enumerate(root.findall(".//health-topic")):
            title = entry.get("title") or entry.findtext("title") or ""
            summary = entry.findtext("full-summary") or entry.findtext("summary") or ""
            if not title or not summary:
                continue
            citation = f"MEDLINE-{title.upper().replace(' ', '')[:24]}-{i:03d}"
            documents.append(
                {
                    "topic": title.strip(),
                    "source": "MedlinePlus",
                    "citation": citation,
                    "text": summary.strip(),
                }
            )

        if not documents:
            logger.warning("download_corpus · live fetch parsed 0 documents, falling back")
            return None

        logger.info("download_corpus · live fetch succeeded", n_documents=len(documents))
        return documents

    except Exception as exc:
        logger.warning("download_corpus · live fetch failed, falling back to offline seed", error=str(exc))
        return None


def _load_offline_seed() -> list[dict[str, str]]:
    from rag.seed_corpus import SEED_TOPICS

    documents = [
        {
            "topic": t["topic"],
            "source": "Aegis Offline Seed Corpus",
            "citation": t["citation"],
            "text": t["text"],
        }
        for t in SEED_TOPICS
    ]
    logger.info("download_corpus · using offline seed corpus", n_documents=len(documents))
    return documents


def _load_medquad(max_documents: int | None) -> list[dict[str, str]] | None:
    """
    Try the bundled MedQuAD corpus (NIH-derived, CC BY 4.0, see
    rag/medquad_source.py). Returns None if no local checkout is
    available so the caller can fall back further.
    """
    from rag.medquad_source import is_available, load_medquad_corpus

    if not is_available():
        logger.info("download_corpus · no local MedQuAD checkout found, skipping")
        return None

    documents = load_medquad_corpus(max_documents=max_documents)
    if not documents:
        return None

    logger.info("download_corpus · using MedQuAD corpus", n_documents=len(documents))
    return documents


def build_corpus(
    offline: bool = False,
    source_url: str | None = None,
    medquad_max_documents: int | None = None,
) -> list[dict[str, str]]:
    source_url = source_url or os.getenv("MEDLINEPLUS_SOURCE_URL", _DEFAULT_SOURCE_URL)

    # Priority order: bundled MedQuAD first — it's real, NIH-sourced,
    # and available offline/deterministically, so it beats attempting
    # a flaky live fetch. Live MedlinePlus fetch is tried only if no
    # local MedQuAD checkout exists. Hand-written seed corpus is the
    # last-resort fallback if neither real source is available.
    documents: list[dict[str, str]] | None = None
    if not offline:
        documents = _load_medquad(medquad_max_documents)

    if documents is None and not offline:
        documents = _try_fetch_medlineplus(source_url)

    if documents is None and offline:
        # --offline still allows the bundled (no-network) MedQuAD source,
        # just skips the live fetch attempt.
        documents = _load_medquad(medquad_max_documents)

    if documents is None:
        documents = _load_offline_seed()

    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description="Download or seed the Aegis RAG corpus.")
    parser.add_argument("--offline", action="store_true", help="Skip network fetch, use bundled seed corpus.")
    parser.add_argument("--source-url", default=None, help="Override MedlinePlus source URL.")
    parser.add_argument(
        "--medquad-max-documents",
        type=int,
        default=None,
        help="Cap the number of MedQuAD QA-pair documents ingested (default: no cap).",
    )
    args = parser.parse_args()

    offline = _is_offline_forced(args.offline)
    documents = build_corpus(
        offline=offline,
        source_url=args.source_url,
        medquad_max_documents=args.medquad_max_documents,
    )

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(documents, indent=2), encoding="utf-8")

    logger.info(
        "download_corpus · wrote corpus",
        path=str(_OUTPUT_PATH),
        n_documents=len(documents),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
