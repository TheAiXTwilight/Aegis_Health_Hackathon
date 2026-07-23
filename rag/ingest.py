"""
rag/ingest.py — Single-command entrypoint for the full knowledge base
build described in docs/corpus_version.md.

Runs, in order:
    1. rag/download_corpus.py  → data/knowledge/raw/corpus.json
    2. rag/build_chroma.py     → data/knowledge/chroma/  (primary retriever)
    3. rag/build_faiss.py      → data/knowledge/faiss.index / .docs (fallback)

Does NOT run rag/export_minilm_onnx.py — the ONNX model export is a
one-time, Mac-only step (see that script's docstring) and the exported
model/tokenizer are expected to already exist at data/knowledge/ before
ingest runs.

Usage:
    python rag/ingest.py                  # live fetch, fall back to seed
    python rag/ingest.py --offline        # force offline seed corpus
    python rag/ingest.py --skip-download  # reuse existing corpus.json
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the full Aegis RAG knowledge base.")
    parser.add_argument("--offline", action="store_true", help="Force offline seed corpus (no network fetch).")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing data/knowledge/raw/corpus.json instead of re-downloading.",
    )
    parser.add_argument(
        "--medquad-max-documents",
        type=int,
        default=None,
        help=(
            "Cap MedQuAD QA-pair documents ingested when it's used as the "
            "corpus source (default: no cap — all ~5,600 usable documents, "
            "~53k chunks. Measured at ~30-35 chunks/sec on this build "
            "hardware, so a full uncapped build takes roughly 25-30 minutes. "
            "Pass a smaller number for a faster iteration/demo build.)"
        ),
    )
    args = parser.parse_args()

    medquad_cap = args.medquad_max_documents if args.medquad_max_documents and args.medquad_max_documents > 0 else None

    from rag.embed import is_available as embed_is_available

    if not embed_is_available():
        logger.error(
            "ingest · ONNX MiniLM model / tokenizer not available at "
            "data/knowledge/minilm.onnx and data/knowledge/tokenizer.json. "
            "Run rag/export_minilm_onnx.py (Mac only) first, or copy the "
            "existing exported model into place."
        )
        return 1

    if not args.skip_download:
        from rag.download_corpus import build_corpus, _RAW_DIR, _OUTPUT_PATH
        import json

        documents = build_corpus(offline=args.offline, medquad_max_documents=medquad_cap)
        _RAW_DIR.mkdir(parents=True, exist_ok=True)
        _OUTPUT_PATH.write_text(json.dumps(documents, indent=2), encoding="utf-8")
        logger.info("ingest · corpus ready", n_documents=len(documents))
    else:
        logger.info("ingest · skipping download, reusing existing corpus.json")

    from rag.build_chroma import build as build_chroma

    n_chroma = build_chroma()
    if n_chroma == 0:
        logger.error("ingest · ChromaDB build failed")
        return 1

    from rag.build_faiss import build as build_faiss

    n_faiss = build_faiss()
    if n_faiss == 0:
        logger.error("ingest · FAISS build failed")
        return 1

    logger.info(
        "ingest · knowledge base build complete",
        n_chroma_vectors=n_chroma,
        n_faiss_vectors=n_faiss,
    )
    print(
        f"\nKnowledge base built successfully.\n"
        f"  ChromaDB vectors: {n_chroma}\n"
        f"  FAISS vectors:    {n_faiss}\n\n"
        f"Next step — append version metadata to docs/corpus_version.md:\n"
        f'  echo "snapshot_date: $(date -u +%Y-%m-%d)" >> docs/corpus_version.md\n'
        f'  echo "source_url: https://medlineplus.gov/xml.html" >> docs/corpus_version.md\n'
        f'  echo "git_commit: $(git rev-parse HEAD)" >> docs/corpus_version.md\n'
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
