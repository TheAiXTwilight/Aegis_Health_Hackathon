"""
rag/medquad_source.py — Offline corpus source built from MedQuAD.

MedQuAD (Ben Abacha & Demner-Fushman, 2019) is a NIH-derived dataset of
medical question-answer pairs collected from 12 NIH/gov health sites,
including MedlinePlus Health Topics. It is distributed under CC BY 4.0
at:

    https://github.com/abachaa/MedQuAD

This module parses a local checkout of that repository into the same
corpus.json schema used by rag/download_corpus.py and rag/chunk.py:

    {"topic": ..., "source": ..., "citation": ..., "text": ...}

Three subsets (ADAM, MedlinePlus Drugs, MedlinePlus Herbs/Supplements)
ship with their <Answer> text stripped in the public MedQuAD release,
to respect MedlinePlus copyright restrictions on those specific pages
per the dataset's own README. Those subsets are skipped here — only
folders with real answer text are ingested.

This is an OFFLINE, BUNDLED source: it requires a local clone of the
MedQuAD repo (see MEDQUAD_LOCAL_PATH below) and makes no network calls
of its own. It sits between the live MedlinePlus XML fetch and the
hand-written seed_corpus.py in download_corpus.py's fallback chain —
richer than the seed corpus, but not a live pull.
"""

from __future__ import annotations

import os
from pathlib import Path
from xml.etree import ElementTree as ET

from loguru import logger

# Subsets with real (non-stripped) answer text.
_INCLUDED_SUBSETS = {
    "1_CancerGov_QA": "NIH National Cancer Institute",
    "2_GARD_QA": "NIH Genetic and Rare Diseases Info Center",
    "3_GHR_QA": "NIH Genetics Home Reference",
    "4_MPlus_Health_Topics_QA": "MedlinePlus Health Topics",
    "5_NIDDK_QA": "NIH NIDDK",
    "6_NINDS_QA": "NIH NINDS",
    "7_SeniorHealth_QA": "NIH SeniorHealth",
    "8_NHLBI_QA_XML": "NIH NHLBI",
    "9_CDC_QA": "CDC",
}

# Explicitly excluded — answers stripped in the public release to
# respect MedlinePlus copyright (see MedQuAD README).
_EXCLUDED_SUBSETS = {
    "10_MPlus_ADAM_QA",
    "11_MPlusDrugs_QA",
    "12_MPlusHerbsSupplements_QA",
}

_DEFAULT_LOCAL_PATH = Path("data/knowledge/sources/medquad")


def _local_path() -> Path:
    override = os.getenv("MEDQUAD_LOCAL_PATH")
    return Path(override) if override else _DEFAULT_LOCAL_PATH


def is_available() -> bool:
    """True if a local MedQuAD checkout is present."""
    root = _local_path()
    return root.exists() and any(root.glob("*/*.xml"))


def _parse_document(path: Path, source_label: str, subset_key: str) -> list[dict[str, str]]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        logger.warning("medquad_source · failed to parse", path=str(path), error=str(exc))
        return []

    doc_id = path.stem  # filename stem — guaranteed unique within a subset,
    # unlike the internal Document id attribute, which some subsets (e.g.
    # CancerGov) reuse across multiple split sub-document files.
    url = root.get("url") or ""
    focus = root.findtext("Focus") or ""

    out: list[dict[str, str]] = []
    for qa in root.findall(".//QAPair"):
        answer = (qa.findtext("Answer") or "").strip()
        question = (qa.findtext("Question") or "").strip()
        pid = qa.get("pid") or "0"
        if not answer:
            continue  # stripped/empty per MedQuAD license note

        topic = focus.strip() or question or f"MedQuAD Document {doc_id}"
        citation = f"MEDQUAD-{subset_key}-{doc_id}-{pid}"
        text = answer
        if question:
            # Keep the question as context prefix; retrieval quality
            # benefits from it, and it's short relative to answer text.
            text = f"{question.strip()} {answer}"

        out.append(
            {
                "topic": topic,
                "source": source_label,
                "citation": citation,
                "text": text,
                "source_url": url,
            }
        )
    return out


def load_medquad_corpus(max_documents: int | None = None) -> list[dict[str, str]]:
    """
    Parse all included MedQuAD subsets from the local checkout into
    corpus documents. Returns [] if no local checkout is found.

    max_documents caps the total number of QA-pair documents returned.
    None means no cap.

    IMPORTANT: when a cap is set, the cap is applied per-subset in
    proportion to each subset's full (uncapped) share of QA-pair
    documents, not sequentially subset-by-subset and not by input
    file count. Subsets vary widely in QA-pairs-per-file (e.g. GARD's
    genetic-disease entries produce many more pairs per file than
    others), so capping by file count does not translate to a
    proportional document cap — capping the actual output count per
    subset is what's needed to keep every subset represented at any
    cap size, including clinically central ones like MedlinePlus
    Health Topics.
    """
    root = _local_path()
    if not root.exists():
        logger.warning("medquad_source · local checkout not found", path=str(root))
        return []

    # First pass: parse every included subset in full so we know each
    # subset's true QA-pair-document yield before capping.
    per_subset_docs: dict[str, list[dict[str, str]]] = {}
    for subset_dir, source_label in _INCLUDED_SUBSETS.items():
        subset_path = root / subset_dir
        if not subset_path.exists():
            continue
        docs: list[dict[str, str]] = []
        for xml_path in sorted(subset_path.glob("*.xml")):
            docs.extend(_parse_document(xml_path, source_label, subset_dir))
        if docs:
            per_subset_docs[subset_dir] = docs

    total_docs = sum(len(docs) for docs in per_subset_docs.values())
    if total_docs == 0:
        return []

    if max_documents is not None and max_documents < total_docs:
        documents: list[dict[str, str]] = []
        for subset_dir, docs in per_subset_docs.items():
            share = len(docs) / total_docs
            # At least 1 doc per subset so nothing is silently dropped.
            take = max(1, round(max_documents * share))
            documents.extend(docs[:take])
        if len(documents) > max_documents:
            documents = documents[:max_documents]
    else:
        documents = [d for docs in per_subset_docs.values() for d in docs]

    logger.info(
        "medquad_source · parsed corpus",
        n_documents=len(documents),
        n_subsets=len(_INCLUDED_SUBSETS),
        excluded_subsets=sorted(_EXCLUDED_SUBSETS),
    )
    return documents
