"""
tools/corpus_version.py — Knowledge base version reader (Phase 3).

Reads version metadata from docs/corpus_version.md and exposes
it for injection into TriageReport.knowledge_base_version and
TriageReport.knowledge_base_date.

Usage (in tools/report_generator.py):

    from tools.corpus_version import get_corpus_version, get_corpus_date

    TriageReport(
        ...
        knowledge_base_version = get_corpus_version(),
        knowledge_base_date    = get_corpus_date(),
    )

docs/corpus_version.md format (required fields):

    snapshot_date: YYYY-MM-DD
    source_url: https://medlineplus.gov/xml.html
    git_commit: <full sha>

Parsing:
    - Lines are scanned for key: value pairs.
    - Keys are case-insensitive.
    - The file may contain any other Markdown content (headings,
      paragraphs, tables) — only lines matching "key: value" are parsed.
    - Both `snapshot_date` and `git_commit` are read; others are ignored
      (source_url is metadata but not surfaced in TriageReport).

Return policy:
    - get_corpus_version() returns the git_commit sha, or None.
    - get_corpus_date()    returns the snapshot_date string, or None.
    - Both return None if the file is absent, malformed, or if the
      required field is not yet populated.

Caching:
    Values are read once at first call and cached for application
    lifetime. A separate _cache_loaded flag tracks whether load has
    been attempted, so a successful load returning None for a missing
    field is distinguishable from "not yet attempted". This separation
    keeps the cached values type-pure (str | None) and avoids sentinel
    values that confuse type checkers.

    Call _reset_cache() in tests that need to reload.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger


# ── Path ───────────────────────────────────────────────────────────

_CORPUS_VERSION_PATH = Path("docs/corpus_version.md")


# ── Module-level cache ─────────────────────────────────────────────
#
# Two separate concerns, two separate variables:
#   _cache_loaded     — has load been attempted?
#   _cached_version   — what value did we load? (str | None)
#   _cached_date      — what value did we load? (str | None)
#
# A None value with _cache_loaded=True means "load succeeded, field absent".
# A None value with _cache_loaded=False means "load not yet attempted".

_cache_loaded:   bool       = False
_cached_version: str | None = None
_cached_date:    str | None = None


def _reset_cache() -> None:
    """Reset cache — for use in tests only."""
    global _cache_loaded, _cached_version, _cached_date
    _cache_loaded   = False
    _cached_version = None
    _cached_date    = None


# ── Parser ─────────────────────────────────────────────────────────

def _parse_corpus_version_file() -> dict[str, str]:
    """
    Parse docs/corpus_version.md and return key→value dict.

    Scans every line for the pattern "key: value" where key contains
    no spaces. Returns only lines that match this pattern. All keys
    are lowercased.

    Returns empty dict if the file does not exist or cannot be read.
    """
    if not _CORPUS_VERSION_PATH.exists():
        logger.debug(
            "corpus_version · file not found",
            path=str(_CORPUS_VERSION_PATH),
        )
        return {}

    fields: dict[str, str] = {}

    try:
        for line in _CORPUS_VERSION_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ": " in line:
                key, _, value = line.partition(": ")
                key   = key.strip().lower()
                value = value.strip()
                # Only simple keys (no spaces) are version metadata fields.
                if key and " " not in key and value:
                    fields[key] = value
    except Exception as exc:
        logger.warning(
            "corpus_version · read failed",
            path=str(_CORPUS_VERSION_PATH),
            error=str(exc),
        )

    return fields


def _ensure_loaded() -> None:
    """
    Load version and date from corpus_version.md on first call.

    Subsequent calls are no-ops. Thread-safe in CPython due to the GIL,
    but not safe across processes — that is acceptable since the cache
    is per-process and the cost of a redundant load is small.
    """
    global _cache_loaded, _cached_version, _cached_date

    if _cache_loaded:
        return

    fields = _parse_corpus_version_file()
    _cached_version = fields.get("git_commit")
    _cached_date    = fields.get("snapshot_date")
    _cache_loaded   = True

    if _cached_version:
        logger.info(
            "corpus_version · loaded",
            git_commit=_cached_version[:12],
            snapshot_date=_cached_date,
        )
    else:
        logger.debug(
            "corpus_version · git_commit not found in file",
            path=str(_CORPUS_VERSION_PATH),
            fields_found=list(fields.keys()),
        )


# ── Public API ─────────────────────────────────────────────────────

def get_corpus_version() -> str | None:
    """
    Return the git commit SHA of the knowledge base corpus, or None.

    Reads docs/corpus_version.md on first call, caches for lifetime.
    """
    _ensure_loaded()
    return _cached_version


def get_corpus_date() -> str | None:
    """
    Return the snapshot date of the knowledge base corpus, or None.

    Reads docs/corpus_version.md on first call, caches for lifetime.
    """
    _ensure_loaded()
    return _cached_date