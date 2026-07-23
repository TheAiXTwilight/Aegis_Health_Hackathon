"""
backend/cache.py — In-memory LRU Result Cache.

Spec 9.3.15:
- Capacity: 128 entries.
- Key from normalized symptoms, medications, X-ray findings, and major lab signals.
- Do not store patient identity in cache values.
- Rehydrate current patient header on cache hit.
- Add cache hit/miss metrics.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any
from loguru import logger
from schemas.state import AegisState


class CacheEntry:
    def __init__(self, value: dict[str, Any]):
        self.value = value
        self.created_at = time.time()


class ResultCache:
    def __init__(self, max_entries: int = 128):
        self.max_entries = max_entries
        self._data: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        if key not in self._data:
            self.misses += 1
            return None
        self.hits += 1
        self._data.move_to_end(key)
        entry = self._data[key]
        logger.info("result_cache · hit", key=key[:12], hits=self.hits, misses=self.misses)
        return entry.value.copy()

    def set(self, key: str, value: dict[str, Any]) -> None:
        blocked = {"patient", "user", "user_id", "patient_name", "patient_dob", "email"}
        sanitized = {k: v for k, v in value.items() if k not in blocked}
        self._data[key] = CacheEntry(value=sanitized)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)
        logger.info("result_cache · set", key=key[:12], cache_size=len(self._data))

    def delete(self, key: str | None) -> bool:
        """Evict one entry by exact key. No-op (returns False) if key is
        None/empty or not present — callers don't need to pre-check."""
        if not key or key not in self._data:
            return False
        del self._data[key]
        logger.info("result_cache · evicted", key=key[:12], cache_size=len(self._data))
        return True

    def clear(self) -> None:
        self._data.clear()
        self.hits = 0
        self.misses = 0
        logger.info("result_cache · cleared")


result_cache = ResultCache(max_entries=128)


def compute_cache_key(state: AegisState) -> str:
    """
    sha256(symptoms + medications + xray_findings + lab_file_hash).hexdigest()
    """
    symptoms = (
        getattr(state, "submitted_symptoms_text", None)
        or state.raw_symptoms_text
        or ""
    ).strip().lower()

    meds = ",".join(
        sorted([m.strip().lower() for m in state.medications_raw if m.strip()])
    )
    xray = ",".join(
        sorted([f.strip().lower() for f in state.xray_findings_raw if f.strip()])
    )

    # ── Lab signal: hash actual file bytes for every uploaded PDF ──────
    lab_pdf_path = getattr(state, "lab_pdf_path", None)

    # Normalise: could be None, a single path string, or a list of paths
    if not lab_pdf_path:
        pdf_paths = []
    elif isinstance(lab_pdf_path, list):
        pdf_paths = [p for p in lab_pdf_path if p]
    else:
        pdf_paths = [lab_pdf_path]

    if pdf_paths:
        file_hasher = hashlib.sha256()
        for path in sorted(pdf_paths):          # sort so order doesn't matter
            try:
                with open(path, "rb") as fh:
                    file_hasher.update(fh.read())
            except (OSError, IOError):
                file_hasher.update(b"unreadable")
        lab_hash = file_hasher.hexdigest()
    else:
        lab_hash = "no_lab"
    # ───────────────────────────────────────────────────────────────────

    raw = f"{symptoms}|{meds}|{xray}|{lab_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()