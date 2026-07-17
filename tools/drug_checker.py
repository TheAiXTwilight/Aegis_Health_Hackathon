"""
tools/drug_checker.py — Drug interaction checker (Step 6).

Phase 3 replacement: SQLite FTS5 database replaces the in-memory
hardcoded interaction table.

Architecture:
    - Drug name resolution: FTS5 fuzzy search against rxnorm_names table
    - Interaction lookup: JOIN on rxcui pairs in interactions table
    - DB path: data/drugs/aegis_drugs.db (built by data/drugs/build_drug_db.py)
    - First-wins canonical key policy (Decision: matches alias map behaviour,
      safer clinically, deterministic)
    - Lazy DB connection — opened once, held for application lifetime
    - Graceful fallback: if DB unavailable, falls back to in-memory
      KNOWN_DRUGS / _INTERACTIONS (preserves existing test coverage)

Public interface unchanged:
    async def run(self, state: AegisState) -> DrugInteractionResult | ToolError

Fallback policy:
    If the SQLite DB is absent or fails to open, the tool logs a warning
    and falls back to the original in-memory lookup. This ensures the
    pipeline never returns ToolError solely due to a missing DB file —
    the placeholder always provides a safe floor.

SQLite DB schema (created by data/drugs/build_drug_db.py):

    CREATE TABLE rxnorm_names (
        rxcui     TEXT NOT NULL,
        name      TEXT NOT NULL,
        tty       TEXT            -- term type: IN, BN, SY, ...
    );
    CREATE VIRTUAL TABLE rxnorm_fts USING fts5(
        name,
        rxcui UNINDEXED,
        content='rxnorm_names',
        content_rowid='rowid'
    );
    CREATE TABLE interactions (
        rxcui_a   TEXT NOT NULL,
        rxcui_b   TEXT NOT NULL,
        severity  TEXT NOT NULL,   -- 'severe' | 'moderate' | 'minor'
        description TEXT NOT NULL
    );

Name resolution strategy:
    1. Exact match (lowercased) against rxnorm_names.name
    2. FTS5 prefix match: name MATCH '<drug>*'
    3. FTS5 fuzzy: name MATCH '<drug>'  (FTS5 tokenizer handles stemming)
    First match at each level wins (first-wins policy).

Duplicate canonical key policy: first-wins (Decision confirmed).
    If the same drug appears twice in medications_raw, the first
    resolved rxcui is kept and subsequent duplicates are skipped.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from loguru import logger

from schemas.drugs import (
    DrugInteraction,
    DrugInteractionResult,
    DrugInteractionSeverity,
)
from schemas.errors import ToolError
from schemas.state import AegisState
from tools.tool_names import TOOL_DRUG_INTERACTION_CHECKER


# ── DB path ────────────────────────────────────────────────────────

_DB_PATH = Path("data/drugs/aegis_drugs.db")


# ── Module-level singleton ─────────────────────────────────────────
#
# Two separate concerns, two separate variables:
#   _db_load_attempted  — have we tried to open the DB?
#   _db_conn            — the open connection, or None if unavailable
#
# A None _db_conn with _db_load_attempted=True means "load failed".
# A None _db_conn with _db_load_attempted=False means "not yet attempted".

_db_load_attempted: bool                       = False
_db_conn:           sqlite3.Connection | None  = None


def _get_db() -> sqlite3.Connection | None:
    """
    Return an open SQLite connection, or None if unavailable.

    Opens once and holds for application lifetime. Thread-safe for
    read-only queries (SQLite WAL mode not required — we never write
    from the tool, only from build_drug_db.py).
    """
    global _db_load_attempted, _db_conn

    if _db_load_attempted:
        return _db_conn

    _db_load_attempted = True

    if not _DB_PATH.exists():
        logger.warning(
            "drug_checker · DB not found · using in-memory fallback",
            path=str(_DB_PATH),
        )
        return None

    try:
        conn = sqlite3.connect(
            str(_DB_PATH),
            check_same_thread=False,  # read-only from multiple async contexts
        )
        conn.row_factory = sqlite3.Row
        # Verify expected tables are present
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {"rxnorm_names", "interactions"}
        if not required.issubset(tables):
            logger.warning(
                "drug_checker · DB missing tables · using in-memory fallback",
                found=sorted(tables),
                required=sorted(required),
            )
            conn.close()
            return None

        _db_conn = conn
        logger.info(
            "drug_checker · SQLite DB opened",
            path=str(_DB_PATH),
        )
        return conn

    except Exception as exc:
        logger.warning(
            "drug_checker · DB open failed · using in-memory fallback",
            error=str(exc),
        )
        return None

# ── In-memory fallback (preserved from placeholder) ────────────────

KNOWN_DRUGS: set[str] = {
    "warfarin",
    "aspirin",
    "ibuprofen",
    "metformin",
    "contrast dye",
    "digoxin",
    "amiodarone",
}

_INTERACTIONS_FALLBACK: dict[frozenset[str], DrugInteraction] = {
    frozenset({"warfarin", "aspirin"}): DrugInteraction(
        drugs=["warfarin", "aspirin"],
        severity=DrugInteractionSeverity.SEVERE,
        description="Warfarin + Aspirin significantly increases bleeding risk.",
    ),
    frozenset({"warfarin", "ibuprofen"}): DrugInteraction(
        drugs=["warfarin", "ibuprofen"],
        severity=DrugInteractionSeverity.SEVERE,
        description="Warfarin + Ibuprofen significantly increases bleeding risk.",
    ),
    frozenset({"aspirin", "ibuprofen"}): DrugInteraction(
        drugs=["aspirin", "ibuprofen"],
        severity=DrugInteractionSeverity.MODERATE,
        description="Concurrent NSAID use increases gastrointestinal bleeding risk.",
    ),
    frozenset({"metformin", "contrast dye"}): DrugInteraction(
        drugs=["metformin", "contrast dye"],
        severity=DrugInteractionSeverity.MODERATE,
        description="Contrast dye may increase lactic acidosis risk with Metformin.",
    ),
    frozenset({"digoxin", "amiodarone"}): DrugInteraction(
        drugs=["digoxin", "amiodarone"],
        severity=DrugInteractionSeverity.SEVERE,
        description=(
            "Amiodarone increases Digoxin concentration — "
            "narrow therapeutic index."
        ),
    ),
}


# ── SQLite name resolution ─────────────────────────────────────────

def _resolve_via_db(
    drug_name: str,
    conn: sqlite3.Connection,
) -> tuple[str, str] | None:
    """
    Resolve a drug name to (rxcui, canonical_name) via SQLite.

    Resolution order (first-wins):
        1. Exact match (case-insensitive)
        2. FTS5 prefix match: name*
        3. FTS5 token match: name (handles stemming/substring)

    Returns None when no match found at any level.
    """
    name_lower = drug_name.lower().strip()

    # Level 1: exact match
    row = conn.execute(
        "SELECT rxcui, name FROM rxnorm_names "
        "WHERE LOWER(name) = ? LIMIT 1",
        (name_lower,),
    ).fetchone()
    if row:
        return row["rxcui"], row["name"]

    # Level 2: FTS5 prefix match
    try:
        row = conn.execute(
            "SELECT rxcui, name FROM rxnorm_fts "
            "WHERE name MATCH ? LIMIT 1",
            (f'"{name_lower}"*',),
        ).fetchone()
        if row:
            return row["rxcui"], row["name"]
    except sqlite3.OperationalError:
        # FTS5 table may not be built — fall through
        pass

    # Level 3: FTS5 token match
    try:
        safe_query = name_lower.replace('"', "")
        row = conn.execute(
            "SELECT rxcui, name FROM rxnorm_fts "
            "WHERE name MATCH ? LIMIT 1",
            (safe_query,),
        ).fetchone()
        if row:
            return row["rxcui"], row["name"]
    except sqlite3.OperationalError:
        pass

    return None


def _lookup_interactions_db(
    rxcui_pairs: list[tuple[str, str, str, str]],  # (rxcui_a, name_a, rxcui_b, name_b)
    conn: sqlite3.Connection,
) -> list[DrugInteraction]:
    """
    Look up drug interactions for all pairs via SQLite.

    Queries both (rxcui_a, rxcui_b) and (rxcui_b, rxcui_a) since
    the interactions table may not guarantee ordering.
    """
    results: list[DrugInteraction] = []

    for rxcui_a, name_a, rxcui_b, name_b in rxcui_pairs:
        row = conn.execute(
            """
            SELECT severity, description FROM interactions
            WHERE (rxcui_a = ? AND rxcui_b = ?)
               OR (rxcui_a = ? AND rxcui_b = ?)
            LIMIT 1
            """,
            (rxcui_a, rxcui_b, rxcui_b, rxcui_a),
        ).fetchone()

        if row:
            try:
                severity = DrugInteractionSeverity(row["severity"].lower())
            except ValueError:
                severity = DrugInteractionSeverity.MINOR

            results.append(
                DrugInteraction(
                    drugs       = [name_a, name_b],
                    severity    = severity,
                    description = row["description"],
                )
            )

    return results


# ── In-memory fallback lookup ─────────────────────────────────────

def _run_in_memory(
    medications: list[str],
) -> tuple[list[str], list[str], list[DrugInteraction]]:
    """
    Resolve and check interactions using the in-memory fallback table.

    Returns (resolved, unresolved, interactions).
    """
    resolved_raw: list[str] = []
    unresolved:   list[str] = []

    for drug in medications:
        if drug in KNOWN_DRUGS:
            resolved_raw.append(drug)
        else:
            unresolved.append(drug)

    # First-wins deduplication
    resolved = list(dict.fromkeys(resolved_raw))

    interactions: list[DrugInteraction] = []
    for i in range(len(resolved)):
        for j in range(i + 1, len(resolved)):
            pair = frozenset({resolved[i], resolved[j]})
            if pair in _INTERACTIONS_FALLBACK:
                interactions.append(_INTERACTIONS_FALLBACK[pair])

    return resolved, unresolved, interactions


# ── SQLite-backed lookup ───────────────────────────────────────────

def _run_via_db(
    medications: list[str],
    conn: sqlite3.Connection,
) -> tuple[list[str], list[str], list[DrugInteraction]]:
    """
    Resolve drug names and check interactions via SQLite.

    Returns (resolved_names, unresolved_names, interactions).
    First-wins deduplication: if the same rxcui resolves from two
    different input strings, only the first is kept.
    """
    resolved_names: list[str]                  = []
    resolved_pairs: list[tuple[str, str, str]] = []  # (input_name, rxcui, canonical)
    unresolved:     list[str]                  = []
    seen_rxcui:     set[str]                   = set()

    for drug in medications:
        match = _resolve_via_db(drug, conn)
        if match:
            rxcui, canonical = match
            if rxcui not in seen_rxcui:
                seen_rxcui.add(rxcui)
                resolved_names.append(canonical)
                resolved_pairs.append((drug, rxcui, canonical))
        else:
            unresolved.append(drug)

    # Build all pairs for interaction lookup
    pair_args: list[tuple[str, str, str, str]] = []
    for i in range(len(resolved_pairs)):
        for j in range(i + 1, len(resolved_pairs)):
            _, rxcui_a, name_a = resolved_pairs[i]
            _, rxcui_b, name_b = resolved_pairs[j]
            pair_args.append((rxcui_a, name_a, rxcui_b, name_b))

    interactions = _lookup_interactions_db(pair_args, conn) if pair_args else []

    return resolved_names, unresolved, interactions


# ── Tool ──────────────────────────────────────────────────────────

class DrugInteractionChecker:
    """
    SQLite FTS5-backed drug interaction checker.

    Falls back to in-memory table if DB is unavailable.
    Does not write to state — pipeline owns state mutation.
    """

    TOOL_NAME = TOOL_DRUG_INTERACTION_CHECKER

    async def run(
        self,
        state: AegisState,
    ) -> DrugInteractionResult | ToolError:

        try:
            medications = [
                drug.strip().lower()
                for drug in state.medications_raw
                if drug.strip()
            ]

            conn = _get_db()

            if conn is not None:
                resolved, unresolved, interactions = _run_via_db(medications, conn)
                source = "sqlite"
            else:
                resolved, unresolved, interactions = _run_in_memory(medications)
                source = "in-memory-fallback"

            logger.info(
                "drug_checker · resolved",
                session_id=state.session_id,
                source=source,
                resolved_count=len(resolved),
                unresolved_count=len(unresolved),
                interactions_count=len(interactions),
            )

            warnings: list[str] = []

            if unresolved:
                warnings.append(
                    f"{len(unresolved)} medication(s) could not be resolved: "
                    + ", ".join(unresolved)
                )

            if interactions:
                warnings.append(
                    f"{len(interactions)} potential drug interaction(s) detected."
                )

            total      = len(resolved) + len(unresolved)
            confidence = len(resolved) / total if total > 0 else 0.0

            return DrugInteractionResult(
                resolved     = resolved,
                unresolved   = unresolved,
                interactions = interactions,
                warnings     = warnings,
                confidence   = confidence,
            )

        except Exception as exc:
            return ToolError(
                tool=TOOL_DRUG_INTERACTION_CHECKER,
                reason=str(exc),
                fatal=False,
            )


async def check(state: AegisState) -> DrugInteractionResult | ToolError:
    """Canonical functional entrypoint."""
    return await DrugInteractionChecker().run(state)