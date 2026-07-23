"""
tools/lab_report_parser.py — Universal lab report parser.

[full docstring same as before, omitted here for brevity — keep the one
you already have]
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from loguru import logger

from schemas.errors import ToolError
from schemas.lab import LabReportResult
from schemas.state import AegisState
from tools.biomarker_knowledge import resolve_canonical_key
from tools.lab_constants import (
    LAB_KEY_CREATININE,
    LAB_KEY_GLUCOSE,
    LAB_KEY_HAEMOGLOBIN,
    LAB_KEY_PLATELETS,
    LAB_KEY_POTASSIUM,
    LAB_KEY_SODIUM,
    LAB_KEY_TROPONIN,
    LAB_KEY_WBC,
)
from tools.lab_thresholds import (
    ABNORMAL_HIGH_GLUCOSE_MG_DL,
    ABNORMAL_HIGH_POTASSIUM_MMOL_L,
    ABNORMAL_HIGH_TROPONIN_NG_ML,
    ABNORMAL_LOW_HAEMOGLOBIN_G_DL,
    CANONICAL_UNITS,
    REFERENCE_RANGES,
)
from tools.tool_names import TOOL_LAB_REPORT_PARSER


# ── PDF detection ─────────────────────────────────────────────────

_PDF_MAGIC = b"%PDF"


def _is_real_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == _PDF_MAGIC
    except OSError:
        return False


# ── Text extraction waterfall ─────────────────────────────────────

def _extract_via_pymupdf(path: Path) -> Optional[str]:
    try:
        import fitz

        text_parts: list[str] = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                text_parts.append(page.get_text("text", sort=True))

        text = "\n".join(text_parts).strip()
        if text:
            logger.debug(
                "lab_report_parser · PyMuPDF extraction succeeded",
                path=str(path), chars=len(text),
            )
            return text
        return None

    except Exception as exc:
        logger.warning(
            "lab_report_parser · PyMuPDF extraction failed",
            path=str(path), error=str(exc),
        )
        return None


def _extract_via_pdfminer(path: Path) -> Optional[str]:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract

        text = pdfminer_extract(str(path)).strip()
        if text:
            logger.debug(
                "lab_report_parser · pdfminer extraction succeeded",
                path=str(path), chars=len(text),
            )
            return text
        return None

    except Exception as exc:
        logger.warning(
            "lab_report_parser · pdfminer extraction failed",
            path=str(path), error=str(exc),
        )
        return None


def _extract_via_easyocr(path: Path) -> Optional[str]:
    try:
        import fitz
        import easyocr
        import numpy as np

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        text_parts: list[str] = []

        with fitz.open(str(path)) as doc:
            for page_num, page in enumerate(doc):
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                results = reader.readtext(img, detail=0)
                page_text = " ".join(results)
                if page_text.strip():
                    text_parts.append(page_text)

        text = "\n".join(text_parts).strip()
        if text:
            logger.info(
                "lab_report_parser · EasyOCR extraction succeeded",
                path=str(path), chars=len(text),
            )
            return text
        return None

    except Exception as exc:
        logger.warning(
            "lab_report_parser · EasyOCR extraction failed",
            path=str(path), error=str(exc),
        )
        return None


def _ocr_enabled() -> bool:
    return os.environ.get("AEGIS_OCR", "").lower() in {"1", "true", "yes"}


def _extract_pdf_text(path: Path) -> Optional[str]:
    text = _extract_via_pymupdf(path)
    if text is not None:
        return text
    text = _extract_via_pdfminer(path)
    if text is not None:
        return text
    if _ocr_enabled():
        return _extract_via_easyocr(path)
    return None


# ── Abnormal value detection (backward-compat) ────────────────────

def _detect_abnormal(key: str, value: float) -> str | None:
    if key == LAB_KEY_HAEMOGLOBIN and value < ABNORMAL_LOW_HAEMOGLOBIN_G_DL:
        return f"Low haemoglobin: {value} g/dL (threshold < {ABNORMAL_LOW_HAEMOGLOBIN_G_DL})"
    if key == LAB_KEY_GLUCOSE and value > ABNORMAL_HIGH_GLUCOSE_MG_DL:
        return f"High glucose: {value} mg/dL (threshold > {ABNORMAL_HIGH_GLUCOSE_MG_DL})"
    if key == LAB_KEY_POTASSIUM and value > ABNORMAL_HIGH_POTASSIUM_MMOL_L:
        return f"High potassium: {value} mmol/L (threshold > {ABNORMAL_HIGH_POTASSIUM_MMOL_L})"
    if key == LAB_KEY_TROPONIN and value > ABNORMAL_HIGH_TROPONIN_NG_ML:
        return f"Elevated troponin: {value} ng/mL (threshold > {ABNORMAL_HIGH_TROPONIN_NG_ML})"
    if key == "total_cholesterol" and value >= 200:
        return f"High total cholesterol: {value} mg/dL (desirable below 200)"
    if key == "ldl_cholesterol" and value >= 130:
        return f"High LDL cholesterol: {value} mg/dL (optimal usually < 100; borderline/high from 130)"
    if key == "triglycerides" and value >= 150:
        return f"High triglycerides: {value} mg/dL (normal below 150)"
    if key == "hdl_cholesterol" and value < 40:
        return f"Low HDL cholesterol: {value} mg/dL (often considered low below 40)"
    if key == "vitamin_d":
        if value < 20:
            return f"Severe vitamin D deficiency: {value} ng/mL (deficient below 20)"
        if value < 30:
            return f"Low vitamin D: {value} ng/mL (insufficient below 30)"
        return None
    if key == "vitamin_b12":
        if value < 200:
            return f"Low vitamin B12: {value} pg/mL (deficient below 200)"
        return None
    if key == "hba1c":
        if value >= 6.5:
            return f"High HbA1c: {value} % (diabetic range at 6.5% or above)"
        if value >= 5.7:
            return f"Borderline HbA1c: {value} % (prediabetic range 5.7–6.4%)"
        return None
    if key == "tsh":
        if value > 4.5:
            return f"High TSH: {value} µIU/mL (elevated above 4.5)"
        if value < 0.4:
            return f"Low TSH: {value} µIU/mL (suppressed below 0.4)"
        return None
    if key == "creatinine":
        if value > 1.3:
            return f"High creatinine: {value} mg/dL (elevated above 1.3)"
        return None
    if key == "mchc":
        if value < 32.0:
            return f"Low MCHC: {value} g/dL (threshold < 32.0)"
        if value > 36.0:
            return f"High MCHC: {value} g/dL (threshold > 36.0)"
        return None
    if key == "mch":
        if value < 27.0:
            return f"Low MCH: {value} pg (threshold < 27.0)"
        if value > 31.0:
            return f"High MCH: {value} pg (threshold > 31.0)"
        return None
    if key == "t3":
        # Values are normalized to ng/mL before reaching this check
        # (see unit_normalizer). Kept in sync with _is_flagged's generic
        # ref-range path so parse-time and report-time never diverge again.
        if value < 0.87:
            return f"Low Total T3: {value} ng/mL (threshold < 0.87)"
        if value > 1.78:
            return f"High Total T3: {value} ng/mL (threshold > 1.78)"
        return None
    return None



# ── Regex patterns ────────────────────────────────────────────────

NUMERIC_PATTERN = re.compile(r"(?<![a-zA-Z0-9])([0-9]+(?:\.[0-9]+)?)(?![a-zA-Z0-9])")

_RANGE_PATTERN = re.compile(
    r"(?:"
    r"(?P<lo>[0-9]+(?:\.[0-9]+)?)\s*[-–—to]{1,3}\s*(?P<hi>[0-9]+(?:\.[0-9]+)?)"
    r"|"
    r"[<≤]\s*(?P<hi_only>[0-9]+(?:\.[0-9]+)?)"
    r"|"
    r"[>≥]\s*(?P<lo_only>[0-9]+(?:\.[0-9]+)?)"
    r"|"
    r"upto\s*(?P<hi_upto>[0-9]+(?:\.[0-9]+)?)"
    r"|"
    r"up to\s*(?P<hi_upto2>[0-9]+(?:\.[0-9]+)?)"
    r")",
    re.IGNORECASE,
)

_RANGE_ONLY_LINE = re.compile(
    r"^\s*"
    r"(?:"
    r"[<≤>≥]\s*[0-9]+(?:\.[0-9]+)?"
    r"|"
    r"[0-9]+(?:\.[0-9]+)?\s*[-–—]\s*[0-9]+(?:\.[0-9]+)?"
    r"|"
    r"upto\s*[0-9]+(?:\.[0-9]+)?"
    r")",
    re.IGNORECASE,
)


# ── List-numbering stripper (unchanged) ───────────────────────────

_LIST_NUMBER_PREFIX_INLINE = re.compile(
    r"(?<=\s)(\d+[.)\]:])\s+(?=[<>≤≥A-Za-z])"
)
_LIST_NUMBER_PREFIX_LEADING = re.compile(
    r"^\s*(\d+[.)\]:])\s+(?=[<>≤≥A-Za-z])"
)


def _strip_list_numbering(line: str) -> str:
    cleaned = _LIST_NUMBER_PREFIX_LEADING.sub("", line)
    cleaned = _LIST_NUMBER_PREFIX_INLINE.sub("", cleaned)
    return cleaned


# ═══════════════════════════════════════════════════════════════════
# FIX A2: Compound medical tokens now include LOOKAHEAD extensions.
# When we hit a token like "25 Hydroxy" in the name, the actual test
# name may continue with parenthetical clarifications and more words
# before the real value: "25 Hydroxy (OH) Vit D    12.4"
# Solution: after finding the leftmost compound token, extend the
# name to include all subsequent NON-NUMERIC tokens until we hit
# a real numeric value with sufficient whitespace before it.
# ═══════════════════════════════════════════════════════════════════
_NAME_EMBEDDED_TOKENS = re.compile(
    r"("
    r"\bTotal\s+T[-\s]?3\b|"
    r"\bTotal\s+T[-\s]?4\b|"
    r"\bFree\s+T[-\s]?3\b|"
    r"\bFree\s+T[-\s]?4\b|"
    r"\bT[-\s]?3\b|"
    r"\bT[-\s]?4\b|"
    r"\bB[-\s]?12\b|"
    r"\bB[-\s]?6\b|"
    r"\bB[-\s]?9\b|"
    r"\bD[-\s]?3\b|"
    r"\bK[-\s]?1\b|"
    r"\bK[-\s]?2\b|"
    r"\bO2\b|"
    r"\bCO2\b|"
    r"\bCA\s?125\b|"
    r"\bCA\s?19[-\s]?9\b|"
    r"\b25[-\s]?OH\b|"
    r"\b17[-\s]?OH\b|"
    r"\bHbA1c\b|"
    r"\bA1C\b|"
    r"\b25\s+Hydroxy\b|"
    r"\bVit\s+D[-\s]?3\b|"
    r"\bVitamin\s+D[-\s]?3\b|"
    r"\bVitamin\s+B[-\s]?12\b|"
    r"\bVit\s+B[-\s]?12\b"
    r")",
    re.IGNORECASE,
)

# After extending past a compound token, continue extending the name
# through non-numeric words separated by 1-2 spaces. Stop when:
#   - we hit a run of 3+ spaces (column boundary in typical PDFs)
#   - we hit a numeric value
#   - we hit end of line
_NAME_CONTINUATION = re.compile(
    r"^\s{1,2}"                        # 1-2 spaces (in-name whitespace)
    r"(?:\([^)]{1,10}\)|[A-Za-z][A-Za-z\-]*)"  # word or short parenthetical
)


SKIP_LINE_FRAGMENTS = [
    "patient", "name", "age", "gender", "sex",
    "collection date", "report date", "sample", "barcode",
    "doctor", "hospital", "clinic", "disclaimer",
    "signature", "authorized",
    "method", "page",
]


def should_skip_lab_line(line: str) -> bool:
    line = line.strip().lower()
    if not line:
        return True
    if len(line) > 220:
        return True
    return any(fragment in line for fragment in SKIP_LINE_FRAGMENTS)


def _extract_unit_near_value(text_after_value: str) -> str:
    if not text_after_value:
        return ""

    snippet = text_after_value[:40].strip()
    if not snippet:
        return ""

    m = re.match(
        r"^([a-zA-Zµμ][a-zA-Zµμ0-9°%.]*"
        r"(?:\s*/\s*[a-zA-Zµμ°%][a-zA-Zµμ0-9°%.]*)?)",
        snippet,
    )
    if not m:
        m = re.match(
            r"^(/\s*[a-zA-Zµμ][a-zA-Zµμ0-9°%.]*)",
            snippet,
        )

    if not m:
        return ""

    candidate = m.group(1).strip()
    candidate = re.sub(r"\s+", "", candidate)
    lower = candidate.lower()

    REJECT_WORDS = {
        "h", "l", "n", "hi", "lo", "high", "low", "normal",
        "abnormal", "range", "ref", "reference", "flag",
        "borderline", "desirable", "text", "value", "result",
        "test", "unit", "units", "method", "note", "notes",
        "sample", "the", "and", "or", "as", "is", "in", "of", "to",
    }
    if lower in REJECT_WORDS:
        return ""
    if len(candidate) > 15:
        return ""
    if len(candidate) == 1 and lower not in {"%"}:
        return ""
    if candidate == "/":
        return ""

    return candidate


def _extract_range_from_tail(tail: str) -> tuple[float | None, float | None]:
    if not tail:
        return (None, None)

    m = _RANGE_PATTERN.search(tail)
    if not m:
        return (None, None)

    lo = m.group("lo")
    hi = m.group("hi")
    hi_only = m.group("hi_only")
    lo_only = m.group("lo_only")
    hi_upto = m.group("hi_upto") or m.group("hi_upto2")

    try:
        if lo and hi:
            return (float(lo), float(hi))
        if hi_only:
            return (None, float(hi_only))
        if lo_only:
            return (float(lo_only), None)
        if hi_upto:
            return (None, float(hi_upto))
    except ValueError:
        pass

    return (None, None)


# ── Multi-line row coalescer ──────────────────────────────────────

def _coalesce_multiline_rows(lines: list[str]) -> list[str]:
    if not lines:
        return lines

    _LEGEND_STARTERS = re.compile(
        r"^\s*(?:sufficiency|insufficiency|deficiency|toxicity|"
        r"optimal|borderline|normal|abnormal|desirable|high|low|"
        r"moderate|near|above|below|prediabet|diabet|excellent|"
        r"good|fair|poor|acceptable|target|goal)\b",
        re.IGNORECASE,
    )

    output: list[str] = []
    continuation_budget = 0

    for line in lines:
        stripped = line.strip()

        if not stripped:
            output.append(line)
            continuation_budget = 0
            continue

        is_continuation = False
        if output and continuation_budget > 0:
            prev = output[-1]
            prev_has_number = bool(NUMERIC_PATTERN.search(prev))
            if prev_has_number:
                if _RANGE_ONLY_LINE.match(stripped):
                    is_continuation = True
                elif _LEGEND_STARTERS.match(stripped):
                    if NUMERIC_PATTERN.search(stripped):
                        is_continuation = True

        if is_continuation:
            output[-1] = output[-1].rstrip() + " " + stripped
            continuation_budget -= 1
        else:
            output.append(line)
            if NUMERIC_PATTERN.search(stripped):
                continuation_budget = 3
            else:
                continuation_budget = 0

    return output


# ═══════════════════════════════════════════════════════════════════
# NOISE FILTER — expanded to catch prose explanations like
# "Cholicalciferol (VitaminD3) is synthesized in the skin from 7
#  dehydrocholestrol in response to sunlight..."
# These are educational footnotes attached to Vitamin D reports.
# ═══════════════════════════════════════════════════════════════════

_NOISE_NAMES_EXACT: set[str] = {
    "excellent control", "good control", "fair control", "poor control",
    "action suggested", "hba1c value between",
    "of diabetes using a cut off point of",
    "optimal", "near optimal", "above optimal", "above desirable",
    "borderline high", "high", "high above", "very high", "desirable",
    "normal", "abnormal", "moderate", "moderately high",
    "insufficiency", "sufficiency", "deficiency", "toxicity",
    "insufficient", "sufficient", "deficient",
    "vitamin d sufficiency", "vitamin d insufficiency",
    "vitamin d deficiency", "vitamin d toxicity",
    "vit d sufficiency", "vit d insufficiency",
    "vit d deficiency", "vit d toxicity",
    "1st trimester", "2nd trimester", "3rd trimester",
    "first trimester", "second trimester", "third trimester",
    "than", "greater than", "less than", "between",
    "if greater than", "if less than",
    "regt no", "reg no", "registration no", "lab no",
    "sample no", "specimen no", "accession no",
    "printed on", "printed by", "verified by", "reviewed by",
    "test", "value", "result", "unit", "units", "range",
    "reference range", "reference", "method", "note", "notes",
    "sample", "specimen", "code",
    "male", "female", "adult", "child", "children",
    "age", "years", "months", "days",
}

_NOISE_REGEXES: list[re.Pattern[str]] = [
    re.compile(r"^(very|extremely|far|slightly|moderately)\s+(high|low|above|below)$", re.IGNORECASE),
    re.compile(r"^(high|low)\s+(above|below|near|around)$", re.IGNORECASE),
    re.compile(r"^(regt|reg|sr|srl|id|s|serial|order)\.?\s*no\.?$", re.IGNORECASE),
    re.compile(r"^\d+(st|nd|rd|th)\s+(trimester|visit|day|month|week|hour|dose)$", re.IGNORECASE),
    re.compile(r"^(above|below|up\s+to|less\s+than|greater\s+than|more\s+than|equal\s+to)$", re.IGNORECASE),
    re.compile(r"^(if|when|for|during|after|before)\s+(greater|less|more|below|above|equal|elevated|reduced|normal|high|low)$", re.IGNORECASE),
    re.compile(r"^(risk|category|group|class|type|grade|stage)\s+(of|category|level|\d+)$", re.IGNORECASE),
    re.compile(r"^(min|max|avg|mean|median|target|goal|desired|ideal)$", re.IGNORECASE),
    re.compile(r"(sufficiency|insufficiency|deficiency|toxicity)\s*$", re.IGNORECASE),
    re.compile(r"ng\s*/\s*ml\s*[-–—]\s*vitamin", re.IGNORECASE),
    re.compile(r"(mg|µg|ug|mcg|ng|pg)\s*/\s*(dl|ml|l)\s*[-–—]\s*vitamin", re.IGNORECASE),
    # NEW: educational/prose explanations following a chemical name
    re.compile(r"\bis\s+synthesized\b", re.IGNORECASE),
    re.compile(r"\bis\s+converted\b", re.IGNORECASE),
    re.compile(r"\bis\s+produced\b", re.IGNORECASE),
    re.compile(r"\bin\s+response\s+to\b", re.IGNORECASE),
    re.compile(r"\bin\s+the\s+(skin|liver|kidney|gut|blood|body)\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+\d+[-\s]*[a-z]", re.IGNORECASE),  # "from 7 dehydro..."
]

_NOISE_SUBSTRINGS: list[str] = [
    "cut off point", "cut-off point", "value between",
    "greater than", "less than",
    "diabetes using", "diabetes mellitus",
    "reference interval", "reference range",
    "risk of", "risk category", "target range",
    "therapeutic range", "critical range",
    "conventional units", "si units",
    "interpretation:", "impression:", "conclusion:",
    "comment:", "remark:", "note:", "advice:", "advised:",
    "clinical significance",
    "method used", "sample type", "specimen type",
    "collected on", "received on", "reported on",
    "printed on", "verified on",
    "authenticated by", "authorised by",
    "referred by", "consultant", "pathologist", "technician",
    "regt no", "sr no", "srl no", "id no", "hospital no",
    "phone", "email", "address", "www.", "http",
    # NEW: educational-footnote fragments
    "synthesized in the",
    "converted in the",
    "produced in the",
    "in response to sunlight",
    "dehydrocholestrol",
    "dehydrocholesterol",
    "cholicalciferol",   # only appears in the prose sentence, not real rows
    "in the skin from",
    "in the liver from",
]


# ═══════════════════════════════════════════════════════════════════
# UNIVERSAL IDENTIFIER/ADMIN-FIELD REJECTION
#
# Denylisting every possible admin-field phrasing ("Lab No/Result No",
# "Acc No", "Barcode Ref", "Sample ID No"...) does not scale — any lab
# format can introduce a new one, as seen with "Lab No/Result No"
# slipping past the exact-match "lab no" entry. Instead we reject
# *structurally*: if identifier/admin words dominate the candidate
# name (regardless of exact phrasing), treat it as administrative
# noise. This catches any lab format past, present, and future.
# ═══════════════════════════════════════════════════════════════════

_IDENTIFIER_WORDS = {
    "no", "num", "number", "id", "code", "ref", "reference",
    "barcode", "registration", "regn", "regt", "reg", "accession",
    "acc", "serial", "srl", "sr", "uhid", "mrn", "ipd", "opd",
    "slip", "receipt", "invoice", "bill", "token", "case",
    "lab", "result", "report", "form", "file", "doc", "document",
    "patient", "visit", "session", "job", "record", "entry",
}

_CONNECTOR_WORDS = {"and", "or", "the", "of", "a", "an"}


def _looks_like_identifier_field(n: str) -> bool:
    """True if the normalized name is dominated by identifier/admin
    tokens — catches any phrasing of "Lab No/Result No", "Acc No",
    "Sample ID No", "Lab Ref No UHID", "Barcode Ref No", etc.,
    without enumerating exact strings.

    Rules:
      1. Empty → True.
      2. Every content word is identifier/connector → True (strict).
      3. ≥ 2 identifier words AND identifier words are ≥ 60% of
         total content words → True (dominant-identifier case,
         catches "Lab No/Result No", "Sample Barcode ID", etc. even
         if a stray non-identifier token slips in from OCR).
      4. ≥ 2 identifier words AND any strong admin marker present
         (no, id, code, uhid, mrn, barcode, accession) → True
         (short admin fields like "Lab No", "Acc No", "Ref Code").

    Kept intentionally structural rather than by exact phrase so any
    lab format's header/ID field is caught universally without ever
    needing a new denylist entry.
    """
    words = n.split()
    if not words:
        return True

    non_connector = [w for w in words if w not in _CONNECTOR_WORDS]
    if not non_connector:
        return True

    identifier_hits = sum(1 for w in non_connector if w in _IDENTIFIER_WORDS)
    total = len(non_connector)

    # Rule 2: strict — every content word is an identifier
    if identifier_hits == total:
        return True

    # Rule 3: dominant identifier ratio
    if identifier_hits >= 2 and (identifier_hits / total) >= 0.6:
        return True

    # Rule 4: strong admin marker plus at least one other identifier
    strong_markers = {"no", "id", "code", "uhid", "mrn", "barcode", "accession"}
    if identifier_hits >= 2 and any(w in strong_markers for w in non_connector):
        return True

    return False


def _is_noise_name(name: str) -> bool:
    n = re.sub(r"[:;,.\-–—_/\\|]+", " ", name).strip().lower()
    n = re.sub(r"\s+", " ", n)
    if not n:
        return True

    if n in _NOISE_NAMES_EXACT:
        return True

    for pat in _NOISE_REGEXES:
        if pat.search(n):
            return True

    if any(frag in n for frag in _NOISE_SUBSTRINGS):
        return True

    if n in {"normal", "high", "low", "optimal", "borderline",
             "abnormal", "critical", "moderate", "severe", "mild",
             "positive", "negative", "trace", "present", "absent",
             "very high", "very low", "high above", "low below",
             "borderline high", "borderline low", "regt no",
             "reg no", "sr no", "srl no", "id no"}:
        return True

    if not re.search(r"[a-zA-Z]{2,}", n):
        return True

    # NEW: reject names longer than 8 words (they're prose, not test names)
    if len(n.split()) > 8:
        return True

    first_words = n.split()[:3]
    if len(first_words) >= 2:
        first_two = " ".join(first_words[:2])
        if first_two in {
            "very high", "very low", "high above", "low below",
            "borderline high", "borderline low", "above optimal",
            "near optimal", "regt no", "reg no", "sr no", "srl no",
            "id no", "s no", "s/no", "s.no",
        }:
            return True

    # UNIVERSAL: any combination of identifier/admin words
    # ("lab no result no", "acc no", "sample id no", "barcode ref no", ...)
    # Rejects structurally rather than by exact phrase, so any lab's
    # header/ID field is caught without needing a new denylist entry.
    if _looks_like_identifier_field(n):
        return True

    # UNIVERSAL: reject bare 1-2 letter alphabetic fragments not
    # already recognized as a real clinical token. Legit short test
    # names (T3, T4, RBC, WBC, TSH, HB...) are handled separately via
    # _NAME_EMBEDDED_TOKENS / resolve_canonical_key upstream; anything
    # this short reaching here is far more likely OCR/column debris
    # (e.g. "Mc", "Sr", "Dt") than an actual biomarker.
    if len(n) <= 2 and n.isalpha():
        return True

    return False


# ── Test-name candidate extraction ────────────────────────────────

_NAME_MAX_LEN = 60
_NAME_MIN_LEN = 2

_NON_ROW_HINTS = {
    "complete blood count", "differential leucocytes count",
    "peripheral smear examination", "biochemistry", "immunology",
    "hormones", "vitamin panel", "lipid profile", "liver function",
    "kidney function", "thyroid function", "urine analysis",
    "microscopy", "physical examination", "chemical examination",
    "hematology", "clinical pathology", "investigation summary",
    "end of report", "----", "===", "___",
    "fully automated", "cell counter", "yumizen",
    "printed", "verified", "reviewed", "consultant", "pathologist",
    "phone", "email", "www", "http",
    "interpretation :", "impression :", "conclusion :",
    "note :", "remark :", "comment :",
    "method used", "sample collected",
}


def _line_is_row_candidate(line: str) -> bool:
    lower = line.lower().strip()
    if not lower or len(lower) < 4:
        return False
    if not NUMERIC_PATTERN.search(line):
        return False
    for hint in _NON_ROW_HINTS:
        if hint in lower:
            return False
    return True


def _extract_test_name(line: str, first_num_pos: int) -> str:
    """
    Extract the test name from a lab row.

    FIX A: compound tokens with digits (T3, T4, B12, 25-OH, etc.)
    FIX A2: after crossing a compound token, keep extending the name
             through subsequent non-numeric words separated by 1-2
             spaces, until a column boundary (3+ spaces) or a real
             numeric value is found. Handles "25 Hydroxy (OH) Vit D".
    """
    prefix_search_start = max(0, first_num_pos - 25)
    search_end = min(len(line), first_num_pos + 15)
    search_region = line[prefix_search_start:search_end]

    extended_end = first_num_pos
    for m in _NAME_EMBEDDED_TOKENS.finditer(search_region):
        abs_start = prefix_search_start + m.start()
        abs_end = prefix_search_start + m.end()
        if abs_start <= first_num_pos < abs_end:
            extended_end = max(extended_end, abs_end)
        elif abs_end == first_num_pos:
            extended_end = max(extended_end, abs_end)
        elif abs_end < first_num_pos and (first_num_pos - abs_end) <= 3:
            gap = line[abs_end:first_num_pos]
            if gap.strip() == "":
                extended_end = max(extended_end, abs_end)

    # ── FIX A2: extend name PAST first_num_pos through non-numeric
    # continuation words until we hit a column boundary or real value ──
    if extended_end > first_num_pos:
        # We consumed the "first number" as part of the name (e.g. "25 Hydroxy")
        # Now continue scanning for more name-like words
        pos = extended_end
        while pos < len(line):
            remainder = line[pos:]
            m = _NAME_CONTINUATION.match(remainder)
            if not m:
                break
            # Check that this word isn't a number
            candidate_word = m.group(0).strip()
            if NUMERIC_PATTERN.match(candidate_word):
                break
            pos += m.end()
            extended_end = pos

    name = line[:extended_end].strip()
    name = re.sub(r"[:;\-–—\s]+$", "", name).strip()
    name = re.sub(r"\s+", " ", name)

    if len(name) < _NAME_MIN_LEN or len(name) > _NAME_MAX_LEN:
        return ""
    if not re.search(r"[a-zA-Z]", name):
        return ""
    return name


def _slugify_extra_key(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r"\s*\([^)]*\)\s*", " ", n)
    n = re.sub(r"[^a-z0-9]+", "_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    return n


# ── Universal row extractor ───────────────────────────────────────

def _extract_rows(text: str) -> list[dict]:
    rows: list[dict] = []
    raw_lines = text.splitlines()
    coalesced = _coalesce_multiline_rows(raw_lines)

    for raw_line in coalesced:
        cleaned = _strip_list_numbering(raw_line)
        line = cleaned.strip()

        if not line:
            continue
        if should_skip_lab_line(line):
            continue
        if not _line_is_row_candidate(line):
            continue

        num_match = NUMERIC_PATTERN.search(line)
        if not num_match:
            continue
        first_num_pos = num_match.start()

        name = _extract_test_name(line, first_num_pos)
        if not name:
            continue

        if _is_noise_name(name):
            logger.debug(
                "lab_report_parser · noise line rejected",
                name=name, line=line,
            )
            continue

        # Diagnostic: log identifier-shaped names that somehow slipped
        # past the noise filter (should be zero — surfaces regressions
        # like the "Lab No/Result No" leak). Kept at WARNING so it
        # shows up in production logs without requiring debug level.
        _norm_probe = re.sub(r"[:;,.\-–—_/\\|]+", " ", name).strip().lower()
        _norm_probe = re.sub(r"\s+", " ", _norm_probe)
        _probe_words = _norm_probe.split()
        _probe_id_hits = sum(1 for w in _probe_words if w in _IDENTIFIER_WORDS)
        if _probe_words and _probe_id_hits >= 2 and _probe_id_hits / len(_probe_words) >= 0.5:
            logger.warning(
                "lab_report_parser · suspicious identifier-shaped name accepted — "
                "please report this so filter can be tightened",
                name=name, normalized=_norm_probe, line=line,
            )

        name_end_in_line = len(name)
        value_match = num_match
        if value_match.start() < name_end_in_line:
            later = NUMERIC_PATTERN.search(line, name_end_in_line)
            if later:
                value_match = later
            else:
                continue

        try:
            value = float(value_match.group(1))
        except ValueError:
            continue
        if value < 0:
            continue
        if value == 0:
            if not any(w in name.lower() for w in ["basophil", "eosinophil", "monocyte"]):
                continue

        tail = line[value_match.end():]
        unit = _extract_unit_near_value(tail) or None
        low, high = _extract_range_from_tail(tail)

        rows.append({
            "name": name,
            "value": value,
            "unit": unit,
            "low": low,
            "high": high,
            "line": line,
        })

    return rows


# ── Text-finding extraction ───────────────────────────────────────

_TEXT_FINDING_LABELS = [
    "rbc morphology", "wbc morphology", "platelet morphology",
    "peripheral smear examination", "peripheral smear",
    "microscopy", "microscopic examination",
    "histopathology", "histopathology report",
    "gross examination", "microscopic findings",
    "biopsy", "biopsy report",
    "cytology", "cytology findings",
    "cytological findings", "cytological examination",
    "fnac", "fnac findings",
    "culture", "culture and sensitivity", "culture sensitivity",
    "sensitivity report", "gram stain", "afb stain",
    "organism isolated",
    "radiologist comment", "radiologist note", "radiologist notes",
    "radiological findings",
    "ultrasound findings", "ultrasound notes", "usg findings",
    "ct findings", "ct report", "mri findings", "mri report",
    "x-ray findings", "xray findings", "chest x-ray findings",
    "ecg findings", "ecg report",
    "echo findings", "echocardiography findings", "doppler findings",
    "impression", "clinical impression",
    "provisional diagnosis", "final diagnosis", "diagnosis",
    "conclusion", "summary of findings", "summary",
    "observation", "observations",
    "clinical significance", "clinical correlation",
    "pathologist note", "pathologist opinion", "opinion",
]

_EXCLUDED_LABELS = {
    "interpretation", "note", "notes", "comment", "comments",
    "advice", "advised", "remark", "remarks",
    "method", "methodology",
    "reference", "references",
    "sample", "specimen",
}


def _extract_text_findings(text: str) -> list[str]:
    findings: list[str] = []
    lines = text.splitlines()
    n = len(lines)

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        lower = line.lower()

        digit_count = sum(1 for c in line if c.isdigit())
        if digit_count >= max(3, len(line) // 4):
            continue

        skip_excluded = False
        for excl in _EXCLUDED_LABELS:
            if lower.startswith(excl + ":") or lower.startswith(excl + " ") or lower == excl:
                skip_excluded = True
                break
        if skip_excluded:
            continue

        matched_label: str | None = None
        for label in _TEXT_FINDING_LABELS:
            if (
                lower == label
                or lower.startswith(label + ":")
                or lower.startswith(label + " -")
                or lower.startswith(label + " ")
            ):
                matched_label = label
                break

        if not matched_label:
            continue

        content = line[len(matched_label):].lstrip(" :-–—\t").strip()

        continuation_parts: list[str] = []
        blank_seen = False
        for j in range(i + 1, min(i + 8, n)):
            nxt = lines[j].strip()
            if not nxt:
                if continuation_parts:
                    blank_seen = True
                continue
            if blank_seen:
                break

            nxt_lower = nxt.lower()
            is_new_label = False
            for lab in _TEXT_FINDING_LABELS + list(_EXCLUDED_LABELS):
                if nxt_lower.startswith(lab):
                    is_new_label = True
                    break
            if is_new_label:
                break

            digit_count_nxt = sum(1 for c in nxt if c.isdigit())
            if digit_count_nxt >= 3:
                break

            if nxt.isupper() and len(nxt) < 60:
                break

            continuation_parts.append(nxt)
            if sum(len(c) for c in continuation_parts) + len(content) > 400:
                break

        if continuation_parts:
            joined = " ".join(continuation_parts).strip()
            if content:
                content = f"{content} {joined}".strip()
            else:
                content = joined

        if not content:
            continue

        if re.fullmatch(r"[\d.,\s%/-]+", content):
            continue
        if content.isupper() and len(content) < 25:
            continue

        content = re.sub(r"^[\d]+[\]\)\.]\s*", "", content)
        content = re.sub(r"^[-•]\s*", "", content)
        content = re.sub(r"\s+", " ", content)

        if len(content) > 300:
            content = content[:297].rstrip() + "..."

        pretty_label = " ".join(w.capitalize() for w in matched_label.split())
        finding_line = f"{pretty_label}: {content}"

        if finding_line not in findings:
            findings.append(finding_line)

    return findings


# ═══════════════════════════════════════════════════════════════════
# FIX D: Biomarkers where labs commonly print multi-tier reference
# legends. For these, ALWAYS prefer the clinical fallback range from
# REFERENCE_RANGES over the parser-extracted tier fragment.
# ═══════════════════════════════════════════════════════════════════
_TIERED_LEGEND_KEYS: set[str] = {
    "triglycerides",
    "ldl_cholesterol",
    "total_cholesterol",
    "vldl_cholesterol",
    "hba1c",
    "vitamin_d",
    "hdl_cholesterol",
}


# ── Main parser ───────────────────────────────────────────────────

def _parse_text(
    text: str,
) -> tuple[
    dict[str, float],
    dict[str, float],
    list[str],
    dict[str, str],
    dict[str, dict[str, float | None]],
    list[str],
]:
    measurements:       dict[str, float] = {}
    extra_measurements: dict[str, float] = {}
    abnormal_values:    list[str]        = []
    units:              dict[str, str]   = {}
    ref_ranges:         dict[str, dict[str, float | None]] = {}

    rows = _extract_rows(text)

    for row in rows:
        raw_name = row["name"]
        value = row["value"]
        unit = row["unit"]
        low = row["low"]
        high = row["high"]

        canonical = resolve_canonical_key(raw_name)

        if canonical is not None:
            if canonical in measurements:
                logger.debug(
                    "lab_report_parser · duplicate canonical key skipped (first-wins)",
                    raw_name=raw_name, canonical=canonical, skipped_value=value,
                )
                continue

            measurements[canonical] = value

            if unit:
                units[canonical] = unit
            if low is not None or high is not None:
                ref_ranges[canonical] = {"low": low, "high": high}

            finding = _detect_abnormal(canonical, value)
            if finding:
                abnormal_values.append(finding)
        else:
            slug = _slugify_extra_key(raw_name)
            if not slug or slug in extra_measurements:
                continue

            extra_measurements[slug] = value

            if unit:
                units[slug] = unit
            if low is not None or high is not None:
                ref_ranges[slug] = {"low": low, "high": high}

    # Fill in canonical fallbacks
    for canonical in measurements:
        if canonical not in units and canonical in CANONICAL_UNITS:
            units[canonical] = CANONICAL_UNITS[canonical]
        if canonical not in ref_ranges and canonical in REFERENCE_RANGES:
            ref_ranges[canonical] = dict(REFERENCE_RANGES[canonical])

    # ═══ FIX D2: Tiered-legend override — always use clinical range ═══
    # For biomarkers with tiered legends, always prefer REFERENCE_RANGES
    # over PDF-extracted ranges. Prevents:
    #   - Triglycerides 130 vs (150,199) → wrongly "low"
    #   - HbA1c 5.7 vs (None, 5.6)      → shows "Normal <5.6%"
    #   - LDL 98 vs (100,129)           → wrongly "low"
    #   - Vitamin D 12.4 vs (30, None)  → keeps clinical (30, 100) with both bounds
    for canonical in list(measurements.keys()):
        if canonical not in _TIERED_LEGEND_KEYS:
            continue
        if canonical not in REFERENCE_RANGES:
            continue

        fallback = REFERENCE_RANGES[canonical]
        current = ref_ranges.get(canonical) or {}

        # ALWAYS override for tiered-legend biomarkers, unless the extracted
        # range EXACTLY matches the fallback (rare but harmless).
        if current != fallback:
            logger.info(
                "lab_report_parser · tiered-legend override",
                key=canonical, extracted=current, override=fallback,
            )
            ref_ranges[canonical] = dict(fallback)

    text_findings = _extract_text_findings(text)

    return (
        measurements,
        extra_measurements,
        abnormal_values,
        units,
        ref_ranges,
        text_findings,
    )


# ── Parser (path-level orchestration) ─────────────────────────────

def _merge_lab_result(
    combined: LabReportResult,
    current: LabReportResult,
) -> LabReportResult:
    for finding in current.abnormal_values:
        if finding not in combined.abnormal_values:
            combined.abnormal_values.append(finding)

    for key, value in current.measurements.items():
        combined.measurements.setdefault(key, value)

    for key, value in current.extra_measurements.items():
        combined.extra_measurements.setdefault(key, value)

    for key, unit in current.units.items():
        combined.units.setdefault(key, unit)

    for key, rng in current.reference_ranges.items():
        combined.reference_ranges.setdefault(key, rng)

    for finding in current.text_findings:
        if finding not in combined.text_findings:
            combined.text_findings.append(finding)

    return combined


def _parse_lab_path(path: Path) -> LabReportResult | ToolError:
    if not path.is_file():
        return ToolError(
            tool=TOOL_LAB_REPORT_PARSER,
            reason=f"Lab report file not found: {path}",
            fatal=False,
        )

    if _is_real_pdf(path):
        logger.info(
            "lab_report_parser · PDF detected · starting extraction waterfall",
            path=str(path),
        )
        text = _extract_pdf_text(path)
        if text is None:
            ocr_hint = "" if _ocr_enabled() else " — set AEGIS_OCR=1 to enable OCR fallback"
            return ToolError(
                tool=TOOL_LAB_REPORT_PARSER,
                reason=f"PDF text extraction failed for {path}{ocr_hint}.",
                fatal=False,
            )
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    (
        measurements,
        extra_measurements,
        abnormal_values,
        units,
        reference_ranges,
        text_findings,
    ) = _parse_text(text)

    logger.info(
        "lab_report_parser · parsing complete",
        path=str(path),
        measurements=len(measurements),
        extra=len(extra_measurements),
        abnormal=len(abnormal_values),
        units=len(units),
        ranges=len(reference_ranges),
        text_findings=len(text_findings),
    )

    return LabReportResult(
        abnormal_values=abnormal_values,
        measurements=measurements,
        extra_measurements=extra_measurements,
        units=units,
        reference_ranges=reference_ranges,
        text_findings=text_findings,
    )


class LabReportParser:
    """Parses laboratory reports into structured LabReportResult."""

    TOOL_NAME = TOOL_LAB_REPORT_PARSER

    async def run(
        self,
        state: AegisState,
    ) -> LabReportResult | ToolError:

        try:
            if not state.lab_pdf_path:
                return ToolError(
                    tool=TOOL_LAB_REPORT_PARSER,
                    reason="No laboratory report path supplied.",
                    fatal=False,
                )

            paths = (
                [Path(p) for p in state.lab_pdf_path]
                if isinstance(state.lab_pdf_path, list)
                else [Path(state.lab_pdf_path)]
            )

            combined = LabReportResult()
            failures: list[str] = []

            for path in paths:
                parsed = _parse_lab_path(path)
                if isinstance(parsed, ToolError):
                    failures.append(parsed.reason)
                    logger.warning(
                        "lab_report_parser · report failed, continuing with remaining reports",
                        path=str(path),
                        reason=parsed.reason,
                        session_id=state.session_id,
                    )
                    continue
                combined = _merge_lab_result(combined, parsed)

            if (
                not combined.measurements
                and not combined.extra_measurements
                and not combined.text_findings
                and failures
            ):
                return ToolError(
                    tool=TOOL_LAB_REPORT_PARSER,
                    reason=f"Failed to parse any lab reports: {failures}",
                    fatal=False,
                )

            for key in combined.measurements:
                if key not in combined.reference_ranges and key in REFERENCE_RANGES:
                    combined.reference_ranges[key] = dict(REFERENCE_RANGES[key])
                if key not in combined.units and key in CANONICAL_UNITS:
                    combined.units[key] = CANONICAL_UNITS[key]

            logger.info(
                "lab_report_parser · combined parsing complete",
                session_id=state.session_id,
                measurements=len(combined.measurements),
                extra=len(combined.extra_measurements),
                abnormal=len(combined.abnormal_values),
                units=len(combined.units),
                ranges=len(combined.reference_ranges),
                text_findings=len(combined.text_findings),
            )

            return combined

        except Exception as exc:
            logger.exception(
                "lab_report_parser · unhandled exception",
                session_id=state.session_id,
            )
            return ToolError(
                tool=TOOL_LAB_REPORT_PARSER,
                reason=f"LabReportParser unhandled exception: {exc}",
                fatal=False,
            )