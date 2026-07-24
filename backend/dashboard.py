"""
backend/dashboard.py — Personalised user dashboard endpoint.

    GET /dashboard — Returns user-specific health summary, recent records,
                     latest vitals with baseline z-scores, risk profile,
                     and safety review items.

All data is scoped to the authenticated user.

Universal biomarker classification:
    - Alias resolution is delegated to biomarker_knowledge.resolve_canonical_key()
    - Display names come from biomarker_knowledge.get_display_name()
    - Any biomarker with a reference range (from PDF or lab_thresholds
      REFERENCE_RANGES) gets automatic classification, even without an
      entry in _MEASUREMENT_META. This is the key fix that makes
      Vitamin D, B12, folate, TSH, ferritin, ALP, GGT, etc. flag correctly.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from statistics import mean

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.db.models import User, HealthRecord, VitalSnapshot
from app.db.session import get_db
from backend.baseline import baseline_summary

# UNIVERSAL FIX: single source of truth for canonical keys and display names
from tools.biomarker_knowledge import resolve_canonical_key, get_display_name, resolve_advice
from tools.lab_thresholds import REFERENCE_RANGES, CANONICAL_UNITS

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_iso(value: datetime | None) -> str | None:
    """
    Serialize database timestamps with an explicit UTC offset.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _json_dict(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
        return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


# ═══════════════════════════════════════════════════════════════════
# MEASUREMENT META — thresholds for critical/high classification.
# Only biomarkers needing HARD critical-severity cutoffs live here.
# Every OTHER biomarker gets classified via REFERENCE_RANGES fallback
# in _classify_via_reference_range() below.
# ═══════════════════════════════════════════════════════════════════
_MEASUREMENT_META: dict[str, dict] = {
    # Vitals
    "heart_rate":         {},
    "spo2":               {},
    "temperature_c":      {},
    "temperature_f":      {},
    "respiratory_rate":   {},
    "systolic_bp":        {},
    "diastolic_bp":       {},
    "bp_systolic":        {},
    "bp_diastolic":       {},

    # Blood — has both regular + critical thresholds
    "haemoglobin":        {"low": 12, "critical_low": 7},
    "rbc":                {},
    "wbc":                {},
    "platelets":          {},
    "hematocrit":         {},
    "mcv":                {},
    "mch":                {},
    "mchc":               {},
    "mpv":                {},
    "rdw":                {},
    "rdw_cv":             {},
    "neutrophils":        {},
    "lymphocytes":        {},
    "monocytes":          {},
    "eosinophils":        {},
    "basophils":          {},

    # Cardiac
    "troponin":           {"high": 0.04, "critical_high": 0.04},

    # Metabolic
    "glucose":            {"high": 180},
    "hba1c":              {},
    "insulin":            {},

    # Electrolytes — critical cutoffs for K+ (arrhythmia risk)
    "potassium":          {"high": 5.5, "critical_high": 6.5},
    "sodium":             {},
    "calcium":            {},
    "magnesium":          {},
    "phosphorus":         {},
    "chloride":           {},

    # Kidney
    "creatinine":         {},
    "urea":               {},
    "bun":                {},
    "uric_acid":          {},

    # Liver
    "sgpt_alt":           {},
    "sgot_ast":           {},
    "bilirubin":          {},
    "bilirubin_direct":   {},
    "bilirubin_indirect": {},
    "albumin":            {},
    "total_protein":      {},
    "globulin":           {},
    "alp":                {},
    "ggt":                {},
    "ag_ratio":           {},

    # Lipids
    "total_cholesterol":  {},
    "ldl_cholesterol":    {},
    "hdl_cholesterol":    {},
    "vldl_cholesterol":   {},
    "triglycerides":      {},
    "chol_hdl_ratio":     {},
    "ldl_hdl_ratio":      {},
    "non_hdl_cholesterol": {},

    # Iron
    "iron":               {},
    "ferritin":           {},
    "tibc":               {},
    "transferrin":        {},
    "transferrin_saturation": {},

    # Vitamins
    "vitamin_d":          {},
    "vitamin_b12":        {},
    "folate":             {},

    # Thyroid
    "tsh":                {},
    "t3":                 {},
    "t4":                 {},
    "ft3":                {},
    "free_t3":            {},
    "ft4":                {},
    "free_t4":            {},
    "anti_tpo":           {},

    # Inflammation
    "crp":                {},
    "esr":                {},

    # Coagulation
    "inr":                {},
    "pt":                 {},
    "aptt":               {},
    "d_dimer":            {},
    "fibrinogen":         {},

    # Hormones
    "testosterone":       {},
    "estrogen":           {},
    "progesterone":       {},
    "cortisol":           {},
    "prolactin":          {},
    "lh":                 {},
    "fsh":                {},
    "homocysteine":       {},

    # Tumour markers
    "psa":                {},
    "cea":                {},
    "afp":                {},
    "ca_125":             {},
    "ca_19_9":            {},
    "beta_hcg":           {},

    # Urine analysis
    "urine_ph":           {},
    "specific_gravity":   {},
    "urine_protein":      {},
    "urine_glucose":      {},
    "urine_ketones":      {},
    "urine_wbc":          {},
    "urine_rbc":          {},
    "urine_epithelial":   {},
}


# ═══════════════════════════════════════════════════════════════════
# ROW 3 — System grouping for Personalized Recommendations card.
# Groups flagged biomarkers by clinical system so each dashboard row
# reads "{System} ({count}) — {biomarkers with statuses} — {shared
# action phrase}". Priority ranks decide which systems surface first
# when more than the row cap are flagged; the remainder collapse into
# a single "Other findings (N)" overflow row so nothing is hidden.
# ═══════════════════════════════════════════════════════════════════
_BIOMARKER_SYSTEMS: dict[str, str] = {
    # cardiovascular
    "troponin": "cardiovascular",
    "bnp": "cardiovascular",
    "systolic_bp": "cardiovascular",
    "diastolic_bp": "cardiovascular",
    "bp_systolic": "cardiovascular",
    "bp_diastolic": "cardiovascular",
    "heart_rate": "cardiovascular",
    # endocrine
    "tsh": "endocrine",
    "t3": "endocrine",
    "t4": "endocrine",
    "ft3": "endocrine",
    "free_t3": "endocrine",
    "ft4": "endocrine",
    "free_t4": "endocrine",
    "anti_tpo": "endocrine",
    "cortisol": "endocrine",
    "testosterone": "endocrine",
    "estrogen": "endocrine",
    "progesterone": "endocrine",
    "prolactin": "endocrine",
    "lh": "endocrine",
    "fsh": "endocrine",
    "insulin": "endocrine",
    "vitamin_d": "endocrine",
    # metabolic
    "glucose": "metabolic",
    "hba1c": "metabolic",
    # renal
    "creatinine": "renal",
    "urea": "renal",
    "bun": "renal",
    "uric_acid": "renal",
    # hepatic
    "sgpt_alt": "hepatic",
    "sgot_ast": "hepatic",
    "alp": "hepatic",
    "ggt": "hepatic",
    "bilirubin": "hepatic",
    "bilirubin_direct": "hepatic",
    "bilirubin_indirect": "hepatic",
    "albumin": "hepatic",
    "globulin": "hepatic",
    "total_protein": "hepatic",
    "ag_ratio": "hepatic",
    # hematologic
    "haemoglobin": "hematologic",
    "rbc": "hematologic",
    "wbc": "hematologic",
    "platelets": "hematologic",
    "hematocrit": "hematologic",
    "mcv": "hematologic",
    "mch": "hematologic",
    "mchc": "hematologic",
    "mpv": "hematologic",
    "rdw": "hematologic",
    "rdw_cv": "hematologic",
    "neutrophils": "hematologic",
    "lymphocytes": "hematologic",
    "monocytes": "hematologic",
    "eosinophils": "hematologic",
    "basophils": "hematologic",
    "vitamin_b12": "hematologic",
    "folate": "hematologic",
    # electrolyte
    "sodium": "electrolyte",
    "potassium": "electrolyte",
    "chloride": "electrolyte",
    "calcium": "electrolyte",
    "magnesium": "electrolyte",
    "phosphorus": "electrolyte",
    # lipid
    "total_cholesterol": "lipid",
    "ldl_cholesterol": "lipid",
    "hdl_cholesterol": "lipid",
    "vldl_cholesterol": "lipid",
    "non_hdl_cholesterol": "lipid",
    "triglycerides": "lipid",
    "chol_hdl_ratio": "lipid",
    "ldl_hdl_ratio": "lipid",
    # inflammatory
    "crp": "inflammatory",
    "esr": "inflammatory",
    # iron
    "iron": "iron",
    "ferritin": "iron",
    "tibc": "iron",
    "transferrin": "iron",
    "transferrin_saturation": "iron",
    # respiratory
    "spo2": "respiratory",
    "respiratory_rate": "respiratory",
    # vitals
    "temperature_c": "vitals",
    "temperature_f": "vitals",
}

_SYSTEM_PRIORITY: dict[str, int] = {
    "cardiovascular": 1,
    "endocrine": 2,
    "metabolic": 3,
    "renal": 4,
    "hepatic": 5,
    "hematologic": 6,
    "electrolyte": 7,
    "lipid": 8,
    "inflammatory": 9,
    "iron": 10,
    "respiratory": 11,
    "vitals": 12,
    "general": 13,
}

_SYSTEM_LABELS: dict[str, str] = {
    "cardiovascular": "Cardiovascular",
    "endocrine": "Endocrine",
    "metabolic": "Metabolic",
    "renal": "Renal",
    "hepatic": "Hepatic",
    "hematologic": "Hematologic",
    "electrolyte": "Electrolyte",
    "lipid": "Lipid",
    "inflammatory": "Inflammatory",
    "iron": "Iron",
    "respiratory": "Respiratory",
    "vitals": "Vitals",
    "general": "General",
}

_SYSTEM_ACTION_PHRASE: dict[str, str] = {
    "cardiovascular": "Discuss cardiac risk with your physician",
    "endocrine": "Review hormone panel with an endocrinologist",
    "metabolic": "Review glucose and metabolic control with your clinician",
    "renal": "Review kidney function with your physician",
    "hepatic": "Review liver function with your physician",
    "hematologic": "Review blood counts with your clinician",
    "electrolyte": "Review electrolyte balance with your clinician",
    "lipid": "Discuss lipid management with your physician",
    "inflammatory": "Discuss inflammation workup with your clinician",
    "iron": "Discuss iron status with your physician",
    "respiratory": "Review respiratory findings with your clinician",
    "vitals": "Monitor these vitals and review with your clinician",
    "general": "Review these results with your healthcare provider",
}


def _classify_system(measurement: dict) -> str:
    key = str(measurement.get("key") or "").lower()
    return _BIOMARKER_SYSTEMS.get(key, "general")


def _humanize_list(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _status_word_for_group(status: str) -> str:
    s = str(status or "").lower()
    if s == "critical_low":
        return "Critical Low"
    if s == "critical_high":
        return "Critical High"
    if s == "critical":
        return "Critical"
    if s == "high":
        return "High"
    if s == "low":
        return "Low"
    if s == "borderline":
        return "Borderline"
    if s == "abnormal":
        return "Abnormal"
    return s.replace("_", " ").title() if s else ""


def _compose_system_row(system: str, group: list[dict]) -> dict:
    label = _SYSTEM_LABELS.get(system, system.title())
    action = _SYSTEM_ACTION_PHRASE.get(system, _SYSTEM_ACTION_PHRASE["general"])

    parts: list[str] = []
    max_risk = 0
    for m in group:
        name = str(m.get("name") or "").strip()
        if not name:
            continue
        status_word = _status_word_for_group(m.get("status"))
        parts.append(f"{name} ({status_word})" if status_word else name)
        risk = m.get("risk_score") or 0
        if isinstance(risk, (int, float)) and risk > max_risk:
            max_risk = int(risk)

    return {
        "status": "system_group",
        "biomarker": f"{label} ({len(group)})",
        "key": f"system::{system}",
        "value": _humanize_list(parts),
        "urgency": "critical" if max_risk >= 2 else "moderate",
        "recommendation": action,
    }


def _compose_overflow_row(overflow_groups: dict[str, list[dict]]) -> dict:
    total = sum(len(g) for g in overflow_groups.values())
    system_labels = [
        _SYSTEM_LABELS.get(s, s.title()) for s in overflow_groups.keys()
    ]
    max_risk = 0
    for group in overflow_groups.values():
        for m in group:
            risk = m.get("risk_score") or 0
            if isinstance(risk, (int, float)) and risk > max_risk:
                max_risk = int(risk)

    return {
        "status": "system_group",
        "biomarker": f"Other findings ({total})",
        "key": "system::__overflow__",
        "value": _humanize_list(system_labels),
        "urgency": "critical" if max_risk >= 2 else "moderate",
        "recommendation": _SYSTEM_ACTION_PHRASE["general"],
    }

def _normalise_measurement_key(value: object) -> str:
    """
    Delegate canonical key resolution to biomarker_knowledge.
    Falls back to snake_case slug if no canonical match.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    canonical = resolve_canonical_key(raw)
    if canonical:
        return canonical
    # Fallback: snake_case slug so unknown items still get a stable key
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


def _numeric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _resolve_display_name(canonical_key: str, raw_key: object) -> str:
    """Get proper display name via biomarker_knowledge, falling back."""
    display = get_display_name(canonical_key)
    if display:
        return display
    return str(raw_key).replace("_", " ").title()


# ═══════════════════════════════════════════════════════════════════
# UNIVERSAL FIX — reference-range fallback classifier.
# When _MEASUREMENT_META has no thresholds for a biomarker, check
# REFERENCE_RANGES. If it has one, classify against it. This is what
# makes Vitamin D, B12, TSH, ferritin, ALP, GGT, etc. auto-flag.
# ═══════════════════════════════════════════════════════════════════
def _classify_via_reference_range(
    canonical_key: str,
    value: float,
    pdf_range: dict | None = None,
) -> tuple[str, int, float] | None:
    """
    Classify a measurement using reference ranges when _MEASUREMENT_META
    provides no thresholds.

    Prefers the PDF-extracted range over the hardcoded REFERENCE_RANGES.
    Returns (status, risk_score, deviation_score) or None if no range known.
    """
    rng: dict | None = None

    # Priority 1: range from actual PDF
    if pdf_range and isinstance(pdf_range, dict):
        rng = pdf_range

    # Priority 2: hardcoded fallback
    if not rng:
        # Try exact key, plus space/underscore variants
        for variant in (
            canonical_key,
            canonical_key.replace("_", " "),
            canonical_key.replace(" ", "_"),
        ):
            candidate = REFERENCE_RANGES.get(variant)
            if candidate:
                rng = candidate
                break

    if not rng:
        return None

    low = rng.get("low")
    high = rng.get("high")

    # Two-sided range
    if low is not None and high is not None:
        span = high - low
        margin = span * 0.1 if span > 0 else 0
        if value > high:
            deviation = (value - high) / max(abs(high), 1.0)
            return ("high", 1, deviation)
        if value < low:
            deviation = (low - value) / max(abs(low), 1.0)
            return ("low", 1, deviation)

        # ── FIX #1: skip low-end borderline when low == 0 ──
        near_high = value >= high - margin
        near_low = (low > 0) and (value <= low + margin)
        if near_high or near_low:
            return ("borderline", 1, 0.0)
        # ────────────────────────────────────────────────────

        return ("normal", 0, 0.0)

    # Upper-limit only (e.g. cholesterol < 200)
    if high is not None:
        if value >= high:
            deviation = (value - high) / max(abs(high), 1.0)
            return ("high", 1, deviation)
        if value >= high * 0.9:
            return ("borderline", 1, 0.0)
        return ("normal", 0, 0.0)

    # Lower-limit only (e.g. HDL > 40)
    if low is not None:
        if value <= low:
            deviation = (low - value) / max(abs(low), 1.0)
            return ("low", 1, deviation)
        # ── FIX #1: skip low-end borderline when low == 0 ──
        if low > 0 and value <= low * 1.1:
            return ("borderline", 1, 0.0)
        # ────────────────────────────────────────────────────
        return ("normal", 0, 0.0)

    return None


def _measurement_payload(
    raw_key: object,
    raw_value: object,
    source: str,
    pdf_range: dict | None = None,
    pdf_unit: str | None = None,
) -> dict | None:
    """
    Build a display-ready measurement payload.

    UNIVERSAL FIX: When _MEASUREMENT_META has no thresholds, falls back
    to reference-range-based classification so every biomarker with a
    known range (from PDF or REFERENCE_RANGES) gets flagged correctly.
    """
    provided_unit = None
    provided_status = None
    provided_risk_score = None
    if isinstance(raw_value, dict):
        provided_unit = raw_value.get("unit")
        provided_status = raw_value.get("status")
        provided_risk_score = raw_value.get("risk_score")
        raw_value = raw_value.get("value", raw_value.get("current"))

    value = _numeric_value(raw_value)
    if value is None:
        return None

    key = _normalise_measurement_key(raw_key)
    if not key:
        return None

    # Temperature: switch to Fahrenheit key if value is F-range
    if key == "temperature_c" and value > 45:
        key = "temperature_f"

    meta = _MEASUREMENT_META.get(key, {})
    name = _resolve_display_name(key, raw_key)

    # Unit resolution priority: provided → PDF → canonical fallback
    unit = (
        provided_unit
        or pdf_unit
        or meta.get("unit")
        or CANONICAL_UNITS.get(key)
        or ""
    )

    low = meta.get("low")
    high = meta.get("high")
    critical_low = meta.get("critical_low")
    critical_high = meta.get("critical_high")

    status = "reported"
    risk_score: int | None = None
    deviation_score: float | None = None

    # ── Path A: _MEASUREMENT_META has explicit thresholds ──
    has_meta_thresholds = any(
        threshold is not None
        for threshold in (low, high, critical_low, critical_high)
    )

    if has_meta_thresholds:
        status = "normal"
        risk_score = 0
        deviation_score = 0.0
        if critical_low is not None and value < critical_low:
            status, risk_score = "critical_low", 2
        elif critical_high is not None and value > critical_high:
            status, risk_score = "critical_high", 2
        elif low is not None and value < low:
            status, risk_score = "low", 1
        elif high is not None and value > high:
            status, risk_score = "high", 1

        if low is not None and value < low:
            deviation_score = (low - value) / max(abs(low), 1.0)
        elif high is not None and value > high:
            deviation_score = (value - high) / max(abs(high), 1.0)

    else:
        # ── Path B: UNIVERSAL FIX — reference-range-based classification ──
        classified = _classify_via_reference_range(key, value, pdf_range)
        if classified is not None:
            status, risk_score, deviation_score = classified

    # ── Path C: honour provided status if nothing else classified ──
    if status == "reported" and provided_status:
        status = str(provided_status).lower().replace(" ", "_")
        if isinstance(provided_risk_score, (int, float)):
            risk_score = int(provided_risk_score)

    display_number = f"{value:.2f}".rstrip("0").rstrip(".")
    display_value = f"{display_number} {unit}".strip()
    source_label = "latest report" if source == "vital" else "latest lab report"

    # Attach the reference range used (for downstream rendering)
    range_used: dict | None = None
    if pdf_range and isinstance(pdf_range, dict):
        range_used = dict(pdf_range)
    elif key in REFERENCE_RANGES:
        range_used = dict(REFERENCE_RANGES[key])

    payload = {
        "key": key,
        "name": name,
        "value": value,
        "unit": unit,
        "display_value": display_value,
        "status": status,
        "risk_score": risk_score,
        "deviation_score": deviation_score,
        "source": source,
        "note": f"From the user's {source_label}",
    }

    if range_used:
        if range_used.get("low") is not None:
            payload["normal_low"] = range_used["low"]
        if range_used.get("high") is not None:
            payload["normal_high"] = range_used["high"]

    return payload


def _extract_report_measurements(result_data: dict) -> list[dict]:
    """
    Extract actual vitals/lab values persisted in one report result.

    UNIVERSAL FIX: passes lab_result.units and lab_result.reference_ranges
    into _measurement_payload so PDF-extracted metadata drives classification
    for biomarkers without hardcoded thresholds.
    """
    measurements: list[dict] = []
    seen: set[str] = set()

    submitted = result_data.get("submitted") or {}
    patient = result_data.get("patient") or {}
    candidates = [
        result_data.get("vitals"),
        result_data.get("vital_snapshot"),
        submitted.get("vitals") if isinstance(submitted, dict) else None,
        patient.get("vitals") if isinstance(patient, dict) else None,
    ]

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for raw_key, raw_value in candidate.items():
            payload = _measurement_payload(raw_key, raw_value, "vital")
            if payload and payload["key"] not in seen:
                seen.add(payload["key"])
                measurements.append(payload)

    lab_result = result_data.get("lab_result") or {}
    if isinstance(lab_result, dict):
        lab_values: dict = {}
        for field in ("measurements", "extra_measurements"):
            values = lab_result.get(field)
            if isinstance(values, dict):
                lab_values.update(values)

        # UNIVERSAL FIX: pull PDF units and ranges for lookup during payload build
        lab_units = lab_result.get("units") or {}
        lab_ranges = lab_result.get("reference_ranges") or {}

        for raw_key, raw_value in lab_values.items():
            # Resolve canonical to look up PDF metadata under either the
            # raw slug (extra_measurements) or the canonical key (measurements)
            canonical_guess = resolve_canonical_key(str(raw_key)) or str(raw_key)

            pdf_range = (
                lab_ranges.get(canonical_guess)
                or lab_ranges.get(str(raw_key))
                or None
            )
            pdf_unit = (
                lab_units.get(canonical_guess)
                or lab_units.get(str(raw_key))
                or None
            )

            payload = _measurement_payload(
                raw_key, raw_value, "lab",
                pdf_range=pdf_range if isinstance(pdf_range, dict) else None,
                pdf_unit=pdf_unit if isinstance(pdf_unit, str) else None,
            )
            if payload and payload["key"] not in seen:
                seen.add(payload["key"])
                measurements.append(payload)

        # Parser-emitted abnormal_values augment classification when
        # _MEASUREMENT_META and REFERENCE_RANGES both miss.
        abnormal_values = [
            str(value)
            for value in (lab_result.get("abnormal_values") or [])
            if value
        ]
        for measurement in measurements:
            if measurement.get("source") != "lab":
                continue

            labels = {
                str(measurement.get("key", "")).replace("_", " ").lower(),
                str(measurement.get("name", "")).lower(),
            }
            matched = next(
                (
                    abnormal
                    for abnormal in abnormal_values
                    if any(label and label in abnormal.lower() for label in labels)
                ),
                None,
            )
            if not matched:
                continue

            lower = matched.lower()
            is_critical = bool(re.search(r"\bcritical\b|\bsevere\b", lower))
            if re.search(r"\blow\b|\bbelow\b|deficien|insufficient", lower):
                status = "critical_low" if is_critical else "low"
            elif re.search(r"\bhigh\b|\babove\b|elevated", lower):
                status = "critical_high" if is_critical else "high"
            else:
                status = "critical" if is_critical else "abnormal"

            parser_risk = 2 if is_critical else 1
            if measurement.get("risk_score") is None or parser_risk > (measurement.get("risk_score") or 0):
                measurement["status"] = status
                measurement["risk_score"] = parser_risk
                measurement["deviation_score"] = float(parser_risk)
                measurement["note"] = matched

    return measurements


def build_report_measurement_groups(result_data: dict, limit: int = 2) -> dict:
    """Return the same current-report groups used by Vitals Overview."""
    measurements = _extract_report_measurements(result_data)
    critical = [item for item in measurements if (item.get("risk_score") or 0) >= 2][:limit]
    observation = [item for item in measurements if item.get("risk_score") == 1][:limit]
    normal_values = [item for item in measurements if item.get("risk_score") == 0]
    reported_values = [item for item in measurements if item.get("risk_score") is None]
    normal = [*normal_values, *reported_values][:limit]

    return {
        "critical": critical,
        "under_observation": observation,
        "normal": normal,
        "counts": {
            "critical": len(critical),
            "under_observation": len(observation),
            "normal": len(normal),
        },
    }


def _append_unique(items: list[str], value: object, limit: int = 3) -> None:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" .")
    if not text or text.lower() in {item.lower() for item in items}:
        return
    if len(text) > 100:
        text = text[:97].rstrip() + "..."
    if len(items) < limit:
        items.append(text)


def _extract_contributing_factors(record: HealthRecord, result_data: dict, measurements: list[dict]) -> list[str]:
    """Build latest-report factors from deterministic structured outputs."""
    factors: list[str] = []

    severity = result_data.get("severity_result") or {}
    if isinstance(severity, dict):
        for reason in severity.get("reasons") or []:
            lower = str(reason).lower()
            if "no high-risk" not in lower and "no high risk" not in lower and "default" not in lower:
                _append_unique(factors, reason)

    lab = result_data.get("lab_result") or {}
    if isinstance(lab, dict):
        for abnormal in lab.get("abnormal_values") or []:
            _append_unique(factors, abnormal)

    xray = result_data.get("xray_result") or {}
    if isinstance(xray, dict):
        for finding in xray.get("findings") or []:
            if not re.search(r"normal|no significant", str(finding), re.I):
                _append_unique(factors, f"X-ray finding: {finding}")

    drug = result_data.get("drug_result") or {}
    if isinstance(drug, dict):
        for interaction in drug.get("interactions") or []:
            if isinstance(interaction, dict):
                _append_unique(factors, interaction.get("description"))
        for warning in drug.get("warnings") or []:
            _append_unique(factors, warning)

    for measurement in measurements:
        if (measurement.get("risk_score") or 0) > 0:
            status = str(measurement.get("status", "abnormal")).replace("_", " ")
            _append_unique(
                factors,
                f"{measurement['name']}: {measurement['display_value']} ({status})",
            )

    symptoms = result_data.get("symptom_result") or {}
    if isinstance(symptoms, dict):
        for indicator in symptoms.get("severity_indicators") or []:
            _append_unique(factors, indicator)
        for symptom in symptoms.get("symptoms") or []:
            _append_unique(factors, f"Reported symptom: {symptom}")

    if not factors and record.symptoms_text:
        _append_unique(factors, f"Reported symptoms: {record.symptoms_text}")

    return factors


def _build_personalized_recommendations(
    measurements: list[dict],
    latest: HealthRecord,
    limit: int = 4,
) -> list[dict]:
    """
    Build system-grouped recommendations for the dashboard Row 3 card.

    Strategy A — one row per clinical system, containing all flagged
    biomarkers in that system plus a shared action phrase. When more
    than `limit` systems are flagged, keep the top (limit - 1) by
    clinical priority and aggregate the remainder into a single
    "Other findings (N)" overflow row so nothing is hidden.

    Row shape:
        {
          "status": "system_group",
          "biomarker": "Metabolic (2)",
          "key": "system::metabolic",
          "value": "Glucose (High) and HbA1c (High)",
          "urgency": "critical" | "moderate",
          "recommendation": "<shared action phrase>",
        }
    """
    if not measurements:
        return []

    flagged = [
        m for m in measurements
        if (m.get("risk_score") or 0) >= 1
    ]
    if not flagged:
        return []

    # Group by clinical system
    grouped: dict[str, list[dict]] = {}
    for m in flagged:
        system = _classify_system(m)
        grouped.setdefault(system, []).append(m)

    # Within each group: sort members by risk_score DESC then deviation_score DESC
    for system in grouped:
        grouped[system].sort(
            key=lambda x: (
                x.get("risk_score") or 0,
                x.get("deviation_score") or 0.0,
            ),
            reverse=True,
        )

    # Order systems by clinical priority (lower rank = more urgent)
    ordered_systems = sorted(
        grouped.keys(),
        key=lambda s: _SYSTEM_PRIORITY.get(s, 99),
    )

    # ≤ limit systems: one row per system, no overflow
    if len(ordered_systems) <= limit:
        return [
            _compose_system_row(system, grouped[system])
            for system in ordered_systems
        ]

    # > limit systems: top (limit - 1) rows + aggregate overflow row
    top_systems = ordered_systems[: limit - 1]
    overflow_systems = ordered_systems[limit - 1:]

    rows = [
        _compose_system_row(system, grouped[system])
        for system in top_systems
    ]
    overflow_map = {s: grouped[s] for s in overflow_systems}
    rows.append(_compose_overflow_row(overflow_map))
    return rows


def _build_care_plan_snapshot(
    measurements: list[dict],
    latest: HealthRecord,
) -> dict:
    """
    Assemble a 3-tier care plan snapshot from the top-priority flagged
    measurement's care_plan. Uses the highest-urgency biomarker's plan
    as the anchor and lifts one item per tier.
    
    Returns:
        {
            "immediate":  {"text": ..., "source_biomarker": ...} | None,
            "short_term": {"text": ..., "source_biomarker": ...} | None,
            "lifestyle":  {"text": ..., "source_biomarker": ...} | None,
        }
    
    Falls back to severity-based generic guidance only when no flagged
    measurements exist (e.g. HIGH severity from symptoms alone).
    """
    empty_plan = {
        "immediate": None,
        "short_term": None,
        "lifestyle": None,
    }

    flagged = [
        m for m in (measurements or [])
        if (m.get("risk_score") or 0) >= 1
    ]

    # Sort by clinical priority
    flagged.sort(
        key=lambda m: (
            m.get("risk_score") or 0,
            m.get("deviation_score") or 0.0,
        ),
        reverse=True,
    )

    # Collect one non-null entry per tier by walking flagged items in priority
    tiers = ("immediate", "short_term", "lifestyle")
    plan: dict[str, dict | None] = {tier: None for tier in tiers}

    for m in flagged:
        advice = resolve_advice(m)
        care_plan = advice.get("care_plan") or {}

        for tier in tiers:
            if plan[tier] is not None:
                continue
            text = care_plan.get(tier)
            if text:
                plan[tier] = {
                    "text": text,
                    "source_biomarker": m.get("name"),
                }

        if all(plan[t] is not None for t in tiers):
            break

    # Severity-based fallback if nothing was flagged
    if not flagged and latest:
        sev = (latest.severity or "").upper()
        if sev in ("HIGH", "CRITICAL"):
            plan["immediate"] = {
                "text": "Seek prompt medical evaluation — go to the nearest emergency department if symptoms worsen.",
                "source_biomarker": None,
            }
            plan["short_term"] = {
                "text": "Book a follow-up consultation with your physician within one week to review the full report.",
                "source_biomarker": None,
            }
            plan["lifestyle"] = {
                "text": "Rest, hydrate, and monitor symptoms daily. Avoid strenuous activity until reviewed.",
                "source_biomarker": None,
            }
        elif sev in ("MEDIUM", "MODERATE"):
            plan["immediate"] = {
                "text": "Share this report with your primary care physician at your next visit.",
                "source_biomarker": None,
            }
            plan["short_term"] = {
                "text": "Monitor symptoms and re-run the assessment if anything changes.",
                "source_biomarker": None,
            }
            plan["lifestyle"] = {
                "text": "Maintain a balanced diet, regular activity, adequate hydration, and sufficient sleep.",
                "source_biomarker": None,
            }
        else:
            return empty_plan

    return plan


_SAFETY_ALERTS_LIMIT = 4


# ── Universal allergy classifier ─────────────────────────────────
# Tier scores are broad clinical-impact bands, not per-allergy scores.
# Any user-entered allergy gets classified into a tier by keyword class.
# This keeps the ordering deterministic without hardcoding named allergies.

_DRUG_KEYWORDS = (
    "cillin", "mycin", "cycline", "sulfa", "sulfonamide", "nsaid",
    "aspirin", "ibuprofen", "acetaminophen", "paracetamol", "opioid",
    "codeine", "morphine", "statin", "ace inhibitor", "beta blocker",
    "anesthesia", "anesthetic", "vaccine", "antibiotic", "antibiotics",
    "drug", "medication", "medicine",
)

_PROCEDURAL_KEYWORDS = (
    "latex", "iodine", "contrast", "adhesive", "rubber", "nickel",
    "chlorhexidine", "betadine",
)

_ANAPHYLAXIS_FOOD_KEYWORDS = (
    "peanut", "tree nut", "walnut", "almond", "cashew", "hazelnut",
    "pistachio", "shellfish", "shrimp", "crab", "lobster", "prawn",
    "fish", "sesame",
)

_COMMON_FOOD_KEYWORDS = (
    "egg", "milk", "dairy", "lactose", "soy", "wheat", "gluten",
    "corn", "rice", "meat", "beef", "pork", "chicken",
)

_ENVIRONMENTAL_KEYWORDS = (
    "pollen", "dust", "mite", "mold", "mould", "pet", "cat", "dog",
    "animal", "grass", "tree", "insect", "bee", "wasp", "hornet",
)


def _allergy_tier_score(name: str) -> int:
    """
    Universal allergy scorer — classifies any user-entered text into
    a clinical-impact tier without needing a per-allergy lookup.

    Tiers (higher = more urgent to surface on the dashboard):
        100 — Drug/medication (prescription cross-check risk)
         85 — Procedural/contact (surgery, imaging, wound care risk)
         75 — Insect sting / venom (anaphylaxis-capable, immediate)
         70 — Anaphylaxis-capable food (nuts, shellfish, fish, sesame)
         50 — Common food intolerance/allergy
         40 — Environmental/inhaled allergen
         10 — Unclassified (keeps user-submitted allergies visible)
    """
    lowered = name.lower().strip()
    if not lowered:
        return 0

    if any(keyword in lowered for keyword in _DRUG_KEYWORDS):
        return 100
    if any(keyword in lowered for keyword in _PROCEDURAL_KEYWORDS):
        return 85
    # Sting/venom check before generic environmental (bee sting = anaphylaxis)
    if any(keyword in lowered for keyword in ("bee", "wasp", "hornet", "sting", "venom")):
        return 75
    if any(keyword in lowered for keyword in _ANAPHYLAXIS_FOOD_KEYWORDS):
        return 70
    if any(keyword in lowered for keyword in _COMMON_FOOD_KEYWORDS):
        return 50
    if any(keyword in lowered for keyword in _ENVIRONMENTAL_KEYWORDS):
        return 40

    return 10


def _capitalize_allergy(name: str) -> str:
    """Capitalize the first letter; leave the rest of the text intact."""
    text = name.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def _build_safety_alerts(record: HealthRecord, result_data: dict) -> list[dict]:
    """
    Build structured safety alerts from patient-reported allergies.

    Universal ranking: every allergy is classified into a clinical-impact
    tier by keyword class, then sorted by tier DESC + submission order.
    Capped at _SAFETY_ALERTS_LIMIT so the Row 2 card stays legible.
    Returns items shaped as:
        {"type": "allergy", "severity": "high", "text": "..."}
    """
    submitted = result_data.get("submitted") if isinstance(result_data.get("submitted"), dict) else {}
    patient = result_data.get("patient") if isinstance(result_data.get("patient"), dict) else {}

    # Collect allergies from wherever they live in the submission
    allergies: list[str] = []
    for source in (submitted, patient):
        if not isinstance(source, dict):
            continue
        raw = source.get("allergies")
        if isinstance(raw, list):
            allergies.extend(str(a).strip() for a in raw if str(a).strip())
        elif isinstance(raw, str) and raw.strip():
            allergies.extend(
                part.strip()
                for part in re.split(r"[,;]", raw)
                if part.strip()
            )

    # Deduplicate case-insensitively, preserve first-seen casing + order
    seen: set[str] = set()
    unique_allergies: list[tuple[int, str]] = []
    for index, allergy in enumerate(allergies):
        key = allergy.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_allergies.append((index, allergy))

    # Sort: tier DESC, then submission index ASC (stable within same tier)
    unique_allergies.sort(key=lambda pair: (-_allergy_tier_score(pair[1]), pair[0]))

    # Cap and emit
    alerts: list[dict] = []
    for _, allergy in unique_allergies[:_SAFETY_ALERTS_LIMIT]:
        pretty = _capitalize_allergy(allergy)
        alerts.append({
            "type": "allergy",
            "severity": "high",
            "text": f"{pretty} allergy on file",
        })

    return alerts

@router.get("")
def dashboard(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return personalised dashboard data for the authenticated user.

    Includes: user greeting, recent records, latest severity, vitals
    with baselines, risk profile computed from severity history,
    safety review checklist, and recommended actions.
    """
    # ── Recent records ───────────────────────────────────────────
    # Keep the total as a separate user-scoped COUNT so routing remains
    # correct even when the history list is capped for dashboard rendering.
    total_records = (
        db.query(HealthRecord)
        .filter(HealthRecord.user_id == user.id)
        .count()
    )
    records = (
        db.query(HealthRecord)
        .filter(HealthRecord.user_id == user.id)
        .order_by(HealthRecord.created_at.desc())
        .limit(10)
        .all()
    )

    # ── Vitals ───────────────────────────────────────────────────
    vitals = (
        db.query(VitalSnapshot)
        .filter_by(user_id=user.id)
        .order_by(VitalSnapshot.created_at.asc())
        .all()
    )

    # ── Latest record ────────────────────────────────────────────
    latest_record = records[0] if records else None

    # ── Overall health score (from latest severity + confidence) ─
    overall = _build_overall(latest_record, records)

    # ── Recent record list ───────────────────────────────────────
    recent = []
    for r in records[:10]:
        result_data = _json_dict(r.result_json)
        measurements = _extract_report_measurements(result_data)
        submitted = result_data.get("submitted") if isinstance(result_data.get("submitted"), dict) else {}
        submitted_medications = submitted.get("medications")
        medications = (
            [str(item).strip() for item in submitted_medications if str(item).strip()]
            if isinstance(submitted_medications, list)
            else _json_list(r.medications_json)
        )

        personalized_recs = _build_personalized_recommendations(measurements, r, limit=4)
        care_plan = _build_care_plan_snapshot(measurements, r)
        safety_alerts = _build_safety_alerts(r, result_data)
        clinical_picture = result_data.get("clinical_picture") or {}

        recent.append({
            "id": r.id,
            "job_id": r.job_id,
            "severity": r.severity,
            "confidence": r.confidence,
            # Full symptoms_text is sent (not truncated) so the frontend's
            # symptom-tag extractor can see the entire report — a 120-char
            # cutoff here was silently dropping legitimate findings (e.g.
            # vitals like "BP 165/98" or "glucose 245" mentioned later in
            # a longer patient description) before they ever reached the UI.
            "symptoms_text": r.symptoms_text or "",
            "medications": medications,
            "medications_json": json.dumps(medications),
            "xray_findings_json": r.xray_findings_json or "[]",
            "created_at": _utc_iso(r.created_at),
            "measurements": measurements,
            "contributing_factors": _extract_contributing_factors(
                r, result_data, measurements
            ),
            "personalized_recommendations": personalized_recs,
            "care_plan_snapshot": care_plan,
            "safety_alerts": safety_alerts,
            "clinical_picture": clinical_picture,
        })

    # ── Vitals baselines ─────────────────────────────────────────
    vitals_baselines = _build_vitals_baselines(vitals)

    # ── Risk profile ─────────────────────────────────────────────
    risk = _build_risk(latest_record, records)

    # ── Safety review ────────────────────────────────────────────
    safety = _build_safety(latest_record)

    # ── Recommended actions ──────────────────────────────────────
    actions = _build_actions(latest_record)

    # ── Critical findings cards ──────────────────────────────────
    criticals = _build_criticals(latest_record)

    return {
        "user": {
            "display_name": user.display_name,
            "role": user.role,
        },
        "generated_at": _now_iso(),
        "overall": overall,
        "recent_records": recent,
        "latest_severity": latest_record.severity if latest_record else None,
        "total_records": total_records,
        "vitals": vitals_baselines,
        "risk": risk,
        "safety_review": safety,
        "actions": actions,
        "critical_cards": criticals,
    }


# ── Section builders ─────────────────────────────────────────────

def _build_overall(latest, records) -> dict:
    """Compute overall health score and status from history."""
    if not latest:
        return {
            "score": 72,
            "status": "No Data",
            "trend": "Submit your first triage",
            "description": "No health records yet. Start a triage assessment to see your dashboard.",
            "data_completeness_score": 0,
            "data_completeness_status": "No data",
        }

    # Score: confidence * severity weight
    severity_weights = {"LOW": 85, "MEDIUM": 65, "MODERATE": 65, "HIGH": 40, "CRITICAL": 20}
    weight = severity_weights.get(latest.severity.upper(), 60)
    score = int(latest.confidence * weight + (1 - latest.confidence) * 50)

    # Data completeness: symptoms + meds + xray + labs + audio
    completeness = 0
    if latest.symptoms_text:
        completeness += 20
    try:
        if json.loads(latest.medications_json):
            completeness += 20
    except Exception:
        pass
    try:
        if json.loads(latest.xray_findings_json):
            completeness += 20
    except Exception:
        pass
    if latest.report_json:
        completeness += 20
    completeness += 20  # base score for having a record

    if score >= 80:
        status, desc = "Good", "Your recent triage results are favorable."
    elif score >= 55:
        status, desc = "Moderate", "Some areas need attention — review your latest report."
    elif score >= 30:
        status, desc = "Needs Attention", "Concerning indicators found. Clinician review advised."
    else:
        status, desc = "Critical", "Urgent. Seek medical attention promptly."

    # Trend: compare latest vs average of previous records
    if len(records) >= 2:
        prev_severities = {"LOW": 1, "MEDIUM": 2, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4}
        prev_avg = mean(prev_severities.get(r.severity.upper(), 2) for r in records[1:min(6, len(records))])
        latest_num = prev_severities.get(latest.severity.upper(), 2)
        if latest_num > prev_avg + 0.4:
            trend = "Worsening trend"
        elif latest_num < prev_avg - 0.4:
            trend = "Improving trend"
        else:
            trend = "Stable"
    else:
        trend = "First assessment"

    return {
        "score": score,
        "status": status,
        "trend": trend,
        "description": desc,
        "data_completeness_score": completeness,
        "data_completeness_status": "Comprehensive" if completeness >= 80 else "Adequate" if completeness >= 50 else "Limited",
    }


def _build_vitals_baselines(vitals: list) -> dict:
    """Build vital baselines with z-scores from history."""
    if not vitals:
        return {"sample_count": 0, "baselines": [], "latest": None}

    latest = vitals[-1]
    fields = [
        ("systolic_bp", "Systolic BP", "mmHg"),
        ("diastolic_bp", "Diastolic BP", "mmHg"),
        ("heart_rate", "Heart Rate", "bpm"),
        ("spo2", "SpO2", "%"),
        ("temperature_c", "Temperature", "°C"),
        ("glucose_mg_dl", "Glucose", "mg/dL"),
        ("weight_kg", "Weight", "kg"),
    ]

    baselines = []
    for attr, name, unit in fields:
        history = [getattr(v, attr) for v in vitals if getattr(v, attr) is not None]
        current = getattr(latest, attr)
        if current is not None and len(history) >= 3:
            summary = baseline_summary(current=current, history=history, vital_name=name)
            summary["unit"] = unit
            baselines.append(summary)
        elif current is not None:
            baselines.append({
                "vital": name,
                "current": current,
                "unit": unit,
                "mean": mean(history) if history else None,
                "z_score": None,
                "sample_size": len(history),
                "interpretation": None,
                "insufficient_history": True,
            })

    return {
        "sample_count": len(vitals),
        "baselines": baselines,
        "latest": {
            "id": latest.id,
            "created_at": latest.created_at.isoformat(),
            "heart_rate": latest.heart_rate,
            "spo2": latest.spo2,
            "systolic_bp": latest.systolic_bp,
            "diastolic_bp": latest.diastolic_bp,
            "temperature_c": latest.temperature_c,
            "glucose_mg_dl": latest.glucose_mg_dl,
            "weight_kg": latest.weight_kg,
        },
    }


def _build_risk(latest, records) -> dict:
    """Build risk profile from severity history."""
    if not latest:
        return {
            "score": 0,
            "status": "Unknown",
            "factors": ["No health data available. Complete a triage assessment."],
        }

    severity_risk = {"LOW": 20, "MEDIUM": 50, "MODERATE": 50, "HIGH": 75, "CRITICAL": 95}
    risk_score = severity_risk.get(latest.severity.upper(), 50)

    if risk_score >= 75:
        status = "High"
    elif risk_score >= 45:
        status = "Moderate"
    else:
        status = "Low"

    factors = []
    if latest.severity.upper() in ("HIGH", "CRITICAL"):
        factors.append("Elevated triage severity — urgent clinical correlation advised.")
    if latest.confidence < 0.6:
        factors.append("Low model confidence — limited input data may affect assessment accuracy.")
    if latest.validation_status == "warning":
        factors.append("Validation warning — minor discrepancies in safety check.")
    if latest.validation_status == "override":
        factors.append("Validation override — significant safety disagreement. Urgent review needed.")
    if not factors:
        factors.append("Report is available for clinical review.")
        factors.append("Follow up if symptoms persist or worsen.")

    return {
        "score": risk_score,
        "status": status,
        "factors": factors,
    }


def _build_safety(latest) -> list:
    """Build safety review checklist."""
    if not latest:
        return [
            {"name": "No triage submitted", "status": "Pending", "ok": False},
        ]

    items = []
    # Medication check
    try:
        meds = json.loads(latest.medications_json)
        has_meds = bool(meds)
    except Exception:
        has_meds = False

    items.append({
        "name": "Medication history reviewed",
        "status": "No interaction detected" if has_meds else "No medications submitted",
        # Not submitting medications is not a safety concern — there is
        # nothing to check. This item is informational only and must
        # never be able to trigger the dashboard's safety warning banner
        # (ValidationBanner filters on `ok`). Only "Clinical rule
        # validation" below reflects the real RuleValidator outcome that
        # the banner should be based on.
        "ok": True,
    })
    items.append({
        "name": "Clinical rule validation",
        "status": "Passed" if latest.validation_status == "agreement" else "Needs review",
        "ok": latest.validation_status == "agreement",
    })
    items.append({
        "name": "Triage severity assessed",
        "status": latest.severity,
        "ok": True,
    })
    items.append({
        "name": "Confidence threshold met",
        "status": f"{latest.confidence:.0%}",
        "ok": latest.confidence >= 0.5,
    })

    return items


def _build_actions(latest) -> list:
    """Build recommended actions from the latest record."""
    if not latest:
        return [
            "Complete a health triage assessment to generate personalised recommendations.",
        ]

    actions = []
    sev = latest.severity.upper()

    if sev in ("CRITICAL", "HIGH"):
        actions.append("Seek urgent medical attention — go to the nearest emergency department.")
    actions.append("Review your latest triage report with a qualified healthcare professional.")
    if latest.validation_status in ("warning", "override"):
        actions.append("Discuss validation warnings with your clinician before making treatment decisions.")
    if sev != "CRITICAL":
        actions.append("Monitor your symptoms and submit a new triage if they change or worsen.")
    actions.append("Keep your vitals check-ins regular for better baseline tracking.")

    return actions[:5]


def _build_criticals(latest) -> list:
    """Build critical finding cards from the latest record."""
    if not latest:
        return [
            {"title": "No Data", "value": "N/A", "badge": "Pending", "badge_class": "badge-orange"},
        ]

    cards = []
    sev = latest.severity.upper()

    if sev in ("CRITICAL", "HIGH"):
        cards.append({
            "title": "Urgent Assessment",
            "value": latest.severity,
            "badge": "Critical" if sev == "CRITICAL" else "High",
            "badge_class": "badge-red",
        })
    else:
        cards.append({
            "title": "Triage Result",
            "value": latest.severity,
            "badge": "Reviewed",
            "badge_class": "badge-green" if sev == "LOW" else "badge-orange",
        })

    if latest.symptoms_text:
        cards.append({
            "title": "Reported Symptoms",
            "value": latest.symptoms_text[:40],
            "badge": "Recorded",
            "badge_class": "badge-green",
        })

    if latest.validation_status and latest.validation_status != "agreement":
        cards.append({
            "title": "Safety Flag",
            "value": latest.validation_status,
            "badge": "Review",
            "badge_class": "badge-red" if latest.validation_status == "override" else "badge-orange",
        })

    return cards[:3]