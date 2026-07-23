# tools/report_analyst.py
"""
Report intelligence layer.

Transforms the raw context dict (built by _build_report_context in
backend/chat.py) into a typed, semantically enriched object that both
the chat answer engine and the question generator reason over.

Why this exists:
  - Eliminates repeated JSON parsing across chat turns
  - Gives downstream code typed fields instead of dict.get() chains
  - to_prompt_block() produces a compact, high-signal text block for
    the enrichment model — the model sees structured values, not 4000
    chars of raw JSON
  - Symptom clustering by body system enables smarter suggestions and
    more specific fallback answers

No business logic lives here — pure parsing and classification.

Fixes vs. earlier version:
  - _clean_symptom() now EXTRACTS an atomic phrase from a blob whenever
    one is present (with word-boundary matching so "dizzy" matches
    "and i feel dizzy sometimes" and returns just "dizzy").
  - _ATOMIC_SYMPTOMS expanded to include short/colloquial forms users
    actually type ("dizzy", "faint", "tired") not only their formal
    equivalents ("dizziness", "fainting", "tiredness").
  - _MAX_SYMPTOM_WORDS reduced from 5 → 3 so borderline-long blobs
    don't leak through as raw "symptoms".
  - _VALUE_RE now requires a word boundary before the number so
    "HbA1c" is never split into "HbA" + value="1c".
  - _parse_lab_finding() searches for value/unit in the post-name
    portion of the string so digits embedded in the name (like the
    "1" in HbA1c) can never be misread as the lab's numeric value.
  - LabFinding.one_line() strips a literal "None" unit token so we
    never render "TSH (6.05None)" as the unit slot.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Severity helpers ──────────────────────────────────────────────

SEVERITY_RANK: dict[str, int] = {
    "LOW": 1, "MODERATE": 2, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4,
}

URGENT_SEVERITIES = {"HIGH", "CRITICAL"}
MODERATE_SEVERITIES = {"MEDIUM", "MODERATE"}


def severity_rank(s: str) -> int:
    return SEVERITY_RANK.get((s or "").upper(), 0)


def is_urgent(severity: str) -> bool:
    return (severity or "").upper() in URGENT_SEVERITIES


def is_moderate(severity: str) -> bool:
    return (severity or "").upper() in MODERATE_SEVERITIES


# ── Symptom display normalization (single source of truth) ─────────
# Any caller that shows a symptom name to the user (chat answers,
# suggested-question chips, report body, etc.) should go through
# display_symptom()/join_symptoms() rather than rendering
# intel.symptom_groups.* strings directly. This is what keeps
# "dizzy, headache is on record" from reappearing in a new call site.
_SYMPTOM_DISPLAY_MAP = {
    "dizzy": "dizziness",
    "faint": "fainting",
    "tired": "fatigue",
    "lightheaded": "lightheadedness",
    "light-headed": "lightheadedness",
    "breathless": "breathlessness",
    "short of breath": "shortness of breath",
    "nauseous": "nausea",
    "queasy": "nausea",
    "throwing up": "vomiting",
    "achy": "aching",
}


def display_symptom(raw: str) -> str:
    """Map a single raw/colloquial symptom string to its proper noun form.

    Two-tier normalization so ANY raw extraction value renders cleanly,
    not just the ones in the alias map:
      1. Known colloquialism/typo → mapped noun form (e.g. "dizzy" → "dizziness").
      2. Anything else → case-normalized to lowercase (extraction can hand
         back "SWOLLEN ANKLES", "Swollen Ankles", etc. with no consistent
         casing; lowercasing here means callers can always safely
         .capitalize() the first word without fighting raw ALL-CAPS text).
    Multi-word acronyms the report might contain (e.g. "COPD", "TSH")
    aren't symptom names, so lowercasing symptom text is safe here even
    though we deliberately don't do this for lab/biomarker names.
    """
    if not raw:
        return ""
    stripped = raw.strip()
    key = stripped.lower()
    if key in _SYMPTOM_DISPLAY_MAP:
        return _SYMPTOM_DISPLAY_MAP[key]
    return key


def join_symptoms(symptoms: list[str], limit: int | None = None) -> str:
    """Comma-join symptoms, each passed through display_symptom()."""
    items = symptoms[:limit] if limit else symptoms
    return ", ".join(display_symptom(s) for s in items if s)


def _resolve_confidence_pct(ctx: dict) -> int:
    """Resolve a 0-100 confidence percentage from context, regardless of
    which key/scale the caller used. Single source of truth for this
    conversion so no caller can silently produce a 0% confidence by
    passing 'confidence' (0-1 scale) instead of 'confidence_percent'
    (0-100 scale), or vice versa.
    """
    if "confidence_percent" in ctx and ctx["confidence_percent"] is not None:
        try:
            return int(round(float(ctx["confidence_percent"])))
        except (TypeError, ValueError):
            return 0
    if "confidence" in ctx and ctx["confidence"] is not None:
        try:
            val = float(ctx["confidence"])
        except (TypeError, ValueError):
            return 0
        # Heuristic: values <= 1 are the 0-1 scale, anything above is
        # already a percentage (defensive against either convention).
        return int(round(val * 100)) if val <= 1 else int(round(val))
    return 0


# ── Finding types ─────────────────────────────────────────────────

@dataclass
class LabFinding:
    name: str
    raw_text: str
    value: str | None
    unit: str | None
    is_abnormal: bool
    direction: str | None   # "high" | "low" | None

    def one_line(self) -> str:
        """
        Render like "HbA1c = 9.2% (↑)" — never leaks a literal "None"
        as a unit even if the parser stored the string "None" instead
        of the Python None singleton.
        """
        parts = [self.name]
        if self.value is not None and self.value != "":
            unit_ok = self.unit and self.unit.lower() not in ("none", "null")
            if unit_ok:
                parts.append(f"= {self.value}{self.unit}")
            else:
                parts.append(f"= {self.value}")
        if self.direction:
            parts.append("(↑)" if self.direction == "high" else "(↓)")
        return " ".join(parts)


@dataclass
class DrugInteraction:
    drugs: list[str]
    severity: str | None
    description: str | None

    def one_line(self) -> str:
        names = " + ".join(self.drugs) if self.drugs else "unknown drugs"
        sev = f"[{self.severity}] " if self.severity else ""
        desc = self.description or "interaction flagged"
        return f"{sev}{names}: {desc}"


@dataclass
class XrayFinding:
    text: str
    is_significant: bool

    def one_line(self) -> str:
        return self.text


@dataclass
class SymptomGroup:
    cardiac: list[str] = field(default_factory=list)
    respiratory: list[str] = field(default_factory=list)
    neurological: list[str] = field(default_factory=list)
    metabolic: list[str] = field(default_factory=list)
    gastrointestinal: list[str] = field(default_factory=list)
    musculoskeletal: list[str] = field(default_factory=list)
    general: list[str] = field(default_factory=list)

    def all_symptoms(self) -> list[str]:
        out: list[str] = []
        for lst in [
            self.cardiac, self.respiratory, self.neurological,
            self.metabolic, self.gastrointestinal,
            self.musculoskeletal, self.general,
        ]:
            out.extend(lst)
        return out

    def dominant_system(self) -> str | None:
        counts = {
            "cardiac": len(self.cardiac),
            "respiratory": len(self.respiratory),
            "neurological": len(self.neurological),
            "metabolic": len(self.metabolic),
            "gastrointestinal": len(self.gastrointestinal),
            "musculoskeletal": len(self.musculoskeletal),
        }
        best = max(counts, key=lambda k: counts[k])
        return best if counts[best] > 0 else None


# ── Symptom cluster keyword sets ──────────────────────────────────

_CARDIAC_KW = (
    "chest pain", "chest tightness", "palpitation", "heart", "bp",
    "blood pressure", "hypertension", "pulse", "heart rate",
    "systolic", "diastolic", "bradycardia", "tachycardia",
)
_RESPIRATORY_KW = (
    "breath", "breathing", "cough", "wheez", "dyspnea", "oxygen",
    "spo2", "saturation", "inhaler", "asthma", "copd",
)
_NEURO_KW = (
    "headache", "dizzy", "dizziness", "faint", "fainting", "seizure",
    "confusion", "memory", "tingling", "numbness", "vision",
    "balance", "lightheaded", "light-headed",
)
_METABOLIC_KW = (
    "glucose", "sugar", "diabetes", "diabetic", "hba1c", "thyroid",
    "tsh", "weight", "bmi", "vitamin d", "cholesterol", "lipid",
    "triglyceride", "insulin",
)
_GI_KW = (
    "nausea", "vomit", "diarrhea", "constipat", "stomach", "abdomen",
    "abdominal", "bowel", "reflux", "heartburn", "bloat",
)
_MSK_KW = (
    "joint", "knee", "back pain", "muscle", "ache", "arthrit",
    "swelling", "stiffness", "mobility",
)

_XRAY_SIGNIFICANT_KW = (
    "consolidat", "effusion", "pneumonia", "pneumothorax", "opacity",
    "cardiomegaly", "infiltrate", "nodule", "mass", "fracture",
)

_DIRECTION_RE = re.compile(
    r"\b(high|elevated|raised|increased|↑|low|reduced|decreased|↓|borderline)\b",
    re.IGNORECASE,
)
_HIGH_WORDS = {"high", "elevated", "raised", "increased", "↑"}
_LOW_WORDS = {"low", "reduced", "decreased", "↓"}

# Number must be preceded by whitespace, ':', '=', or the start of the
# string. This prevents matching "1c" inside "HbA1c" as the numeric value.
_VALUE_RE = re.compile(
    r"(?:^|[\s:=])(\d+\.?\d*(?:/\d+\.?\d*)?)\s*([a-zA-Z/%µ]+(?:/[a-zA-Z]+)?)?",
    re.IGNORECASE,
)


# ── Symptom cleaning ──────────────────────────────────────────────
# The pipeline's symptom_result.symptoms field frequently contains long
# free-text blobs (e.g. the entire submitted history). Without cleaning
# these blobs would leak into suggested-question templates and produce
# ugly text like "Could my i have been feeling very tired for the past
# week be a sign of something serious?".
#
# _clean_symptom() does one of three things:
#   1. If any known atomic phrase (e.g. "chest tightness", "dizzy")
#      appears in the raw text — with a word boundary — extract and
#      return that phrase alone. This is the PRIMARY path.
#   2. If the raw string is short (<= 3 words), return it as-is.
#   3. Otherwise return None — the blob is rejected and never appears
#      as a single-symptom label anywhere.

_ATOMIC_SYMPTOMS = (
    # Cardiac
    "chest tightness", "chest pain", "chest pressure",
    "palpitations", "irregular heartbeat", "rapid heartbeat",
    "high blood pressure", "low blood pressure",
    # Respiratory
    "shortness of breath", "difficulty breathing", "breathlessness",
    "cough", "dry cough", "productive cough", "wheezing",
    "sore throat", "runny nose",
    # Constitutional
    "fever", "high fever", "low fever",
    "fatigue", "tiredness", "tired", "weakness", "exhaustion",
    # Neurological — BOTH formal and colloquial forms so user-typed
    # "and i feel dizzy sometimes" gets collapsed to just "dizzy".
    "headache", "migraine",
    "dizziness", "dizzy", "lightheadedness", "lightheaded",
    "vertigo", "fainting", "faint", "blackout",
    "confusion", "memory loss", "tingling", "numbness",
    "blurred vision", "vision changes",
    # GI
    "nausea", "vomiting", "diarrhea", "constipation",
    "abdominal pain", "stomach pain",
    # MSK
    "back pain", "joint pain",
    # Metabolic / other
    "swelling", "rash", "itching",
    "weight loss", "weight gain", "appetite loss",
    "insomnia", "sleep problems",
    "high glucose", "low glucose",
)

# Match longest phrases first so "chest tightness" wins over "chest".
_ATOMIC_SYMPTOMS_SORTED = tuple(sorted(_ATOMIC_SYMPTOMS, key=len, reverse=True))

_MAX_SYMPTOM_WORDS = 3


def _clean_symptom(raw: str) -> str | None:
    """
    Return a clean atomic symptom label, or None if the raw string is a
    free-text blob that cannot be reduced to a clean label.

    Priority:
      1. Extract the longest matching atomic phrase using word-boundary
         regex — collapses long blobs into a clean short label
         (e.g. "and i feel dizzy sometimes" → "dizzy").
      2. If the raw string is already short (<= 3 words), return it.
      3. Otherwise reject.
    """
    if not raw:
        return None
    text = re.sub(r"\s+", " ", raw.strip()).lower()
    if not text:
        return None

    # Rule 1 (primary): extract atomic phrase with word-boundary match.
    for phrase in _ATOMIC_SYMPTOMS_SORTED:
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            return phrase

    # Rule 2: short blob
    words = text.split()
    if len(words) <= _MAX_SYMPTOM_WORDS:
        return text

    # Rule 3: reject
    return None


# ── Main intelligence object ──────────────────────────────────────

@dataclass
class ReportIntelligence:
    """
    A fully parsed, semantically enriched view of one completed report.

    Build via ReportIntelligence.from_context(context_dict) where
    context_dict is what _build_report_context() returns in chat.py.
    """

    # Identity
    job_id: str
    severity: str
    severity_rank: int
    confidence_pct: int
    validation_status: str | None

    # Symptoms
    symptom_groups: SymptomGroup
    symptom_duration: str | None
    severity_indicators: list[str]
    severity_reasons: list[str]

    # Labs
    abnormal_labs: list[LabFinding]
    normal_labs: list[LabFinding]

    # Medications
    medications: list[str]
    drug_interactions: list[DrugInteraction]
    drug_warnings: list[str]

    # Imaging
    xray_findings: list[XrayFinding]

    # Raw text
    report_text: str

    # Computed flags (set by __post_init__)
    has_cardiac_risk: bool = False
    has_respiratory_risk: bool = False
    has_neuro_risk: bool = False
    has_metabolic_risk: bool = False
    has_significant_xray: bool = False
    has_drug_interaction: bool = False
    has_abnormal_labs: bool = False

    def __post_init__(self) -> None:
        self.has_cardiac_risk = bool(self.symptom_groups.cardiac) or any(
            any(kw in lab.name.lower() for kw in ("bp", "pressure", "cardiac", "troponin"))
            for lab in self.abnormal_labs
        )
        self.has_respiratory_risk = bool(self.symptom_groups.respiratory) or any(
            any(kw in lab.name.lower() for kw in ("oxygen", "spo2", "co2"))
            for lab in self.abnormal_labs
        )
        self.has_neuro_risk = bool(self.symptom_groups.neurological)
        self.has_metabolic_risk = bool(self.symptom_groups.metabolic) or any(
            any(kw in lab.name.lower() for kw in ("glucose", "hba1c", "thyroid", "tsh", "vitamin"))
            for lab in self.abnormal_labs
        )
        self.has_significant_xray = any(f.is_significant for f in self.xray_findings)
        self.has_drug_interaction = bool(self.drug_interactions)
        self.has_abnormal_labs = bool(self.abnormal_labs)

    # ── Factory ───────────────────────────────────────────────────

    @classmethod
    def from_context(cls, ctx: dict[str, Any]) -> "ReportIntelligence":
        severity = str(ctx.get("severity") or "UNKNOWN").upper()

        # ── Symptom clustering ───────────────────────────────────
        # Collect candidate symptoms from BOTH the extracted list AND
        # by scanning the raw reported_symptoms free-text for known
        # atomic symptom phrases. _clean_symptom() then filters and
        # normalises each candidate so long blobs never leak through.
        raw_symptoms: list[str] = []

        extracted = ctx.get("extracted_symptoms") or []
        if isinstance(extracted, list):
            raw_symptoms.extend(str(s) for s in extracted if s)

        reported = str(ctx.get("reported_symptoms") or "")
        if reported and reported.lower() not in ("none reported", ""):
            reported_lower = reported.lower()
            # Scan the free text for any known atomic symptom
            for phrase in _ATOMIC_SYMPTOMS_SORTED:
                if re.search(r"\b" + re.escape(phrase) + r"\b", reported_lower):
                    if phrase not in raw_symptoms:
                        raw_symptoms.append(phrase)
            # Also accept short comma-separated items as-is
            for chunk in re.split(r"[,;]", reported):
                chunk = chunk.strip()
                if chunk and len(chunk.split()) <= _MAX_SYMPTOM_WORDS:
                    if chunk not in raw_symptoms:
                        raw_symptoms.append(chunk)

        groups = _cluster_symptoms(raw_symptoms)

        # ── Lab parsing ──────────────────────────────────────────
        abnormal_raw: list[str] = ctx.get("lab_abnormal_values") or []
        measurements: dict = ctx.get("lab_measurements") or {}
        abnormal_labs = [_parse_lab_finding(t, is_abnormal=True) for t in abnormal_raw if t]
        normal_labs = [
            _parse_lab_finding(f"{k}: {v}", is_abnormal=False)
            for k, v in measurements.items()
            if not any(t.lower() in k.lower() for t in abnormal_raw)
        ]

        # ── Drug interactions ────────────────────────────────────
        raw_interactions = ctx.get("drug_interactions") or []
        interactions: list[DrugInteraction] = []
        for item in raw_interactions:
            if isinstance(item, dict):
                drugs_val = item.get("drugs")
                interactions.append(DrugInteraction(
                    drugs=drugs_val if isinstance(drugs_val, list) else [],
                    severity=item.get("severity"),
                    description=item.get("description"),
                ))

        # ── X-ray findings ───────────────────────────────────────
        xray_raw: list[str] = ctx.get("xray_findings") or []
        xray_findings = [_parse_xray_finding(t) for t in xray_raw if t]
        xray_notes = str(ctx.get("xray_notes") or "")
        if xray_notes and not any(xray_notes in f.text for f in xray_findings):
            xray_findings.append(_parse_xray_finding(xray_notes))

        return cls(
            job_id=str(ctx.get("job_id") or ""),
            severity=severity,
            severity_rank=severity_rank(severity),
            confidence_pct=_resolve_confidence_pct(ctx),
            validation_status=ctx.get("validation_status"),
            symptom_groups=groups,
            symptom_duration=ctx.get("symptom_duration"),
            severity_indicators=list(ctx.get("severity_indicators") or []),
            severity_reasons=list(ctx.get("severity_reasons") or []),
            abnormal_labs=abnormal_labs,
            normal_labs=normal_labs,
            medications=list(ctx.get("medications") or []),
            drug_interactions=interactions,
            drug_warnings=list(ctx.get("drug_warnings") or []),
            xray_findings=xray_findings,
            report_text=str(ctx.get("report_text") or "")[:4000],
        )

    # ── Convenience accessors ─────────────────────────────────────

    def top_abnormal_lab(self) -> LabFinding | None:
        return self.abnormal_labs[0] if self.abnormal_labs else None

    def top_interaction(self) -> DrugInteraction | None:
        labeled = [i for i in self.drug_interactions if i.severity]
        return labeled[0] if labeled else (
            self.drug_interactions[0] if self.drug_interactions else None
        )

    def all_symptoms(self) -> list[str]:
        return self.symptom_groups.all_symptoms()

    def short_lab_summary(self, n: int = 3) -> str:
        if not self.abnormal_labs:
            return "no abnormal labs recorded"
        return "; ".join(lab.one_line() for lab in self.abnormal_labs[:n])

    def short_symptom_summary(self, n: int = 5) -> str:
        symptoms = self.all_symptoms()
        if not symptoms:
            return "no symptoms recorded"
        return ", ".join(symptoms[:n])

    def short_medication_summary(self, n: int = 4) -> str:
        if not self.medications:
            return "no medications on record"
        return ", ".join(self.medications[:n])

    def to_prompt_block(self) -> str:
        """
        Compact, high-signal structured text block for the enrichment
        prompt. Every line references an actual report value — no filler.
        This is the ONLY thing the enrichment model sees so it must
        contain everything needed without exceeding the small context.
        """
        lines: list[str] = []

        lines.append(f"SEVERITY: {self.severity} ({self.confidence_pct}% confidence)")
        if self.validation_status:
            lines.append(f"VALIDATION: {self.validation_status}")

        if self.symptom_groups.cardiac:
            lines.append(f"CARDIAC SYMPTOMS: {', '.join(self.symptom_groups.cardiac)}")
        if self.symptom_groups.respiratory:
            lines.append(f"RESPIRATORY SYMPTOMS: {', '.join(self.symptom_groups.respiratory)}")
        if self.symptom_groups.neurological:
            lines.append(f"NEUROLOGICAL SYMPTOMS: {', '.join(self.symptom_groups.neurological)}")
        if self.symptom_groups.metabolic:
            lines.append(f"METABOLIC SYMPTOMS: {', '.join(self.symptom_groups.metabolic)}")
        if self.symptom_groups.gastrointestinal:
            lines.append(f"GI SYMPTOMS: {', '.join(self.symptom_groups.gastrointestinal)}")
        if self.symptom_groups.general:
            lines.append(f"OTHER SYMPTOMS: {', '.join(self.symptom_groups.general)}")
        if self.symptom_duration:
            lines.append(f"SYMPTOM DURATION: {self.symptom_duration}")

        if self.severity_reasons:
            lines.append(f"SEVERITY REASONS: {'; '.join(self.severity_reasons[:3])}")
        if self.severity_indicators:
            lines.append(f"SEVERITY INDICATORS: {'; '.join(self.severity_indicators[:3])}")

        for lab in self.abnormal_labs[:6]:
            unit_ok = lab.unit and lab.unit.lower() not in ("none", "null")
            if lab.value and unit_ok:
                val = f"{lab.value}{lab.unit}"
            elif lab.value:
                val = str(lab.value)
            else:
                val = "value not recorded"
            direction = f" ({lab.direction})" if lab.direction else ""
            lines.append(f"ABNORMAL LAB — {lab.name}: {val}{direction}")

        for lab in self.normal_labs[:3]:
            unit_ok = lab.unit and lab.unit.lower() not in ("none", "null")
            if lab.value and unit_ok:
                val = f"{lab.value}{lab.unit}"
            elif lab.value:
                val = str(lab.value)
            else:
                val = "recorded"
            lines.append(f"NORMAL LAB — {lab.name}: {val}")

        if self.medications:
            lines.append(f"MEDICATIONS: {', '.join(self.medications)}")

        for interaction in self.drug_interactions[:2]:
            drugs = " + ".join(interaction.drugs) if interaction.drugs else "unknown"
            sev = f" [{interaction.severity}]" if interaction.severity else ""
            desc = interaction.description or "interaction flagged"
            lines.append(f"DRUG INTERACTION{sev}: {drugs} — {desc}")

        for warning in self.drug_warnings[:2]:
            lines.append(f"DRUG WARNING: {warning}")

        for finding in self.xray_findings[:3]:
            sig = " [SIGNIFICANT]" if finding.is_significant else ""
            lines.append(f"XRAY FINDING{sig}: {finding.text}")

        return "\n".join(lines)


# ── Private helpers ───────────────────────────────────────────────

def _cluster_symptoms(symptoms: list[str]) -> SymptomGroup:
    """
    Cluster cleaned atomic symptoms into body-system groups.
    Every input is passed through _clean_symptom() first so long
    medical-history blobs cannot leak into any group.
    """
    groups = SymptomGroup()

    seen: set[str] = set()
    for symptom in symptoms:
        cleaned = _clean_symptom(symptom)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)

        s = cleaned.lower()
        assigned = False
        for kw in _CARDIAC_KW:
            if kw in s:
                groups.cardiac.append(cleaned)
                assigned = True
                break
        if assigned:
            continue
        for kw in _RESPIRATORY_KW:
            if kw in s:
                groups.respiratory.append(cleaned)
                assigned = True
                break
        if assigned:
            continue
        for kw in _NEURO_KW:
            if kw in s:
                groups.neurological.append(cleaned)
                assigned = True
                break
        if assigned:
            continue
        for kw in _METABOLIC_KW:
            if kw in s:
                groups.metabolic.append(cleaned)
                assigned = True
                break
        if assigned:
            continue
        for kw in _GI_KW:
            if kw in s:
                groups.gastrointestinal.append(cleaned)
                assigned = True
                break
        if assigned:
            continue
        for kw in _MSK_KW:
            if kw in s:
                groups.musculoskeletal.append(cleaned)
                assigned = True
                break
        if not assigned:
            groups.general.append(cleaned)
    return groups


def _parse_lab_finding(text: str, *, is_abnormal: bool) -> LabFinding:
    """
    Parse a raw lab string into a LabFinding.

    Handles:
      "HbA1c: 9.2% (high)"            → name=HbA1c,     value=9.2, unit=%
      "TSH 6.05 mIU/L elevated"       → name=TSH,       value=6.05, unit=mIU/L
      "Severe vitamin D = 12.4 ng/mL" → name=vitamin D, value=12.4, unit=ng/mL

    Name is derived from the substring BEFORE the first colon or the
    first isolated number, so alphanumeric identifiers like "HbA1c"
    survive intact. Value/unit are then searched in the REMAINDER of
    the string so we never grab digits embedded in the name.
    """
    text = text.strip()

    # Direction
    dir_match = _DIRECTION_RE.search(text)
    direction: str | None = None
    if dir_match:
        word = dir_match.group(1).lower()
        direction = "high" if word in _HIGH_WORDS else "low"

    # Name / remainder split
    if ":" in text:
        name_raw, rest = text.split(":", 1)
    else:
        # Split at the first isolated number (preceded by whitespace or =)
        split_match = re.search(r"[\s=]\d", text)
        if split_match:
            name_raw = text[: split_match.start()]
            rest = text[split_match.start():]
        else:
            name_raw = text
            rest = ""

    # Value & unit — search only in the post-name portion so a name
    # like "HbA1c" (with embedded digit) can never yield value="1"
    value: str | None = None
    unit: str | None = None
    search_target = rest if rest else text
    val_match = _VALUE_RE.search(search_target)
    if val_match:
        value = val_match.group(1)
        raw_unit = val_match.group(2)
        if raw_unit and raw_unit.lower() not in ("none", "null"):
            unit = raw_unit

    # Strip leading severity qualifiers from name
    name_raw = re.sub(
        r"^(elevated|raised|high|low|reduced|abnormal|borderline|severe|critical)\s+",
        "",
        name_raw,
        flags=re.IGNORECASE,
    ).strip(" -–—(),")
    name = name_raw if name_raw else text[:40]

    return LabFinding(
        name=name, raw_text=text, value=value,
        unit=unit, is_abnormal=is_abnormal, direction=direction,
    )


def _parse_xray_finding(text: str) -> XrayFinding:
    t = text.lower()
    significant = any(kw in t for kw in _XRAY_SIGNIFICANT_KW)
    return XrayFinding(text=text.strip(), is_significant=significant)