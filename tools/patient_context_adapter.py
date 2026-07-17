"""
tools/patient_context_adapter.py — Patient-context-driven advice adapter.

Modifies advice, care plans, and severity based on patient allergies,
medical conditions, medications, pregnancy status, age, and immune
compromise. Emits safety warnings for top-of-report display.

Design principles:
  - Universal: driven by three registries (allergies, conditions,
    safety escalations); adding a new rule is one dict entry.
  - Fail-safe: any exception returns the original advice unchanged.
  - Additive: never removes advice content; only replaces phrases,
    appends context, or issues warnings.
  - Auditable: every modification records an entry in
    'personalization_applied' so downstream can trace what changed.
  - Composable: multiple rules can apply to the same advice item.

Consumed by:
  tools.report_generator._build_deterministic_report() — advice
  adaptation loop and safety warnings rendering.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from loguru import logger
from tools.unit_normalizer import normalize_for_comparison  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# NORMALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════

def _norm(text: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation edges."""
    if not text:
        return ""
    lowered = re.sub(r"\s+", " ", str(text).lower()).strip()
    lowered = lowered.strip(".,;:!?")
    return lowered


def _tokenize_free_text(text: str) -> list[str]:
    """Split free-text allergies into comma/semicolon/newline tokens."""
    if not text:
        return []
    tokens = re.split(r"[,;\n/&]+|(?:\band\b)", str(text), flags=re.IGNORECASE)
    return [_norm(t) for t in tokens if _norm(t)]


def _contains_any(text: str, keywords: list[str]) -> bool:
    """
    Universal, robust keyword matcher for free-text patient input.

    Strategy (applied in order per keyword):
      1. Normalize both sides: lowercase, collapse whitespace, unify
         separators (hyphens, underscores, slashes → space).
      2. Match with word-boundary anchors (\b) to block mid-word hits
         like 'ra' inside 'warfarin'.
      3. Allow common English inflections on the keyword tail:
         optional trailing 's', 'es', 'ed', 'er', 'ing', 'ism', 'ist'
         so 'sjogren' matches 'sjogrens', 'nsaid' matches 'nsaids',
         'diabetic' matches 'diabetics' — without a hardcoded suffix list.
      4. Multi-word keywords: separator normalization on both sides means
         'beta-lactam', 'beta lactam', 'beta_lactam' all match the
         keyword 'beta-lactam'.

    Fail-safe: regex errors per keyword are logged and skipped.
    Never raises. Never returns True on error.
    """
    if not text:
        return False

    def _normalize(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[-_/\\]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    text_norm = _normalize(str(text))
    _INFLECTION_SUFFIX = r"(?:ism|ist|ing|ers?|e?s|e?d)?"

    for kw in keywords:
        if not kw:
            continue
        try:
            kw_norm = _normalize(str(kw))
            if not kw_norm:
                continue
            kw_pattern_body = re.escape(kw_norm).replace(r"\ ", r"\s+")
            pattern = (
                r"\b"
                + kw_pattern_body
                + _INFLECTION_SUFFIX
                + r"\b"
            )
            if re.search(pattern, text_norm):
                return True
        except re.error:
            logger.warning(
                "patient_context_adapter · _contains_any: "
                "invalid keyword pattern skipped",
                keyword=kw,
            )
            continue

    return False


# ═══════════════════════════════════════════════════════════════════
# ALLERGY REGISTRY
# ═══════════════════════════════════════════════════════════════════
_ALLERGY_RULES: list[dict[str, Any]] = [
    {
        "id": "shellfish",
        "supersedes": ["fish"],
        "allergen_keywords": [
            "shellfish", "shell fish", "prawn", "prawns", "shrimp",
            "crab", "lobster", "seafood",
        ],
        "strip_from_advice": [
            "fatty fish", "salmon", "mackerel", "sardines", "tuna",
            "seafood", "shellfish",
        ],
        "alternative_suggestion": (
            "For omega-3 intake, since you reported a shellfish/seafood "
            "allergy, consider flaxseed, chia seeds, walnuts, or a "
            "physician-approved algae-based omega-3 supplement instead."
        ),
        "safety_warning": None,
    },
    {
        "id": "fish",
        "supersedes": [],
        "allergen_keywords": ["fish", "salmon", "tuna", "mackerel"],
        "strip_from_advice": [
            "fatty fish", "salmon", "mackerel", "sardines", "tuna",
            "fish oil", "seafood",
        ],
        "alternative_suggestion": (
            "For omega-3 intake, since you reported a fish allergy, "
            "consider flaxseed, chia seeds, walnuts, or a physician-"
            "approved algae-based omega-3 supplement."
        ),
        "safety_warning": None,
    },
    {
        "id": "peanuts",
        "allergen_keywords": ["peanut", "peanuts", "groundnut", "groundnuts"],
        "strip_from_advice": ["peanut", "peanuts", "peanut butter"],
        "alternative_suggestion": (
            "For plant-protein and healthy-fat intake, since peanuts are "
            "excluded due to allergy, consider seeds (sunflower, pumpkin) "
            "or physician-cleared alternatives."
        ),
        "safety_warning": None,
    },
    {
        "id": "tree_nuts",
        "allergen_keywords": [
            "tree nut", "tree nuts", "walnut", "walnuts", "almond",
            "almonds", "cashew", "cashews", "brazil nut", "brazil nuts",
            "pecan", "pecans", "pistachio", "pistachios", "hazelnut",
        ],
        "strip_from_advice": [
            "walnuts", "almonds", "brazil nuts", "cashews", "nuts",
            "nut butter",
        ],
        "alternative_suggestion": (
            "For healthy fats and selenium, since tree nuts are excluded, "
            "consider seeds, avocado, olive oil, or physician-cleared "
            "supplements."
        ),
        "safety_warning": None,
    },
    {
        "id": "dairy",
        "allergen_keywords": [
            "dairy", "milk", "lactose", "casein", "whey",
        ],
        "strip_from_advice": [
            "milk", "yogurt", "cheese", "dairy", "fortified milk",
        ],
        "alternative_suggestion": (
            "For calcium and Vitamin D, since dairy is excluded, consider "
            "fortified plant milks, leafy greens, tofu, or physician-"
            "approved supplements."
        ),
        "safety_warning": None,
    },
    {
        "id": "eggs",
        "allergen_keywords": ["egg", "eggs", "egg white", "egg yolk"],
        "strip_from_advice": ["eggs", "egg yolks", "egg whites"],
        "alternative_suggestion": (
            "For protein and B12 intake, since eggs are excluded, "
            "consider legumes, dairy (if tolerated), or physician-"
            "approved supplements."
        ),
        "safety_warning": None,
    },
    {
        "id": "gluten",
        "allergen_keywords": ["gluten", "wheat", "celiac", "coeliac"],
        "strip_from_advice": [
            "whole grains", "wheat", "wheat bran", "whole wheat",
        ],
        "alternative_suggestion": (
            "For fiber and B vitamins, since gluten is excluded, consider "
            "gluten-free whole grains (quinoa, brown rice, oats certified "
            "gluten-free), legumes, and vegetables."
        ),
        "safety_warning": None,
    },
    {
        "id": "soy",
        "allergen_keywords": ["soy", "soya", "soybean"],
        "strip_from_advice": ["soy", "tofu", "soy milk", "edamame"],
        "alternative_suggestion": (
            "For plant protein, since soy is excluded, consider legumes, "
            "quinoa, or physician-approved alternatives."
        ),
        "safety_warning": None,
    },
    {
        "id": "penicillin",
        "allergen_keywords": [
            "penicillin", "amoxicillin", "ampicillin", "beta-lactam",
        ],
        "strip_from_advice": [],
        "alternative_suggestion": None,
        "safety_warning": (
            "You reported a penicillin allergy. If any recommendation in "
            "this report leads to antibiotic evaluation, ensure your "
            "physician is aware of this allergy before any prescription."
        ),
    },
    {
        "id": "sulfa",
        "allergen_keywords": ["sulfa", "sulfonamide", "sulphonamide"],
        "strip_from_advice": [],
        "alternative_suggestion": None,
        "safety_warning": (
            "You reported a sulfa/sulfonamide allergy. Ensure your "
            "physician is aware if any sulfa-based medication is being "
            "considered."
        ),
    },
    {
        "id": "aspirin_nsaids",
        "allergen_keywords": [
            "aspirin", "nsaid", "nsaids", "ibuprofen", "naproxen",
        ],
        "strip_from_advice": ["aspirin", "nsaids", "ibuprofen"],
        "alternative_suggestion": None,
        "safety_warning": (
            "You reported an aspirin/NSAID allergy. Any pain-management "
            "or anti-inflammatory suggestion should be discussed with "
            "your physician before use."
        ),
    },
    {
        "id": "iodine",
        "allergen_keywords": ["iodine", "iodinated"],
        "strip_from_advice": ["iodized salt", "seaweed", "iodine"],
        "alternative_suggestion": (
            "Since iodine is excluded due to allergy, dietary iodine "
            "sources should be avoided and any iodinated contrast imaging "
            "must be discussed with your physician."
        ),
        "safety_warning": (
            "You reported an iodine allergy. Any imaging using iodinated "
            "contrast must be pre-approved by your physician."
        ),
    },
    {
        "id": "latex",
        "allergen_keywords": ["latex"],
        "strip_from_advice": [],
        "alternative_suggestion": None,
        "safety_warning": (
            "You reported a latex allergy. Ensure any healthcare visit "
            "uses latex-free supplies."
        ),
    },
    {
        "id": "contrast_dye",
        "allergen_keywords": ["contrast dye", "contrast media", "gadolinium"],
        "strip_from_advice": [],
        "alternative_suggestion": None,
        "safety_warning": (
            "You reported a contrast dye allergy. Any recommendation for "
            "contrast-enhanced imaging must be discussed with your "
            "physician before proceeding."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════
# MEDICAL CONDITION REGISTRY
# ═══════════════════════════════════════════════════════════════════
_CONDITION_RULES: list[dict[str, Any]] = [
    {
        "id": "diabetes",
        "condition_keywords": [
            "diabetes", "diabetic", "type 1 diabetes", "type 2 diabetes",
            "t1dm", "t2dm", "iddm", "niddm",
        ],
        "applies_to_biomarker_keys": ["glucose", "hba1c"],
        "applies_to_statuses": ["high", "critical_high", "borderline"],
        "advice_replacements": {
            "screen for diabetes": (
                "review your existing diabetes management plan with your "
                "endocrinologist"
            ),
            "possible diabetes": "your existing diabetes",
            "consider HbA1c testing": (
                "discuss dose adjustments with your endocrinologist"
            ),
        },
        "advice_appends": [
            "Given your existing diabetes, monitor blood glucose more "
            "frequently until reviewed with your physician."
        ],
        "care_plan_appends": {
            "immediate": [
                "Share this result with your endocrinologist within 1 week "
                "for possible dose or regimen adjustment.",
            ],
        },
        "safety_warning": None,
    },
    {
        "id": "hypertension",
        "condition_keywords": [
            "hypertension", "high blood pressure", "htn",
        ],
        "applies_to_biomarker_keys": ["bp_systolic", "bp_diastolic"],
        "applies_to_statuses": ["high", "critical_high", "borderline"],
        "advice_replacements": {
            "consider antihypertensive evaluation": (
                "review your current antihypertensive regimen with your physician"
            ),
        },
        "advice_appends": [
            "Continue home BP monitoring and log daily readings for your physician."
        ],
        "care_plan_appends": {},
        "safety_warning": None,
    },
    {
        "id": "hypothyroidism",
        "condition_keywords": [
            "hypothyroidism", "hypothyroid", "levothyroxine", "thyroxine therapy",
        ],
        "applies_to_biomarker_keys": ["tsh", "t3", "t4", "free_t3", "free_t4"],
        "applies_to_statuses": ["high", "critical_high", "low", "critical_low", "borderline"],
        "advice_replacements": {
            "possible hypothyroidism": (
                "your existing hypothyroidism — your levothyroxine dose may "
                "need adjustment"
            ),
        },
        "advice_appends": [
            "Do not stop or change thyroid medication without medical guidance."
        ],
        "care_plan_appends": {
            "immediate": [
                "Discuss this thyroid result with your endocrinologist to "
                "review medication dosing.",
            ],
        },
        "safety_warning": None,
    },
    {
        "id": "hyperthyroidism",
        "condition_keywords": [
            "hyperthyroidism", "hyperthyroid", "graves disease",
            "carbimazole", "methimazole", "antithyroid",
        ],
        "applies_to_biomarker_keys": ["tsh", "t3", "t4", "free_t3", "free_t4"],
        "applies_to_statuses": ["high", "critical_high", "low", "critical_low"],
        "advice_appends": [
            "Given your existing hyperthyroidism, medication dose may need "
            "adjustment — discuss urgently with your endocrinologist."
        ],
        "care_plan_appends": {
            "immediate": [
                "Contact your endocrinologist to review antithyroid "
                "medication dose based on this result.",
            ],
        },
        "safety_warning": None,
    },
    {
        "id": "ckd",
        "condition_keywords": [
            "ckd", "chronic kidney disease", "kidney disease",
            "renal insufficiency", "renal failure", "on dialysis",
        ],
        "applies_to_biomarker_keys": [
            "creatinine", "urea", "bun", "potassium", "sodium", "phosphorus",
        ],
        "applies_to_statuses": ["high", "critical_high", "low", "critical_low", "borderline"],
        "advice_replacements": {
            "drink 2-3 liters of water": (
                "follow your nephrologist's fluid intake guidance"
            ),
            "drink plenty of water": (
                "follow your nephrologist's fluid intake guidance"
            ),
            "ensure adequate hydration": (
                "follow your nephrologist's fluid intake guidance"
            ),
        },
        "advice_appends": [
            "Given your kidney disease, avoid NSAIDs and any nephrotoxic "
            "medications without physician approval."
        ],
        "care_plan_appends": {
            "immediate": [
                "Share this result with your nephrologist for review in "
                "the context of your kidney disease.",
            ],
        },
        "safety_warning": None,
    },
    {
        "id": "anticoagulation",
        "condition_keywords": [
            "warfarin", "coumadin", "apixaban", "rivaroxaban", "dabigatran",
            "edoxaban", "eliquis", "xarelto", "pradaxa", "anticoagulation",
            "anticoagulant", "blood thinner", "blood thinners",
        ],
        "applies_to_biomarker_keys": ["platelets", "inr", "pt", "aptt"],
        "applies_to_statuses": ["low", "critical_low", "high", "critical_high", "borderline"],
        "advice_replacements": {
            "avoid aspirin and NSAIDs": (
                "strictly avoid aspirin and NSAIDs given your anticoagulation therapy"
            ),
        },
        "advice_appends": [
            "Given your anticoagulation therapy, discuss this result "
            "urgently with your prescribing physician."
        ],
        "care_plan_appends": {
            "immediate": [
                "Contact your prescribing physician within 24–48 hours to "
                "review anticoagulation given this result.",
            ],
        },
        "safety_warning": None,
    },
    {
        "id": "pregnancy",
        "condition_keywords": ["pregnant", "pregnancy", "gestation"],
        "applies_to_biomarker_keys": None,
        "applies_to_statuses": None,
        "advice_appends": [
            "Given your pregnancy, discuss all medication and supplement "
            "recommendations with your obstetrician before starting."
        ],
        "care_plan_appends": {},
        "safety_warning": (
            "You reported pregnancy. All medication, supplement, imaging, "
            "and treatment recommendations must be pre-approved by your "
            "obstetrician for pregnancy safety."
        ),
    },
    {
        "id": "cardiovascular",
        "condition_keywords": [
            "coronary artery disease", "cad", "heart disease",
            "myocardial infarction", "heart attack", "angina",
            "cardiac disease", "cardiovascular disease",
        ],
        "applies_to_biomarker_keys": [
            "ldl_cholesterol", "hdl_cholesterol", "total_cholesterol",
            "triglycerides", "troponin", "bp_systolic", "bp_diastolic",
        ],
        "applies_to_statuses": ["high", "critical_high", "low", "critical_low", "borderline"],
        "advice_appends": [
            "Given your cardiovascular history, discuss this result "
            "urgently with your cardiologist."
        ],
        "care_plan_appends": {
            "immediate": [
                "Book cardiology review within 1–2 weeks given your "
                "cardiac history.",
            ],
        },
        "safety_warning": None,
    },
    {
        "id": "chemotherapy",
        "condition_keywords": [
            "cancer", "chemotherapy", "chemo", "oncology treatment",
            "radiation therapy", "immunotherapy",
        ],
        "applies_to_biomarker_keys": [
            "wbc", "neutrophils", "haemoglobin", "platelets",
        ],
        "applies_to_statuses": ["low", "critical_low"],
        "advice_appends": [
            "Given your ongoing cancer treatment, this result may require "
            "urgent oncology review — do not delay contact."
        ],
        "care_plan_appends": {
            "immediate": [
                "Contact your oncology team urgently to review this result.",
            ],
        },
        "safety_warning": None,
    },
    {
        "id": "autoimmune",
        "condition_keywords": [
            "lupus", "sle", "rheumatoid arthritis", "ra",
            "autoimmune", "sjogren", "vasculitis",
        ],
        "applies_to_biomarker_keys": ["crp", "esr"],
        "applies_to_statuses": ["high", "critical_high"],
        "advice_appends": [
            "Given your autoimmune condition, elevated inflammatory "
            "markers may indicate a flare — discuss with your rheumatologist."
        ],
        "care_plan_appends": {
            "immediate": [
                "Contact your rheumatologist to evaluate for possible "
                "autoimmune flare.",
            ],
        },
        "safety_warning": None,
    },
    {
        "id": "copd_asthma",
        "condition_keywords": ["copd", "asthma", "emphysema", "bronchitis"],
        "applies_to_biomarker_keys": ["spo2", "respiratory_rate"],
        "applies_to_statuses": ["low", "critical_low", "high", "critical_high"],
        "advice_appends": [
            "Given your respiratory condition, monitor SpO2 more frequently "
            "and act early on any worsening."
        ],
        "care_plan_appends": {},
        "safety_warning": None,
    },
    {
        "id": "liver_disease",
        "condition_keywords": [
            "liver disease", "hepatitis", "cirrhosis", "fatty liver",
            "nafld", "alcoholic liver",
        ],
        "applies_to_biomarker_keys": [
            "sgpt_alt", "sgot_ast", "bilirubin", "albumin", "alp", "ggt",
        ],
        "applies_to_statuses": ["high", "critical_high"],
        "advice_appends": [
            "Given your liver disease, compare this result to your baseline "
            "with your hepatologist to distinguish chronic changes from "
            "acute deterioration."
        ],
        "care_plan_appends": {},
        "safety_warning": None,
    },
    {
        "id": "immunocompromised",
        "condition_keywords": [
            "immunocompromised", "immunosuppressed", "transplant",
            "hiv", "on steroids", "prednisone",
        ],
        "applies_to_biomarker_keys": ["wbc", "neutrophils"],
        "applies_to_statuses": ["low", "critical_low"],
        "advice_appends": [
            "Given your immunocompromised state, any signs of infection "
            "should trigger urgent medical evaluation."
        ],
        "care_plan_appends": {
            "immediate": [
                "Monitor closely for infection signs and seek prompt care "
                "if febrile.",
            ],
        },
        "safety_warning": None,
    },
]


# ═══════════════════════════════════════════════════════════════════
# SAFETY ESCALATION REGISTRY — cross-cutting rules
# ═══════════════════════════════════════════════════════════════════
_SAFETY_ESCALATION_RULES: list[dict[str, Any]] = [
    {
        "id": "anticoagulated_low_platelets",
        "requires_conditions": ["anticoagulation"],
        "requires_biomarker": {"key": "platelets", "op": "<", "value": 100},
        "severity_boost": 2,
        "safety_warning": (
            "You reported anticoagulation therapy and your platelet count "
            "is below 100,000/µL — urgent physician review is required "
            "due to significantly elevated bleeding risk."
        ),
        "safety_warning_short": (
            "Anticoagulation + low platelets — urgent bleeding risk review"
        ),
        "care_plan_appends": {
            "immediate": [
                "Contact your prescribing physician urgently to discuss "
                "anticoagulation given your low platelet count.",
            ],
        },
    },
    {
        "id": "diabetic_severe_hyperglycemia",
        "requires_conditions": ["diabetes"],
        "requires_biomarker": {"key": "glucose", "op": ">", "value": 300},
        "severity_boost": 2,
        "safety_warning": (
            "You reported diabetes and your glucose is above 300 mg/dL — "
            "urgent physician review is required to prevent diabetic emergency."
        ),
        "safety_warning_short": (
            "Diabetes + glucose >300 — urgent review, prevent emergency"
        ),
        "care_plan_appends": {
            "immediate": [
                "Contact your physician urgently. If experiencing extreme "
                "thirst, vomiting, or confusion, go to an emergency department.",
            ],
        },
    },
    {
        "id": "immunocompromised_neutropenia",
        "requires_conditions": ["immunocompromised", "chemotherapy"],
        "requires_biomarker": {"key": "neutrophils", "op": "<", "value": 1.5},
        "severity_boost": 2,
        "safety_warning": (
            "Low neutrophils combined with your immunocompromised state "
            "significantly increases infection risk — urgent review needed."
        ),
        "safety_warning_short": (
            "Low neutrophils + immunocompromised — high infection risk"
        ),
        "care_plan_appends": {
            "immediate": [
                "Avoid crowds, monitor temperature, and seek immediate "
                "medical care if febrile (T > 100.4°F / 38°C).",
            ],
        },
    },
    {
        "id": "pregnancy_low_platelets_fever",
        "requires_conditions": ["pregnancy"],
        "requires_biomarker": {"key": "platelets", "op": "<", "value": 150},
        "severity_boost": 2,
        "safety_warning": (
            "Low platelets during pregnancy require urgent obstetric review "
            "to rule out pre-eclampsia or other complications."
        ),
        "safety_warning_short": (
            "Pregnancy + low platelets — urgent obstetric review"
        ),
        "care_plan_appends": {
            "immediate": [
                "Contact your obstetrician urgently to review this result.",
            ],
        },
    },
    {
        "id": "cardiovascular_elevated_troponin",
        "requires_conditions": ["cardiovascular"],
        "requires_biomarker": {"key": "troponin", "op": ">", "value": 0.04},
        "severity_boost": 2,
        "safety_warning": (
            "Elevated troponin with your known cardiovascular history is a "
            "medical emergency. Seek immediate emergency care."
        ),
        "safety_warning_short": (
            "Cardiac history + high troponin — emergency, call 911"
        ),
        "care_plan_appends": {
            "immediate": [
                "Call emergency services immediately — do not delay.",
            ],
        },
    },
]


# ═══════════════════════════════════════════════════════════════════
# AGE HELPER
# ═══════════════════════════════════════════════════════════════════

def _parse_age_from_dob(dob: str | None) -> int | None:
    """Parse patient age from DOB string. Returns None if unparseable."""
    if not dob:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            born = datetime.strptime(str(dob).strip(), fmt)
            today = datetime.now()
            age = today.year - born.year - (
                (today.month, today.day) < (born.month, born.day)
            )
            if 0 <= age <= 130:
                return age
        except ValueError:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API — CONTEXT NORMALIZATION
# ═══════════════════════════════════════════════════════════════════

def normalize_patient_context(state) -> dict:
    """
    Parse state.patient_allergies (free text) and
    state.patient_medical_conditions (list) into structured normalized form.

    Returns:
      {
        "allergies":         [rule_id, ...],
        "conditions":        [rule_id, ...],
        "meds":              [str, ...],
        "pregnant":          bool,
        "immunocompromised": bool,
        "age":               int | None,
        "sex":               str | None,
        "raw_allergies":     str,
        "raw_conditions":    list[str],
      }

    Fail-safe: any exception returns a minimal empty context.
    """
    try:
        allergies_text   = getattr(state, "patient_allergies", None) or ""
        conditions_list  = getattr(state, "patient_medical_conditions", None) or []
        dob              = getattr(state, "patient_dob", None)
        sex              = getattr(state, "patient_sex", None)
        meds             = list(getattr(state, "medications_raw", None) or [])

        matched_allergies: list[str] = []
        for rule in _ALLERGY_RULES:
            if _contains_any(allergies_text, rule["allergen_keywords"]):
                matched_allergies.append(rule["id"])

        matched_conditions: list[str] = []
        combined_conditions_text = " ".join(str(c) for c in conditions_list)
        combined_conditions_text += " " + " ".join(meds)

        for rule in _CONDITION_RULES:
            if _contains_any(combined_conditions_text, rule["condition_keywords"]):
                matched_conditions.append(rule["id"])

        pregnant = "pregnancy" in matched_conditions
        immunocompromised = (
            "immunocompromised" in matched_conditions
            or "chemotherapy" in matched_conditions
        )

        age = _parse_age_from_dob(dob)

        ctx = {
            "allergies":         matched_allergies,
            "conditions":        matched_conditions,
            "meds":              meds,
            "pregnant":          pregnant,
            "immunocompromised": immunocompromised,
            "age":               age,
            "sex":               str(sex).lower() if sex else None,
            "raw_allergies":     allergies_text,
            "raw_conditions":    conditions_list,
        }

        logger.info(
            "patient_context_adapter · normalized context",
            allergies=matched_allergies,
            conditions=matched_conditions,
            pregnant=pregnant,
            immunocompromised=immunocompromised,
            age=age,
        )

        return ctx

    except Exception:
        logger.exception(
            "patient_context_adapter · normalization failed; using empty context"
        )
        return {
            "allergies": [], "conditions": [], "meds": [],
            "pregnant": False, "immunocompromised": False,
            "age": None, "sex": None,
            "raw_allergies": "", "raw_conditions": [],
        }


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API — ADVICE ADAPTATION
# ═══════════════════════════════════════════════════════════════════

def _apply_allergy_rules(
    advice: dict,
    context: dict,
    audit: list[str],
) -> dict:
    """
    Apply allergy rules to an advice dict.

    Universal guarantees:
      1. STRIP-GATE: alternative_suggestion appended ONLY when stripping
         actually removed content from THIS recommendation.
      2. SUPERSEDES: more-specific rules suppress less-specific ones for
         alternative_suggestion. Both still strip.
      3. DEDUPLICATION: each unique alternative_suggestion appears at most once.
      4. PUNCTUATION CLEANUP: single universal pass after all stripping.
      5. SAFE TYPES: all values coerced to str before processing.
      6. EMPTY GUARD: no alternative appended to an empty recommendation.

    Fail-safe: any exception returns advice unchanged.
    """
    if not context.get("allergies"):
        return advice

    matched_ids   = set(context["allergies"])
    matched_rules = [r for r in _ALLERGY_RULES if r["id"] in matched_ids]
    if not matched_rules:
        return advice

    superseded_ids: set[str] = set()
    for rule in matched_rules:
        for sup_id in (rule.get("supersedes") or []):
            if sup_id in matched_ids:
                superseded_ids.add(sup_id)

    def _safe_str(v) -> str:
        if v is None:
            return ""
        if isinstance(v, list):
            return ". ".join(str(i) for i in v if i)
        return str(v)

    recommendation = _safe_str(advice.get("recommendation"))
    raw_care_plan  = advice.get("care_plan") or {}
    care_plan: dict[str, str] = {k: _safe_str(v) for k, v in raw_care_plan.items()}
    rules_that_stripped_rec: set[str] = set()

    for rule in matched_rules:
        strip_items = rule.get("strip_from_advice") or []
        rec_changed = False

        for item in strip_items:
            if not item:
                continue
            sep_pattern = re.sub(r"[-_\s]+", r"[-_\\s]+", re.escape(item))
            pat = re.compile(sep_pattern, re.IGNORECASE)

            if pat.search(recommendation):
                recommendation = pat.sub("", recommendation)
                rec_changed = True

            for bucket in list(care_plan.keys()):
                if pat.search(care_plan[bucket]):
                    care_plan[bucket] = pat.sub("", care_plan[bucket])

        if rec_changed:
            rules_that_stripped_rec.add(rule["id"])
            audit.append(f"allergy:{rule['id']}:stripped")

    # Punctuation cleanup
    r = recommendation
    r = re.sub(r"\s*/\s*(?=[\s,.]|$)", " ", r)
    r = re.sub(r"(?<![^\s])\s*/\s*", " ", r)
    r = re.sub(r"\(\s*\)", "", r)
    r = re.sub(r"(,\s*){2,}", ", ", r)
    r = re.sub(r"\bor\s*,", "", r)
    r = re.sub(r"\band\s*,", "", r)
    r = re.sub(r",\s*\.", ".", r)
    r = re.sub(r",\s*$", "", r)
    r = re.sub(r"\s{2,}", " ", r)
    r = r.strip().strip(".,;").strip()
    recommendation = r

    appended_alts: list[str] = []
    alt_audit_ids: list[str] = []
    rec_is_meaningful = bool(recommendation.strip().strip(".,;"))

    for rule in matched_rules:
        rule_id = rule["id"]
        alt     = (rule.get("alternative_suggestion") or "").strip()

        if not alt:
            continue
        if rule_id in superseded_ids:
            continue
        if rule_id not in rules_that_stripped_rec:
            continue
        if alt in appended_alts:
            continue
        if not rec_is_meaningful:
            continue

        appended_alts.append(alt)
        alt_audit_ids.append(rule_id)

    for alt, rid in zip(appended_alts, alt_audit_ids):
        recommendation = f"{recommendation.rstrip('.')}. {alt}"
        audit.append(f"allergy:{rid}:alternative_added")

    advice["recommendation"] = recommendation.strip()
    advice["care_plan"]      = care_plan
    return advice


def _apply_condition_rules(
    advice: dict,
    context: dict,
    biomarker_item: dict,
    audit: list[str],
) -> dict:
    """
    Apply condition rules to an advice dict.

    Care plan entries from condition rules are stored as separate list
    items rather than concatenated onto a single string. This prevents
    multiple condition appends from producing awkward multi-sentence
    bullets in the rendered care plan.

    The care_plan dict uses list[str] values here; _render_care_plan_html
    already handles both str and list values via _safe_str coercion.
    """
    if not context.get("conditions"):
        return advice

    matched_ids   = set(context["conditions"])
    matched_rules = [r for r in _CONDITION_RULES if r["id"] in matched_ids]
    if not matched_rules:
        return advice

    biomarker_key    = str(biomarker_item.get("key") or "").lower()
    biomarker_status = str(biomarker_item.get("status") or "").lower()

    recommendation = str(advice.get("recommendation") or "")

    # Care plan: normalise to dict[str, list[str]] for clean per-entry storage.
    # Existing string values are wrapped in a list so the shape is consistent.
    raw_care_plan = advice.get("care_plan") or {}
    care_plan: dict[str, list[str]] = {}
    for k, v in raw_care_plan.items():
        if isinstance(v, list):
            care_plan[k] = [str(i) for i in v if i]
        elif v:
            care_plan[k] = [str(v)]
        else:
            care_plan[k] = []

    for rule in matched_rules:
        # Check biomarker applicability
        applies_keys = rule.get("applies_to_biomarker_keys")
        if applies_keys is not None and biomarker_key not in applies_keys:
            continue

        # Check status applicability
        applies_statuses = rule.get("applies_to_statuses")
        if applies_statuses is not None and biomarker_status not in applies_statuses:
            continue

        # Apply phrase replacements to recommendation text
        replacements = rule.get("advice_replacements") or {}
        for old_phrase, new_phrase in replacements.items():
            if old_phrase.lower() in recommendation.lower():
                pattern = re.compile(re.escape(old_phrase), re.IGNORECASE)
                recommendation = pattern.sub(new_phrase, recommendation)
                audit.append(f"condition:{rule['id']}:replaced")

        # Append condition-specific advice sentences
        for append_text in (rule.get("advice_appends") or []):
            if append_text not in recommendation:
                recommendation = f"{recommendation.rstrip('.')}. {append_text}"
                audit.append(f"condition:{rule['id']}:appended")

        # Append to care plan buckets — each entry as a separate list item
        cp_appends = rule.get("care_plan_appends") or {}
        for bucket, extras in cp_appends.items():
            if bucket not in care_plan:
                care_plan[bucket] = []
            existing_set = set(care_plan[bucket])
            for extra in (extras or []):
                extra = str(extra).strip()
                if extra and extra not in existing_set:
                    care_plan[bucket].append(extra)
                    existing_set.add(extra)
                    audit.append(f"condition:{rule['id']}:care_plan:{bucket}")

    advice["recommendation"] = recommendation.strip()
    advice["care_plan"]      = care_plan
    return advice


def adapt_advice_for_patient(
    advice: dict,
    patient_context: dict,
    biomarker_item: dict,
) -> dict:
    """
    Adapt an advice dict for a specific patient context.

    Args:
        advice:          The dict returned by resolve_advice().
        patient_context: The dict returned by normalize_patient_context().
        biomarker_item:  The measurement item this advice applies to.

    Returns:
        A modified advice dict with the same shape plus:
          - "safety_warnings":         list[str]
          - "personalization_applied": list[str]

    Fail-safe: any exception returns the original advice unchanged.
    """
    try:
        modified = {
            "recommendation": str(advice.get("recommendation") or ""),
            "care_plan":      dict(advice.get("care_plan") or {}),
            "source":         advice.get("source"),
        }

        audit:           list[str] = []
        safety_warnings: list[str] = []

        modified = _apply_allergy_rules(modified, patient_context, audit)

        matched_allergy_ids = set(patient_context.get("allergies") or [])
        for rule in _ALLERGY_RULES:
            if rule["id"] in matched_allergy_ids and rule.get("safety_warning"):
                warning = rule["safety_warning"]
                if warning not in safety_warnings:
                    safety_warnings.append(warning)

        modified = _apply_condition_rules(
            modified, patient_context, biomarker_item, audit,
        )

        matched_condition_ids = set(patient_context.get("conditions") or [])
        for rule in _CONDITION_RULES:
            if rule["id"] in matched_condition_ids and rule.get("safety_warning"):
                warning = rule["safety_warning"]
                if warning not in safety_warnings:
                    safety_warnings.append(warning)

        modified["safety_warnings"]         = safety_warnings
        modified["personalization_applied"] = audit

        return modified

    except Exception:
        logger.exception(
            "patient_context_adapter · adapt_advice failed; returning original"
        )
        return {
            **advice,
            "safety_warnings":         [],
            "personalization_applied": [],
        }


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API — SAFETY ESCALATION
# ═══════════════════════════════════════════════════════════════════

def _biomarker_condition_met(condition: dict, measurements: list[dict]) -> bool:
    """
    Check if any measurement satisfies a biomarker threshold condition.

    Routes all value/threshold comparisons through normalize_for_comparison()
    so threshold registry values (clinical shorthand) are always compared
    against values in the same canonical unit.

    Fail-safe: any conversion or comparison error skips that measurement.
    """
    from tools.unit_normalizer import normalize_for_comparison, _resolve_canonical_key

    key_wanted = str(condition.get("key") or "").lower()
    op         = condition.get("op")
    threshold  = condition.get("value")

    if not key_wanted or not op or threshold is None:
        return False

    try:
        threshold_f = float(threshold)
    except (TypeError, ValueError):
        return False

    for m in measurements:
        m_key = str(m.get("key") or "").lower()

        if m_key != key_wanted:
            if _resolve_canonical_key(m_key) != _resolve_canonical_key(key_wanted):
                continue

        raw_value = m.get("value")
        raw_unit  = m.get("unit") or m.get("units") or ""

        try:
            norm = normalize_for_comparison(m_key, raw_value, raw_unit)
            v    = norm.value
        except Exception:
            try:
                v = float(raw_value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue

        if v is None:
            continue

        try:
            if op == ">"  and v >  threshold_f: return True
            if op == "<"  and v <  threshold_f: return True
            if op == ">=" and v >= threshold_f: return True
            if op == "<=" and v <= threshold_f: return True
            if op == "==" and v == threshold_f: return True
        except TypeError:
            continue

    return False


def compute_safety_escalations(
    all_measurements: list[dict],
    patient_context: dict,
) -> dict:
    """
    Evaluate cross-cutting safety escalation rules that combine patient
    conditions with biomarker findings.

    Returns:
      {
        "safety_warnings":   list[str],
        "severity_boost":    int,           # capped at 3
        "care_plan_appends": dict[str, list[str]],
        "matched_rule_ids":  list[str],
      }

    Fail-safe: any exception returns an empty result.
    """
    result = {
        "safety_warnings": [],
        "severity_boost":  0,
        "care_plan_appends": {
            "immediate": [], "short_term": [], "lifestyle": [],
            "follow_up": [], "long_term": [],
        },
        "matched_rule_ids": [],
    }

    try:
        conditions = set(patient_context.get("conditions") or [])
        if not conditions and not patient_context.get("pregnant"):
            return result

        for rule in _SAFETY_ESCALATION_RULES:
            required = set(rule.get("requires_conditions") or [])
            if required and not (required & conditions):
                continue

            biomarker_cond = rule.get("requires_biomarker") or {}
            if not _biomarker_condition_met(biomarker_cond, all_measurements):
                continue

            rid = rule["id"]
            result["matched_rule_ids"].append(rid)
            result["severity_boost"] += int(rule.get("severity_boost") or 0)

            warning = rule.get("safety_warning")
            if warning and warning not in result["safety_warnings"]:
                result["safety_warnings"].append(warning)

            cp_appends = rule.get("care_plan_appends") or {}
            for bucket, extras in cp_appends.items():
                if bucket not in result["care_plan_appends"]:
                    continue
                for extra in extras:
                    if extra not in result["care_plan_appends"][bucket]:
                        result["care_plan_appends"][bucket].append(extra)

        if result["severity_boost"] > 3:
            result["severity_boost"] = 3

        if result["matched_rule_ids"]:
            logger.info(
                "patient_context_adapter · safety escalations matched",
                rules=result["matched_rule_ids"],
                boost=result["severity_boost"],
            )

        return result

    except Exception:
        logger.exception(
            "patient_context_adapter · safety escalation failed"
        )
        return result


__all__ = [
    "normalize_patient_context",
    "adapt_advice_for_patient",
    "compute_safety_escalations",
]