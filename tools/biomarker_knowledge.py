"""
tools/biomarker_knowledge.py — Clinical advice knowledge base + universal
biomarker resolver.

Single source of truth for:
    1. Biomarker canonical-key resolution (used by parser + dashboard)
    2. Patient-facing recommendations and care-plan content
    3. Display name formatting with medical abbreviations
    4. Universal fallback for unknown biomarkers

Design principles:
    - Data-driven: adding a biomarker = one dict entry, no code changes
    - Universal fallback: NO biomarker is ever silently dropped
    - Fuzzy matching: handles lab-format variance without hardcoded aliases
    - Single source of truth: parser + dashboard + report all use the
      same resolver so biomarker names never drift across files

Consumed by:
    tools.lab_report_parser  → resolve_canonical_key() for parsed rows
    backend.dashboard        → resolve_canonical_key() + get_display_name()
    tools.report_generator   → resolve_advice() for flagged items
"""
from __future__ import annotations

import re
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# EXACT ALIAS MAP — fast-path lookup for known name variants.
# Falls through to fuzzy matcher if no exact hit.
# ═══════════════════════════════════════════════════════════════════
_KB_ALIASES: dict[str, str] = {
    # ── Vitals ──
    "heart_rate": "heart_rate",
    "heart rate": "heart_rate",
    "pulse": "heart_rate",
    "pulse rate": "heart_rate",
    "spo2": "spo2",
    "sp o2": "spo2",
    "sp 02": "spo2",
    "oxygen saturation": "spo2",
    "oxygen_saturation": "spo2",
    "o2 saturation": "spo2",
    "temperature": "temperature_c",
    "temperature c": "temperature_c",
    "temperature_c": "temperature_c",
    "temperature f": "temperature_f",
    "temperature_f": "temperature_f",
    "body temperature": "temperature_c",
    "respiratory rate": "respiratory_rate",
    "respiratory_rate": "respiratory_rate",
    "resp rate": "respiratory_rate",
    "breathing rate": "respiratory_rate",
    "systolic bp": "bp_systolic",
    "systolic_bp": "bp_systolic",
    "systolic": "bp_systolic",
    "systolic blood pressure": "bp_systolic",
    "diastolic bp": "bp_diastolic",
    "diastolic_bp": "bp_diastolic",
    "diastolic": "bp_diastolic",
    "diastolic blood pressure": "bp_diastolic",
    "bp_systolic": "bp_systolic",
    "bp_diastolic": "bp_diastolic",

    # ── Blood / CBC — core ──
    "haemoglobin": "haemoglobin",
    "hemoglobin": "haemoglobin",
    "hgb": "haemoglobin",
    "hb": "haemoglobin",
    "hemoglobin hb": "haemoglobin",
    "haemoglobin hb": "haemoglobin",
    # WBC — expanded aliases including new variants
    "wbc": "wbc",
    "white blood cells": "wbc",
    "white blood cell": "wbc",
    "white blood cell count": "wbc",
    "white blood cells count": "wbc",
    "total white blood cell count": "wbc",
    "total white blood cells count": "wbc",
    "white cell count": "wbc",
    "leucocyte count": "wbc",
    "leukocyte count": "wbc",
    "leucocytes count": "wbc",
    "leukocytes count": "wbc",
    "leucocytes": "wbc",
    "leukocytes": "wbc",
    "tlc": "wbc",
    "total leucocyte count": "wbc",
    "total leukocyte count": "wbc",
    "total leucocytes count": "wbc",
    "total leukocytes count": "wbc",
    "total leucocytes count tlc": "wbc",
    "total leukocytes count tlc": "wbc",
    "total wbc": "wbc",
    "total wbc count": "wbc",
    # RBC
    "rbc": "rbc",
    "red blood cells": "rbc",
    "red blood cell": "rbc",
    "red blood cell count": "rbc",
    "red cell count": "rbc",
    "erythrocytes": "rbc",
    "total rbc": "rbc",
    "rbc count": "rbc",
    "total rbc count": "rbc",
    # Platelets
    "platelets": "platelets",
    "platelet count": "platelets",
    "plt": "platelets",
    "thrombocytes": "platelets",
    "thrombocyte count": "platelets",
    "total platelet count": "platelets",
    # Hematocrit
    "hematocrit": "hematocrit",
    "haematocrit": "hematocrit",
    "hct": "hematocrit",
    "pcv": "hematocrit",
    "packed cell volume": "hematocrit",

    # ── CBC indices ──
    "mcv": "mcv",
    "mean corpuscular volume": "mcv",
    "mch": "mch",
    "mean corpuscular hemoglobin": "mch",
    "mean corpuscular haemoglobin": "mch",
    "mchc": "mchc",
    "mean corpuscular hemoglobin concentration": "mchc",
    "mean corpuscular haemoglobin concentration": "mchc",
    "rdw_cv": "rdw_cv",
    "rdw cv": "rdw_cv",
    "rdw": "rdw_cv",
    "red cell distribution width": "rdw_cv",
    "mpv": "mpv",
    "mean platelet volume": "mpv",
    "pdw": "pdw",

    # ── WBC differentials ──
    "neutrophils": "neutrophils",
    "neutrophil": "neutrophils",
    "polymorphs": "neutrophils",
    "lymphocytes": "lymphocytes",
    "lymphocyte": "lymphocytes",
    "monocytes": "monocytes",
    "monocyte": "monocytes",
    "eosinophils": "eosinophils",
    "eosinophil": "eosinophils",
    "basophils": "basophils",
    "basophil": "basophils",

    # ── Metabolic / glycemic ──
    "glucose": "glucose",
    "blood glucose": "glucose",
    "blood sugar": "glucose",
    "fasting glucose": "glucose",
    "fasting blood sugar": "glucose",
    "fasting blood glucose": "glucose",
    "random glucose": "glucose",
    "random blood sugar": "glucose",
    "postprandial glucose": "glucose",
    "post prandial glucose": "glucose",
    "ppbs": "glucose",
    "fbs": "glucose",
    "rbs": "glucose",
    "mean blood glucose": "glucose",
    "mean blood glucose level": "glucose",
    "hba1c": "hba1c",
    "hb a1c": "hba1c",
    "a1c": "hba1c",
    "glycated hemoglobin": "hba1c",
    "glycated haemoglobin": "hba1c",
    "glycosylated hemoglobin": "hba1c",
    "glycosylated haemoglobin": "hba1c",
    "insulin": "insulin",
    "fasting insulin": "insulin",

    # ── Electrolytes ──
    "potassium": "potassium",
    "k": "potassium",
    "k+": "potassium",
    "serum potassium": "potassium",
    "sodium": "sodium",
    "na": "sodium",
    "na+": "sodium",
    "serum sodium": "sodium",
    "chloride": "chloride",
    "cl": "chloride",
    "cl-": "chloride",
    "serum chloride": "chloride",
    "calcium": "calcium",
    "ca": "calcium",
    "serum calcium": "calcium",
    "total calcium": "calcium",
    "phosphorus": "phosphorus",
    "phosphate": "phosphorus",
    "serum phosphorus": "phosphorus",
    "magnesium": "magnesium",
    "mg": "magnesium",
    "serum magnesium": "magnesium",

    # ── Cardiac ──
    "troponin": "troponin",
    "troponin i": "troponin",
    "troponin t": "troponin",
    "trop i": "troponin",
    "trop t": "troponin",
    "trop-i": "troponin",
    "trop-t": "troponin",
    "ctni": "troponin",
    "ctnt": "troponin",
    "hs troponin": "troponin",
    "high sensitivity troponin": "troponin",
    "bnp": "bnp",
    "nt probnp": "bnp",
    "nt-probnp": "bnp",
    "b type natriuretic peptide": "bnp",

    # ── Kidney / Liver ──
    "creatinine": "creatinine",
    "creat": "creatinine",
    "cr": "creatinine",
    "serum creatinine": "creatinine",
    "urea": "urea",
    "blood urea": "urea",
    "serum urea": "urea",
    "bun": "bun",
    "blood urea nitrogen": "bun",
    "uric acid": "uric_acid",
    "uric_acid": "uric_acid",
    "serum uric acid": "uric_acid",
    "sgpt": "sgpt_alt",
    "alt": "sgpt_alt",
    "sgpt alt": "sgpt_alt",
    "sgpt (alt)": "sgpt_alt",
    "alanine aminotransferase": "sgpt_alt",
    "alanine transaminase": "sgpt_alt",
    "sgot": "sgot_ast",
    "ast": "sgot_ast",
    "sgot ast": "sgot_ast",
    "sgot (ast)": "sgot_ast",
    "aspartate aminotransferase": "sgot_ast",
    "aspartate transaminase": "sgot_ast",
    "bilirubin": "bilirubin",
    "total bilirubin": "bilirubin",
    "serum bilirubin": "bilirubin",
    "bilirubin total": "bilirubin",
    "bilirubin direct": "bilirubin_direct",
    "direct bilirubin": "bilirubin_direct",
    "conjugated bilirubin": "bilirubin_direct",
    "bilirubin indirect": "bilirubin_indirect",
    "indirect bilirubin": "bilirubin_indirect",
    "unconjugated bilirubin": "bilirubin_indirect",
    "albumin": "albumin",
    "serum albumin": "albumin",
    "globulin": "globulin",
    "serum globulin": "globulin",
    "total protein": "total_protein",
    "serum total protein": "total_protein",
    "serum protein": "total_protein",
    "protein total": "total_protein",
    "ag ratio": "ag_ratio",
    "a g ratio": "ag_ratio",
    "a/g ratio": "ag_ratio",
    "albumin globulin ratio": "ag_ratio",
    "alp": "alp",
    "alkaline phosphatase": "alp",
    "serum alkaline phosphatase": "alp",
    "ggt": "ggt",
    "gamma gt": "ggt",
    "gamma glutamyl transferase": "ggt",
    "gamma glutamyl transpeptidase": "ggt",

    # ── Iron studies ──
    "iron": "iron",
    "sr iron": "iron",
    "sr. iron": "iron",
    "serum iron": "iron",
    "ferritin": "ferritin",
    "serum ferritin": "ferritin",
    "tibc": "tibc",
    "total iron binding capacity": "tibc",
    "transferrin": "transferrin",
    "transferrin saturation": "transferrin_saturation",

    # ── Lipids ──
    "total cholesterol": "total_cholesterol",
    "cholesterol": "total_cholesterol",
    "serum cholesterol": "total_cholesterol",
    "serum cholesterol total": "total_cholesterol",
    "cholesterol total": "total_cholesterol",
    "ldl": "ldl_cholesterol",
    "ldl cholesterol": "ldl_cholesterol",
    "ldl-c": "ldl_cholesterol",
    "low density lipoprotein": "ldl_cholesterol",
    "low density lipoprotein cholesterol": "ldl_cholesterol",
    "hdl": "hdl_cholesterol",
    "hdl cholesterol": "hdl_cholesterol",
    "hdl-c": "hdl_cholesterol",
    "high density lipoprotein": "hdl_cholesterol",
    "high density lipoprotein cholesterol": "hdl_cholesterol",
    "vldl": "vldl_cholesterol",
    "vldl cholesterol": "vldl_cholesterol",
    "vldl-c": "vldl_cholesterol",
    "very low density lipoprotein": "vldl_cholesterol",
    "triglycerides": "triglycerides",
    "triglyceride": "triglycerides",
    "tg": "triglycerides",
    "serum triglycerides": "triglycerides",
    "non hdl cholesterol": "non_hdl_cholesterol",
    "non-hdl cholesterol": "non_hdl_cholesterol",
    "chol hdl ratio": "chol_hdl_ratio",
    "cholesterol hdl ratio": "chol_hdl_ratio",
    "chol/hdl ratio": "chol_hdl_ratio",
    "total cholesterol hdl ratio": "chol_hdl_ratio",
    "ldl hdl ratio": "ldl_hdl_ratio",
    "ldl/hdl ratio": "ldl_hdl_ratio",

    # ── Vitamins ──
    "vitamin d": "vitamin_d",
    "vit d": "vitamin_d",
    "vitamin d3": "vitamin_d",
    "vit d3": "vitamin_d",
    "vitamin d total": "vitamin_d",
    "vit d total": "vitamin_d",
    "25 oh vitamin d": "vitamin_d",
    "25 oh vit d": "vitamin_d",
    "25oh vitamin d": "vitamin_d",
    "25oh vit d": "vitamin_d",
    "25 hydroxyvitamin d": "vitamin_d",
    "25 hydroxy vitamin d": "vitamin_d",
    "25 hydroxy vitamin d3": "vitamin_d",
    "25 hydroxy vit d": "vitamin_d",
    "25 hydroxy vit d3": "vitamin_d",
    "25 hydroxy oh vit d": "vitamin_d",
    "hydroxy vit d": "vitamin_d",
    "hydroxy vitamin d": "vitamin_d",
    "vitamin d 25 hydroxy": "vitamin_d",
    "serum 25 oh vitamin d": "vitamin_d",
    "serum vitamin d": "vitamin_d",
    "s 25 oh vitamin d": "vitamin_d",
    "calcidiol": "vitamin_d",
    "cholecalciferol": "vitamin_d",
    "vitamin b12": "vitamin_b12",
    "vit b12": "vitamin_b12",
    "b12": "vitamin_b12",
    "vitamin b 12": "vitamin_b12",
    "vitamin b12 level": "vitamin_b12",
    "cobalamin": "vitamin_b12",
    "cyanocobalamin": "vitamin_b12",
    "serum vitamin b12": "vitamin_b12",
    "folate": "folate",
    "folic acid": "folate",
    "vitamin b9": "folate",
    "serum folate": "folate",
    "red cell folate": "folate",

    # ── Thyroid — expanded T3/T4 variants for hyphen-normalization ──
    "tsh": "tsh",
    "thyroid stimulating hormone": "tsh",
    "s tsh": "tsh",
    "serum tsh": "tsh",
    "t3": "t3",
    "total t3": "t3",
    "total t 3": "t3",
    "total t-3": "t3",
    "triiodothyronine": "t3",
    "total triiodothyronine": "t3",
    "t4": "t4",
    "total t4": "t4",
    "total t 4": "t4",
    "total t-4": "t4",
    "thyroxine": "t4",
    "total thyroxine": "t4",
    "ft3": "free_t3",
    "free t3": "free_t3",
    "free t 3": "free_t3",
    "free t-3": "free_t3",
    "free triiodothyronine": "free_t3",
    "ft4": "free_t4",
    "free t4": "free_t4",
    "free t 4": "free_t4",
    "free t-4": "free_t4",
    "free thyroxine": "free_t4",
    "anti tpo": "anti_tpo",
    "anti-tpo": "anti_tpo",
    "anti thyroid peroxidase": "anti_tpo",

    # ── Inflammation ──
    "crp": "crp",
    "c reactive protein": "crp",
    "c-reactive protein": "crp",
    "hs crp": "crp",
    "hs-crp": "crp",
    "esr": "esr",
    "erythrocyte sedimentation rate": "esr",

    # ── Coagulation ──
    "pt": "pt",
    "prothrombin time": "pt",
    "inr": "inr",
    "international normalized ratio": "inr",
    "aptt": "aptt",
    "activated partial thromboplastin time": "aptt",
    "d dimer": "d_dimer",
    "d-dimer": "d_dimer",
    "fibrinogen": "fibrinogen",

    # ── Hormones ──
    "cortisol": "cortisol",
    "serum cortisol": "cortisol",
    "morning cortisol": "cortisol",
    "am cortisol": "cortisol",
    "testosterone": "testosterone",
    "total testosterone": "testosterone",
    "serum testosterone": "testosterone",
    "estrogen": "estrogen",
    "estradiol": "estrogen",
    "progesterone": "progesterone",
    "prolactin": "prolactin",
    "prl": "prolactin",
    "serum prolactin": "prolactin",
    "lh": "lh",
    "luteinizing hormone": "lh",
    "fsh": "fsh",
    "follicle stimulating hormone": "fsh",
    "homocysteine": "homocysteine",

    # ── Tumor markers ──
    "psa": "psa",
    "prostate specific antigen": "psa",
    "total psa": "psa",
    "free psa": "psa",
    "ca 125": "ca_125",
    "ca-125": "ca_125",
    "ca125": "ca_125",
    "ca 19 9": "ca_19_9",
    "ca 19-9": "ca_19_9",
    "ca19-9": "ca_19_9",
    "cea": "cea",
    "carcinoembryonic antigen": "cea",
    "afp": "afp",
    "alpha fetoprotein": "afp",
    "beta hcg": "beta_hcg",

    # ── Urine analysis (numeric portion) ──
    "urine ph": "urine_ph",
    "specific gravity": "specific_gravity",
    "urine protein": "urine_protein",
    "urine glucose": "urine_glucose",
    "urine ketones": "urine_ketones",
    "urine wbc": "urine_wbc",
    "urine rbc": "urine_rbc",
    "pus cells": "urine_wbc",
    "epithelial cells": "urine_epithelial",
}


def _kb_key(raw_key: str) -> str:
    """Normalise any incoming key to the canonical KB key."""
    normalised = re.sub(r"[^a-z0-9]+", "_", str(raw_key or "").lower()).strip("_")
    return _KB_ALIASES.get(normalised.replace("_", " "), _KB_ALIASES.get(normalised, normalised))


# ═══════════════════════════════════════════════════════════════════
# FUZZY MATCHER — token-based scoring for unknown name formats.
# ═══════════════════════════════════════════════════════════════════
_CANONICAL_TOKENS: dict[str, dict[str, set[str]]] = {
    # ── Vitamins ──
    "vitamin_d": {
        "signal":   {"vitamin", "vit", "d", "25", "oh", "hydroxy", "d3",
                     "calcidiol", "cholecalciferol", "hydroxyvitamin"},
        "required": {"d"},
        "excluded": {"b12", "b9", "b6", "b1", "b2", "b3", "b5",
                     "cobalamin", "folate", "folic",
                     "c", "e", "k", "a"},
    },
    "vitamin_b12": {
        "signal":   {"vitamin", "vit", "b12", "b", "12",
                     "cobalamin", "cyanocobalamin", "methylcobalamin"},
        "required": {"b12"},
        "excluded": {"folate", "folic", "b9", "d", "b6", "c"},
    },
    "folate": {
        "signal":   {"folate", "folic", "acid", "vitamin", "b9"},
        "required": {"folate"},
        "excluded": {"b12", "cobalamin", "d"},
    },

    # ── Thyroid ──
    "tsh": {
        "signal":   {"tsh", "thyroid", "stimulating", "hormone"},
        "required": {"tsh"},
        "excluded": {"t3", "t4"},
    },
    "t3": {
        "signal":   {"t3", "total", "triiodothyronine"},
        "required": {"t3"},
        "excluded": {"free", "ft3", "reverse"},
    },
    "t4": {
        "signal":   {"t4", "total", "thyroxine"},
        "required": {"t4"},
        "excluded": {"free", "ft4"},
    },
    "free_t3": {
        "signal":   {"free", "ft3", "t3", "triiodothyronine"},
        "required": {"free"},
        "excluded": set(),
    },
    "free_t4": {
        "signal":   {"free", "ft4", "t4", "thyroxine"},
        "required": {"free"},
        "excluded": set(),
    },

    # ── Lipids ──
    "ldl_cholesterol": {
        "signal":   {"ldl", "cholesterol", "low", "density", "lipoprotein"},
        "required": {"ldl"},
        "excluded": {"hdl", "vldl", "ratio"},
    },
    "hdl_cholesterol": {
        "signal":   {"hdl", "cholesterol", "high", "density", "lipoprotein"},
        "required": {"hdl"},
        "excluded": {"ldl", "vldl", "ratio", "non"},
    },
    "vldl_cholesterol": {
        "signal":   {"vldl", "cholesterol", "very", "low", "density"},
        "required": {"vldl"},
        "excluded": {"ldl", "hdl", "ratio"},
    },
    "total_cholesterol": {
        "signal":   {"total", "serum", "cholesterol"},
        "required": {"cholesterol"},
        "excluded": {"ldl", "hdl", "vldl", "ratio", "non"},
    },
    "triglycerides": {
        "signal":   {"triglycerides", "triglyceride", "tg", "serum"},
        "required": {"triglyceride"},
        "excluded": set(),
    },

    # ── CBC ──
    "haemoglobin": {
        "signal":   {"hemoglobin", "haemoglobin", "hb", "hgb"},
        "required": {"hemoglobin"},
        "excluded": {"glycated", "glycosylated", "hba1c", "a1c",
                     "mean", "corpuscular", "mch", "mchc"},
    },
    "hba1c": {
        "signal":   {"hba1c", "a1c", "glycated", "glycosylated",
                     "hemoglobin", "haemoglobin"},
        "required": {"a1c"},
        "excluded": set(),
    },
    "wbc": {
        "signal":   {"wbc", "white", "blood", "cells", "leucocytes",
                     "leukocytes", "tlc", "total", "count"},
        "required": {"wbc"},
        "excluded": {"differential", "neutrophil", "lymphocyte",
                     "monocyte", "eosinophil", "basophil"},
    },
    "rbc": {
        "signal":   {"rbc", "red", "blood", "cells", "erythrocytes"},
        "required": {"rbc"},
        "excluded": {"mch", "mchc", "mcv", "mpv", "rdw",
                     "hemoglobin", "haemoglobin", "platelet"},
    },
    "platelets": {
        "signal":   {"platelets", "platelet", "plt", "thrombocytes",
                     "count"},
        "required": {"platelet"},
        "excluded": {"mpv", "pdw"},
    },
    "hematocrit": {
        "signal":   {"hematocrit", "haematocrit", "hct", "pcv",
                     "packed", "cell", "volume"},
        "required": {"hematocrit"},
        "excluded": set(),
    },

    # ── Kidney / Liver ──
    "creatinine": {
        "signal":   {"creatinine", "creat", "cr", "serum"},
        "required": {"creatinine"},
        "excluded": {"clearance", "urine"},
    },
    "urea": {
        "signal":   {"urea", "blood", "serum"},
        "required": {"urea"},
        "excluded": {"nitrogen", "bun"},
    },
    "bun": {
        "signal":   {"bun", "blood", "urea", "nitrogen"},
        "required": {"bun"},
        "excluded": set(),
    },
    "uric_acid": {
        "signal":   {"uric", "acid", "serum"},
        "required": {"uric"},
        "excluded": set(),
    },
    "sgpt_alt": {
        "signal":   {"sgpt", "alt", "alanine", "aminotransferase"},
        "required": {"alt"},
        "excluded": {"sgot", "ast"},
    },
    "sgot_ast": {
        "signal":   {"sgot", "ast", "aspartate", "aminotransferase"},
        "required": {"ast"},
        "excluded": {"sgpt", "alt"},
    },
    "bilirubin": {
        "signal":   {"bilirubin", "total", "serum"},
        "required": {"bilirubin"},
        "excluded": {"direct", "indirect", "conjugated", "unconjugated"},
    },
    "bilirubin_direct": {
        "signal":   {"bilirubin", "direct", "conjugated"},
        "required": {"direct"},
        "excluded": {"indirect", "unconjugated"},
    },
    "bilirubin_indirect": {
        "signal":   {"bilirubin", "indirect", "unconjugated"},
        "required": {"indirect"},
        "excluded": {"direct", "conjugated"},
    },
    "albumin": {
        "signal":   {"albumin", "serum"},
        "required": {"albumin"},
        "excluded": {"globulin", "ratio", "urine", "microalbumin"},
    },
    "globulin": {
        "signal":   {"globulin", "serum"},
        "required": {"globulin"},
        "excluded": {"albumin", "ratio", "immuno"},
    },
    "total_protein": {
        "signal":   {"total", "protein", "serum"},
        "required": {"protein"},
        "excluded": {"albumin", "globulin", "urine", "c-reactive", "reactive"},
    },
    "ag_ratio": {
        "signal":   {"a", "g", "ratio", "albumin", "globulin", "ag"},
        "required": {"ratio"},
        "excluded": {"chol", "ldl", "hdl"},
    },
    "alp": {
        "signal":   {"alp", "alkaline", "phosphatase"},
        "required": {"alkaline"},
        "excluded": set(),
    },
    "ggt": {
        "signal":   {"ggt", "gamma", "gt", "glutamyl", "transferase"},
        "required": {"ggt"},
        "excluded": set(),
    },

    # ── Metabolic ──
    "glucose": {
        "signal":   {"glucose", "sugar", "blood", "fasting", "random",
                     "fbs", "rbs", "ppbs", "postprandial"},
        "required": {"glucose"},
        "excluded": {"urine", "tolerance", "hba1c", "a1c"},
    },
    "insulin": {
        "signal":   {"insulin", "serum", "fasting"},
        "required": {"insulin"},
        "excluded": set(),
    },

    # ── Iron studies ──
    "iron": {
        "signal":   {"iron", "serum", "sr"},
        "required": {"iron"},
        "excluded": {"binding", "tibc", "ferritin", "transferrin"},
    },
    "ferritin": {
        "signal":   {"ferritin", "serum"},
        "required": {"ferritin"},
        "excluded": set(),
    },
    "tibc": {
        "signal":   {"tibc", "total", "iron", "binding", "capacity"},
        "required": {"tibc"},
        "excluded": set(),
    },

    # ── Electrolytes ──
    "sodium": {
        "signal":   {"sodium", "na", "serum"},
        "required": {"sodium"},
        "excluded": {"urine"},
    },
    "potassium": {
        "signal":   {"potassium", "k", "serum"},
        "required": {"potassium"},
        "excluded": {"urine"},
    },
    "chloride": {
        "signal":   {"chloride", "cl", "serum"},
        "required": {"chloride"},
        "excluded": {"urine"},
    },
    "calcium": {
        "signal":   {"calcium", "ca", "serum", "total"},
        "required": {"calcium"},
        "excluded": {"ionized", "urine", "corrected"},
    },
    "phosphorus": {
        "signal":   {"phosphorus", "phosphate", "p", "serum"},
        "required": {"phosphor"},
        "excluded": {"alkaline"},
    },
    "magnesium": {
        "signal":   {"magnesium", "mg", "serum"},
        "required": {"magnesium"},
        "excluded": {"urine"},
    },

    # ── Cardiac ──
    "troponin": {
        "signal":   {"troponin", "trop", "hs", "ctni", "ctnt", "i", "t"},
        "required": {"troponin"},
        "excluded": set(),
    },
    "bnp": {
        "signal":   {"bnp", "nt", "probnp", "natriuretic", "peptide"},
        "required": {"bnp"},
        "excluded": set(),
    },

    # ── Inflammation ──
    "crp": {
        "signal":   {"crp", "c", "reactive", "protein", "hs"},
        "required": {"crp"},
        "excluded": {"total"},
    },
    "esr": {
        "signal":   {"esr", "erythrocyte", "sedimentation", "rate"},
        "required": {"esr"},
        "excluded": set(),
    },

    # ── Coagulation ──
    "pt": {
        "signal":   {"pt", "prothrombin", "time"},
        "required": {"prothrombin"},
        "excluded": {"aptt", "activated", "inr", "partial"},
    },
    "inr": {
        "signal":   {"inr", "international", "normalized", "ratio"},
        "required": {"inr"},
        "excluded": set(),
    },
    "aptt": {
        "signal":   {"aptt", "activated", "partial", "thromboplastin", "time"},
        "required": {"aptt"},
        "excluded": set(),
    },
    "d_dimer": {
        "signal":   {"d", "dimer"},
        "required": {"dimer"},
        "excluded": set(),
    },

    # ── Hormones ──
    "cortisol": {
        "signal":   {"cortisol", "serum", "morning", "am", "pm"},
        "required": {"cortisol"},
        "excluded": {"urine", "saliva"},
    },
    "testosterone": {
        "signal":   {"testosterone", "total", "free", "serum"},
        "required": {"testosterone"},
        "excluded": set(),
    },
    "prolactin": {
        "signal":   {"prolactin", "prl", "serum"},
        "required": {"prolactin"},
        "excluded": set(),
    },
    "lh": {
        "signal":   {"lh", "luteinizing", "hormone"},
        "required": {"lh"},
        "excluded": set(),
    },
    "fsh": {
        "signal":   {"fsh", "follicle", "stimulating", "hormone"},
        "required": {"fsh"},
        "excluded": set(),
    },

    # ── Tumor markers ──
    "psa": {
        "signal":   {"psa", "prostate", "specific", "antigen", "total", "free"},
        "required": {"psa"},
        "excluded": set(),
    },
    "cea": {
        "signal":   {"cea", "carcinoembryonic", "antigen"},
        "required": {"cea"},
        "excluded": set(),
    },
    "afp": {
        "signal":   {"afp", "alpha", "fetoprotein", "feto"},
        "required": {"afp"},
        "excluded": set(),
    },
}


# Alternative token sets that also satisfy the "required" check.
_REQUIRED_VARIANTS: dict[str, list[set[str]]] = {
    "vitamin_b12":        [{"b12"}, {"cobalamin"}, {"b", "12"}],
    "folate":             [{"folate"}, {"folic"}],
    "haemoglobin":        [{"hemoglobin"}, {"haemoglobin"}, {"hb"}, {"hgb"}],
    "hba1c":              [{"hba1c"}, {"a1c"}, {"glycated"}, {"glycosylated"}],
    "wbc":                [{"wbc"}, {"leucocytes"}, {"leukocytes"}, {"tlc"},
                           {"white", "blood"}, {"white", "cell"}],
    "rbc":                [{"rbc"}, {"erythrocytes"}, {"red", "blood"},
                           {"red", "cell"}],
    "platelets":          [{"platelets"}, {"platelet"}, {"plt"}, {"thrombocytes"}],
    "hematocrit":         [{"hematocrit"}, {"haematocrit"}, {"hct"}, {"pcv"}],
    "bun":                [{"bun"}, {"blood", "urea", "nitrogen"}],
    "alp":                [{"alp"}, {"alkaline", "phosphatase"}],
    "ggt":                [{"ggt"}, {"gamma", "gt"}, {"gamma", "glutamyl"}],
    "glucose":            [{"glucose"}, {"sugar"}, {"fbs"}, {"rbs"}, {"ppbs"}],
    "iron":               [{"iron"}],
    "sodium":             [{"sodium"}, {"na"}],
    "potassium":          [{"potassium"}, {"k"}],
    "chloride":           [{"chloride"}, {"cl"}],
    "phosphorus":         [{"phosphorus"}, {"phosphate"}, {"phosphor"}],
    "troponin":           [{"troponin"}, {"trop"}, {"ctni"}, {"ctnt"}],
    "bnp":                [{"bnp"}, {"probnp"}, {"natriuretic"}],
    "crp":                [{"crp"}, {"reactive", "protein"}],
    "esr":                [{"esr"}, {"sedimentation"}],
    "pt":                 [{"pt"}, {"prothrombin"}],
    "aptt":               [{"aptt"}, {"activated", "partial"}],
    "prolactin":          [{"prolactin"}, {"prl"}],
    "lh":                 [{"lh"}, {"luteinizing"}],
    "fsh":                [{"fsh"}, {"follicle"}],
    "psa":                [{"psa"}, {"prostate", "specific"}],
    "afp":                [{"afp"}, {"fetoprotein"}, {"feto", "protein"}],
    "free_t3":            [{"free", "t3"}, {"ft3"}],
    "free_t4":            [{"free", "t4"}, {"ft4"}],
    "bilirubin_direct":   [{"direct", "bilirubin"}, {"conjugated", "bilirubin"}],
    "bilirubin_indirect": [{"indirect", "bilirubin"}, {"unconjugated", "bilirubin"}],
    "ag_ratio":           [{"a", "g", "ratio"}, {"albumin", "globulin", "ratio"},
                           {"ag", "ratio"}],
    "d_dimer":            [{"d", "dimer"}, {"ddimer"}],
    "tibc":               [{"tibc"}, {"binding", "capacity"}],
    "insulin":            [{"insulin"}],
}

def _tokenize(name: str) -> set[str]:
    """Extract normalized word tokens from a raw biomarker name."""
    lowered = name.lower()
    # Preserve compound medical tokens as single tokens
    lowered = re.sub(r"\bb\s+12\b", "b12", lowered)
    lowered = re.sub(r"\bb\s+6\b",  "b6",  lowered)
    lowered = re.sub(r"\bb\s+9\b",  "b9",  lowered)
    lowered = re.sub(r"\bt\s+3\b",  "t3",  lowered)
    lowered = re.sub(r"\bt\s+4\b",  "t4",  lowered)
    lowered = re.sub(r"\bd\s+3\b",  "d3",  lowered)
    raw_tokens = re.split(r"[^a-z0-9]+", lowered)
    return {t for t in raw_tokens if t}


def _matches_required(candidate_key: str, input_tokens: set[str]) -> bool:
    """Required-token check: uses variants if defined, else strict."""
    if candidate_key in _REQUIRED_VARIANTS:
        return any(variant.issubset(input_tokens)
                   for variant in _REQUIRED_VARIANTS[candidate_key])
    spec = _CANONICAL_TOKENS.get(candidate_key, {})
    required = spec.get("required", set())
    if not required:
        return True
    return required.issubset(input_tokens)


def _matches_excluded(candidate_key: str, input_tokens: set[str]) -> bool:
    """Return True if any excluded token is in the input (→ disqualify)."""
    spec = _CANONICAL_TOKENS.get(candidate_key, {})
    excluded = spec.get("excluded", set())
    return bool(excluded & input_tokens)


def _fuzzy_score(candidate_key: str, input_tokens: set[str]) -> float:
    """Score how well input_tokens match candidate_key's signal tokens."""
    if not _matches_required(candidate_key, input_tokens):
        return 0.0
    if _matches_excluded(candidate_key, input_tokens):
        return 0.0

    spec = _CANONICAL_TOKENS.get(candidate_key, {})
    signal = spec.get("signal", set())
    if not signal:
        return 0.0

    overlap = signal & input_tokens
    if not overlap:
        return 0.0

    signal_coverage = len(overlap) / len(signal)
    input_coverage  = len(overlap) / max(len(input_tokens), 1)
    return (signal_coverage * 0.6) + (input_coverage * 0.4)


def _normalize_for_alias_lookup(name: str) -> str:
    """
    Normalise a raw name for exact-match against _KB_ALIASES.

    Steps:
      1. Lowercase + strip whitespace
      2. Strip ALL parenthetical clarifications
      3. Normalise separators (hyphens, dashes, underscores, etc.) to spaces
      4. Collapse whitespace
      5. Rejoin broken medical tokens: "t 3" → "t3", "b 12" → "b12", etc.
         This is CRITICAL because step 3 converts "T-3" to "t 3" which then
         fails alias lookup against "total t3". The rejoin step ensures
         "Total T-3" normalises to "total t3" (which IS in _KB_ALIASES).
    """
    lowered = name.lower().strip()
    # Strip ALL parenthetical clarifications (anywhere in string)
    lowered = re.sub(r"\s*\([^)]*\)\s*", " ", lowered)
    # Normalise separators to spaces
    lowered = re.sub(r"[-–—_/\\.,;:]+", " ", lowered)
    # Collapse whitespace
    lowered = re.sub(r"\s+", " ", lowered).strip()

    # ═══ Rejoin broken medical tokens ═══
    # After hyphen→space conversion, "T-3" becomes "t 3" which then
    # fails alias lookup. Rejoin these compound tokens so
    # "Total T-3" → "total t3", "Vitamin B-12" → "vitamin b12", etc.
    lowered = re.sub(r"\bt\s+3\b",    "t3",    lowered)
    lowered = re.sub(r"\bt\s+4\b",    "t4",    lowered)
    lowered = re.sub(r"\bb\s+12\b",   "b12",   lowered)
    lowered = re.sub(r"\bb\s+6\b",    "b6",    lowered)
    lowered = re.sub(r"\bb\s+9\b",    "b9",    lowered)
    lowered = re.sub(r"\bd\s+3\b",    "d3",    lowered)
    lowered = re.sub(r"\bk\s+1\b",    "k1",    lowered)
    lowered = re.sub(r"\bk\s+2\b",    "k2",    lowered)
    lowered = re.sub(r"\b25\s+oh\b",  "25oh",  lowered)
    lowered = re.sub(r"\b17\s+oh\b",  "17oh",  lowered)
    lowered = re.sub(r"\bhb\s+a1c\b", "hba1c", lowered)
    lowered = re.sub(r"\ba\s+1c\b",   "a1c",   lowered)
    lowered = re.sub(r"\bca\s+125\b", "ca125",  lowered)
    lowered = re.sub(r"\bca\s+19\s+9\b", "ca19 9", lowered)

    return lowered


def resolve_canonical_key(
    raw_name: str,
    min_confidence: float = 0.35,
) -> str | None:
    """
    Universal biomarker name resolver — SINGLE ENTRY POINT.

    Returns the canonical key (e.g. "vitamin_d") if the raw name maps
    to a known biomarker, or None if confidence is too low.

    Resolution waterfall:
      1. Exact alias match (space form)     → _KB_ALIASES
      2. Underscore form alias match        → _KB_ALIASES
      3. Fuzzy token-overlap match          → _CANONICAL_TOKENS
      4. Return None → caller keeps raw key
    """
    if not raw_name or not isinstance(raw_name, str):
        return None

    # Path 1 + 2: exact alias match
    normalised = _normalize_for_alias_lookup(raw_name)
    if normalised in _KB_ALIASES:
        return _KB_ALIASES[normalised]
    underscored = normalised.replace(" ", "_")
    if underscored in _KB_ALIASES:
        return _KB_ALIASES[underscored]

    # Path 3: fuzzy token match
    input_tokens = _tokenize(raw_name)
    if not input_tokens:
        return None

    best_key: str | None = None
    best_score: float = 0.0
    second_best_score: float = 0.0

    for candidate_key in _CANONICAL_TOKENS:
        score = _fuzzy_score(candidate_key, input_tokens)
        if score > best_score:
            second_best_score = best_score
            best_score = score
            best_key = candidate_key
        elif score > second_best_score:
            second_best_score = score

    if best_score >= min_confidence and (best_score - second_best_score) >= 0.10:
        return best_key
    return None


# ═══════════════════════════════════════════════════════════════════
# DISPLAY NAMES — canonical key → properly-cased medical name.
# ═══════════════════════════════════════════════════════════════════
_DISPLAY_NAMES: dict[str, str] = {
    "vitamin_d":           "Vitamin D",
    "vitamin_b12":         "Vitamin B12",
    "folate":              "Folate",
    "tsh":                 "TSH",
    "t3":                  "Total T3",
    "t4":                  "Total T4",
    "free_t3":             "Free T3",
    "free_t4":             "Free T4",
    "anti_tpo":            "Anti-TPO",
    "ldl_cholesterol":     "LDL Cholesterol",
    "hdl_cholesterol":     "HDL Cholesterol",
    "vldl_cholesterol":    "VLDL Cholesterol",
    "total_cholesterol":   "Serum Cholesterol Total",
    "triglycerides":       "Triglycerides",
    "chol_hdl_ratio":      "Chol HDL Ratio",
    "ldl_hdl_ratio":       "LDL HDL Ratio",
    "non_hdl_cholesterol": "Non-HDL Cholesterol",
    "haemoglobin":         "Haemoglobin",
    "hba1c":               "HbA1c",
    "wbc":                 "White Blood Cells",
    "rbc":                 "RBC",
    "platelets":           "Platelets",
    "hematocrit":          "Hematocrit",
    "mcv":                 "MCV",
    "mch":                 "MCH",
    "mchc":                "MCHC",
    "mpv":                 "MPV",
    "pdw":                 "PDW",
    "rdw_cv":              "RDW",
    "rdw":                 "RDW",
    "neutrophils":         "Neutrophils",
    "lymphocytes":         "Lymphocytes",
    "monocytes":           "Monocytes",
    "eosinophils":         "Eosinophils",
    "basophils":           "Basophils",
    "sgpt_alt":            "SGPT ALT",
    "sgot_ast":            "SGOT AST",
    "alp":                 "ALP",
    "ggt":                 "GGT",
    "bilirubin":           "Bilirubin",
    "bilirubin_direct":    "Bilirubin Direct",
    "bilirubin_indirect":  "Bilirubin Indirect",
    "albumin":             "Albumin",
    "globulin":            "Globulin",
    "total_protein":       "Total Protein",
    "ag_ratio":            "A/G Ratio",
    "creatinine":          "Creatinine",
    "urea":                "Urea",
    "bun":                 "BUN",
    "uric_acid":           "Uric Acid",
    "iron":                "Iron",
    "ferritin":            "Ferritin",
    "tibc":                "TIBC",
    "transferrin":         "Transferrin",
    "transferrin_saturation": "Transferrin Saturation",
    "sodium":              "Sodium",
    "potassium":           "Potassium",
    "chloride":            "Chloride",
    "calcium":             "Calcium",
    "magnesium":           "Magnesium",
    "phosphorus":          "Phosphorus",
    "glucose":             "Glucose",
    "insulin":             "Insulin",
    "troponin":            "Troponin",
    "bnp":                 "BNP",
    "crp":                 "CRP",
    "esr":                 "ESR",
    "pt":                  "PT",
    "inr":                 "INR",
    "aptt":                "aPTT",
    "d_dimer":             "D-Dimer",
    "fibrinogen":          "Fibrinogen",
    "cortisol":            "Cortisol",
    "testosterone":        "Total Testosterone",
    "estrogen":            "Estrogen",
    "progesterone":        "Progesterone",
    "prolactin":           "Prolactin",
    "lh":                  "LH",
    "fsh":                 "FSH",
    "homocysteine":        "Homocysteine",
    "psa":                 "PSA",
    "ca_125":              "CA 125",
    "ca_19_9":             "CA 19-9",
    "cea":                 "CEA",
    "afp":                 "AFP",
    "beta_hcg":            "Beta hCG",
    "heart_rate":          "Heart Rate",
    "spo2":                "SpO2",
    "temperature_c":       "Temperature",
    "temperature_f":       "Temperature",
    "respiratory_rate":    "Respiratory Rate",
    "systolic_bp":         "Systolic Blood Pressure",
    "diastolic_bp":        "Diastolic Blood Pressure",
    "bp_systolic":         "Systolic Blood Pressure",
    "bp_diastolic":        "Diastolic Blood Pressure",
    "urine_ph":            "Urine pH",
    "specific_gravity":    "Specific Gravity",
    "urine_protein":       "Urine Protein",
    "urine_glucose":       "Urine Glucose",
    "urine_ketones":       "Urine Ketones",
    "urine_wbc":           "Urine WBC",
    "urine_rbc":           "Urine RBC",
    "urine_epithelial":    "Urine Epithelial Cells",
}


def get_display_name(canonical_key: str) -> str:
    """
    Return a properly-cased display name for a canonical key.
    Falls back to Title Case for unknown keys.
    """
    if not canonical_key:
        return ""
    if canonical_key in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[canonical_key]

    _UPPER_TOKENS = {
        "tsh", "hdl", "ldl", "vldl", "sgot", "sgpt", "alt", "ast",
        "hba1c", "a1c", "wbc", "rbc", "mcv", "mch", "mchc", "mpv",
        "pdw", "rdw", "cv", "alp", "ggt", "bun", "crp", "esr",
        "psa", "cea", "afp", "lh", "fsh", "bnp", "pt", "inr", "aptt",
        "hcg", "spo2", "bp", "hb", "hgb", "ca", "na", "cl", "mg",
        "tibc", "pcv", "hct", "tlc", "fbs", "rbs", "ppbs", "b12",
    }
    parts = canonical_key.replace("_", " ").split()
    out = []
    for p in parts:
        if p.lower() in _UPPER_TOKENS:
            if p.lower() == "hba1c":
                out.append("HbA1c")
            elif p.lower() == "spo2":
                out.append("SpO2")
            elif p.lower() == "aptt":
                out.append("aPTT")
            else:
                out.append(p.upper())
        else:
            out.append(p.capitalize())
    return " ".join(out)


# ═══════════════════════════════════════════════════════════════════
# BIOMARKER KNOWLEDGE BASE — clinical advice per (biomarker, status).
# ═══════════════════════════════════════════════════════════════════
BIOMARKER_KNOWLEDGE: dict[str, dict[str, dict[str, Any]]] = {

    # ═══════════════════ THYROID ═══════════════════
    "tsh": {
        "high": {
            "recommendation": "Elevated TSH suggests possible hypothyroidism — consult an endocrinologist for confirmation with Free T3, Free T4, and thyroid antibody testing.",
            "care_plan": {
                "immediate": "Book an appointment with an endocrinologist within the next 1–2 weeks to discuss thyroid function.",
                "short_term": "Get a full thyroid panel including Free T3, Free T4, and anti-TPO antibodies to confirm the diagnosis.",
                "lifestyle": "Ensure adequate dietary iodine (iodized salt, dairy, seafood) and selenium (Brazil nuts, eggs).",
                "follow_up": "Repeat TSH in 6–8 weeks after clinical review or start of any prescribed therapy.",
                "long_term": None,
            },
        },
        "critical_high": {
            "recommendation": "TSH is markedly elevated — urgent endocrinology review is recommended to evaluate for overt hypothyroidism.",
            "care_plan": {
                "immediate": "Seek endocrinology consultation as soon as possible — do not delay beyond one week.",
                "short_term": "Complete Free T3, Free T4, anti-TPO, and thyroglobulin antibody testing.",
                "lifestyle": "Avoid sudden dietary iodine excess until reviewed by a physician.",
                "follow_up": "Repeat TSH every 6 weeks until stable on treatment.",
                "long_term": "Lifelong thyroid monitoring will likely be required.",
            },
        },
        "low": {
            "recommendation": "Low TSH may indicate hyperthyroidism — clinical evaluation and repeat thyroid panel are recommended.",
            "care_plan": {
                "immediate": "Consult an endocrinologist promptly to evaluate possible hyperthyroidism.",
                "short_term": "Order Free T3, Free T4, and TSH-receptor antibody testing.",
                "lifestyle": "Avoid excessive iodine intake and limit stimulants (caffeine) until reviewed.",
                "follow_up": "Repeat thyroid panel in 4–6 weeks per specialist guidance.",
                "long_term": None,
            },
        },
        "critical_low": {
            "recommendation": "TSH is markedly suppressed — urgent endocrinology review is recommended to rule out overt hyperthyroidism or thyrotoxicosis.",
            "care_plan": {
                "immediate": "Book urgent endocrinology consultation within the week.",
                "short_term": "Complete Free T3, Free T4, TSH-receptor antibody, and thyroid ultrasound.",
                "lifestyle": "Avoid iodine-rich supplements and excessive caffeine until evaluated.",
                "follow_up": "Repeat thyroid panel every 4 weeks until stable.",
                "long_term": "Long-term thyroid monitoring will likely be required.",
            },
        },
    },

    "t3": {
        "high": {
            "recommendation": "Elevated T3 may indicate hyperthyroidism — please discuss with your physician for a full thyroid evaluation.",
            "care_plan": {"immediate": "Consult endocrinology for evaluation of hyperthyroid symptoms.", "short_term": "Complete Free T3, Free T4, TSH, and antibody panel.", "lifestyle": "Avoid iodine supplements and stimulants until reviewed.", "follow_up": "Repeat thyroid panel in 4–6 weeks.", "long_term": None},
        },
        "low": {
            "recommendation": "Low T3 may indicate hypothyroidism or non-thyroidal illness — please review with your physician.",
            "care_plan": {"immediate": "Discuss with your physician within the next 1–2 weeks.", "short_term": "Complete a full thyroid panel including antibodies.", "lifestyle": "Ensure adequate iodine and selenium intake.", "follow_up": "Recheck thyroid panel in 6–8 weeks.", "long_term": None},
        },
    },

    "t4": {
        "high": {
            "recommendation": "Elevated T4 may indicate hyperthyroidism — please review with an endocrinologist.",
            "care_plan": {"immediate": "Book endocrinology consultation within 1–2 weeks.", "short_term": "Complete Free T4, Free T3, TSH, and antibody testing.", "lifestyle": "Avoid iodine excess and stimulants until reviewed.", "follow_up": "Repeat thyroid panel in 4–6 weeks.", "long_term": None},
        },
        "low": {
            "recommendation": "Low T4 may indicate hypothyroidism — please discuss with your physician for confirmation.",
            "care_plan": {"immediate": "Consult endocrinology within 1–2 weeks.", "short_term": "Complete full thyroid panel including antibodies.", "lifestyle": "Include iodine-rich foods; avoid extreme calorie restriction.", "follow_up": "Recheck thyroid panel in 6–8 weeks.", "long_term": None},
        },
    },

    # ═══════════════════ VITAMINS ═══════════════════
    "vitamin_d": {
        "low": {
            "recommendation": "Vitamin D deficiency detected — consider daily supplementation (typically 1000–2000 IU) and increased sun exposure. Recheck in 3 months.",
            "care_plan": {"immediate": "Begin daily Vitamin D supplementation (typically 1000–2000 IU) after physician confirmation.", "short_term": "Aim for 15–20 minutes of midday sunlight exposure 3–4 times per week.", "lifestyle": "Include Vitamin D–rich foods regularly: fatty fish (salmon, mackerel), fortified milk, egg yolks.", "follow_up": "Recheck Vitamin D level in 3 months to confirm response to supplementation.", "long_term": "Maintain a maintenance dose annually if deficiency was severe."},
        },
        "critical_low": {
            "recommendation": "Vitamin D is severely deficient — physician-guided high-dose supplementation is recommended.",
            "care_plan": {"immediate": "Consult your physician within 1 week to discuss high-dose Vitamin D therapy (often 50,000 IU weekly for 8 weeks).", "short_term": "Complete calcium, phosphate, and PTH testing to assess metabolic impact.", "lifestyle": "Increase safe sun exposure and dietary Vitamin D sources.", "follow_up": "Recheck Vitamin D at 8–12 weeks.", "long_term": "Continue maintenance dose lifelong if repeatedly deficient."},
        },
        "borderline": {
            "recommendation": "Vitamin D is on the low side — dietary sources (fatty fish, fortified milk) and moderate sun exposure are recommended.",
            "care_plan": {"immediate": None, "short_term": "Discuss whether a low-dose supplement (600–1000 IU) is appropriate.", "lifestyle": "Include Vitamin D–rich foods and 10–15 minutes of daily sunlight.", "follow_up": "Recheck Vitamin D in 6 months.", "long_term": None},
        },
        "high": {
            "recommendation": "Vitamin D is elevated — review supplementation with your physician to avoid toxicity.",
            "care_plan": {"immediate": "Stop Vitamin D supplements and consult your physician.", "short_term": "Check serum calcium to rule out hypercalcemia.", "lifestyle": "Avoid additional Vitamin D–fortified foods and supplements until reviewed.", "follow_up": "Recheck Vitamin D and calcium in 4–8 weeks.", "long_term": None},
        },
    },

    "vitamin_b12": {
        "low": {
            "recommendation": "Vitamin B12 is suboptimal — include B12-rich foods (meat, dairy, eggs) or discuss supplementation with your doctor.",
            "care_plan": {"immediate": None, "short_term": "Discuss B12 supplementation (oral or injectable) with your physician if deficiency is confirmed.", "lifestyle": "Increase B12-rich foods: meat, poultry, fish, dairy, eggs, and fortified cereals.", "follow_up": "Recheck Vitamin B12 in 3 months and check homocysteine or MMA if symptoms persist.", "long_term": "Vegetarians and older adults may need ongoing B12 supplementation."},
        },
        "critical_low": {
            "recommendation": "Vitamin B12 is severely low — evaluate for pernicious anemia and consider injectable B12 therapy.",
            "care_plan": {"immediate": "Consult your physician within 1 week to discuss urgent B12 replacement.", "short_term": "Complete intrinsic factor antibody, homocysteine, and MMA testing.", "lifestyle": "Increase B12-rich foods alongside supplementation.", "follow_up": "Recheck B12 in 4–8 weeks after therapy.", "long_term": "Lifelong B12 monitoring may be required."},
        },
        "borderline": {
            "recommendation": "Vitamin B12 is borderline — dietary review is recommended.",
            "care_plan": {"immediate": None, "short_term": "Include B12-rich foods; discuss low-dose supplementation with physician.", "lifestyle": "Regular intake of meat, dairy, eggs, or fortified plant milks.", "follow_up": "Recheck B12 in 6 months.", "long_term": None},
        },
    },

    "folate": {
        "low": {"recommendation": "Low folate detected — dietary review and folate supplementation may be recommended.", "care_plan": {"immediate": None, "short_term": "Discuss folate supplementation with your physician.", "lifestyle": "Increase folate-rich foods: leafy greens, legumes, fortified grains, citrus fruits.", "follow_up": "Recheck folate in 2–3 months.", "long_term": None}},
    },

    # ═══════════════════ METABOLIC / GLYCEMIC ═══════════════════
    "glucose": {
        "high": {"recommendation": "Elevated glucose detected — reduce refined carbohydrates and sugars, increase physical activity, and consider HbA1c testing to screen for diabetes.", "care_plan": {"immediate": "Reduce simple sugars, sweets, and refined carbohydrates immediately.", "short_term": "Get HbA1c and fasting glucose testing done to screen for prediabetes or diabetes.", "lifestyle": "Adopt a balanced diet emphasizing whole grains, vegetables, lean protein, and healthy fats. Aim for at least 150 minutes of moderate exercise per week.", "follow_up": "Follow up with primary care physician in 4–6 weeks with repeat glucose testing.", "long_term": "Establish a diabetes prevention plan if HbA1c is in the prediabetic range."}},
        "critical_high": {"recommendation": "Glucose is markedly elevated — urgent medical evaluation is recommended to rule out diabetic emergency.", "care_plan": {"immediate": "Seek medical attention promptly; if symptoms of diabetic ketoacidosis (extreme thirst, vomiting, confusion) are present, go to an emergency department immediately.", "short_term": "Complete HbA1c, ketones, and full metabolic panel.", "lifestyle": "Strict dietary control and hydration until reviewed by a physician.", "follow_up": "Weekly physician review until glucose is stabilised.", "long_term": "Establish long-term diabetes management plan with an endocrinologist."}},
        "low": {"recommendation": "Glucose is lower than normal — eat a balanced meal soon and discuss recurrent low readings with your physician.", "care_plan": {"immediate": "Consume a fast-acting carbohydrate (juice, glucose tablets) if symptomatic.", "short_term": "Investigate cause with your physician if this recurs.", "lifestyle": "Avoid long fasting periods; eat regular balanced meals.", "follow_up": "Recheck fasting and postprandial glucose within 2 weeks.", "long_term": None}},
        "borderline": {"recommendation": "Glucose is at a borderline level — dietary review and periodic monitoring are recommended.", "care_plan": {"immediate": None, "short_term": "Discuss with your physician and consider HbA1c testing.", "lifestyle": "Limit sugary drinks and refined carbs; increase physical activity.", "follow_up": "Recheck fasting glucose in 3 months.", "long_term": None}},
    },

    "hba1c": {
        "high": {"recommendation": "HbA1c is outside optimal range — dietary review and physician consultation for diabetes screening is advised.", "care_plan": {"immediate": None, "short_term": "Consult a physician for structured diabetes screening and possible glucose tolerance testing.", "lifestyle": "Reduce refined carbohydrates, prioritise whole grains, increase physical activity to 150+ min/week.", "follow_up": "Recheck HbA1c every 3 months until stable.", "long_term": "Establish a long-term glycemic management plan."}},
        "critical_high": {"recommendation": "HbA1c indicates poorly controlled or established diabetes — urgent physician review is required.", "care_plan": {"immediate": "Book physician appointment within 1 week to initiate or intensify diabetes management.", "short_term": "Complete full metabolic panel, lipid panel, and kidney function testing.", "lifestyle": "Strict dietary control, regular exercise, and diabetes education.", "follow_up": "Recheck HbA1c every 3 months.", "long_term": "Long-term diabetes monitoring including annual eye and foot examinations."}},
        "borderline": {"recommendation": "HbA1c is at a borderline level — dietary review and lifestyle changes are advised.", "care_plan": {"immediate": None, "short_term": "Discuss prediabetes screening with your physician.", "lifestyle": "Reduce refined sugars and carbs; increase daily activity.", "follow_up": "Recheck HbA1c in 3–6 months.", "long_term": None}},
    },

    "insulin": {
        "high": {"recommendation": "Elevated insulin may indicate insulin resistance — dietary and lifestyle review recommended.", "care_plan": {"immediate": None, "short_term": "Consult physician for insulin resistance workup and HbA1c testing.", "lifestyle": "Reduce refined carbs and sugars; increase physical activity and strength training.", "follow_up": "Recheck insulin, glucose, and HbA1c in 3 months.", "long_term": None}},
    },

    # ═══════════════════ LIPIDS ═══════════════════
    "ldl_cholesterol": {
        "high": {"recommendation": "Elevated LDL cholesterol — adopt a heart-healthy diet (low saturated fat), increase soluble fiber, and consider a lipid-lowering plan with your physician.", "care_plan": {"immediate": None, "short_term": "Discuss LDL cholesterol management with your physician; lipid-lowering therapy may be considered.", "lifestyle": "Adopt a heart-healthy diet: reduce saturated and trans fats, increase soluble fiber (oats, beans, apples).", "follow_up": "Recheck lipid profile in 3 months after lifestyle or medication changes.", "long_term": "Ongoing cardiovascular risk monitoring is recommended."}},
        "critical_high": {"recommendation": "LDL cholesterol is severely elevated — physician consultation is strongly recommended, likely including lipid-lowering therapy.", "care_plan": {"immediate": "Book physician appointment within 1–2 weeks.", "short_term": "Discuss statin therapy and cardiovascular risk assessment.", "lifestyle": "Strict heart-healthy diet, weight management, and regular aerobic exercise.", "follow_up": "Recheck lipid panel in 8–12 weeks after initiating therapy.", "long_term": "Long-term cardiovascular risk monitoring and management."}},
        "borderline": {"recommendation": "LDL cholesterol is at a borderline value — heart-healthy lifestyle changes are recommended.", "care_plan": {"immediate": None, "short_term": "Discuss cardiovascular risk with your physician.", "lifestyle": "Reduce saturated fat; increase soluble fiber, aerobic exercise, and healthy fats.", "follow_up": "Recheck lipid profile in 3–6 months.", "long_term": None}},
    },

    "hdl_cholesterol": {"low": {"recommendation": "Low HDL cholesterol — increase aerobic exercise, quit smoking if applicable, and include healthy fats (olive oil, nuts).", "care_plan": {"immediate": None, "short_term": "Discuss overall cardiovascular risk with your physician.", "lifestyle": "Increase aerobic exercise (30 min/day, 5 days/week). Include healthy fats: olive oil, avocado, nuts, and fatty fish.", "follow_up": "Recheck lipid profile in 3–6 months.", "long_term": None}}},
    "triglycerides": {
        "high": {"recommendation": "Elevated triglycerides — limit alcohol, reduce sugar intake, and increase omega-3-rich foods.", "care_plan": {"immediate": None, "short_term": "Limit alcohol, sugary drinks, and refined carbohydrates.", "lifestyle": "Increase omega-3 intake through fatty fish (2–3 servings/week) or physician-approved fish oil.", "follow_up": "Recheck triglycerides in 3 months alongside full lipid panel.", "long_term": None}},
        "critical_high": {"recommendation": "Triglycerides are severely elevated — urgent physician review is recommended to reduce risk of pancreatitis.", "care_plan": {"immediate": "Consult your physician within 1 week; strict dietary changes are needed.", "short_term": "Complete full lipid, glucose, and liver panels.", "lifestyle": "Strict low-fat, low-sugar diet; eliminate alcohol.", "follow_up": "Recheck lipid panel in 4–8 weeks.", "long_term": "Long-term triglyceride management may require medication."}},
        "borderline": {"recommendation": "Triglycerides are borderline high — dietary changes are recommended.", "care_plan": {"immediate": None, "short_term": "Reduce sugar, alcohol, and refined carbs.", "lifestyle": "Increase omega-3 foods and regular exercise.", "follow_up": "Recheck lipid panel in 3–6 months.", "long_term": None}},
    },
    "total_cholesterol": {
        "high": {"recommendation": "Total cholesterol above range — a comprehensive lipid management plan is advised.", "care_plan": {"immediate": None, "short_term": "Discuss a comprehensive lipid management plan with your physician.", "lifestyle": "Reduce dietary cholesterol from red meat and full-fat dairy; increase plant-based fiber.", "follow_up": "Recheck lipid panel in 3 months.", "long_term": None}},
        "borderline": {"recommendation": "Total cholesterol is at a borderline level — dietary review is recommended.", "care_plan": {"immediate": None, "short_term": "Discuss cardiovascular risk with your physician.", "lifestyle": "Adopt a heart-healthy diet; reduce saturated fats.", "follow_up": "Recheck lipid panel in 3–6 months.", "long_term": None}},
    },
    "vldl_cholesterol": {"high": {"recommendation": "Elevated VLDL is often linked to elevated triglycerides — dietary review and lipid panel follow-up are recommended.", "care_plan": {"immediate": None, "short_term": "Reduce sugar and refined carbohydrate intake.", "lifestyle": "Increase physical activity and omega-3 intake.", "follow_up": "Recheck lipid panel in 3 months.", "long_term": None}}},
    "non_hdl_cholesterol": {"high": {"recommendation": "Non-HDL cholesterol is elevated — cardiovascular risk review with your physician is recommended.", "care_plan": {"immediate": None, "short_term": "Discuss cardiovascular risk stratification with your physician.", "lifestyle": "Heart-healthy diet; reduce saturated and trans fats.", "follow_up": "Recheck lipid panel in 3 months.", "long_term": None}}},
    "chol_hdl_ratio": {"high": {"recommendation": "Elevated cholesterol/HDL ratio suggests increased cardiovascular risk — lifestyle changes are recommended.", "care_plan": {"immediate": None, "short_term": "Discuss overall cardiovascular risk with your physician.", "lifestyle": "Increase aerobic exercise, adopt heart-healthy diet, avoid smoking.", "follow_up": "Recheck lipid panel in 3–6 months.", "long_term": None}}},
    "ldl_hdl_ratio": {"high": {"recommendation": "Elevated LDL/HDL ratio suggests atherogenic lipid pattern — cardiovascular risk assessment recommended.", "care_plan": {"immediate": None, "short_term": "Discuss lipid management with your physician.", "lifestyle": "Heart-healthy diet with soluble fiber; regular aerobic exercise.", "follow_up": "Recheck lipid panel in 3–6 months.", "long_term": None}}},

    # ═══════════════════ CBC / BLOOD ═══════════════════
    "haemoglobin": {
        "low": {"recommendation": "Low haemoglobin may indicate anemia — iron-rich foods (leafy greens, red meat), and iron studies are recommended.", "care_plan": {"immediate": None, "short_term": "Get iron studies (serum iron, ferritin, TIBC) to evaluate possible iron deficiency anemia.", "lifestyle": "Include iron-rich foods: leafy greens, red meat, lentils, and pair with Vitamin C for better absorption.", "follow_up": "Recheck haemoglobin and iron levels in 2–3 months after dietary or supplement changes.", "long_term": None}},
        "critical_low": {"recommendation": "Haemoglobin is critically low — urgent medical evaluation is required; transfusion or intravenous iron may be needed.", "care_plan": {"immediate": "Seek medical attention promptly; if severely symptomatic (dizziness, breathlessness, chest pain), go to an emergency department.", "short_term": "Complete iron studies, reticulocyte count, and evaluate for blood loss.", "lifestyle": "Iron-rich diet alongside physician-directed treatment.", "follow_up": "Weekly haemoglobin monitoring until stable.", "long_term": "Long-term anemia management plan with cause identification."}},
        "high": {"recommendation": "Elevated haemoglobin — discuss with your physician to evaluate underlying causes (dehydration, smoking, or hematologic conditions).", "care_plan": {"immediate": None, "short_term": "Consult physician for evaluation; may require additional CBC and erythropoietin testing.", "lifestyle": "Maintain hydration; avoid smoking.", "follow_up": "Recheck haemoglobin in 4–8 weeks.", "long_term": None}},
        "borderline": {"recommendation": "Haemoglobin is at a borderline level — dietary iron and B12 review is recommended.", "care_plan": {"immediate": None, "short_term": "Include iron-rich foods and consider B12/folate intake.", "lifestyle": "Balanced diet with leafy greens, lean protein.", "follow_up": "Recheck haemoglobin in 3 months.", "long_term": None}},
    },
    "wbc": {
        "high": {"recommendation": "Elevated WBC may indicate infection or inflammation — clinical correlation and repeat CBC are advised.", "care_plan": {"immediate": "Discuss elevated WBC with a physician to rule out infection or inflammation.", "short_term": "Repeat CBC in 2–4 weeks; if persistent, consider peripheral smear and further workup.", "lifestyle": "Maintain general hygiene and monitor for infection symptoms.", "follow_up": "Repeat CBC as advised by physician.", "long_term": None}},
        "low": {"recommendation": "Low WBC — evaluate for immune status, infection risk, and consider follow-up CBC.", "care_plan": {"immediate": "Consult a physician to evaluate low WBC; monitor for signs of infection (fever, chills).", "short_term": "Repeat CBC and consider bone marrow evaluation if persistently low.", "lifestyle": "Avoid exposure to infections; maintain good hygiene.", "follow_up": "Repeat CBC as advised by physician.", "long_term": None}},
        "borderline": {"recommendation": "WBC is at a borderline level — monitor for signs of infection and repeat CBC.", "care_plan": {"immediate": None, "short_term": "Recheck CBC in 4–6 weeks.", "lifestyle": "Maintain good hygiene and adequate sleep.", "follow_up": "Repeat CBC as advised.", "long_term": None}},
    },
    "platelets": {
        "low": {"recommendation": "Low platelet count — avoid injury/trauma, consult hematology to determine underlying cause.", "care_plan": {"immediate": "Avoid trauma, aspirin, and NSAIDs. Consult hematology promptly.", "short_term": "Complete peripheral smear and evaluate for underlying cause.", "lifestyle": "Avoid contact sports and high-risk activities until reviewed.", "follow_up": "Repeat CBC as advised by hematologist.", "long_term": None}},
        "critical_low": {"recommendation": "Platelet count is critically low — urgent hematology consultation is required due to bleeding risk.", "care_plan": {"immediate": "Seek urgent medical attention; avoid all trauma, aspirin, and anticoagulants.", "short_term": "Hospitalisation may be needed for platelet transfusion or evaluation.", "lifestyle": "Strict activity restrictions until platelets recover.", "follow_up": "Daily to weekly monitoring per hematology guidance.", "long_term": "Long-term hematology follow-up."}},
        "high": {"recommendation": "Elevated platelets — clinical review for underlying cause (inflammation, reactive thrombocytosis) is advised.", "care_plan": {"immediate": None, "short_term": "Consult physician to evaluate underlying cause.", "lifestyle": "Maintain hydration and avoid smoking.", "follow_up": "Repeat CBC in 4–6 weeks to check for persistence.", "long_term": None}},
        "borderline": {"recommendation": "Platelet count is borderline — monitor and repeat CBC.", "care_plan": {"immediate": None, "short_term": "Recheck CBC in 4–6 weeks.", "lifestyle": "Maintain hydration and balanced nutrition.", "follow_up": "Repeat CBC as advised.", "long_term": None}},
    },
    "rbc": {
        "low": {"recommendation": "Low RBC count — evaluate for anemia; iron studies and B12/folate levels are recommended.", "care_plan": {"immediate": None, "short_term": "Complete iron studies, B12, folate, and reticulocyte count.", "lifestyle": "Iron-rich diet with Vitamin C for better absorption.", "follow_up": "Recheck CBC in 2–3 months.", "long_term": None}},
        "high": {"recommendation": "Elevated RBC — discuss with physician to evaluate causes such as dehydration or polycythemia.", "care_plan": {"immediate": None, "short_term": "Consult physician for further evaluation.", "lifestyle": "Maintain hydration; avoid smoking.", "follow_up": "Recheck CBC in 4–8 weeks.", "long_term": None}},
        "borderline": {"recommendation": "RBC is at a borderline level — dietary iron review and monitoring recommended.", "care_plan": {"immediate": None, "short_term": "Include iron-rich foods and B12/folate sources.", "lifestyle": "Balanced diet with leafy greens and lean protein.", "follow_up": "Recheck CBC in 3 months.", "long_term": None}},
    },
    "hematocrit": {
        "low": {"recommendation": "Low hematocrit suggests possible anemia — please discuss with your physician.", "care_plan": {"immediate": None, "short_term": "Complete iron studies and full anemia workup.", "lifestyle": "Iron-rich diet with Vitamin C for better absorption.", "follow_up": "Recheck CBC in 2–3 months.", "long_term": None}},
        "high": {"recommendation": "Elevated hematocrit — please review with your physician (may reflect dehydration or polycythemia).", "care_plan": {"immediate": None, "short_term": "Consult physician for evaluation.", "lifestyle": "Maintain hydration.", "follow_up": "Recheck CBC in 4–8 weeks.", "long_term": None}},
        "borderline": {"recommendation": "Hematocrit is at a borderline value — monitor and maintain hydration.", "care_plan": {"immediate": None, "short_term": "Recheck CBC in 3 months.", "lifestyle": "Maintain adequate hydration and balanced iron intake.", "follow_up": "Repeat CBC as advised.", "long_term": None}},
    },
    "mcv": {"high": {"recommendation": "Elevated MCV suggests macrocytic changes — B12/folate deficiency or liver/thyroid causes should be evaluated.", "care_plan": {"immediate": None, "short_term": "Check B12, folate, TSH, and liver function.", "lifestyle": "Balanced diet with B12- and folate-rich foods; limit alcohol.", "follow_up": "Recheck CBC and workup in 2–3 months.", "long_term": None}}, "low": {"recommendation": "Low MCV suggests microcytic changes — iron deficiency or thalassemia trait should be evaluated.", "care_plan": {"immediate": None, "short_term": "Complete iron studies and hemoglobin electrophoresis if indicated.", "lifestyle": "Iron-rich diet with Vitamin C for absorption.", "follow_up": "Recheck CBC in 2–3 months.", "long_term": None}}},
    "mch": {"high": {"recommendation": "Elevated MCH may reflect macrocytic anemia — clinical review is recommended.", "care_plan": {"immediate": None, "short_term": "Check B12, folate, and reticulocyte count.", "lifestyle": "Balanced diet including B12/folate; limit alcohol.", "follow_up": "Recheck CBC in 2–3 months.", "long_term": None}}, "low": {"recommendation": "Low MCH may reflect microcytic anemia — iron studies are recommended.", "care_plan": {"immediate": None, "short_term": "Complete iron studies.", "lifestyle": "Iron-rich diet with Vitamin C.", "follow_up": "Recheck CBC in 2–3 months.", "long_term": None}}},
    "mchc": {"low": {"recommendation": "Low MCHC may indicate hypochromic anemia — iron studies are recommended.", "care_plan": {"immediate": None, "short_term": "Complete iron studies.", "lifestyle": "Iron-rich diet.", "follow_up": "Recheck CBC in 2–3 months.", "long_term": None}}},
    "rdw_cv": {"high": {"recommendation": "Elevated RDW suggests variability in red cell size — evaluate for early iron, B12, or folate deficiency.", "care_plan": {"immediate": None, "short_term": "Complete iron, B12, and folate studies.", "lifestyle": "Balanced diet with iron and B-vitamin sources.", "follow_up": "Recheck CBC in 2–3 months.", "long_term": None}}},
    "mpv": {"high": {"recommendation": "Elevated MPV may reflect increased platelet turnover — clinical correlation is recommended.", "care_plan": {"immediate": None, "short_term": "Discuss with physician; may require follow-up CBC.", "lifestyle": "Maintain hydration and balanced diet.", "follow_up": "Recheck CBC in 4–6 weeks.", "long_term": None}}, "low": {"recommendation": "Low MPV may occur with certain marrow conditions — clinical review is advised.", "care_plan": {"immediate": None, "short_term": "Discuss with physician.", "lifestyle": "Balanced nutrition.", "follow_up": "Recheck CBC as advised.", "long_term": None}}},
    "neutrophils": {"high": {"recommendation": "Elevated neutrophils may indicate bacterial infection, stress, or inflammation — clinical review is recommended.", "care_plan": {"immediate": "Discuss with physician if febrile or symptomatic.", "short_term": "Repeat CBC and evaluate for infection source.", "lifestyle": "Maintain hygiene; rest and hydration if unwell.", "follow_up": "Repeat CBC in 2–4 weeks.", "long_term": None}}, "low": {"recommendation": "Low neutrophils increase infection risk — clinical evaluation is recommended.", "care_plan": {"immediate": "Consult physician; monitor for signs of infection.", "short_term": "Repeat CBC and consider peripheral smear.", "lifestyle": "Avoid crowds and infection exposure; strict hygiene.", "follow_up": "Repeat CBC as advised.", "long_term": None}}, "borderline": {"recommendation": "Neutrophils are at a borderline level — repeat CBC recommended.", "care_plan": {"immediate": None, "short_term": "Recheck CBC in 4–6 weeks.", "lifestyle": "Maintain hygiene and adequate nutrition.", "follow_up": "Repeat CBC as advised.", "long_term": None}}},
    "lymphocytes": {"high": {"recommendation": "Elevated lymphocytes may indicate viral infection or chronic immune stimulation — clinical review recommended.", "care_plan": {"immediate": None, "short_term": "Discuss with physician; consider peripheral smear if persistent.", "lifestyle": "Rest, hydration, and monitor for infection symptoms.", "follow_up": "Repeat CBC in 2–4 weeks.", "long_term": None}}, "low": {"recommendation": "Low lymphocytes may reflect immune compromise or acute illness — clinical evaluation recommended.", "care_plan": {"immediate": "Consult physician if persistent or symptomatic.", "short_term": "Repeat CBC and evaluate immune status.", "lifestyle": "Avoid infection exposure; balanced diet and rest.", "follow_up": "Repeat CBC as advised.", "long_term": None}}, "borderline": {"recommendation": "Lymphocyte count is borderline — repeat CBC recommended.", "care_plan": {"immediate": None, "short_term": "Recheck CBC in 4–6 weeks.", "lifestyle": "Adequate rest, hydration, and nutrition.", "follow_up": "Repeat CBC as advised.", "long_term": None}}},
    "monocytes": {"high": {"recommendation": "Elevated monocytes may indicate chronic infection or inflammation — clinical review recommended.", "care_plan": {"immediate": None, "short_term": "Discuss with physician for evaluation.", "lifestyle": "Balanced diet and rest.", "follow_up": "Repeat CBC in 4–6 weeks.", "long_term": None}}, "low": {"recommendation": "Low monocytes are uncommon and usually not clinically significant — physician review recommended if persistent.", "care_plan": {"immediate": None, "short_term": "Recheck CBC.", "lifestyle": "Maintain overall health.", "follow_up": "Repeat CBC as advised.", "long_term": None}}},
    "eosinophils": {"high": {"recommendation": "Elevated eosinophils may indicate allergic reaction, parasitic infection, or asthma — clinical review recommended.", "care_plan": {"immediate": None, "short_term": "Discuss with physician; consider allergy testing or stool examination if indicated.", "lifestyle": "Avoid known allergens; balanced diet.", "follow_up": "Repeat CBC in 4–6 weeks.", "long_term": None}}},
    "basophils": {"high": {"recommendation": "Elevated basophils are uncommon — clinical review recommended if persistent.", "care_plan": {"immediate": None, "short_term": "Discuss with physician.", "lifestyle": "Balanced diet and rest.", "follow_up": "Repeat CBC as advised.", "long_term": None}}, "borderline": {"recommendation": "Basophils are at a borderline value — no immediate concern; repeat CBC if clinically indicated.", "care_plan": {"immediate": None, "short_term": "Recheck CBC only if symptomatic.", "lifestyle": "Maintain overall health.", "follow_up": "Repeat CBC as advised.", "long_term": None}}},

    # ═══════════════════ ELECTROLYTES / CARDIAC ═══════════════════
    "potassium": {"high": {"recommendation": "Potassium is elevated — clinical evaluation is recommended; avoid high-potassium foods until reviewed.", "care_plan": {"immediate": "Consult your physician within 24–48 hours; avoid bananas, oranges, and salt substitutes.", "short_term": "Repeat electrolytes and kidney function testing.", "lifestyle": "Limit high-potassium foods until cleared.", "follow_up": "Recheck potassium as advised.", "long_term": None}}, "critical_high": {"recommendation": "Potassium is critically elevated — urgent medical attention is required due to cardiac arrhythmia risk.", "care_plan": {"immediate": "Go to an emergency department immediately — this is a medical emergency.", "short_term": "Cardiac monitoring and urgent treatment will be required.", "lifestyle": "Strict low-potassium diet until stabilised.", "follow_up": "Frequent electrolyte monitoring under physician care.", "long_term": "Long-term kidney function monitoring."}}, "low": {"recommendation": "Low potassium — dietary sources (bananas, potatoes, leafy greens) and physician review are recommended.", "care_plan": {"immediate": "Consult physician if symptomatic (muscle weakness, cramps, palpitations).", "short_term": "Repeat electrolytes and evaluate cause.", "lifestyle": "Include potassium-rich foods; avoid excessive caffeine or diuretics.", "follow_up": "Recheck potassium in 2–4 weeks.", "long_term": None}}},
    "sodium": {"high": {"recommendation": "Sodium is elevated — increase water intake and consult physician to evaluate cause.", "care_plan": {"immediate": "Increase water intake unless fluid-restricted; consult physician.", "short_term": "Repeat electrolytes and evaluate hydration status.", "lifestyle": "Maintain adequate hydration and moderate salt intake.", "follow_up": "Recheck sodium in 1–2 weeks.", "long_term": None}}, "low": {"recommendation": "Sodium is low — please consult your physician; underlying causes should be evaluated.", "care_plan": {"immediate": "Consult physician within 1 week (or urgently if symptomatic).", "short_term": "Evaluate cause: medications, kidney or endocrine issues.", "lifestyle": "Avoid excessive water intake; balanced diet with adequate salt.", "follow_up": "Recheck sodium as advised.", "long_term": None}}},
    "chloride": {"high": {"recommendation": "Elevated chloride — clinical review with electrolyte panel is recommended.", "care_plan": {"immediate": None, "short_term": "Repeat electrolytes; evaluate for acid-base disorder.", "lifestyle": "Maintain hydration.", "follow_up": "Recheck electrolytes as advised.", "long_term": None}}, "low": {"recommendation": "Low chloride — clinical evaluation for underlying cause is recommended.", "care_plan": {"immediate": None, "short_term": "Repeat electrolytes.", "lifestyle": "Maintain hydration and balanced diet.", "follow_up": "Recheck electrolytes as advised.", "long_term": None}}},
    "calcium": {"high": {"recommendation": "Elevated calcium — parathyroid and vitamin D evaluation recommended.", "care_plan": {"immediate": "Consult physician; avoid calcium supplements until reviewed.", "short_term": "Check PTH, vitamin D, and phosphorus.", "lifestyle": "Adequate hydration; limit calcium supplements.", "follow_up": "Recheck calcium in 2–4 weeks.", "long_term": None}}, "low": {"recommendation": "Low calcium — dietary review and clinical evaluation recommended.", "care_plan": {"immediate": None, "short_term": "Check vitamin D, magnesium, and PTH.", "lifestyle": "Include calcium-rich foods: dairy, leafy greens, fortified foods.", "follow_up": "Recheck calcium in 4–8 weeks.", "long_term": None}}},
    "troponin": {"high": {"recommendation": "Elevated troponin suggests possible cardiac injury — urgent cardiology evaluation is required.", "care_plan": {"immediate": "Seek emergency medical evaluation immediately.", "short_term": "Cardiac workup including ECG, echocardiography, and repeat troponin.", "lifestyle": "Rest and avoid physical exertion until cleared.", "follow_up": "Follow cardiology guidance closely.", "long_term": "Long-term cardiovascular risk management."}}, "critical_high": {"recommendation": "Troponin is critically elevated — this may indicate a heart attack. Emergency care is required immediately.", "care_plan": {"immediate": "Call emergency services immediately — do not delay.", "short_term": "Hospital admission with continuous cardiac monitoring.", "lifestyle": "Complete rest until medically cleared.", "follow_up": "Cardiology-directed follow-up.", "long_term": "Long-term cardiac rehabilitation and risk management."}}},

    # ═══════════════════ KIDNEY / LIVER ═══════════════════
    "creatinine": {"high": {"recommendation": "Elevated creatinine — hydration status review and nephrology consultation to assess renal function are advised.", "care_plan": {"immediate": "Ensure adequate hydration (2–3 liters of water daily, unless restricted by physician).", "short_term": "Consult nephrology to evaluate renal function and calculate eGFR.", "lifestyle": "Avoid nephrotoxic drugs (NSAIDs) and excessive protein intake.", "follow_up": "Repeat renal panel in 4–6 weeks.", "long_term": "Long-term kidney function monitoring."}}, "critical_high": {"recommendation": "Creatinine is severely elevated — urgent nephrology consultation is required to assess for acute kidney injury.", "care_plan": {"immediate": "Seek medical attention within 24–48 hours.", "short_term": "Complete comprehensive kidney workup including imaging.", "lifestyle": "Strict avoidance of nephrotoxic medications.", "follow_up": "Frequent monitoring under nephrology care.", "long_term": "Ongoing kidney function surveillance."}}, "low": {"recommendation": "Creatinine is lower than typical — often not clinically concerning, but discuss with your physician if persistent.", "care_plan": {"immediate": None, "short_term": "Discuss with physician if concerning.", "lifestyle": "Maintain balanced nutrition.", "follow_up": "Recheck at next routine visit.", "long_term": None}}},
    "urea": {"high": {"recommendation": "Elevated urea — evaluate hydration, kidney function, and dietary protein intake.", "care_plan": {"immediate": "Ensure adequate hydration.", "short_term": "Complete full kidney function panel.", "lifestyle": "Moderate protein intake and adequate fluid intake.", "follow_up": "Recheck in 4–6 weeks.", "long_term": None}}},
    "bun": {"high": {"recommendation": "Elevated BUN — evaluate for dehydration and kidney function.", "care_plan": {"immediate": "Ensure adequate hydration.", "short_term": "Complete kidney function panel.", "lifestyle": "Adequate hydration; moderate protein intake.", "follow_up": "Recheck in 4–6 weeks.", "long_term": None}}},
    "uric_acid": {"high": {"recommendation": "Elevated uric acid — dietary changes and physician review for gout risk are recommended.", "care_plan": {"immediate": None, "short_term": "Discuss with physician; may require additional evaluation.", "lifestyle": "Limit red meat, seafood, alcohol; increase water intake.", "follow_up": "Recheck uric acid in 3 months.", "long_term": "Long-term uric acid monitoring if history of gout."}}},
    "sgpt_alt": {"high": {"recommendation": "Liver enzymes elevated — avoid alcohol, review medications, and follow up with LFT panel.", "care_plan": {"immediate": "Avoid alcohol and review all medications and supplements with your physician.", "short_term": "Discuss hepatology workup: viral hepatitis panel, ultrasound, and metabolic screening.", "lifestyle": "Adopt a liver-friendly diet: reduce processed foods, sugar, and saturated fats.", "follow_up": "Recheck LFT panel in 4–8 weeks.", "long_term": "Ongoing liver health monitoring."}}},
    "sgot_ast": {"high": {"recommendation": "AST is elevated — avoid alcohol and discuss with your physician; may require liver evaluation.", "care_plan": {"immediate": "Avoid alcohol and review medications with physician.", "short_term": "Complete liver panel and hepatitis screening.", "lifestyle": "Liver-friendly diet; adequate hydration.", "follow_up": "Recheck LFT panel in 4–8 weeks.", "long_term": None}}},
    "bilirubin": {"high": {"recommendation": "Elevated bilirubin — please consult your physician for liver evaluation.", "care_plan": {"immediate": "Consult physician within 1–2 weeks.", "short_term": "Complete liver panel, hepatitis screening, and abdominal ultrasound.", "lifestyle": "Avoid alcohol until reviewed.", "follow_up": "Recheck LFT panel in 4–8 weeks.", "long_term": None}}},
    "bilirubin_direct": {"high": {"recommendation": "Elevated direct bilirubin suggests possible cholestasis — liver and biliary evaluation is recommended.", "care_plan": {"immediate": "Consult physician for liver evaluation.", "short_term": "Complete LFT panel and abdominal ultrasound.", "lifestyle": "Avoid alcohol and hepatotoxic medications.", "follow_up": "Recheck LFT in 4–8 weeks.", "long_term": None}}},
    "bilirubin_indirect": {"high": {"recommendation": "Elevated indirect bilirubin may suggest hemolysis or Gilbert's syndrome — clinical evaluation recommended.", "care_plan": {"immediate": None, "short_term": "Complete LFT, CBC, and reticulocyte count.", "lifestyle": "Adequate hydration; avoid extended fasting.", "follow_up": "Recheck LFT in 4–8 weeks.", "long_term": None}}},
    "albumin": {"low": {"recommendation": "Low albumin may indicate nutritional or liver/kidney issues — please review with your physician.", "care_plan": {"immediate": None, "short_term": "Complete liver, kidney, and nutritional evaluation.", "lifestyle": "Balanced diet with adequate protein.", "follow_up": "Recheck albumin in 4–8 weeks.", "long_term": None}}},
    "globulin": {"low": {"recommendation": "Low globulin — evaluate for underlying immune or liver conditions with your physician.", "care_plan": {"immediate": None, "short_term": "Complete protein electrophoresis if persistent.", "lifestyle": "Balanced diet with adequate protein.", "follow_up": "Recheck total protein and globulin in 2–3 months.", "long_term": None}}, "high": {"recommendation": "Elevated globulin may suggest chronic infection, inflammation, or immune disorder — clinical review recommended.", "care_plan": {"immediate": None, "short_term": "Complete protein electrophoresis and inflammatory markers.", "lifestyle": "Balanced nutrition and adequate rest.", "follow_up": "Recheck in 2–3 months.", "long_term": None}}},
    "total_protein": {"low": {"recommendation": "Low total protein — nutritional or liver/kidney evaluation is recommended.", "care_plan": {"immediate": None, "short_term": "Complete liver, kidney panels and nutritional assessment.", "lifestyle": "Increase protein intake through lean meat, dairy, legumes, eggs.", "follow_up": "Recheck total protein in 2–3 months.", "long_term": None}}, "high": {"recommendation": "Elevated total protein — clinical review with protein electrophoresis is recommended.", "care_plan": {"immediate": None, "short_term": "Complete protein electrophoresis and clinical evaluation.", "lifestyle": "Maintain hydration.", "follow_up": "Recheck in 2–3 months.", "long_term": None}}},
    "ag_ratio": {"high": {"recommendation": "Elevated A/G ratio may indicate low globulin or immune deficiency — clinical review recommended.", "care_plan": {"immediate": None, "short_term": "Discuss with physician; consider protein electrophoresis.", "lifestyle": "Balanced diet with adequate protein.", "follow_up": "Recheck in 2–3 months.", "long_term": None}}, "low": {"recommendation": "Low A/G ratio may suggest chronic inflammation, liver disease, or immunoglobulin excess — clinical review recommended.", "care_plan": {"immediate": None, "short_term": "Complete protein electrophoresis and liver function testing.", "lifestyle": "Balanced nutrition; adequate hydration.", "follow_up": "Recheck in 2–3 months.", "long_term": None}}},
    "alp": {"high": {"recommendation": "Elevated ALP may indicate liver or bone disorder — clinical evaluation recommended.", "care_plan": {"immediate": None, "short_term": "Complete LFT, GGT, and vitamin D testing.", "lifestyle": "Balanced diet; avoid alcohol.", "follow_up": "Recheck in 4–8 weeks.", "long_term": None}}},
    "ggt": {"high": {"recommendation": "Elevated GGT may suggest liver or biliary disorder — clinical evaluation recommended.", "care_plan": {"immediate": "Avoid alcohol and review medications with physician.", "short_term": "Complete LFT panel and hepatobiliary imaging if indicated.", "lifestyle": "Liver-friendly diet; eliminate alcohol.", "follow_up": "Recheck LFT in 4–8 weeks.", "long_term": None}}},

    # ═══════════════════ IRON STUDIES ═══════════════════
    "iron": {"low": {"recommendation": "Low serum iron — evaluate for iron-deficiency anemia; iron-rich diet and supplementation may be needed.", "care_plan": {"immediate": None, "short_term": "Complete full iron studies (ferritin, TIBC, transferrin saturation).", "lifestyle": "Iron-rich foods (leafy greens, red meat, lentils); pair with Vitamin C for absorption.", "follow_up": "Recheck iron and haemoglobin in 2–3 months.", "long_term": None}}, "high": {"recommendation": "Elevated iron — please review with your physician; may require ferritin and transferrin saturation testing.", "care_plan": {"immediate": None, "short_term": "Complete full iron studies to rule out hemochromatosis.", "lifestyle": "Avoid iron supplements; moderate red meat consumption.", "follow_up": "Recheck iron studies in 3 months.", "long_term": None}}},
    "ferritin": {"low": {"recommendation": "Low ferritin indicates depleted iron stores — iron supplementation and dietary review are recommended.", "care_plan": {"immediate": None, "short_term": "Discuss iron supplementation with your physician.", "lifestyle": "Iron-rich diet (leafy greens, red meat, lentils, tofu) with Vitamin C.", "follow_up": "Recheck ferritin and CBC in 3 months.", "long_term": "Investigate underlying causes if recurrent (menstrual loss, GI bleeding)."}}, "high": {"recommendation": "Elevated ferritin may indicate inflammation or iron overload — please evaluate with your physician.", "care_plan": {"immediate": None, "short_term": "Evaluate for hemochromatosis with transferrin saturation testing.", "lifestyle": "Avoid iron supplements and reduce red meat intake.", "follow_up": "Recheck ferritin in 4–6 weeks.", "long_term": None}}},

    # ═══════════════════ INFLAMMATION ═══════════════════
    "crp": {"high": {"recommendation": "Elevated CRP suggests inflammation — clinical correlation is recommended to identify the underlying cause.", "care_plan": {"immediate": None, "short_term": "Consult physician to evaluate for infection, inflammation, or autoimmune conditions.", "lifestyle": "Anti-inflammatory diet (fruits, vegetables, omega-3s); regular exercise.", "follow_up": "Repeat CRP as advised by physician.", "long_term": None}}},
    "esr": {"high": {"recommendation": "Elevated ESR suggests inflammation — please review with your physician to determine cause.", "care_plan": {"immediate": None, "short_term": "Evaluate for infection, inflammation, or autoimmune conditions.", "lifestyle": "Anti-inflammatory diet and regular exercise.", "follow_up": "Repeat ESR as advised by physician.", "long_term": None}}},

    # ═══════════════════ VITALS ═══════════════════
    "bp_systolic": {"high": {"recommendation": "Systolic blood pressure is elevated — reduce sodium, monitor daily, and consult your physician.", "care_plan": {"immediate": "Monitor blood pressure at home twice daily and keep a log.", "short_term": "Consult a physician to evaluate for hypertension.", "lifestyle": "Reduce dietary sodium (<2 g/day), limit alcohol, increase potassium-rich foods. Regular aerobic exercise and stress-reduction.", "follow_up": "Follow up with physician in 4 weeks with BP log.", "long_term": "Long-term cardiovascular risk monitoring."}}, "critical_high": {"recommendation": "Systolic blood pressure is critically high — urgent medical evaluation is required.", "care_plan": {"immediate": "Seek urgent medical attention; if symptomatic go to emergency department.", "short_term": "Physician-directed antihypertensive management.", "lifestyle": "Strict low-sodium diet, weight management, regular exercise.", "follow_up": "Frequent physician monitoring until controlled.", "long_term": "Long-term cardiovascular and kidney function monitoring."}}, "low": {"recommendation": "Systolic blood pressure is low — if symptomatic (dizziness, fainting), please consult your physician.", "care_plan": {"immediate": "Ensure adequate hydration and salt intake; consult physician if symptomatic.", "short_term": "Evaluate for underlying causes.", "lifestyle": "Adequate hydration, gradual position changes.", "follow_up": "Recheck BP as advised.", "long_term": None}}},
    "bp_diastolic": {"high": {"recommendation": "Diastolic blood pressure is elevated — lifestyle changes and physician evaluation are recommended.", "care_plan": {"immediate": "Monitor blood pressure at home twice daily.", "short_term": "Consult physician for hypertension evaluation.", "lifestyle": "Reduce sodium, limit alcohol, regular aerobic exercise, stress management.", "follow_up": "Follow up with physician in 4 weeks with BP log.", "long_term": "Long-term cardiovascular monitoring."}}, "critical_high": {"recommendation": "Diastolic blood pressure is critically high — urgent medical evaluation is required.", "care_plan": {"immediate": "Seek urgent medical attention.", "short_term": "Physician-directed antihypertensive management.", "lifestyle": "Strict low-sodium diet, weight management, regular exercise.", "follow_up": "Frequent monitoring until controlled.", "long_term": "Long-term cardiovascular monitoring."}}, "low": {"recommendation": "Diastolic blood pressure is low — if symptomatic (dizziness), please consult your physician.", "care_plan": {"immediate": "Ensure adequate hydration; consult physician if symptomatic.", "short_term": "Evaluate for underlying causes.", "lifestyle": "Adequate hydration; gradual position changes.", "follow_up": "Recheck BP as advised.", "long_term": None}}},
    "heart_rate": {"high": {"recommendation": "Heart rate is elevated — evaluate for cause; consult physician if persistent.", "care_plan": {"immediate": "Rest, hydrate, and monitor pulse; consult physician if persistent.", "short_term": "Complete ECG and thyroid panel.", "lifestyle": "Regular exercise, adequate sleep, limit caffeine.", "follow_up": "Recheck at next physician visit.", "long_term": None}}, "low": {"recommendation": "Heart rate is low — if symptomatic (dizziness, fatigue), please consult your physician.", "care_plan": {"immediate": "Consult physician if symptomatic.", "short_term": "Consider ECG evaluation.", "lifestyle": "Review medications with physician.", "follow_up": "Follow physician guidance.", "long_term": None}}},
    "spo2": {"low": {"recommendation": "Oxygen saturation is low — please consult your physician promptly.", "care_plan": {"immediate": "Seek medical evaluation promptly; if severely low (<90%), seek emergency care.", "short_term": "Complete pulmonary evaluation.", "lifestyle": "Avoid smoking; ensure adequate ventilation.", "follow_up": "Follow physician guidance closely.", "long_term": "Ongoing respiratory monitoring."}}, "critical_low": {"recommendation": "Oxygen saturation is critically low — emergency medical attention is required.", "care_plan": {"immediate": "Call emergency services immediately.", "short_term": "Hospital admission and oxygen therapy will be required.", "lifestyle": "Complete rest until medically cleared.", "follow_up": "Pulmonary and cardiac follow-up.", "long_term": "Long-term respiratory monitoring."}}},
    "temperature": {"high": {"recommendation": "Elevated temperature suggests fever — monitor closely, hydrate, and consult physician if persistent.", "care_plan": {"immediate": "Rest, hydrate, and monitor temperature regularly.", "short_term": "Consult physician if fever persists >3 days or exceeds 103°F.", "lifestyle": "Adequate rest, fluids, and light meals.", "follow_up": "Recheck if symptoms persist.", "long_term": None}}, "critical_high": {"recommendation": "Temperature is critically elevated — seek urgent medical attention.", "care_plan": {"immediate": "Seek emergency medical care.", "short_term": "Physician-directed evaluation for infection or hyperthermia.", "lifestyle": "Complete rest and hydration.", "follow_up": "Follow physician guidance.", "long_term": None}}, "low": {"recommendation": "Temperature is low — if symptomatic or persistent, consult your physician.", "care_plan": {"immediate": "Warm up and monitor; seek medical attention if severe.", "short_term": "Consult physician if persistent.", "lifestyle": "Dress warmly, maintain adequate nutrition.", "follow_up": "Recheck as advised.", "long_term": None}}},
    "respiratory_rate": {"high": {"recommendation": "Respiratory rate is elevated — evaluate for underlying cause; consult physician if persistent.", "care_plan": {"immediate": "Rest and monitor; seek medical evaluation if persistent.", "short_term": "Complete pulmonary and cardiac evaluation.", "lifestyle": "Avoid smoking; maintain fitness.", "follow_up": "Follow physician guidance.", "long_term": None}}, "low": {"recommendation": "Respiratory rate is low — if symptomatic, consult your physician promptly.", "care_plan": {"immediate": "Seek medical attention if symptomatic.", "short_term": "Physician evaluation.", "lifestyle": "Review medications with physician.", "follow_up": "Follow physician guidance.", "long_term": None}}},

    # ═══════════════════ X-RAY / IMAGING ═══════════════════
    "pneumonia": {"high": {"recommendation": "Pneumonia detected on chest X-ray — urgent evaluation and treatment required.", "care_plan": {"immediate": "Seek medical evaluation immediately.", "short_term": "CBC, CRP, and sputum culture may be ordered.", "lifestyle": "Rest, hydration, and avoid contact with immunocompromised individuals.", "follow_up": "Follow-up chest X-ray in 4–6 weeks.", "long_term": "Annual flu and pneumococcal vaccination recommended."}}, "critical": {"recommendation": "Pneumonia detected on chest X-ray — urgent evaluation and treatment required.", "care_plan": {"immediate": "Seek medical evaluation immediately.", "short_term": "CBC, CRP, and sputum culture may be ordered.", "lifestyle": "Rest, hydration, and avoid contact with immunocompromised individuals.", "follow_up": "Follow-up chest X-ray in 4–6 weeks.", "long_term": "Annual flu and pneumococcal vaccination recommended."}}},
    "consolidation": {"high": {"recommendation": "Pulmonary consolidation detected — may indicate pneumonia or other lung pathology requiring urgent clinical review.", "care_plan": {"immediate": "Seek medical evaluation promptly.", "short_term": "Blood work and possibly sputum cultures.", "lifestyle": "Rest, hydration, and respiratory hygiene.", "follow_up": "Follow-up imaging in 4–6 weeks.", "long_term": None}}, "critical": {"recommendation": "Pulmonary consolidation detected — urgent clinical review required.", "care_plan": {"immediate": "Seek medical evaluation promptly.", "short_term": "Blood work and possibly sputum cultures.", "lifestyle": "Rest, hydration, and respiratory hygiene.", "follow_up": "Follow-up imaging in 4–6 weeks.", "long_term": None}}},
    "cardiomegaly": {"high": {"recommendation": "Cardiomegaly detected — cardiology evaluation recommended.", "care_plan": {"immediate": "Book cardiology consultation within 1–2 weeks.", "short_term": "Echocardiography and ECG recommended.", "lifestyle": "Low-sodium diet, regular moderate exercise, weight management.", "follow_up": "Follow cardiology guidance.", "long_term": "Long-term cardiovascular monitoring."}}, "critical": {"recommendation": "Cardiomegaly detected — cardiology evaluation recommended.", "care_plan": {"immediate": "Book cardiology consultation within 1–2 weeks.", "short_term": "Echocardiography and ECG recommended.", "lifestyle": "Low-sodium diet, moderate exercise, weight management.", "follow_up": "Follow cardiology guidance.", "long_term": "Long-term cardiovascular monitoring."}}},
    "pleural_effusion": {"high": {"recommendation": "Pleural effusion detected — clinical evaluation needed.", "care_plan": {"immediate": "Seek medical evaluation; if breathless, seek urgent care.", "short_term": "Diagnostic thoracentesis may be recommended.", "lifestyle": "Rest and monitor breathing.", "follow_up": "Follow-up imaging as directed.", "long_term": None}}, "critical": {"recommendation": "Pleural effusion detected — clinical evaluation needed.", "care_plan": {"immediate": "Seek medical evaluation; if breathless, seek urgent care.", "short_term": "Diagnostic thoracentesis may be recommended.", "lifestyle": "Rest and monitor breathing.", "follow_up": "Follow-up imaging as directed.", "long_term": None}}},
    "pneumothorax": {"high": {"recommendation": "Pneumothorax detected — medical emergency requiring immediate treatment.", "care_plan": {"immediate": "Seek emergency medical care immediately.", "short_term": "Hospital admission for chest tube or observation.", "lifestyle": "Complete rest; avoid air travel and diving.", "follow_up": "Follow-up chest X-ray as directed.", "long_term": "Pulmonary follow-up for recurrence."}}, "critical": {"recommendation": "Pneumothorax detected — medical emergency.", "care_plan": {"immediate": "Seek emergency medical care immediately.", "short_term": "Hospital admission for chest tube or observation.", "lifestyle": "Complete rest; avoid air travel and diving.", "follow_up": "Follow-up chest X-ray as directed.", "long_term": "Pulmonary follow-up for recurrence."}}},
    "pulmonary_edema": {"high": {"recommendation": "Pulmonary edema detected — urgent medical evaluation required.", "care_plan": {"immediate": "Seek emergency medical evaluation.", "short_term": "Cardiac workup including ECG, echo, and BNP.", "lifestyle": "Strict fluid and sodium restriction.", "follow_up": "Frequent physician monitoring.", "long_term": "Long-term cardiac and respiratory management."}}, "critical": {"recommendation": "Pulmonary edema detected — urgent evaluation required.", "care_plan": {"immediate": "Seek emergency medical evaluation.", "short_term": "Cardiac workup including ECG, echo, and BNP.", "lifestyle": "Strict fluid and sodium restriction.", "follow_up": "Frequent physician monitoring.", "long_term": "Long-term cardiac and respiratory management."}}},
    "atelectasis": {"high": {"recommendation": "Atelectasis detected — clinical correlation recommended.", "care_plan": {"immediate": "Consult physician.", "short_term": "Incentive spirometry and deep breathing exercises.", "lifestyle": "Avoid prolonged bed rest.", "follow_up": "Follow-up imaging if symptoms persist.", "long_term": None}}, "critical": {"recommendation": "Atelectasis detected — clinical correlation recommended.", "care_plan": {"immediate": "Consult physician.", "short_term": "Incentive spirometry and deep breathing exercises.", "lifestyle": "Avoid prolonged bed rest.", "follow_up": "Follow-up imaging if symptoms persist.", "long_term": None}}},
    "infiltrates": {"high": {"recommendation": "Pulmonary infiltrates detected — may indicate infection or inflammation.", "care_plan": {"immediate": "Seek medical evaluation if febrile.", "short_term": "Blood work and possibly sputum cultures.", "lifestyle": "Rest, hydration, respiratory hygiene.", "follow_up": "Follow-up imaging in 4–6 weeks.", "long_term": None}}, "critical": {"recommendation": "Pulmonary infiltrates detected — clinical evaluation recommended.", "care_plan": {"immediate": "Seek medical evaluation if febrile.", "short_term": "Blood work and sputum cultures.", "lifestyle": "Rest, hydration, respiratory hygiene.", "follow_up": "Follow-up imaging in 4–6 weeks.", "long_term": None}}},
    "nodule_mass": {"high": {"recommendation": "Pulmonary nodule/mass detected — further imaging and specialist referral recommended.", "care_plan": {"immediate": "Book pulmonology/oncology consultation within 1–2 weeks.", "short_term": "CT chest scan for characterisation.", "lifestyle": "Avoid smoking.", "follow_up": "Follow specialist guidance.", "long_term": "Long-term surveillance imaging."}}, "critical": {"recommendation": "Pulmonary nodule/mass detected — further imaging and referral recommended.", "care_plan": {"immediate": "Book pulmonology/oncology consultation within 1–2 weeks.", "short_term": "CT chest scan.", "lifestyle": "Avoid smoking.", "follow_up": "Follow specialist guidance.", "long_term": "Long-term surveillance imaging."}}},
    "fracture": {"high": {"recommendation": "Fracture detected — orthopedic evaluation recommended.", "care_plan": {"immediate": "Seek medical evaluation; immobilise affected area.", "short_term": "Orthopedic consultation.", "lifestyle": "Rest; adequate calcium and Vitamin D.", "follow_up": "Follow-up X-ray to confirm healing.", "long_term": "Bone density screening if recurrent."}}, "critical": {"recommendation": "Fracture detected — orthopedic evaluation recommended.", "care_plan": {"immediate": "Seek medical evaluation; immobilise affected area.", "short_term": "Orthopedic consultation.", "lifestyle": "Rest; adequate calcium and Vitamin D.", "follow_up": "Follow-up X-ray to confirm healing.", "long_term": "Bone density screening if recurrent."}}},
}


# ═══════════════════════════════════════════════════════════════════
# UNIVERSAL FALLBACK — enhanced with PDF reference ranges
# ═══════════════════════════════════════════════════════════════════

_STATUS_PHRASE = {
    "critical_high": "critically high",
    "critical_low":  "critically low",
    "critical":      "at a critical level",
    "high":          "higher than the normal range",
    "low":           "lower than the normal range",
    "borderline":    "at a borderline value",
    "abnormal":      "outside the normal range",
}


def universal_fallback(item: dict) -> dict:
    """
    Generate safe generic-but-personalized advice for any biomarker not
    covered by BIOMARKER_KNOWLEDGE. Uses the item's actual value, unit,
    and reference range from the PDF so the advice references real
    numbers instead of generic prose.
    """
    name = item.get("name") or item.get("key") or "This measurement"
    status = str(item.get("status") or "abnormal").lower()
    display_value = item.get("display_value") or ""
    value = item.get("value")
    unit = item.get("unit") or ""
    normal_low = item.get("normal_low")
    normal_high = item.get("normal_high")

    # Build the value-and-range clause dynamically
    value_clause = ""
    if display_value:
        value_clause = f" is {display_value}"
    elif value is not None:
        num_str = f"{value:.2f}".rstrip("0").rstrip(".")
        value_clause = f" is {num_str} {unit}".rstrip()

    range_clause = ""
    unit_suffix = f" {unit}" if unit else ""
    if normal_low is not None and normal_high is not None:
        direction = "above" if status in ("high", "critical_high") else "below" if status in ("low", "critical_low") else "outside"
        range_clause = f", {direction} the reference range of {normal_low}–{normal_high}{unit_suffix} shown on your report"
    elif normal_high is not None:
        range_clause = (
            f", above the reference limit of {normal_high}{unit_suffix} shown on your report"
            if status in ("high", "critical_high")
            else f", within the reference limit of {normal_high}{unit_suffix} shown on your report"
        )
    elif normal_low is not None:
        range_clause = (
            f", below the reference minimum of {normal_low}{unit_suffix} shown on your report"
            if status in ("low", "critical_low")
            else f", above the reference minimum of {normal_low}{unit_suffix} shown on your report"
        )

    severity_descriptor = {
        "critical_high": "This represents a critically high value that needs prompt medical attention.",
        "critical_low":  "This represents a critically low value that needs prompt medical attention.",
        "critical":      "This represents a critical value that needs prompt medical attention.",
        "high":          "This represents an elevated value that should be reviewed clinically.",
        "low":           "This represents a reduced value that should be reviewed clinically.",
        "borderline":    "This is at a borderline level worth monitoring.",
        "abnormal":      "This value is outside the expected reference range.",
    }.get(status, "This value should be reviewed clinically.")

    recommendation = (
        f"{name}{value_clause}{range_clause}. {severity_descriptor} "
        f"Please discuss this specific finding with your physician for personalized "
        f"guidance based on your full medical history and other test results."
    )

    immediate_line = f"Share the {name} result ({display_value}) with your healthcare provider at your next visit." if display_value else f"Share the {name} result with your healthcare provider at your next visit."
    short_term_line = f"Ask your physician whether a repeat {name} test or additional workup is recommended given this value."
    follow_up_line = f"Follow your physician's guidance on retesting intervals for {name}."
    if normal_low is not None and normal_high is not None:
        follow_up_line = f"Follow your physician's guidance on retesting {name}; the reference range on this report is {normal_low}–{normal_high}{unit_suffix}."
    lifestyle_line = "Maintain a balanced diet, regular physical activity, adequate hydration, and sufficient sleep to support overall health while awaiting clinical review."

    return {
        "recommendation": recommendation,
        "care_plan": {
            "immediate": immediate_line,
            "short_term": short_term_line,
            "lifestyle": lifestyle_line,
            "follow_up": follow_up_line,
            "long_term": None,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# RESOLVER — advice waterfall
# ═══════════════════════════════════════════════════════════════════

def resolve_advice(item: dict) -> dict:
    """
    Resolve advice for a single flagged measurement item.

    Waterfall:
      1. BIOMARKER_KNOWLEDGE for (key, status)
      2. Fallback status variants (critical_high→high, critical_low→low)
      3. Universal fallback — always returns valid advice, NEVER None
    """
    raw_key = item.get("key") or item.get("name") or ""
    status = str(item.get("status") or "").lower()

    canonical = _kb_key(raw_key)
    kb_entry = BIOMARKER_KNOWLEDGE.get(canonical, {})

    advice = kb_entry.get(status)
    if advice is None and status == "critical_high":
        advice = kb_entry.get("high")
    if advice is None and status == "critical_low":
        advice = kb_entry.get("low")
    if advice is None and status in ("abnormal", "critical"):
        advice = kb_entry.get("high") or kb_entry.get("low")

    if advice is not None:
        return {
            "recommendation": advice.get("recommendation", ""),
            "care_plan": dict(advice.get("care_plan") or {}),
            "source": "knowledge",
        }

    fallback = universal_fallback(item)
    return {**fallback, "source": "fallback"}


__all__ = [
    "BIOMARKER_KNOWLEDGE",
    "resolve_advice",
    "universal_fallback",
    "resolve_canonical_key",
    "get_display_name",
]