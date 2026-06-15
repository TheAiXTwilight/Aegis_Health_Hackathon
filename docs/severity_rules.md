# Aegis Health — Severity Rules

Single authoritative reference for all severity rule constants.
Tests assert on these constants — never on reason strings.
Rule constants are defined in tools/severity_scorer.py.
Thresholds are defined in tools/lab_thresholds.py.


## Rule Evaluation

Rules are evaluated in descending priority order.
All fired rules are collected. The highest-priority fired rule
determines the overall severity level.

    highest_priority_rule always equals triggered_rules[0]
    len(reasons) always equals len(triggered_rules)
    triggered_rules always contains at least one entry


## HIGH Rules (priority 130–190)

Any HIGH rule firing sets severity to HIGH regardless of MEDIUM rules.

Constant                        Priority    Trigger                                             Confidence
RULE_CHEST_PAIN_AND_SOB         190         Chest pain + shortness of breath in symptom data    0.97
RULE_CRITICAL_LAB_TROPONIN      180         Troponin > 0.04 ng/mL                              0.99
RULE_CRITICAL_LAB_HAEMOGLOBIN   170         Haemoglobin < 7.0 g/dL                             0.98
RULE_CRITICAL_LAB_POTASSIUM     160         Potassium > 6.5 mmol/L                             0.98
RULE_XRAY_PNEUMOTHORAX          150         "Pneumothorax" in X-ray findings                   0.97
RULE_XRAY_PULMONARY_EDEMA       140         "Pulmonary Edema" in X-ray findings                0.95
RULE_SEVERE_DRUG_INTERACTION    130         Any interaction with severity=SEVERE                0.95


## MEDIUM Rules (priority 50–90)

MEDIUM rules only fire when no HIGH rule has fired.
RULE_PROLONGED_SYMPTOMS and RULE_MODERATE_DRUG_INTERACTION
check ctx.any_high_fired at evaluation time and return False
immediately if any HIGH rule has already fired.

Constant                        Priority    Trigger                                                         Confidence
RULE_ABNORMAL_LAB_ANY           90          Any abnormal value in lab_result.abnormal_values                0.86
RULE_XRAY_CARDIOMEGALY          80          "Cardiomegaly" in X-ray findings                               0.84
RULE_XRAY_PLEURAL_EFFUSION      75          "Pleural Effusion" in X-ray findings                           0.82
RULE_XRAY_CONSOLIDATION         70          "Consolidation" in X-ray findings                              0.80
RULE_PROLONGED_SYMPTOMS         60          Duration contains "week" or "month", no HIGH fired             0.75
RULE_MODERATE_DRUG_INTERACTION  50          Any interaction with severity=MODERATE, no HIGH fired           0.78


## LOW Fallback

Constant            Trigger                         Confidence
RULE_DEFAULT_LOW    No other rules fired            0.80

RULE_DEFAULT_LOW is not in _RULES.
It is inserted by the evaluator when fired is empty after
evaluating all 13 real rules.


## Thresholds

All numeric thresholds live in tools/lab_thresholds.py.

Constant                            Value   Unit        Used by
CRITICAL_HAEMOGLOBIN_G_DL           7.0     g/dL        RULE_CRITICAL_LAB_HAEMOGLOBIN
CRITICAL_POTASSIUM_MMOL_L           6.5     mmol/L      RULE_CRITICAL_LAB_POTASSIUM
CRITICAL_TROPONIN_NG_ML             0.04    ng/mL       RULE_CRITICAL_LAB_TROPONIN
ABNORMAL_LOW_HAEMOGLOBIN_G_DL       12.0    g/dL        RULE_ABNORMAL_LAB_ANY (via parser)
ABNORMAL_HIGH_POTASSIUM_MMOL_L      5.5     mmol/L      RULE_ABNORMAL_LAB_ANY (via parser)
ABNORMAL_HIGH_TROPONIN_NG_ML        0.04    ng/mL       RULE_ABNORMAL_LAB_ANY (via parser)
ABNORMAL_HIGH_GLUCOSE_MG_DL         180.0   mg/dL       RULE_ABNORMAL_LAB_ANY (via parser)


## Invariants (enforced by schema and scorer)

- highest_priority_rule == triggered_rules[0] always
- len(reasons) == len(triggered_rules) always
- triggered_rules contains at least one entry always
- RULE_DEFAULT_LOW appears only when no other rule fires
- RULE_PROLONGED_SYMPTOMS never appears alongside any HIGH rule
- RULE_MODERATE_DRUG_INTERACTION never appears alongside any HIGH rule


## ALL_RULE_CONSTANTS

Auto-derived in tools/severity_scorer.py:

    ALL_RULE_CONSTANTS = [r.constant for r in _RULES] + [RULE_DEFAULT_LOW]

Current value (14 constants, priority descending):

    RULE_CHEST_PAIN_AND_SOB
    RULE_CRITICAL_LAB_TROPONIN
    RULE_CRITICAL_LAB_HAEMOGLOBIN
    RULE_CRITICAL_LAB_POTASSIUM
    RULE_XRAY_PNEUMOTHORAX
    RULE_XRAY_PULMONARY_EDEMA
    RULE_SEVERE_DRUG_INTERACTION
    RULE_ABNORMAL_LAB_ANY
    RULE_XRAY_CARDIOMEGALY
    RULE_XRAY_PLEURAL_EFFUSION
    RULE_XRAY_CONSOLIDATION
    RULE_PROLONGED_SYMPTOMS
    RULE_MODERATE_DRUG_INTERACTION
    RULE_DEFAULT_LOW


## Contributing Tools per Rule

Constant                        Contributing Tool
RULE_CHEST_PAIN_AND_SOB         SymptomExtractor
RULE_CRITICAL_LAB_TROPONIN      LabReportParser
RULE_CRITICAL_LAB_HAEMOGLOBIN   LabReportParser
RULE_CRITICAL_LAB_POTASSIUM     LabReportParser
RULE_XRAY_PNEUMOTHORAX          XRayProcessor
RULE_XRAY_PULMONARY_EDEMA       XRayProcessor
RULE_SEVERE_DRUG_INTERACTION    DrugInteractionChecker
RULE_ABNORMAL_LAB_ANY           LabReportParser
RULE_XRAY_CARDIOMEGALY          XRayProcessor
RULE_XRAY_PLEURAL_EFFUSION      XRayProcessor
RULE_XRAY_CONSOLIDATION         XRayProcessor
RULE_PROLONGED_SYMPTOMS         SymptomExtractor
RULE_MODERATE_DRUG_INTERACTION  DrugInteractionChecker
RULE_DEFAULT_LOW                (none)