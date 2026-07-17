import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { getDashboard, getJobResult } from "../../services/api";
import "./Dashboard.css";

/* ═══════════════════════════════════════════════════════════════════
   AEGIS HEALTH — INTELLIGENT CLINICAL DASHBOARD
   Every insight is context-aware — generated from actual symptoms,
   vitals abnormalities, medications, and findings. No static text.
   ═══════════════════════════════════════════════════════════════════ */

/* ───────────────────────────────────────────────────────────────────
   INTELLIGENT CONTENT GENERATORS
   ─────────────────────────────────────────────────────────────────── */

function generateVitalsObservations(measurements) {
  if (!measurements || !measurements.length) {
    return { critical: [], stable: [], insufficient: [] };
  }

  const critical = [];
  const stable = [];

  for (const measurement of measurements) {
    const riskScore = Number.isFinite(measurement.risk_score)
      ? measurement.risk_score
      : null;
    const status = measurement.status || "reported";
    const isLow = status.includes("low");
    const z = riskScore === null
      ? null
      : riskScore === 0
        ? 0
        : (isLow ? -riskScore : riskScore);
    const item = {
      key: measurement.key,
      name: measurement.name || measurement.vital || "Clinical Measurement",
      current: measurement.display_value || `${measurement.current ?? measurement.value ?? "—"} ${measurement.unit || ""}`.trim(),
      numericValue: measurement.value ?? measurement.current,
      unit: measurement.unit || "",
      status,
      category: riskScore >= 2 ? "critical" : riskScore === 1 ? "observation" : riskScore === 0 ? "normal" : "reported",
      z,
      note: measurement.note || "From the latest report",
      samples: 1,
    };

    if (riskScore >= 2) critical.push(item);
    else stable.push(item);
  }

  return { critical, stable, insufficient: [] };
}

/* ───────────────────────────────────────────────────────────────────
   PRIORITY VITAL SELECTOR
   Shared filter: picks up to N vitals distributed by clinical priority.
   Fill order: Critical → Observational → Normal
   Within each category: sorted by risk_score desc, then |deviation| desc.

   Used by:
     • SidebarVitals (double-purpose) — asks for 8 slots with previous filter
     • ComparisonTable (double-purpose) — asks for 6 slots with previous filter
     • VitalsOverviewRow (single-purpose) — asks for 5 slots, no previous filter

   If previous report is provided, vitals missing from previous are SKIPPED
   (next priority fills the slot) — ensures comparison table always has
   valid previous values for every row shown.
   ─────────────────────────────────────────────────────────────────── */

function selectPriorityVitals(currentMeasurements, previousMeasurements = null, maxSlots = 7) {
  if (!Array.isArray(currentMeasurements) || currentMeasurements.length === 0) {
    return [];
  }

  // Build lookup of previous measurements by key (for skip-if-missing logic)
  const previousByKey = previousMeasurements
    ? new Map(previousMeasurements.map((m) => [m.key, m]))
    : null;

  // Rank helper — higher risk_score first, then higher |deviation_score|
  const rank = (arr) =>
    arr.sort((a, b) => {
      const ra = a?.risk_score ?? 0;
      const rb = b?.risk_score ?? 0;
      if (rb !== ra) return rb - ra;
      const da = Math.abs(a?.deviation_score ?? 0);
      const db = Math.abs(b?.deviation_score ?? 0);
      return db - da;
    });

  // Bucket vitals by category based on risk_score
  const critical = [];
  const observational = [];
  const normal = [];

  for (const m of currentMeasurements) {
    // Skip if we need previous but it's missing for this vital
    if (previousByKey && !previousByKey.has(m.key)) continue;

    const rs = Number.isFinite(m.risk_score) ? m.risk_score : null;
    if (rs === null) continue;

    if (rs >= 2) critical.push(m);
    else if (rs === 1) observational.push(m);
    else if (rs === 0) normal.push(m);
  }

  // Priority fill: critical first, then observational, then normal.
  // All three categories are always in play — if critical + observational
  // don't fill all slots, normal fills the rest.
  const ranked = [...rank(critical), ...rank(observational), ...rank(normal)];
  return ranked.slice(0, maxSlots);
}

/* ───────────────────────────────────────────────────────────────────
   SYMPTOM EXTRACTOR (hybrid: backend-first, regex-fallback)

   Extracts clinical symptom keywords from free-text patient input.

   PRIMARY: If the report includes a pre-computed `symptoms_extracted`
   array from the backend (proper clinical NLP), use it directly.

   FALLBACK: Regex-based extraction with:
     • Negation detection ("no fever", "denies chest pain")
     • Vital-with-value extraction ("BP 165/98", "glucose 245")
     • Multi-word phrase priority ("chest tightness" before "chest")
     • Expanded coverage (40+ patterns across all major systems)

   Returns array of normalized display strings (max N).
   ─────────────────────────────────────────────────────────────────── */

const SYMPTOM_PATTERNS = [
  // ── Cardiovascular ──
  { pattern: /chest\s+(pain|tight|pressure|discomfort|heavy|heaviness)/i, tag: "Chest tightness" },
  { pattern: /palpitat|racing\s+heart|fast\s+heart|heart\s+pound/i, tag: "Palpitations" },
  { pattern: /dizz|light[-\s]?head|feel\s+faint|faint(ing|ed|ness)?\b/i, tag: "Dizziness" },
  { pattern: /syncope|passed\s+out|fainted/i, tag: "Fainting" },

  // ── Respiratory ──
  { pattern: /short(ness)?\s+of\s+breath|dyspnea|hard\s+to\s+breathe|breathless|can'?t\s+breathe/i, tag: "Shortness of breath" },
  { pattern: /wheez/i, tag: "Wheezing" },
  { pattern: /cough(ing)?/i, tag: "Cough" },
  { pattern: /(sore|scratchy)\s+throat|throat\s+(pain|ache)/i, tag: "Sore throat" },
  { pattern: /runny\s+nose|congestion|stuffy|stuffed\s+up|nasal\s+(drip|congest)/i, tag: "Congestion" },
  { pattern: /sneez/i, tag: "Sneezing" },

  // ── Neurological ──
  { pattern: /headache|head\s+(pain|ache)|migraine/i, tag: "Headache" },
  { pattern: /confus|disorient|foggy|brain\s+fog/i, tag: "Confusion" },
  { pattern: /numb(ness)?|tingl|pins\s+and\s+needles/i, tag: "Numbness" },
  { pattern: /blurr|blurry|vision\s+(problem|change|loss)|double\s+vision|can'?t\s+see/i, tag: "Vision changes" },
  { pattern: /seizure|convuls/i, tag: "Seizure" },

  // ── Ear / Nose / Throat ──
  { pattern: /ear\s+(pain|ache)|earache/i, tag: "Ear pain" },
  { pattern: /ringing.*ear|tinnitus/i, tag: "Ringing in ears" },
  { pattern: /hearing\s+loss|can'?t\s+hear|deaf/i, tag: "Hearing loss" },

  // ── Constitutional ──
  { pattern: /fever|febrile|running\s+temperature|temp(erature)?\s+(is\s+)?(high|10\d|9[89])/i, tag: "Fever" },
  { pattern: /chills|shiver|shivering/i, tag: "Chills" },
  { pattern: /night\s+sweat/i, tag: "Night sweats" },
  { pattern: /fatigue|tired|exhaust|weak(ness)?|low\s+energy|worn\s+out|lethargy|lethargic/i, tag: "Fatigue" },
  { pattern: /(?<!night\s)sweat|perspir/i, tag: "Sweating" },
  { pattern: /weight\s+(loss|drop|dropping)|losing\s+weight/i, tag: "Weight loss" },
  { pattern: /weight\s+gain|gaining\s+weight/i, tag: "Weight gain" },

  // ── GI ──
  { pattern: /nausea|nauseous|feel\s+sick|queasy/i, tag: "Nausea" },
  { pattern: /vomit|throw(ing)?\s+up|throwing\s+up/i, tag: "Vomiting" },
  { pattern: /diarrhea|loose\s+stool|watery\s+stool/i, tag: "Diarrhea" },
  { pattern: /constipat/i, tag: "Constipation" },
  { pattern: /abdominal\s+pain|stomach\s+(pain|ache|cramp)|belly\s+(pain|ache)/i, tag: "Abdominal pain" },
  { pattern: /appetite\s+loss|no\s+appetite|not\s+eating|lost\s+appetite|reduced\s+appetite/i, tag: "Loss of appetite" },
  { pattern: /heartburn|acid\s+reflux|indigestion/i, tag: "Heartburn" },
  { pattern: /bloat/i, tag: "Bloating" },

  // ── Musculoskeletal ──
  { pattern: /joint\s+(pain|ache|swelling)|arthralg/i, tag: "Joint pain" },
  { pattern: /muscle\s+(pain|ache|cramp)|myalg|body\s+ache/i, tag: "Body aches" },
  { pattern: /back\s+(pain|ache)/i, tag: "Back pain" },
  { pattern: /neck\s+(pain|stiff|ache)/i, tag: "Neck pain" },
  { pattern: /leg\s+(pain|cramp|ache)/i, tag: "Leg pain" },
  { pattern: /(swelling|swollen|edema)\s+(in\s+)?(feet|ankle|leg|hand)/i, tag: "Swelling" },
  { pattern: /(?<!no\s)swelling|swollen(?!\s*lymph)/i, tag: "Swelling" },

  // ── Skin ──
  { pattern: /rash|itch|itchy|hives|urticaria/i, tag: "Rash / Itching" },
  { pattern: /bruis/i, tag: "Bruising" },

  // ── Urinary ──
  { pattern: /burning.*urin|painful.*urin|dysuria/i, tag: "Painful urination" },
  { pattern: /frequent\s+urin|urinating\s+(a\s+lot|often)|urgency/i, tag: "Frequent urination" },
  { pattern: /blood.*urin|hematuria/i, tag: "Blood in urine" },

  // ── Sleep / Mental ──
  { pattern: /insomnia|can'?t\s+sleep|trouble\s+sleep|sleepless/i, tag: "Insomnia" },
  { pattern: /anxi|worried|panic|nervous/i, tag: "Anxiety" },
  { pattern: /depress|feeling\s+sad|hopeless|down/i, tag: "Low mood" },
];

// Vital-with-value extractors — pull specific readings out of prose.
// Emit tags like "BP 165/98" so the actual value is preserved.
const VITAL_VALUE_EXTRACTORS = [
  {
    pattern: /\b(?:bp|blood\s+pressure)\s*(?:is|was|of|:|=|@|around|about)?\s*(\d{2,3})\s*[\/\\]\s*(\d{2,3})/i,
    build: (m) => `BP ${m[1]}/${m[2]}`,
  },
  {
    pattern: /\b(?:glucose|sugar|blood\s+sugar)\s*(?:is|was|of|:|=|around|about)?\s*(\d{2,3})/i,
    build: (m) => `Glucose ${m[1]}`,
  },
  {
    pattern: /\b(?:heart\s+rate|hr|pulse)\s*(?:is|was|of|:|=|around|about)?\s*(\d{2,3})/i,
    build: (m) => `HR ${m[1]}`,
  },
  {
    pattern: /\b(?:spo2|spo₂|oxygen|o2\s+sat|oxygen\s+sat)\s*(?:is|was|of|:|=|around|about)?\s*(\d{2,3})/i,
    build: (m) => `SpO₂ ${m[1]}%`,
  },
  {
    pattern: /\btemp(?:erature)?\s*(?:is|was|of|:|=|around|about)?\s*(\d{2,3}(?:\.\d)?)/i,
    build: (m) => `Temp ${m[1]}°`,
  },
];

// Detects negation windows so we don't tag "no fever" as "Fever".
// Returns true if the match position falls within a negation phrase.
function isNegatedAt(text, matchIndex) {
  // Look back up to 30 chars for a negation cue
  const window = text.slice(Math.max(0, matchIndex - 30), matchIndex).toLowerCase();
  return /\b(no|not|without|denies|deny|denied|negative\s+for|neg\.?\s+for|absence\s+of|ruled\s+out)\s+[a-z\s]{0,20}$/i.test(window);
}

function extractSymptomTagsFromText(text, maxTags = 6) {
  if (!text || typeof text !== "string") return [];
  const found = [];
  const seen = new Set();

  // ── Pass 1: vital values with numbers (highest signal) ──
  for (const { pattern, build } of VITAL_VALUE_EXTRACTORS) {
    const m = text.match(pattern);
    if (m && typeof m.index === "number" && !isNegatedAt(text, m.index)) {
      const tag = build(m);
      if (!seen.has(tag)) {
        seen.add(tag);
        found.push(tag);
        if (found.length >= maxTags) return found;
      }
    }
  }

  // ── Pass 2: symptom keywords ──
  for (const { pattern, tag } of SYMPTOM_PATTERNS) {
    const m = text.match(pattern);
    if (m && typeof m.index === "number" && !isNegatedAt(text, m.index)) {
      if (!seen.has(tag)) {
        seen.add(tag);
        found.push(tag);
        if (found.length >= maxTags) return found;
      }
    }
  }

  return found;
}

// Public entry point — uses backend-provided array if available,
// otherwise falls back to regex extraction.
//
// IMPORTANT: this is wrapped in try/catch so that ANY unexpected input
// shape or regex edge case (unusual unicode, extremely long text, etc.)
// degrades gracefully instead of throwing inside render — a throw here
// would blank the entire Quick Signals card for every symptom type, not
// just the one that triggered it.
function extractSymptomTags(reportOrText, maxTags = 6) {
  try {
    // If caller passed a report object, prefer backend-provided array
    if (reportOrText && typeof reportOrText === "object" && !Array.isArray(reportOrText)) {
      const backendTags = reportOrText.symptoms_extracted;
      if (Array.isArray(backendTags) && backendTags.length > 0) {
        // Backend already normalized these — trust and slice
        const cleaned = backendTags.filter(Boolean).map(String).slice(0, maxTags);
        if (cleaned.length > 0) return cleaned;
      }
      // Fallback to regex on the free-text field
      return extractSymptomTagsFromText(reportOrText.symptoms_text || "", maxTags);
    }
    // Legacy: caller passed raw string directly
    return extractSymptomTagsFromText(String(reportOrText || ""), maxTags);
  } catch (err) {
    // Never let extraction failure hide the underlying report — fall back
    // to a single raw-text pseudo-tag so the user still sees *something*.
    if (typeof console !== "undefined" && console.warn) {
      console.warn("extractSymptomTags failed, falling back to raw text:", err);
    }
    const raw =
      reportOrText && typeof reportOrText === "object" && !Array.isArray(reportOrText)
        ? reportOrText.symptoms_text
        : reportOrText;
    return raw ? [String(raw)] : [];
  }
}

function generateSymptomBasedPrecautions(symptomsText, severity, xrayFindings, medications) {
  const precautions = [];
  const sx = (symptomsText || "").toLowerCase();

  if (/cough/i.test(sx)) {
    precautions.push({
      text: "Monitor cough progression",
      detail:
        "Track if cough becomes productive (with mucus), bloody, or persists beyond 2 weeks. Seek evaluation if accompanied by chest pain or breathing difficulty.",
      icon: "🫁",
    });
    precautions.push({
      text: "Avoid respiratory irritants",
      detail: "Limit exposure to smoke, dust, strong fumes. Use a humidifier if the air is dry. Stay hydrated to help thin mucus.",
      icon: "💨",
    });
  }

  if (/fever|temperature|38/i.test(sx)) {
    precautions.push({
      text: "Monitor body temperature",
      detail: "Check temperature every 4-6 hours. Seek medical attention if fever exceeds 39.5°C (103°F) or persists beyond 3 days.",
      icon: "🌡",
    });
    precautions.push({
      text: "Stay well hydrated",
      detail:
        "Increased fluid loss during fever. Drink water, herbal teas, or electrolyte solutions. Monitor for signs of dehydration (dark urine, dizziness).",
      icon: "💧",
    });
  }

  if (/breath|wheez|shortness|dyspnea/i.test(sx)) {
    precautions.push({
      text: "Monitor breathing difficulty",
      detail:
        "Note if shortness of breath occurs at rest or with minimal activity. Seek emergency care if you cannot complete a sentence without gasping.",
      icon: "🫁",
    });

    if (severity === "HIGH" || severity === "CRITICAL") {
      precautions.push({
        text: "Keep emergency medications accessible",
        detail:
          "If prescribed an inhaler or rescue medication, keep it within reach at all times. Ensure family members know where it is located.",
        icon: "💊",
      });
    }
  }

  if (/chest pain|chest tight|chest pressure|radiating/i.test(sx)) {
    precautions.push({
      text: "DO NOT ignore chest symptoms",
      detail:
        "Chest pain or pressure, especially if radiating to the arm, jaw, or back, requires immediate medical evaluation. Call emergency services.",
      icon: "⚠",
    });
    precautions.push({
      text: "Avoid physical exertion",
      detail: "Rest until evaluated by a clinician. Physical strain could worsen underlying cardiac conditions.",
      icon: "🛌",
    });
  }

  if (/pain|ache|burning/i.test(sx) && !/chest pain/i.test(sx)) {
    precautions.push({
      text: "Track pain patterns",
      detail:
        "Note what makes the pain better or worse, time of day it peaks, and any associated symptoms. This helps clinicians narrow the diagnosis.",
      icon: "📋",
    });
  }

  if (xrayFindings && xrayFindings.length && !/no significant/i.test(xrayFindings[0] || "")) {
    precautions.push({
      text: "Follow up on imaging findings",
      detail: `X-ray noted: ${xrayFindings[0]}. Discuss follow-up imaging or specialist referral with your clinician.`,
      icon: "🩻",
    });
  }

  if (medications && medications.length >= 2) {
    precautions.push({
      text: "Review medication interactions",
      detail: `You are taking ${medications.length} medications. Always inform every healthcare provider of your complete medication list to avoid interactions.`,
      icon: "💊",
    });
  }

  if (precautions.length < 3) {
    precautions.push({
      text: "Keep a symptom diary",
      detail: "Recording when symptoms occur, their severity, and any triggers helps build a clearer clinical picture over time.",
      icon: "📝",
    });
    precautions.push({
      text: "Know when to seek emergency care",
      detail:
        "Warning signs: severe pain, difficulty breathing, confusion, uncontrolled bleeding, loss of consciousness. When in doubt, seek care.",
      icon: "🚨",
    });
  }

  return precautions.slice(0, 5);
}

function generateClinicalSummary(symptomsText, severity, confidence, trend, xrayFindings, medications, recordCount) {
  const sx = (symptomsText || "").toLowerCase();
  const parts = [];

  if (/cough/i.test(sx) && /fever/i.test(sx)) {
    parts.push(`The combination of ${/chest/i.test(sx) ? "chest discomfort, " : ""}cough, and fever suggests a respiratory process that warrants attention.`);

    if (/consolidation|infiltrate/i.test((xrayFindings || []).join(" "))) {
      parts.push(`Chest imaging shows ${xrayFindings[0].toLowerCase()}, which may indicate an infectious or inflammatory pulmonary condition.`);
    } else {
      parts.push(
        `Chest imaging was ${xrayFindings && xrayFindings.length ? xrayFindings[0].toLowerCase() : "not performed"}, providing ${
          xrayFindings && xrayFindings.length && /no significant/i.test(xrayFindings[0])
            ? "reassurance regarding"
            : xrayFindings && xrayFindings.length
              ? "additional context for"
              : "no additional information for"
        } the assessment.`
      );
    }
  } else if (/chest pain/i.test(sx)) {
    parts.push(
      `Chest pain, especially ${/radiating/i.test(sx) ? "with radiation" : "when described as sharp or pressure-like"}, requires careful clinical evaluation to rule out cardiac, pulmonary, or musculoskeletal causes.`
    );
  } else if (/fatigue|body ache|joint/i.test(sx)) {
    parts.push(
      `The pattern of ${/joint/i.test(sx) ? "joint pain" : "fatigue and body aches"} suggests a systemic process. The severity was assessed as ${severity} with ${Math.round(
        confidence * 100
      )}% confidence based on the submitted clinical data.`
    );
  } else {
    parts.push(`Based on the submitted clinical information, the overall severity is assessed as ${severity} with ${Math.round(confidence * 100)}% confidence.`);
  }

  if (recordCount >= 2) {
    if (trend === "improving") {
      parts.push("Encouragingly, your trajectory shows improvement from previous assessments, suggesting that current management or natural recovery is progressing.");
    } else if (trend === "worsening") {
      parts.push(
        `Of note, the trend across ${recordCount} assessments shows an upward trajectory. This pattern warrants discussion with a healthcare provider to determine if intervention is needed.`
      );
    } else {
      parts.push(`Your clinical picture has remained stable across ${recordCount} assessments. Continued monitoring is recommended to detect any changes early.`);
    }
  }

  if (medications && medications.length) {
    const medNames = medications.slice(0, 3).join(", ");
    parts.push(
      `You are currently taking ${medications.length > 1 ? `${medications.length} medications` : "medication"} (${medNames}${
        medications.length > 3 ? ", and others" : ""
      }), for which no contraindications were identified in the current analysis.`
    );
  }

  return parts.join(" ");
}

function generateRecommendedActions(severity, trend, symptomsText, vitalCriticalCount) {
  const actions = [];
  const sx = (symptomsText || "").toLowerCase();

  if (severity === "CRITICAL" || severity === "HIGH" || /chest pain|severe|radiating|unconscious/i.test(sx)) {
    actions.push({
      text: "Seek immediate medical evaluation",
      detail:
        severity === "CRITICAL"
          ? "Go to the nearest emergency department or call emergency services now."
          : "Schedule an urgent appointment with your healthcare provider within 24 hours.",
      urgent: true,
    });
  }

  actions.push({
    text: "Share this report with your clinician",
    detail: "Bring the complete triage report to your appointment. The findings, trends, and citations provide clinical context for your provider.",
  });

  if (trend === "worsening" && severity !== "CRITICAL") {
    actions.push({
      text: "Schedule a follow-up within 7 days",
      detail: "The worsening trend suggests timely re-evaluation is warranted. Earlier if symptoms intensify.",
    });
  }

  if (/cough|fever|breath|wheez/i.test(sx)) {
    actions.push({
      text: "Monitor respiratory symptoms daily",
      detail: "Track cough frequency, sputum color, fever pattern, and any breathing difficulty. Record changes to share with your clinician.",
    });
  }

  if (vitalCriticalCount > 0) {
    actions.push({
      text: `Review ${vitalCriticalCount} vital sign${vitalCriticalCount > 1 ? "s" : ""} under observation`,
      detail: `${vitalCriticalCount} vital reading${vitalCriticalCount > 1 ? "s are" : " is"} significantly outside your personal baseline. Discuss these with your clinician.`,
    });
  }

  return actions.slice(0, 4);
}

/* ───────────────────────────────────────────────────────────────────
   CONSTANTS
   ─────────────────────────────────────────────────────────────────── */

const SM = { LOW: 1, MEDIUM: 2, MODERATE: 2, HIGH: 3, CRITICAL: 4 };
const SC = { LOW: "#10b981", MEDIUM: "#f59e0b", MODERATE: "#f59e0b", HIGH: "#ef4444", CRITICAL: "#991b1b" };
const SBG = {
  LOW: "rgba(16,185,129,0.04)",
  MEDIUM: "rgba(245,158,11,0.04)",
  MODERATE: "rgba(245,158,11,0.04)",
  HIGH: "rgba(239,68,68,0.04)",
  CRITICAL: "rgba(153,27,27,0.04)",
};

/* ═══════════════════════════════════════════════════════════════════
   CLINICAL SVG ICONS — professional monochrome line icons
   Replaces emoji lookup for vital rows. Each icon is a lucide-style
   stroke-based SVG that scales cleanly and inherits color from CSS.
   ═══════════════════════════════════════════════════════════════════ */

const ClinicalIcons = {
  heart: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
    </svg>
  ),
  lung: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6.081 20c1.612 0 2.919-1.335 2.919-2.98v-6.53c0-.72-.284-1.41-.789-1.919L4.211 4.7C3.756 4.244 3 4.596 3 5.25V17c0 1.657 1.343 3 3 3z" />
      <path d="M17.919 20c-1.612 0-2.919-1.335-2.919-2.98v-6.53c0-.72.284-1.41.789-1.919L19.789 4.7c.455-.455 1.211-.104 1.211.55V17c0 1.657-1.343 3-3 3z" />
      <path d="M12 3v18" />
    </svg>
  ),
  thermometer: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0z" />
    </svg>
  ),
  stethoscope: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3" />
      <path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4" />
      <circle cx="20" cy="10" r="2" />
    </svg>
  ),
  droplet: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2v6" />
      <path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5S5 13 5 15a7 7 0 0 0 7 7z" />
    </svg>
  ),
  microscope: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 18h8" />
      <path d="M3 22h18" />
      <path d="M14 22a7 7 0 1 0 0-14h-1" />
      <path d="M9 14h2" />
      <path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2" />
      <path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3" />
    </svg>
  ),
  flask: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 2v7.31" />
      <path d="M14 9.3V1.99" />
      <path d="M8.5 2h7" />
      <path d="M14 9.3a6.5 6.5 0 1 1-4 0" />
      <path d="M5.58 16.5h12.85" />
    </svg>
  ),
  cog: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16Z" />
      <path d="M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z" />
      <path d="M12 2v2" />
      <path d="M12 20v2" />
      <path d="m4.93 4.93 1.41 1.41" />
      <path d="m17.66 17.66 1.41 1.41" />
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="m6.34 17.66-1.41 1.41" />
      <path d="m19.07 4.93-1.41 1.41" />
    </svg>
  ),
  sugar: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="8" width="18" height="8" rx="2" />
      <path d="M3 12h18" />
      <path d="M7 8v8" />
      <path d="M11 8v8" />
      <path d="M15 8v8" />
      <path d="M19 8v8" />
    </svg>
  ),
  kidney: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17.9 3.5C15.7 3.5 14 5.3 14 7.6c0 1.2.5 2.3 1.3 3.1-1.4 1-2.3 2.6-2.3 4.5 0 3 2.4 5.3 5.3 5.3S23.6 18.2 23.6 15.2c0-1.9-.9-3.5-2.3-4.5.8-.8 1.3-1.9 1.3-3.1C22.6 5.3 20.9 3.5 18.7 3.5h-.8z" transform="translate(-4 0)" />
    </svg>
  ),
  liver: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12c0-4 3-8 9-8s9 4 9 8-4 7-9 7-9-3-9-7z" />
      <path d="M12 12v7" />
      <path d="M8 10h1" />
    </svg>
  ),
  bolt: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z" />
    </svg>
  ),
  link: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
    </svg>
  ),
  pill: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z" />
      <path d="m8.5 8.5 7 7" />
    </svg>
  ),
  flame: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z" />
    </svg>
  ),
  dna: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 15c6.667-6 13.333 0 20-6" />
      <path d="M9 22c1.798-1.998 2.518-3.995 2.807-5.993" />
      <path d="M15 2c-1.798 1.998-2.518 3.995-2.807 5.993" />
      <path d="m17 6-2.5-2.5" />
      <path d="m14 8-1-1" />
      <path d="m7 18 2.5 2.5" />
      <path d="m3.5 14.5.5.5" />
      <path d="m20 9 .5.5" />
      <path d="m6.5 12.5 1 1" />
      <path d="m16.5 10.5 1 1" />
      <path d="m10 16 1.5 1.5" />
    </svg>
  ),
  chart: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3v18h18" />
      <path d="m19 9-5 5-4-4-3 3" />
    </svg>
  ),
};

// Vital name → { icon: React element, color: stroke color }
// Colors follow clinical system palette:
//   red    #ef4444 — cardio, hematology, iron, inflammation
//   blue   #2563ff — respiratory, kidney, electrolytes
//   amber  #f59e0b — temperature, glucose/metabolic, vitamins
//   purple #7c3aed — endocrine, liver, hormones, tumor markers
//   green  #10b981 — HDL (good cholesterol)

const VITAL_ICON_MAP = {
  // ── Cardiovascular / Circulatory ──
  "Heart Rate": { icon: ClinicalIcons.heart, color: "#ef4444" },
  "Blood Pressure": { icon: ClinicalIcons.stethoscope, color: "#10b981" },
  "Systolic BP": { icon: ClinicalIcons.stethoscope, color: "#10b981" },
  "Diastolic BP": { icon: ClinicalIcons.stethoscope, color: "#10b981" },
  Troponin: { icon: ClinicalIcons.heart, color: "#ef4444" },

  // ── Respiratory ──
  SpO2: { icon: ClinicalIcons.lung, color: "#2563ff" },
  "SpO₂": { icon: ClinicalIcons.lung, color: "#2563ff" },
  "Oxygen Saturation": { icon: ClinicalIcons.lung, color: "#2563ff" },
  "Respiratory Rate": { icon: ClinicalIcons.lung, color: "#f59e0b" },

  // ── Temperature ──
  Temperature: { icon: ClinicalIcons.thermometer, color: "#f59e0b" },
  "Body Temperature": { icon: ClinicalIcons.thermometer, color: "#f59e0b" },

  // ── Hematology (red cells) ──
  Hemoglobin: { icon: ClinicalIcons.droplet, color: "#ef4444" },
  Haemoglobin: { icon: ClinicalIcons.droplet, color: "#ef4444" },
  RBC: { icon: ClinicalIcons.droplet, color: "#ef4444" },
  "Red Blood Cells": { icon: ClinicalIcons.droplet, color: "#ef4444" },
  Platelets: { icon: ClinicalIcons.droplet, color: "#ef4444" },
  Hematocrit: { icon: ClinicalIcons.droplet, color: "#ef4444" },

  // ── Hematology (indices / microscopy) ──
  MCV: { icon: ClinicalIcons.microscope, color: "#2563ff" },
  MCH: { icon: ClinicalIcons.microscope, color: "#2563ff" },
  MCHC: { icon: ClinicalIcons.microscope, color: "#2563ff" },
  RDW: { icon: ClinicalIcons.microscope, color: "#2563ff" },
  MPV: { icon: ClinicalIcons.microscope, color: "#2563ff" },

  // ── Hematology (white cells) ──
  WBC: { icon: ClinicalIcons.flask, color: "#2563ff" },
  "White Blood Cells": { icon: ClinicalIcons.flask, color: "#2563ff" },
  Neutrophils: { icon: ClinicalIcons.flask, color: "#2563ff" },
  Lymphocytes: { icon: ClinicalIcons.flask, color: "#2563ff" },
  Monocytes: { icon: ClinicalIcons.flask, color: "#2563ff" },
  Eosinophils: { icon: ClinicalIcons.flask, color: "#2563ff" },
  Basophils: { icon: ClinicalIcons.flask, color: "#2563ff" },
  "Reactive Lymphocytes": { icon: ClinicalIcons.flask, color: "#2563ff" },

  // ── Endocrine / Hormonal ──
  TSH: { icon: ClinicalIcons.cog, color: "#7c3aed" },
  T3: { icon: ClinicalIcons.cog, color: "#7c3aed" },
  T4: { icon: ClinicalIcons.cog, color: "#7c3aed" },
  "Free T3": { icon: ClinicalIcons.cog, color: "#7c3aed" },
  "Free T4": { icon: ClinicalIcons.cog, color: "#7c3aed" },
  Cortisol: { icon: ClinicalIcons.cog, color: "#7c3aed" },
  Insulin: { icon: ClinicalIcons.cog, color: "#7c3aed" },

  // ── Metabolic (glucose) ──
  Glucose: { icon: ClinicalIcons.sugar, color: "#f59e0b" },
  HbA1c: { icon: ClinicalIcons.sugar, color: "#f59e0b" },

  // ── Lipid Panel ──
  Cholesterol: { icon: ClinicalIcons.droplet, color: "#f59e0b" },
  LDL: { icon: ClinicalIcons.droplet, color: "#ef4444" },
  HDL: { icon: ClinicalIcons.droplet, color: "#10b981" },
  Triglycerides: { icon: ClinicalIcons.droplet, color: "#f59e0b" },

  // ── Renal (kidney) ──
  Creatinine: { icon: ClinicalIcons.kidney, color: "#2563ff" },
  BUN: { icon: ClinicalIcons.kidney, color: "#2563ff" },
  Urea: { icon: ClinicalIcons.kidney, color: "#2563ff" },
  "Uric Acid": { icon: ClinicalIcons.kidney, color: "#2563ff" },

  // ── Hepatic (liver) ──
  ALT: { icon: ClinicalIcons.liver, color: "#7c3aed" },
  AST: { icon: ClinicalIcons.liver, color: "#7c3aed" },
  ALP: { icon: ClinicalIcons.liver, color: "#7c3aed" },
  GGT: { icon: ClinicalIcons.liver, color: "#7c3aed" },
  Bilirubin: { icon: ClinicalIcons.liver, color: "#f59e0b" },
  Albumin: { icon: ClinicalIcons.liver, color: "#7c3aed" },
  Globulin: { icon: ClinicalIcons.liver, color: "#7c3aed" },
  "Total Protein": { icon: ClinicalIcons.liver, color: "#7c3aed" },
  "A/G Ratio": { icon: ClinicalIcons.liver, color: "#7c3aed" },

  // ── Electrolytes ──
  Sodium: { icon: ClinicalIcons.bolt, color: "#2563ff" },
  Potassium: { icon: ClinicalIcons.bolt, color: "#2563ff" },
  Calcium: { icon: ClinicalIcons.bolt, color: "#2563ff" },
  Magnesium: { icon: ClinicalIcons.bolt, color: "#2563ff" },
  Chloride: { icon: ClinicalIcons.bolt, color: "#2563ff" },
  Bicarbonate: { icon: ClinicalIcons.bolt, color: "#2563ff" },

  // ── Iron studies ──
  Iron: { icon: ClinicalIcons.link, color: "#ef4444" },
  Ferritin: { icon: ClinicalIcons.link, color: "#ef4444" },
  TIBC: { icon: ClinicalIcons.link, color: "#ef4444" },

  // ── Vitamins ──
  "Vitamin D": { icon: ClinicalIcons.pill, color: "#f59e0b" },
  "Vitamin B12": { icon: ClinicalIcons.pill, color: "#f59e0b" },

  // ── Inflammation markers ──
  CRP: { icon: ClinicalIcons.flame, color: "#ef4444" },
  ESR: { icon: ClinicalIcons.flame, color: "#ef4444" },

  // ── Tumor markers ──
  PSA: { icon: ClinicalIcons.dna, color: "#7c3aed" },
};

// Smart fallback — guess icon + color by keyword when name isn't in map.
function inferVitalIconMeta(name) {
  const n = String(name || "").toLowerCase();
  if (/heart|pulse|troponin|cardiac/.test(n)) return { icon: ClinicalIcons.heart, color: "#ef4444" };
  if (/breath|respir|oxygen|spo|lung/.test(n)) return { icon: ClinicalIcons.lung, color: "#2563ff" };
  if (/temp|fever/.test(n)) return { icon: ClinicalIcons.thermometer, color: "#f59e0b" };
  if (/pressure|systol|diastol/.test(n)) return { icon: ClinicalIcons.stethoscope, color: "#10b981" };
  if (/hemoglobin|haemoglobin|rbc|red blood|platelet|hematocrit/.test(n)) return { icon: ClinicalIcons.droplet, color: "#ef4444" };
  if (/iron|ferritin|tibc/.test(n)) return { icon: ClinicalIcons.link, color: "#ef4444" };
  if (/wbc|white blood|neutro|lympho|mono|eosino|baso/.test(n)) return { icon: ClinicalIcons.flask, color: "#2563ff" };
  if (/mcv|mch|mchc|rdw|mpv/.test(n)) return { icon: ClinicalIcons.microscope, color: "#2563ff" };
  if (/tsh|t3|t4|thyroid|cortisol|insulin|hormone/.test(n)) return { icon: ClinicalIcons.cog, color: "#7c3aed" };
  if (/glucose|hba1c|sugar/.test(n)) return { icon: ClinicalIcons.sugar, color: "#f59e0b" };
  if (/hdl/.test(n)) return { icon: ClinicalIcons.droplet, color: "#10b981" };
  if (/ldl/.test(n)) return { icon: ClinicalIcons.droplet, color: "#ef4444" };
  if (/cholesterol|triglycer|lipid/.test(n)) return { icon: ClinicalIcons.droplet, color: "#f59e0b" };
  if (/creatinine|bun|urea|uric|renal|kidney/.test(n)) return { icon: ClinicalIcons.kidney, color: "#2563ff" };
  if (/alt|ast|alp|ggt|bilirubin|albumin|globulin|protein|hepatic|liver|a\/g/.test(n)) return { icon: ClinicalIcons.liver, color: "#7c3aed" };
  if (/sodium|potassium|calcium|magnesium|chloride|bicarbonate|electrolyte/.test(n)) return { icon: ClinicalIcons.bolt, color: "#2563ff" };
  if (/vitamin|b12/.test(n)) return { icon: ClinicalIcons.pill, color: "#f59e0b" };
  if (/crp|esr|inflammation/.test(n)) return { icon: ClinicalIcons.flame, color: "#ef4444" };
  if (/psa|marker/.test(n)) return { icon: ClinicalIcons.dna, color: "#7c3aed" };
  return { icon: ClinicalIcons.chart, color: "#64748b" };
}

// Public helper — always returns { icon, color } for any vital name
function getVitalIconMeta(name) {
  return VITAL_ICON_MAP[name] || inferVitalIconMeta(name);
}

/* ═══════════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════════ */

export default function DashboardPage() {
  const nav = useNavigate();
  const [sp] = useSearchParams();
  const jid = sp.get("jobId");

  const [ld, sl] = useState(true);
  const [dash, sd] = useState(null);
  const [co, setCo] = useState(false);
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        let dd = await getDashboard();

        // Key Insights can arrive immediately after completion. Keep a
        // non-static one-report fallback while persistence finishes.
        if ((!dd || !dd.recent_records?.length) && jid) {
          const result = await getJobResult(jid);
          if (result) {
            dd = {
              user: {
                display_name: result.patient?.name || "User",
                role: "user",
              },
              recent_records: [
                {
                  id: jid,
                  job_id: jid,
                  severity: result.report?.severity || "MEDIUM",
                  confidence: result.report?.confidence || 0,
                  symptoms_text: result.submitted?.symptoms_text || "",
                  medications_json: JSON.stringify(result.submitted?.medications || []),
                  xray_findings_json: JSON.stringify(result.submitted?.xray_findings || []),
                  measurements: [],
                  contributing_factors: result.severity_result?.reasons || [],
                  created_at: new Date().toISOString(),
                },
              ],
              risk: { status: result.report?.severity || "Medium", factors: [] },
              safety_review: [],
              actions: [],
            };
          }
        }

        if (!cancelled) sd(dd);
      } catch {
        if (!cancelled) {
          sd({ user: {}, recent_records: [], risk: {}, safety_review: [], actions: [] });
        }
      } finally {
        if (!cancelled) sl(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [jid]);

  // ── Auto-bold and Title-Case key medical/biomarker terms in Clinical Picture Summary rows ──
  useEffect(() => {
    if (ld || !dash) return;

    const medicalTerms = [
      'TSH', 'HbA1c', 'WBC', 'RBC', 'Hemoglobin', 'Haemoglobin', 'Platelets',
      'Glucose', 'Vitamin D', 'Vitamin B12', 'Cholesterol', 'LDL', 'HDL',
      'Triglycerides', 'Creatinine', 'BUN', 'Sodium', 'Potassium', 'Calcium',
      'Magnesium', 'Albumin', 'Globulin', 'Bilirubin', 'ALT', 'AST', 'ALP',
      'GGT', 'Reactive Lymphocytes', 'Peripheral Smear', 'White Cell Differential',
      'Protein Markers', 'Nutritional Depletion', 'Hypothyroidism',
      'Hyperthyroidism', 'Liver Function', 'Kidney Function', 'A/G Ratio',
      'Total Protein', 'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils',
      'Basophils', 'MCV', 'MCH', 'MCHC', 'RDW', 'MPV', 'ESR', 'CRP', 'Ferritin',
      'Iron', 'TIBC', 'PSA', 'Troponin', 'SpO2', 'SpO₂', 'Oxygen Saturation',
      'Heart Rate', 'Blood Pressure', 'Respiratory Rate', 'Body Temperature',
      'Uric Acid', 'Urea', 'Chloride', 'Bicarbonate', 'Anion Gap',
      'Free T3', 'Free T4', 'T3', 'T4', 'Cortisol', 'Insulin',
      'Viral Infection', 'Bacterial Infection', 'Anemia', 'Diabetes',
      'Hypertension', 'Hypotension', 'Inflammation', 'Dehydration',
      'Subclinical', 'Overt'
    ];

    // Sort longest-first so multi-word terms match before single-word subsets
    const sortedTerms = [...medicalTerms].sort((a, b) => b.length - a.length);

    const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

    // Title-case helper — keeps acronyms and mixed-case medical tokens intact
    const toTitleCase = (str) => {
      return str
        .split(/(\s+|\/|-)/)
        .map((word) => {
          if (/^[A-Z0-9]+$/.test(word)) return word;
          if (/[A-Z]/.test(word) && /[a-z]/.test(word)) return word;
          if (!/[a-zA-Z]/.test(word)) return word;
          return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
        })
        .join('');
    };

    const timeoutId = setTimeout(() => {
      document.querySelectorAll('.clinical-picture-row').forEach((row) => {
        const originalText = row.textContent;
        let html = originalText;

        sortedTerms.forEach((term) => {
          const regex = new RegExp(`\\b(${escapeRegex(term)})\\b`, 'gi');
          html = html.replace(regex, (match) => {
            const titled = toTitleCase(match);
            return `<span class="clinical-term">${titled}</span>`;
          });
        });

        row.innerHTML = html;
      });
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [ld, dash]);

  if (ld) {
    return (
      <div className="dashv2-page">
        <div className="dashv2-loading">
          <div className="dashv2-spinner" />
          <span className="dashv2-loading-text">Preparing your health intelligence...</span>
        </div>
      </div>
    );
  }

  const u = dash?.user || {};
  const recs = dash?.recent_records || [];

  // ── Determine which report is the "current view" (L) ──
  // recs is ordered newest → oldest. Default to latest unless jobId
  // in URL selects a specific report.
  const latest = recs[0];
  const selectedIdx = jid ? recs.findIndex((r) => r.job_id === jid || r.id === jid) : 0;
  const L = selectedIdx >= 0 ? recs[selectedIdx] : latest;
  const isLatest = !L || L === latest || selectedIdx === 0;

  // ── Determine the comparison target (P) based on selection ──
  //   • Only 1 report total        → P = null (single-purpose)
  //   • L is the latest report     → P = recs[1] (immediately previous)
  //   • L is an older report       → P = latest = recs[0]
  let P = null;
  if (recs.length >= 2) {
    if (isLatest) {
      P = recs[1];
    } else {
      P = latest;
    }
  }

  const isMulti = !!P; // double-purpose only when we have a valid comparison

  const sevN = L ? SM[L.severity] || 2 : 0;

  // Trend calculation — direction depends on comparison direction:
  //   • Latest-vs-Previous : trend = current(L) vs previous(P). Higher L = worsening.
  //   • Older-vs-Latest    : reversed. If chosen L is lower severity
  //                          than latest P, patient has since worsened.
  let trend = "stable";
  if (isMulti) {
    const d = (SM[L.severity] || 2) - (SM[P.severity] || 2);
    if (isLatest) {
      if (d < 0) trend = "improving";
      else if (d > 0) trend = "worsening";
    } else {
      if (d > 0) trend = "improving";
      else if (d < 0) trend = "worsening";
    }
  }

  const hour = new Date().getHours();
  const grt = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";
  const bc = `dashv2-body${co ? " dashv2-body-with-chat" : ""}`;

  let meds = Array.isArray(L?.medications) ? L.medications : [];
  if (!meds.length) {
    try {
      const parsed = JSON.parse(L?.medications_json || "[]");
      meds = Array.isArray(parsed) ? parsed : [];
    } catch {
      meds = [];
    }
  }
  meds = meds.map((medication) => String(medication).trim()).filter(Boolean);

  let xry = [];
  try {
    xry = JSON.parse(L?.xray_findings_json || "[]");
  } catch {
    xry = [];
  }

  const vitalObs = generateVitalsObservations(L?.measurements || []);

  const bc2 = `${bc}${isMulti ? " dashv2-body-with-sidebar" : ""}`;
  const rl = { High: 3, Moderate: 2, Low: 1 }[dash?.risk?.status] || 2;
  const selectedChatReport = L;

  return (
    <div className="dashv2-page">
      <img src="/heart.png" alt="" className="heart-image" />

      <Strip
        grt={grt}
        name={u.display_name}
        sevN={sevN}
        recs={recs}
        L={L}
        nav={nav}
        co={co}
        onTog={() => setCo((open) => !open)}
      />

      <div className={bc2}>
        <div className="dashv2-rail">
          <NewTriage recs={recs} nav={nav} />
          <History recs={recs} nav={nav} activeId={L?.job_id || L?.id} />
          <NeedSupport nav={nav} onChat={() => setCo(true)} />
        </div>

        <div className="dashv2-stage">
          <ValidationBanner dash={dash} />
          <ClinicalSummaryCard L={L} sev={L?.severity} recs={recs} nav={nav} />

          <div className="dashv2-triple-row">
            <SafetyOrCarePlanCard alerts={L?.safety_alerts} plan={L?.care_plan_snapshot} />
            <TrendCard recs={recs} trend={trend} />
            <RiskCard risk={dash?.risk} rl={rl} recs={recs} />
          </div>

          <div className="dashv2-card-grid">
            <PersonalizedRecommendationsCard recs={L?.personalized_recommendations || []} />
            <ClinicalPictureSummaryCard picture={L?.clinical_picture} />
          </div>

          {isMulti ? (
            <ComparisonTable L={L} P={P} isLatestSelected={isLatest} />
          ) : (
            <VitalsOverviewRow
              currentReport={L}
              vitalObs={vitalObs}
              quickSignals={<QuickSignalsCard L={L} recs={recs} trend={trend} />}
            />
          )}
        </div>

        {isMulti && (
          <div className="dashv2-sidebar">
            <SidebarVitals
              vitalObs={vitalObs}
              currentReport={L}
              previousReport={P}
              nav={nav}
            />
            <QuickSignalsCard L={L} recs={recs} trend={trend} variant="sidebar" />
          </div>
        )}

        <div
          className="dashv2-chat-panel"
          aria-hidden={!co}
          inert={co ? undefined : ""}
          style={{
            opacity: co ? 1 : 0.001,
            pointerEvents: co ? "auto" : "none",
          }}
        >
          <Chat
            key={selectedChatReport?.job_id || selectedChatReport?.id}
            dash={dash}
            report={selectedChatReport}
            isOpen={co}
            onClose={() => setCo(false)}
          />
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   AMBIENT STRIP
   ═══════════════════════════════════════════════════════════════════ */

function Strip({ grt, name, sevN, recs, L, nav, co, onTog }) {
  const tint = sevN >= 3 ? "#ef4444" : sevN >= 2 ? "#f59e0b" : "#10b981";

  return (
    <div className="dashv2-strip">
      <div className="dashv2-strip-left">
        <span className="dashv2-strip-greeting">
          {grt}, <strong>{name || "there"}</strong>
        </span>
      </div>

      <div className="dashv2-strip-right">
        {[
          { label: "Reports", value: recs.length },
          { label: "Status", value: L?.severity || recs[0]?.severity || "—", accent: tint },
        ].map((st, i) => (
          <div key={i} className="dashv2-strip-stat">
            <span className="dashv2-strip-stat-label">{st.label}</span>
            <span className="dashv2-strip-stat-val" style={{ color: st.accent || "#0d2167" }}>
              {st.value}
            </span>
          </div>
        ))}

        <button className="dashv2-strip-btn" onClick={() => nav("/medical-form")}>
          + New Triage
        </button>

        <button
          className={`dashv2-strip-chat-btn${co ? " dashv2-strip-chat-btn-active" : ""}`}
          onClick={onTog}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          {co ? "Close Chat" : "Chat"}
        </button>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   LEFT RAIL
   ═══════════════════════════════════════════════════════════════════ */

function NewTriage({ recs, nav }) {
  return (
    <div className="dashv2-card dashv2-triage-card">
      <div className="dashv2-triage-top">
        <svg className="dashv2-triage-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563ff" strokeWidth="2.2">
          <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
          <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
        </svg>
        <span className="dashv2-triage-slogan">Every report sharpens the clinical picture.</span>
      </div>

      <span className="dashv2-triage-label">{recs.length === 0 ? "Start your health journey" : "New assessment"}</span>

      <button className="dashv2-triage-btn" onClick={() => nav("/medical-form")}>
        Begin Triage
      </button>
    </div>
  );
}

function History({ recs, nav, activeId }) {
  const [ex, se] = useState(null);
  if (!recs || !recs.length) return null;

  const bg = (s) =>
    s === "CRITICAL"
      ? "dashv2-sevbg-CRITICAL dashv2-sev-CRITICAL"
      : s === "HIGH"
        ? "dashv2-sevbg-HIGH dashv2-sev-HIGH"
        : s === "MEDIUM" || s === "MODERATE"
          ? "dashv2-sevbg-MODERATE dashv2-sev-MODERATE"
          : "dashv2-sevbg-LOW dashv2-sev-LOW";

  return (
    <div className="dashv2-card dashv2-history-card">
      <span className="dashv2-history-label">Report History</span>

      <div className="dashv2-history-list">
        {recs.slice(0, 12).map((r) => {
          const o = ex === r.id;
          const isActive = activeId && (r.job_id === activeId || r.id === activeId);
          const dt = new Date(r.created_at);
          const ds = dt.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
          const ts = dt.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });

          return (
            <div key={r.id}>
              <div
                className={`dashv2-history-row${o || isActive ? " dashv2-history-row-active" : ""}`}
                onClick={() => se(o ? null : r.id)}
              >
                <div className="dashv2-history-left">
                  <div className="dashv2-history-file-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                      <polyline points="14 2 14 8 20 8" />
                      <line x1="16" y1="13" x2="8" y2="13" />
                      <line x1="16" y1="17" x2="8" y2="17" />
                    </svg>
                  </div>

                  <div className="dashv2-history-text-stack">
                    <span className="dashv2-history-name">Report - {ds}</span>
                    <span className="dashv2-history-sub">
                      {ds} • {ts}
                    </span>
                  </div>
                </div>

                <span className={`dashv2-history-sev-badge ${bg(r.severity)}`}>{r.severity}</span>
              </div>

              {o && (
                <div className="dashv2-history-preview">
                  <p className="dashv2-history-preview-symptoms">{r.symptoms_text || "No symptoms recorded."}</p>

                  <div className="dashv2-history-preview-meta">
                    <span>
                      Confidence: <strong>{Math.round((r.confidence || 0.85) * 100)}%</strong>
                    </span>
                  </div>

                  <div className="dashv2-history-preview-actions">
                    <button className="dashv2-history-preview-btn" onClick={() => nav(`/dashboard?jobId=${r.job_id || r.id}`)}>
                      Open Report →
                    </button>

                    <button
                      className="dashv2-history-preview-btn-outline"
                      onClick={() => {
                        window.open(`/export/pdf/${r.job_id}`, "_blank");
                      }}
                    >
                      Download PDF
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <button className="dashv2-history-viewall" onClick={() => nav("/report")}>
        View all reports →
      </button>
    </div>
  );
}

function NeedSupport({ nav, onChat }) {
  return (
    <div className="dashv2-card dashv2-support-card">
      <div className="dashv2-support-text">
        <span className="dashv2-support-title">Need help?</span>
        <span className="dashv2-support-sub">Chat with support or request a callback.</span>
      </div>

      <button
        className="dashv2-support-btn"
        onClick={() => (onChat ? onChat() : nav("/support"))}
        aria-label="Chat with support"
        title="Chat with support"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M3 18v-6a9 9 0 0 1 18 0v6" />
          <path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z" />
        </svg>
      </button>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   CLINICAL SUMMARY CARD
   ═══════════════════════════════════════════════════════════════════ */

function factorIcon(f) {
  const s = (f || "").toLowerCase();
  if (/heart|pulse/.test(s)) return "❤️";
  if (/spo2|spo₂|spo\s*2|oxygen|o₂|o2/.test(s)) return "🩸";
  if (/temp|fever/.test(s)) return "🌡️";
  if (/pressure/.test(s)) return "🩺";
  if (/respirat|breath/.test(s)) return "🫁";
  return "•";
}

function generatePrimaryFactors(L) {
  // ── PRIORITY 1: derive from actual measurements (top 3 by risk_score) ──
  const measurements = Array.isArray(L?.measurements) ? L.measurements : [];
  const ranked = measurements
    .filter((m) => (m?.risk_score ?? 0) >= 1)
    .sort((a, b) => {
      const ra = a?.risk_score ?? 0;
      const rb = b?.risk_score ?? 0;
      if (rb !== ra) return rb - ra;
      const da = a?.deviation_score ?? 0;
      const db = b?.deviation_score ?? 0;
      return db - da;
    });

  if (ranked.length > 0) {
    return ranked.slice(0, 3).map((m) => {
      const status = String(m.status || "").toLowerCase();
      let prefix = "";
      if (status.includes("critical")) {
        prefix = status.includes("low") ? "Critical Low " : "Critical High ";
      } else if (status.includes("low")) {
        prefix = "Low ";
      } else if (status.includes("high") || status.includes("elevated")) {
        prefix = "High ";
      } else if (status.includes("borderline")) {
        prefix = "Borderline ";
      }
      return `${prefix}${m.name}: ${m.display_value}`.trim();
    });
  }

  // ── PRIORITY 2: filtered contributing_factors (skip generic noise) ──
  const GENERIC_PATTERNS = [
    /abnormal laboratory values detected/i,
    /abnormal lab values/i,
    /no significant/i,
    /default/i,
    /reported symptoms:/i,
  ];

  const raw = Array.isArray(L?.contributing_factors) ? L.contributing_factors : [];
  const filtered = raw.filter(
    (f) => !GENERIC_PATTERNS.some((rx) => rx.test(String(f || "")))
  );

  const cleaned = filtered
    .map((f) => {
      let s = String(f || "").trim();
      s = s.replace(/\s*\([^)]*\)\s*/g, " ").trim();
      s = s.replace(/\s+/g, " ");
      if (s.length > 55) {
        s = s.slice(0, 52).replace(/\s+\S*$/, "") + "…";
      }
      return s;
    })
    .filter(Boolean);

  if (cleaned.length > 0) return cleaned.slice(0, 3);

  // ── PRIORITY 3: fallback to symptoms ──
  if (L?.symptoms_text) {
    return [`Reported: ${String(L.symptoms_text).slice(0, 50)}`];
  }

  return [];
}

function ClinicalSummaryCard({ L, sev, recs, nav }) {
  if (!L) {
    return (
      <div className="dashv2-card" style={{ textAlign: "center", padding: "40px", color: "#7182b1", fontSize: "14px" }}>
        No health data yet. Submit your first triage to unlock clinical analysis.
      </div>
    );
  }

  const c = SC[sev] || "#2563ff";
  const factors = generatePrimaryFactors(L);
  const hasMeasurements = Array.isArray(L.measurements) && L.measurements.length > 0;
  const keyTakeaway = `Your latest ${hasMeasurements ? "report measurements and symptoms" : "reported symptoms and clinical findings"} indicate a ${(sev || "medium").toLowerCase()} health risk at this time. Continue monitoring and follow recommended actions.`;

  return (
    <div className="dashv2-card dashv2-analysis-card">
      <div className="dashv2-card-header">
        <span className="dashv2-card-title">
          <span className="dashv2-card-title-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2563ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 2h6a1 1 0 0 1 1 1v1h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2V3a1 1 0 0 1 1-1z" />
              <path d="M9 4h6" />
              <line x1="8" y1="11" x2="16" y2="11" />
              <line x1="8" y1="15" x2="13" y2="15" />
            </svg>
          </span>
          Clinical Summary
        </span>
      </div>

      <div className="dashv2-summary-grid">
        {/* Column 1 — Overall Severity */}
        <div>
          <span className="dashv2-summary-col-label">Overall Severity</span>
          <div className="dashv2-summary-severity-val" style={{ color: c }}>
            {sev}
          </div>
          <div className="dashv2-summary-confidence">
            <strong>{Math.round((L.confidence || 0.85) * 100)}%</strong> Confidence
          </div>
          <span className="dashv2-summary-reports-chip">
            {recs.length >= 2 ? `Stable across ${recs.length} reports` : "Based on 1 report"}
          </span>
        </div>

        {/* Column 2 — Key Clinical Takeaway (no "Why this matters") */}
        <div>
          <span className="dashv2-summary-col-label">Key Clinical Takeaway</span>
          <p className="dashv2-summary-takeaway-text">{keyTakeaway}</p>
        </div>

        {/* Column 3 — Primary Contributing Factors (simplified) */}
        <div>
          <span className="dashv2-summary-col-label">Primary Contributing Factors</span>
          <div className="dashv2-summary-factors-list">
            {factors.length ? (
              factors.map((f, i) => (
                <div key={i} className="dashv2-summary-factor-row">
                  <span className="dashv2-summary-factor-bullet">•</span>
                  <span className="dashv2-summary-factor-text">{f}</span>
                </div>
              ))
            ) : (
              <div className="dashv2-summary-factor-row">
                <span className="dashv2-summary-factor-bullet">•</span>
                <span className="dashv2-summary-factor-text">No significant factors flagged</span>
              </div>
            )}
          </div>

          <button className="dashv2-summary-full-report-btn" onClick={() => nav(`/report?jobId=${L.job_id || L.id}`)}>
            View Full Report →
          </button>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   VALIDATION BANNER
   ═══════════════════════════════════════════════════════════════════ */

function ValidationBanner({ dash }) {
  const sf = dash?.safety_review || [];
  const wn = sf.filter((s) => !s.ok);

  if (wn.length === 0 && sf.length > 0) {
    return (
      <div className="dashv2-banner dashv2-banner-ok">
        <span className="dashv2-banner-icon">✓</span>
        <div className="dashv2-banner-body">
          <span className="dashv2-banner-title">Safety Validation Passed</span>
          <span className="dashv2-banner-sub">Rule-based and AI assessments are in agreement. All safety checks completed.</span>
        </div>
      </div>
    );
  }

  if (wn.length > 0) {
    const io = wn.some((w) => w.status && w.status.toLowerCase().includes("override"));
    return (
      <div className={`dashv2-banner ${io ? "dashv2-banner-override" : "dashv2-banner-warning"}`}>
        <span className="dashv2-banner-icon">{io ? "⚠" : "⚡"}</span>
        <div className="dashv2-banner-body">
          <span className="dashv2-banner-title">{io ? "Safety Override — Urgent Review Required" : "Safety Warning — Clinician Review Advised"}</span>
          <span className="dashv2-banner-sub">
            {io ? "Significant disagreement detected. Urgent clinical review required." : "Minor discrepancies between rule-based and AI assessments."}
          </span>
        </div>
      </div>
    );
  }

  return null;
}

/* ═══════════════════════════════════════════════════════════════════
   VITALS UNDER OBSERVATION (unused legacy — kept for compatibility)
   ═══════════════════════════════════════════════════════════════════ */

function VitalsCriticalObs({ critical }) {
  return (
    <div className="dashv2-card" style={{ borderLeft: "3px solid #ef4444" }}>
      <div className="dashv2-card-header">
        <span className="dashv2-card-title">🔴 Vitals Under Observation</span>
        <span className="dashv2-card-badge" style={{ background: "rgba(239,68,68,0.08)", color: "#ef4444" }}>
          {critical.length} flagged
        </span>
      </div>

      <p className="dashv2-analysis-paragraph" style={{ color: "#ef4444", fontWeight: 600, marginBottom: "12px" }}>
        These readings are significantly outside your personal baseline range and require clinical attention.
      </p>

      {critical.map((v, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "12px",
            padding: "10px 12px",
            borderRadius: "10px",
            background: "rgba(239,68,68,0.04)",
            marginBottom: "6px",
          }}
        >
          <span style={{ fontSize: "20px" }}>📊</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: "13px", fontWeight: 800, color: "#0d2167" }}>{v.name}</div>
            <div style={{ fontSize: "11px", color: "#7182b1" }}>{v.note}</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: "17px", fontWeight: 900, color: "#ef4444" }}>{v.current}</div>
            <div style={{ fontSize: "10px", color: "#ef4444", fontWeight: 700 }}>
              {v.z > 0 ? "+" : ""}
              {v.z.toFixed(1)}σ · N={v.samples}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function VitalsStable({ stable }) {
  return (
    <div className="dashv2-card">
      <div className="dashv2-card-header">
        <span className="dashv2-card-title">🟢 Vitals in Range</span>
        <span className="dashv2-card-badge" style={{ background: "rgba(16,185,129,0.06)", color: "#10b981" }}>
          Stable
        </span>
      </div>

      <div className="dashv2-vitals-grid">
        {stable.map((v, i) => (
          <div key={i} className="dashv2-vital-chip">
            <span className="dashv2-vital-name" style={{ fontSize: "11px" }}>
              {v.name}
            </span>
            <span className="dashv2-vital-current" style={{ fontSize: "16px" }}>
              {v.current}
            </span>
            <span style={{ fontSize: "9px", color: "#10b981", fontWeight: 700 }}>
              {v.z > 0 ? "+" : ""}
              {v.z.toFixed(1)}σ
            </span>
            <span className="dashv2-vital-badge" style={{ color: "#10b981", background: "rgba(16,185,129,0.08)" }}>
              N={v.samples}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function VitalsInsufficient({ insufficient }) {
  return (
    <div className="dashv2-card">
      <div className="dashv2-card-header">
        <span className="dashv2-card-title">⚪ Building Baselines</span>
        <span className="dashv2-card-badge" style={{ background: "rgba(107,114,128,0.06)", color: "#6b7280" }}>
          Insufficient
        </span>
      </div>

      <p className="dashv2-analysis-paragraph" style={{ marginBottom: "8px" }}>
        These vitals need more readings before personal baselines can be established. Continue your daily check-ins.
      </p>

      <div className="dashv2-vitals-grid">
        {insufficient.map((v, i) => (
          <div key={i} className="dashv2-vital-chip">
            <span className="dashv2-vital-name" style={{ fontSize: "11px" }}>
              {v.name}
            </span>
            <span className="dashv2-vital-current" style={{ fontSize: "15px" }}>
              {v.current}
            </span>
            <span style={{ fontSize: "9px", color: "#9ca3af", fontWeight: 600 }}>{v.note}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   PERSONALIZED RECOMMENDATIONS (from BIOMARKER_KNOWLEDGE)
   Replaces the old "Recommended Actions" card.
   ═══════════════════════════════════════════════════════════════════ */

function PersonalizedRecommendationsCard({ recs }) {
  const items = (recs || []).slice(0, 3);

  return (
    <div className="dashv2-card dashv2-actions-card">
      <div className="dashv2-card-header dashv2-thirdrow-header">
        <span className="dashv2-card-title dashv2-thirdrow-card-title">
          <span className="dashv2-card-title-icon dashv2-thirdrow-card-title-icon">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2563ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 2h6a1 1 0 0 1 1 1v1h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2V3a1 1 0 0 1 1-1z" />
              <path d="M9 4h6" />
              <path d="M9 13l2 2 4-4" />
            </svg>
          </span>
          Personalized Recommendations
        </span>
      </div>

      <div className="dashv2-info-list dashv2-actions-list-tight">
        {items.length === 0 && (
          <div className="dashv2-action-row">
            <span className="dashv2-action-copy">
              <div className="dashv2-action-title">No flagged biomarkers</div>
              <div className="dashv2-action-detail">
                All measured values are within their reference ranges. Continue routine monitoring.
              </div>
            </span>
          </div>
        )}

        {items.map((rec, i) => {
          const isGroup = rec.status === "system_group";
          const systemLabel = isGroup
            ? String(rec.biomarker || "").replace(/\s*\(\d+\)\s*$/, "")
            : rec.biomarker;
          const titleLine = isGroup && rec.value
            ? `${systemLabel}: ${rec.value}`
            : rec.biomarker;

          return (
            <div key={rec.key || i} className="dashv2-action-row">
              <span
                style={{
                  fontSize: "12px",
                  fontWeight: 600,
                  color: "#0d2167",
                  paddingTop: "2px",
                  textAlign: "right",
                  paddingRight: "2px",
                }}
              >
                {i + 1}.
              </span>

              <div className="dashv2-action-copy">
                <div
                  className="dashv2-action-title"
                  style={{ fontWeight: 600 }}
                >
                  {titleLine}
                  {!isGroup && rec.value && (
                    <span style={{ color: "#7182b1", fontWeight: 500, marginLeft: 6 }}>
                      · {rec.value}
                    </span>
                  )}
                </div>
                <div className="dashv2-action-detail">{rec.recommendation}</div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="dashv2-inline-note">
        <span>ⓘ</span>
        <span>Always confirm biomarker-specific guidance with your physician.</span>
      </div>
    </div>
  );
}


/* ═══════════════════════════════════════════════════════════════════
   CLINICAL PICTURE SUMMARY (from clinical_picture_synthesizer)
   Replaces CarePlanSnapshotCard in Row 3.
   ═══════════════════════════════════════════════════════════════════ */

function ClinicalPictureSummaryCard({ picture }) {
  // ── collect findings, best-confidence first ─────────────────────────
  const confident = Array.isArray(picture?.confident_findings)
    ? picture.confident_findings
    : [];
  const differential = Array.isArray(picture?.differential_findings)
    ? picture.differential_findings
    : [];

  const allFindings = [...confident, ...differential]
    .filter((f) => f && (f.narrative || f.narrative_short || f.id))
    .sort((a, b) => (b.confidence || 0) - (a.confidence || 0));

  // ── shorten a narrative to its first complete sentence ──────────────
  const shortenNarrative = (text) => {
    if (!text) return "";
    let s = String(text).trim();
    const stop = s.search(/[.!?](\s|$)/);
    if (stop > 20) s = s.slice(0, stop + 1).trim();
    s = s
      .replace(
        /,?\s*(and\s+)?(clinical correlation.*|viral serology may be considered.*|is recommended\.?|are recommended\.?|should be considered\.?|warrants? clinical.*|may be considered\.?)$/i,
        ""
      )
      .trim()
      .replace(/[,\s]+$/, "");
    if (s && !/[.!?]$/.test(s)) s += ".";
    return s;
  };

  // ── LAYOUT BUDGET ───────────────────────────────────────────────────
  const MAX_ROWS_HARD = 5;
  const MAX_LINES = 8;
  const CHARS_PER_LINE = 115;

  const linesFor = (txt) => {
    if (!txt) return 0;
    if (txt.length <= CHARS_PER_LINE) return 1;
    if (txt.length <= CHARS_PER_LINE * 2) return 2;
    return 3;
  };

  const pickTextForFinding = (f) => {
    const full = shortenNarrative(f.narrative || "");
    const short = f.narrative_short || "";
    const idFallback = String(f.id || "").replace(/_/g, " ");

    const candidates = [full, short, idFallback].filter(
      (t) => t && t.length >= 12
    );
    if (!candidates.length) return "";

    const threeLineMax = CHARS_PER_LINE * 3;
    const fits = candidates.filter((t) => t.length <= threeLineMax);
    if (fits.length) {
      return fits.sort((a, b) => b.length - a.length)[0];
    }
    return candidates.sort((a, b) => a.length - b.length)[0];
  };

  // ── PACK ROWS ───────────────────────────────────────────────────────
  const displayRows = [];
  let linesUsed = 0;
  let cursor = 0;

  for (; cursor < allFindings.length; cursor++) {
    if (displayRows.length >= MAX_ROWS_HARD) break;

    const f = allFindings[cursor];
    const text = pickTextForFinding(f);
    if (!text) continue;

    const cost = linesFor(text);

    if (linesUsed + cost > MAX_LINES) break;

    displayRows.push({ finding: f, text });
    linesUsed += cost;
  }

  const overflowCount = allFindings.length - cursor;
  const isEmpty = displayRows.length === 0;

  // ── confidence badge ── unified pill style (matches Report History) ──
  const renderBadge = (conf) => {
    const pct = Math.round((Number(conf) || 0) * 100);
    const [color, bg] =
      pct >= 70
        ? ["#10b981", "rgba(16,185,129,0.15)"]
        : pct >= 40
          ? ["#f59e0b", "rgba(245,158,11,0.16)"]
          : ["#2563ff", "rgba(37,99,255,0.15)"];
    return (
      <span
        className="dashv2-clinical-pill"
        style={{
          background: bg,
          color,
          marginLeft: 8,
          flexShrink: 0,
          alignSelf: "flex-start",
          marginTop: 2,
          whiteSpace: "nowrap",
        }}
      >
        {pct}%
      </span>
    );
  };

  const numStyle = {
    fontSize: "12px",
    fontWeight: 600,
    color: "#0d2167",
    paddingTop: "2px",
    textAlign: "right",
    paddingRight: "2px",
    flexShrink: 0,
  };

  return (
    <div className="dashv2-card dashv2-precautions-card">
      {/* ── header ── */}
      <div className="dashv2-card-header dashv2-thirdrow-header">
        <span className="dashv2-card-title dashv2-thirdrow-card-title">
          <span
            className="dashv2-card-title-icon dashv2-thirdrow-card-title-icon"
            style={{ background: "rgba(124,58,237,0.12)" }}
          >
            <svg
              width="15" height="15" viewBox="0 0 24 24"
              fill="none" stroke="#7c3aed"
              strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="9" />
              <path d="M12 7v5l3 2" />
            </svg>
          </span>
          Clinical Picture Summary
        </span>
      </div>

      {/* ── body ── */}
      <div className="dashv2-info-list dashv2-actions-list-tight">

        {isEmpty && (
          <div className="dashv2-action-row">
            <span style={numStyle}>1.</span>
            <div className="dashv2-action-copy">
              <div className="dashv2-action-detail">
                Clinical picture synthesis not available for this report.
              </div>
            </div>
          </div>
        )}

        {displayRows.map(({ finding, text }, i) => (
          <div key={finding.id || i} className="dashv2-action-row">
            <span style={numStyle}>{i + 1}.</span>
            <div
              className="dashv2-action-copy"
              style={{ display: "flex", alignItems: "flex-start" }}
            >
              <div
                className="dashv2-action-detail clinical-picture-row"
                title={finding.narrative || text}
                style={{ flex: 1, minWidth: 0 }}
              >
                {text}
              </div>
              {renderBadge(finding.confidence)}
            </div>
          </div>
        ))}

        {overflowCount > 0 && (
          <div className="dashv2-action-row">
            <span style={numStyle}>{displayRows.length + 1}.</span>
            <div className="dashv2-action-copy">
              <div
                className="dashv2-action-detail"
                style={{ fontStyle: "italic", color: "#7182b1" }}
              >
                {overflowCount} additional pattern
                {overflowCount === 1 ? "" : "s"} considered — see full report.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   VISUAL CARDS
   ═══════════════════════════════════════════════════════════════════ */

function SafetyOrCarePlanCard({ alerts, plan }) {
  const safetyAlerts = Array.isArray(alerts) ? alerts.filter((a) => a && a.text) : [];
  const hasAlerts = safetyAlerts.length > 0;

  // Priority 1: Safety Alerts
  if (hasAlerts) {
    return (
      <div className="dashv2-card">
        <div className="dashv2-mini-card-header">
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span className="dashv2-mini-card-icon" aria-hidden="true">⚠️</span>
            <span className="dashv2-mini-card-title">Safety Alerts</span>
          </div>
          <span
            className="dashv2-card-badge"
            style={{ background: "rgba(239,68,68,0.15)", color: "#ef4444" }}
          >
            {safetyAlerts.length} {safetyAlerts.length === 1 ? "alert" : "alerts"}
          </span>
        </div>

        <div className="dashv2-summary-factors-list">
          {safetyAlerts.map((alert, i) => (
            <div key={i} className="dashv2-summary-factor-row">
              <span className="dashv2-summary-factor-bullet">•</span>
              <span className="dashv2-summary-factor-text">{alert.text}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Priority 2: Care Plan (from care_plan_snapshot)
  const planItems = [];
  if (plan?.immediate?.text) planItems.push(plan.immediate.text);
  if (plan?.short_term?.text) planItems.push(plan.short_term.text);
  const hasPlan = planItems.length > 0;

  return (
    <div className="dashv2-card">
      <div className="dashv2-mini-card-header">
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span className="dashv2-mini-card-icon" aria-hidden="true">📋</span>
          <span className="dashv2-mini-card-title">Care Plan</span>
        </div>
        {hasPlan && (
          <span
            className="dashv2-card-badge"
            style={{ background: "rgba(37,99,255,0.15)", color: "#2563ff" }}
          >
            Next steps
          </span>
        )}
      </div>

      {hasPlan ? (
        <div className="dashv2-summary-factors-list">
          {planItems.map((item, i) => (
            <div key={i} className="dashv2-summary-factor-row">
              <span className="dashv2-summary-factor-bullet">•</span>
              <span className="dashv2-summary-factor-text">{item}</span>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ fontSize: "12.5px", color: "#262626", fontWeight: 500, lineHeight: 1.5 }}>
          No specific care actions from the latest report.
        </div>
      )}
    </div>
  );
}

function TrendCard({ recs, trend }) {
  const tc = trend === "improving" ? "#10b981" : trend === "worsening" ? "#ef4444" : "#2563ff";
  const label = trend === "improving" ? "IMPROVING" : trend === "worsening" ? "WORSENING" : "STABLE";

  return (
    <div className="dashv2-card">
      <div className="dashv2-mini-card-header">
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span className="dashv2-mini-card-icon" aria-hidden="true">
            📈
          </span>
          <span className="dashv2-mini-card-title">Trend</span>
        </div>
      </div>

      <div className="dashv2-trend-val" style={{ color: tc }}>
        {label}
      </div>
      <div className="dashv2-trend-sub">{recs.length >= 2 ? `Across last ${recs.length} reports` : "Based on 1 report"}</div>
    </div>
  );
}

function RiskCard({ rl, recs }) {
  const rc = rl >= 3 ? "#ef4444" : rl >= 2 ? "#f59e0b" : "#10b981";
  const rlbl = rl >= 3 ? "HIGH" : rl >= 2 ? "MODERATE" : "LOW";

  return (
    <div className="dashv2-card">
      <div className="dashv2-mini-card-header">
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span className="dashv2-mini-card-icon" aria-hidden="true">
            🛡️
          </span>
          <span className="dashv2-mini-card-title">Risk Profile</span>
        </div>
      </div>

      <div className="dashv2-risk-val" style={{ color: rc }}>
        {rlbl}
      </div>
      <div className="dashv2-risk-sub">{recs.length >= 2 ? "Multiple reports analyzed" : "Based on 1 report"}</div>
    </div>
  );
}

function statusForVital(v) {
  const status = (v.status || "reported").toLowerCase();
  if (status === "not_reported") {
    return { label: "Not reported", color: "#475569", bg: "rgba(148,163,184,0.18)" };
  }
  if (v.category === "critical") {
    return { label: status.includes("low") ? "Low" : "High", color: "#ef4444", bg: "rgba(239,68,68,0.15)" };
  }
  if (v.category === "observation") {
    return { label: status.includes("low") ? "Low" : "Elevated", color: "#f59e0b", bg: "rgba(245,158,11,0.16)" };
  }
  if (v.category === "normal") {
    return { label: "Good", color: "#10b981", bg: "rgba(16,185,129,0.15)" };
  }
  return { label: "Reported", color: "#2563ff", bg: "rgba(37,99,255,0.15)" };
}

function VitalsInlineRow({ vitals }) {
  return (
    <div className="dashv2-vitals-hrow">
      {vitals.map((v, i) => {
        const { icon, color } = getVitalIconMeta(v.name);
        const st = statusForVital(v);
        const val = v.numericValue ?? (v.current || "—").split(" ")[0];
        const unit = v.unit || (v.current || "").split(" ").slice(1).join(" ");

        return (
          <div key={i} className="dashv2-vital-hchip">
            <div className="dashv2-vital-hchip-top">
              <span
                className="dashv2-vital-hchip-icon"
                aria-hidden="true"
                style={{ color, display: "inline-flex", alignItems: "center", justifyContent: "center" }}
              >
                {icon}
              </span>
              {v.name}
            </div>

            <div className="dashv2-vital-hchip-val">
              {val}
              <span> {unit}</span>
            </div>

            <span className="dashv2-vital-hchip-badge" style={{ background: st.bg, color: st.color }}>
              {st.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   VITALS OVERVIEW ROW (SINGLE-PURPOSE dashboard, 5 vitals)
   Uses shared selectPriorityVitals() with NO previous filter.
   Flat list — no category headings.
   ═══════════════════════════════════════════════════════════════════ */

function VitalsOverviewRow({ currentReport, vitalObs, quickSignals }) {
  // Single-purpose: pick top 6 by priority, no previous filter
  const currentMeasurements = currentReport?.measurements || [];
  const selected = selectPriorityVitals(currentMeasurements, null, 6);

  // Convert to the display format the chip row expects
  const actual = selected.map((m) => {
    const rs = Number.isFinite(m.risk_score) ? m.risk_score : null;
    return {
      key: m.key,
      name: m.name || m.vital || "Clinical Measurement",
      current: m.display_value || `${m.value ?? "—"} ${m.unit || ""}`.trim(),
      numericValue: m.value,
      unit: m.unit || "",
      status: m.status || "reported",
      category: rs >= 2 ? "critical" : rs === 1 ? "observation" : rs === 0 ? "normal" : "reported",
    };
  });

  // If we have fewer than 6 real vitals, pad with placeholders so layout
  // stays consistent (6 chips across).
  const existingNames = new Set(actual.map((vital) => vital.name));
  const placeholderNames = [
    "Heart Rate",
    "SpO2",
    "Temperature",
    "Blood Pressure",
    "Respiratory Rate",
    "Weight",
  ];
  const placeholders = placeholderNames
    .filter((name) => !existingNames.has(name))
    .slice(0, Math.max(0, 6 - actual.length))
    .map((name) => ({
      name,
      current: "—",
      numericValue: "—",
      unit: "",
      status: "not_reported",
      category: "reported",
    }));
  const all = [...actual, ...placeholders].slice(0, 6);

  return (
    <div className="dashv2-card dashv2-vitals-row-card">
      <div>
        <div className="dashv2-card-header">
          <span className="dashv2-card-title">❤️ Vitals Overview</span>
        </div>

        {all.length > 0 ? (
          <VitalsInlineRow vitals={all} />
        ) : (
          <p className="dashv2-analysis-paragraph">No vitals data available yet. Vitals from your triage submissions will appear here once processed.</p>
        )}
      </div>

      {quickSignals}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   QUICK SIGNALS CARD
   Auto-extracts symptom tags from patient free-text via
   extractSymptomTags(). When 2+ symptoms are found, renders compact
   pill tags instead of raw prose — handles long descriptions
   gracefully. Full original text is preserved as a browser tooltip
   on hover of the tag group.

   Extractor is hybrid: uses backend-provided `symptoms_extracted`
   array if present on the report object, otherwise falls back to
   improved regex extraction (with negation detection + vital-value
   extraction). Passing the full `L` report enables the auto-upgrade.
   ═══════════════════════════════════════════════════════════════════ */

function QuickSignalsCard({ L, trend, variant }) {
  if (!L) return null;

  const tc = trend === "improving" ? "#10b981" : trend === "worsening" ? "#ef4444" : "#2563ff";
  const tl = trend === "improving" ? "Improving" : trend === "worsening" ? "Worsening" : "Stable";

  // Extract structured symptom tags — pass the full report so the extractor
  // can use backend-provided `symptoms_extracted` if available, otherwise
  // falls back to regex extraction on `symptoms_text`.
  const symptomText = L.symptoms_text || "";
  // Cap raised from 5 → 8: real-world reports often combine several
  // symptoms with several vital-value readings (e.g. fatigue + fever +
  // chest tightness + dizziness + BP + glucose = 6 legitimate tags).
  // A cap of 5 was silently dropping the last valid finding.
  const tags = extractSymptomTags(L, 8);
  // Only switch to pill view when we actually have 2+ usable tags —
  // an empty or single-item result always falls back to raw text so
  // the card is never left blank for symptoms our patterns don't cover.
  const useTagView = tags.length >= 2;

  return (
    <div className={`dashv2-card dashv2-quicksig-card${variant === "sidebar" ? " dashv2-quicksig-card-tall" : ""}`}>
      <div className="dashv2-card-header">
        <span className="dashv2-card-title">⚡ Quick Signals</span>
      </div>

      <div className={`dashv2-quicksig-body${variant === "sidebar" ? " dashv2-quicksig-body-sidebar" : ""}`}>
        <div className="dashv2-quicksig-section dashv2-quicksig-section-top">
          <div className="dashv2-quicksig-label">
            {useTagView ? "Reported Symptoms" : "Reported Symptom"}
          </div>

          {useTagView ? (
            <div
              className="dashv2-quicksig-symptom-tags"
              title={symptomText}
              style={{
                display: "flex",
                flexWrap: "wrap",
                rowGap: "10px",
                columnGap: "10px",
                marginTop: "10px",
              }}
            >
              {tags.map((tag, i) => (
                <span
                  key={i}
                  style={{
                    fontSize: "11.5px",
                    fontWeight: 600,
                    color: "#0d2167",
                    background: "rgba(37,99,255,0.10)",
                    padding: "4px 10px",
                    borderRadius: "999px",
                    whiteSpace: "nowrap",
                    lineHeight: 1.3,
                  }}
                >
                  {tag}
                </span>
              ))}
            </div>
          ) : (
            <div className="dashv2-quicksig-symptom" title={symptomText}>
              {symptomText || "—"}
            </div>
          )}
        </div>

        <div className="dashv2-quicksig-section dashv2-quicksig-section-mid">
          <div className="dashv2-quicksig-grid">
            <div>
              <div className="dashv2-quicksig-label">Confidence</div>
              <div className="dashv2-quicksig-val" style={{ color: "#10b981" }}>
                {Math.round((L.confidence || 0.85) * 100)}%
              </div>
            </div>

            <div>
              <div className="dashv2-quicksig-label">Trend</div>
              <div className="dashv2-quicksig-val" style={{ color: tc }}>
                {tl}
              </div>
            </div>
          </div>
        </div>

        <div className="dashv2-quicksig-section dashv2-quicksig-section-bottom">
          <div className="dashv2-quicksig-foot">
            <div className="dashv2-quicksig-label">Last Updated</div>
            <div style={{ fontSize: "12.5px", fontWeight: 800, color: "#0d2167" }}>
              Today, {new Date(L.created_at).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   SIDEBAR VITALS (2+ reports)
   Uses shared selectPriorityVitals() with 7 slots.
   Fill order: Critical → Observational → Normal (variable per category).
   All three categories are always in play; normal fills whatever slots
   remain after critical + observational.
   ═══════════════════════════════════════════════════════════════════ */

function SidebarVitals({ vitalObs, currentReport, previousReport, nav }) {
  // Sidebar uses 8 slots (has room for one more than the comparison table)
  const currentMeasurements = currentReport?.measurements || [];
  const previousMeasurements = previousReport?.measurements || null;
  const selected = selectPriorityVitals(currentMeasurements, previousMeasurements, 8);

  // Group the selected 7 by category for display
  const crit = [];
  const obs = [];
  const normal = [];

  for (const m of selected) {
    const rs = Number.isFinite(m.risk_score) ? m.risk_score : null;
    const item = {
      key: m.key,
      name: m.name || m.vital || "Clinical Measurement",
      current: m.display_value || `${m.value ?? "—"} ${m.unit || ""}`.trim(),
      status: m.status || "reported",
      category: rs >= 2 ? "critical" : rs === 1 ? "observation" : rs === 0 ? "normal" : "reported",
    };
    if (rs >= 2) crit.push(item);
    else if (rs === 1) obs.push(item);
    else normal.push(item);
  }

  const empty = selected.length === 0;

  const renderRow = (v, i) => {
    const { icon, color } = getVitalIconMeta(v.name);
    const st = statusForVital(v);

    return (
      <div key={v.key || i} className="dashv2-sbvitals-row">
        <span
          className="dashv2-sbvitals-icon"
          aria-hidden="true"
          style={{
            color,
            background: "rgba(255,255,255,0.35)",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          {icon}
        </span>

        <div>
          <div className="dashv2-sbvitals-name">{v.name}</div>
          <div className="dashv2-sbvitals-val">{v.current}</div>
        </div>

        <span className="dashv2-sbvitals-badge" style={{ background: st.bg, color: st.color }}>
          {st.label}
        </span>
      </div>
    );
  };

  return (
    <div className="dashv2-card dashv2-vitals-overview-card">
      <div className="dashv2-card-header">
        <span className="dashv2-card-title">❤️ Vitals Overview</span>
        <span style={{ color: "#9ca3af" }}>ⓘ</span>
      </div>

      <div className="dashv2-sbvitals-scroll">
        {empty && (
          <p className="dashv2-analysis-paragraph">
            No comparable vitals available across current and comparison reports.
          </p>
        )}

        {!empty && crit.length > 0 && (
          <>
            <div className="dashv2-sbvitals-group-label">
              <span className="dashv2-sbvitals-dot" style={{ background: "#ef4444" }} />
              Critical ({crit.length})
            </div>
            {crit.map((v, i) => renderRow(v, i))}
          </>
        )}

        {!empty && obs.length > 0 && (
          <>
            <div className="dashv2-sbvitals-group-label">
              <span className="dashv2-sbvitals-dot" style={{ background: "#f59e0b" }} />
              Under Observation ({obs.length})
            </div>
            {obs.map((v, i) => renderRow(v, i))}
          </>
        )}

        {!empty && normal.length > 0 && (
          <>
            <div className="dashv2-sbvitals-group-label">
              <span className="dashv2-sbvitals-dot" style={{ background: "#10b981" }} />
              Normal ({normal.length})
            </div>
            {normal.map((v, i) => renderRow(v, i))}
          </>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   COMPARISON TABLE  (LAYOUT FIXED — 6 vitals max)
   Row layout: Overall Severity + up to 6 priority vitals = 7 rows max.

   Header direction convention — left column always = chronologically
   earlier, right column always = chronologically later. Header labels
   reflect the user's current context:
     • Latest selected → "Previous → Current"
     • Older selected  → "Current → Latest"  (chosen report is on the left)
   ═══════════════════════════════════════════════════════════════════ */

function buildMeasurementComparisonRows(currentReport, previousReport) {
  // Comparison table is layout-locked → only 6 vitals allowed
  const currentMeasurements = currentReport?.measurements || [];
  const previousMeasurements = previousReport?.measurements || [];
  const selected = selectPriorityVitals(currentMeasurements, previousMeasurements, 6);

  const previousByKey = new Map(previousMeasurements.map((m) => [m.key, m]));
  const rows = [];

  for (const current of selected) {
    const previous = previousByKey.get(current.key);
    if (!previous) continue; // safety — selector already filters this

    const currentValue = Number(current.value);
    const previousValue = Number(previous.value);
    if (!Number.isFinite(currentValue) || !Number.isFinite(previousValue)) continue;

    const delta = currentValue - previousValue;
    const currentDeviation = current.deviation_score == null ? null : Number(current.deviation_score);
    const previousDeviation = previous.deviation_score == null ? null : Number(previous.deviation_score);
    let status = "Stable";

    if (Number.isFinite(currentDeviation) && Number.isFinite(previousDeviation)) {
      if (currentDeviation < previousDeviation - 0.0001) status = "Improved";
      else if (currentDeviation > previousDeviation + 0.0001) status = "Worsened";
    } else if (Number.isFinite(current.risk_score) && Number.isFinite(previous.risk_score)) {
      if (current.risk_score < previous.risk_score) status = "Improved";
      else if (current.risk_score > previous.risk_score) status = "Worsened";
    }

    const roundedDelta = Math.round(Math.abs(delta) * 100) / 100;
    const change = Math.abs(delta) < 0.0001
      ? "No change"
      : `${delta > 0 ? "↑" : "↓"} ${roundedDelta}${current.unit ? ` ${current.unit}` : ""}`;

    rows.push({
      param: current.name,
      prev: previous.display_value,
      cur: current.display_value,
      change,
      status,
    });
  }

  return rows;
}

function ComparisonTable({ L, P, isLatestSelected }) {
  // Left column always = chronologically earlier; right = chronologically later.
  //   • Latest selected → L is latest, P is one before latest.
  //     Left = P (earlier), Right = L (later).
  //   • Older selected  → L is the selected (older) report, P is the latest.
  //     Left = L (earlier), Right = P (later).
  const leftReport  = isLatestSelected ? P : L;
  const rightReport = isLatestSelected ? L : P;

  const rows = [];
  const sevD = (SM[rightReport.severity] || 2) - (SM[leftReport.severity] || 2);

  rows.push({
    param: "Overall Severity",
    prev: leftReport.severity,
    cur: rightReport.severity,
    change: sevD === 0 ? "No change" : `${sevD < 0 ? "↓" : "↑"} ${Math.abs(sevD)} level`,
    status: sevD < 0 ? "Improved" : sevD > 0 ? "Worsened" : "Stable",
  });

  // Pass right-report as "current" (later in time) and left-report as
  // "previous" (earlier in time). Keeps status comparisons chronologically
  // consistent regardless of which report the user has selected.
  rows.push(...buildMeasurementComparisonRows(rightReport, leftReport));

  // Cap at 7 total rows (1 Overall Severity + 6 priority vitals) — LOCKED
  const cappedRows = rows.slice(0, 7);
  const hasVitalRows = cappedRows.length > 1;

  // Consistent 2-color + neutral logic for both Change and Status
  const getChangeColor = (status) => {
    if (status === "Improved") return "#10b981";
    if (status === "Worsened") return "#ef4444";
    return "#262626"; // neutral for Stable / No change
  };

  const getStatusPill = (status) => {
    if (status === "Improved") {
      return { bg: "rgba(16,185,129,0.15)", c: "#10b981" };
    }
    if (status === "Worsened") {
      return { bg: "rgba(239,68,68,0.15)", c: "#ef4444" };
    }
    // Stable / neutral — muted style matching Report History neutral pills
    return { bg: "rgba(37,99,255,0.15)", c: "#2563ff" };
  };

  // Header labels reflect the user's context:
  //   • Latest selected → left = "Previous", right = "Current"
  //   • Older selected  → left = "Current" (the chosen report they're viewing),
  //                       right = "Latest" (what it's being compared against)
  const leftHeaderLabel  = isLatestSelected ? "Previous" : "Current";
  const rightHeaderLabel = isLatestSelected ? "Current"  : "Latest";

  const leftDate  = new Date(leftReport.created_at).toLocaleDateString("en-US",  { month: "short", day: "numeric", year: "numeric" });
  const rightDate = new Date(rightReport.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });

  return (
    <div className="dashv2-card">
      <div className="dashv2-card-header">
        <span className="dashv2-card-title">📊 Severity &amp; Vitals Comparison</span>
      </div>

      <div className="dashv2-comparison-grid">
        <div className="dashv2-comparison-grid-head">
          <div className="dashv2-comp-head dashv2-comp-head-param">Parameter</div>
          <div className="dashv2-comp-head dashv2-comp-head-center">
            {leftHeaderLabel} ({leftDate})
          </div>
          <div className="dashv2-comp-head dashv2-comp-head-center dashv2-comp-head-current">
            <span className="dashv2-comp-head-flow">→</span>
            <span>{rightHeaderLabel} ({rightDate})</span>
          </div>
          <div className="dashv2-comp-head dashv2-comp-head-center">Change</div>
          <div className="dashv2-comp-head dashv2-comp-head-center">Status</div>
        </div>

        <div className="dashv2-comparison-grid-body">
          {cappedRows.map((r, i) => {
            const pill = getStatusPill(r.status);
            const changeColor = getChangeColor(r.status);

            return (
              <div key={i} className="dashv2-comparison-grid-row">
                <div className="dashv2-comp-param">{r.param}</div>
                <div className="dashv2-comp-prev">{r.prev}</div>
                <div className="dashv2-comp-cur">{r.cur}</div>
                <div className="dashv2-comp-change" style={{ color: changeColor }}>{r.change}</div>
                <div className="dashv2-comp-status-cell">
                  <span className="dashv2-comp-status-badge" style={{ background: pill.bg, color: pill.c }}>
                    {r.status}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {!hasVitalRows && (
        <p className="dashv2-analysis-paragraph" style={{ marginTop: "10px" }}>
          Vitals comparison will appear here once vitals data is available across reports.
        </p>
      )}

      <div className="dashv2-comparison-caption">
        {isLatestSelected
          ? "Current report compared with the immediately previous report."
          : "Selected report compared with the latest report."}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   CHAT PANEL
   ═══════════════════════════════════════════════════════════════════ */

function Chat({ dash, report, isOpen, onClose }) {
  const MX = 7;
  const [msgs, sm] = useState([]);
  const [inp, si] = useState("");
  const [sug, ss] = useState([]);
  const [trn, st] = useState(0);
  const [snd, sn] = useState(false);
  const [init, sInit] = useState(true);
  const [limitReached, sLimit] = useState(false);
  const br = useRef(null);
  const loadedRef = useRef(null);

  useEffect(() => {
    br.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs]);

  const jid = report?.job_id || report?.id;

  // Fetch chat state (prior messages, turn count, suggestions) whenever
  // the panel opens for this report. The component itself stays mounted
  // between opens (only opacity/pointer-events toggle), so relying on
  // mount-time fetch alone would leave `trn`/messages stale after the
  // user closes and reopens without switching reports. Refetch keys off
  // (jid, isOpen) so re-opening always pulls the latest server truth.
  useEffect(() => {
    if (!jid || !isOpen) return;
    const fetchKey = `${jid}:${isOpen}`;
    if (loadedRef.current === fetchKey) return;

    let cancelled = false;
    sInit(true);

    (async () => {
      try {
        const resp = await fetch(`/queue/chat/${encodeURIComponent(jid)}/init`, {
          credentials: "include",
        });
        if (!resp.ok) throw new Error("init failed");
        const data = await resp.json();
        if (cancelled) return;
        sm((data.messages || []).map((m) => ({ role: m.role, content: m.content })));
        st(data.turn || 0);
        ss(data.suggested_questions || []);
        sLimit(Boolean(data.limit_reached));
        loadedRef.current = fetchKey;
      } catch {
        if (!cancelled) {
          ss([
            "What are the most critical factors in my report?",
            "What should I do next?",
            "Is my condition getting worse?",
          ]);
          loadedRef.current = fetchKey;
        }
      } finally {
        if (!cancelled) sInit(false);
      }
    })();

    return () => { cancelled = true; };
  }, [jid, isOpen]);

  // Math.max keeps tl from ever going negative if the server and
  // client counters briefly disagree (mirrors the report-page panel)
  const tl = Math.max(0, MX - trn);
  const un = dash?.user?.display_name?.split(" ")[0] || "there";

  const send = async (override) => {
    const t = (override ?? inp).trim();
    if (!t || tl <= 0 || snd) return;

    sm((prev) => [...prev, { role: "user", content: t }]);
    si("");
    ss([]); // clear stale suggestions immediately while the new answer loads
    sn(true);

    try {
      if (!jid) throw new Error("No selected report is available for chat.");

      const resp = await fetch("/queue/chat", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: jid, message: t }),
      });
      const data = await resp.json().catch(() => ({}));

      if (resp.ok) {
        sm((prev) => [...prev, { role: "assistant", content: data.answer, severityDelta: data.severity_delta }]);
        // Trust the server's turn count but never regress — protects
        // against out-of-order responses (fast repeated sends) leaving
        // the badge showing a lower count than turns actually used.
        st((prev) => Math.max(prev, data.turn || 0));
        ss(data.suggested_questions || []);
        if ((data.turn || 0) >= MX) {
          sLimit(true);
        }
      } else {
        sm((prev) => [...prev, { role: "assistant", content: data.detail || "I couldn't process that. Please try again." }]);
      }
    } catch {
      sm((prev) => [...prev, { role: "assistant", content: "The follow-up service is currently unavailable." }]);
    } finally {
      sn(false);
    }
  };

  return (
    <div className="dashv2-card dashv2-chat-panel-card">
      <div className="dashv2-chat-panel-header">
        <div className="dashv2-chat-panel-header-left">
          <div className="dashv2-chat-panel-header-text">
            <span className="dashv2-chat-panel-header-title">Health Assistant</span>
            <span className="dashv2-chat-panel-header-sub">Powered by your data</span>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span
            className="dashv2-card-badge"
            style={{
              background: tl <= 1 ? "rgba(239,68,68,0.15)" : "rgba(37,99,255,0.15)",
              color: tl <= 1 ? "#ef4444" : "#2563ff",
            }}
          >
            {tl}/{MX}
          </span>
          <button className="dashv2-chat-panel-close" onClick={onClose}>
            ✕
          </button>
        </div>
      </div>

      <div className="dashv2-chat-panel-messages">
        {init && msgs.length === 0 && (
          <p className="dashv2-chat-panel-placeholder">Loading report context…</p>
        )}

        {!init && msgs.length === 0 && limitReached && (
          <p className="dashv2-chat-panel-placeholder">
            You've used all {MX} follow-up questions for this report. Start a new assessment if your
            symptoms or clinical information changed.
          </p>
        )}

        {!init && msgs.length === 0 && !limitReached && (
          <p className="dashv2-chat-panel-placeholder">Hello{un !== "there" ? ` ${un}` : ""}. I can see your clinical data. Ask about your health picture.</p>
        )}

        {msgs.map((m, i) => (
          <div key={i} className={`dashv2-chat-panel-msg dashv2-chat-panel-msg-${m.role}`}>
            <div className="dashv2-chat-panel-bubble">
              {m.content}
              {m.severityDelta && m.severityDelta !== "unchanged" && (
                <span className={`dashv2-chat-panel-sev-tag dashv2-chat-panel-sev-${m.severityDelta}`}>
                  Severity: {m.severityDelta}
                </span>
              )}
            </div>
          </div>
        ))}

        {snd && (
          <div className="dashv2-chat-panel-msg dashv2-chat-panel-msg-assistant">
            <div className="dashv2-chat-panel-bubble dashv2-chat-panel-typing">
              <span className="dashv2-chat-panel-dot" />
              <span className="dashv2-chat-panel-dot" />
              <span className="dashv2-chat-panel-dot" />
            </div>
          </div>
        )}

        <div ref={br} />
      </div>

      {sug.length > 0 && !limitReached && (
        <div className="dashv2-chat-panel-suggested">
          {sug.map((q, i) => (
            <button
              key={i}
              className="dashv2-chat-panel-suggested-btn"
              onClick={() => send(q)}
              disabled={snd || tl <= 0}
            >
              {q}
            </button>
          ))}
        </div>
      )}

      <div className="dashv2-chat-panel-input-row">
        <textarea
          className="dashv2-chat-panel-input"
          placeholder={tl > 0 ? "Ask about your health..." : "Limit reached"}
          value={inp}
          onChange={(e) => si(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          disabled={tl <= 0 || snd}
          rows={1}
        />

        <button className="dashv2-chat-panel-send" onClick={() => send()} disabled={!inp.trim() || tl <= 0 || snd}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </div>
    </div>
  );
}