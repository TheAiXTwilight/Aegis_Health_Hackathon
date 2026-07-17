import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getRecords, getRecord } from '../../services/api';
import './KeyInsights.css';

/**
 * KeyInsights — conditionally renders:
 *   - First report:  Single-Purpose Personalized Report
 *   - 2nd+ report:   Double-Purpose Comparative Report (current vs previous)
 *
 * Props:
 *   - jobId:          current job ID
 *   - recordId:       current health record ID (from result)
 *   - reportText:     the report markdown text
 *   - resultData:     structured result payload
 *   - onClose:        callback to close the panel
 */
export default function KeyInsights({ jobId, recordId, reportText, resultData, onClose }) {
  const navigate = useNavigate();
  const [previousRecords, setPreviousRecords] = useState([]);
  const [previousRecord, setPreviousRecord] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchHistory() {
      try {
        const data = await getRecords();
        const records = data.records || data || [];
        // Exclude current record
        const prev = records.filter((r) => r.id !== recordId && r.job_id !== jobId);
        setPreviousRecords(prev);

        // If there's at least one previous record, fetch the most recent one
        if (prev.length > 0) {
          const latestPrev = prev[0]; // already sorted desc by backend
          const fullRecord = await getRecord(latestPrev.id);
          setPreviousRecord(fullRecord);
        }
      } catch (err) {
        console.error('Failed to load previous records for comparison:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchHistory();
  }, [recordId, jobId]);

  if (loading) {
    return (
      <div className="key-insights-overlay">
        <div className="key-insights-card">
          <div className="ki-loading">Loading insights...</div>
        </div>
      </div>
    );
  }

  const isFirstReport = previousRecords.length === 0;

  return (
    <div className="key-insights-overlay" onClick={onClose}>
      <div className="key-insights-card" onClick={(e) => e.stopPropagation()}>
        <div className="ki-header">
          <div className="ki-title-group">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2563ff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2l2.4 7.6 7.6 2.4-7.6 2.4L12 22l-2.4-7.6L2 12l7.6-2.4L12 2z" />
            </svg>
            <h2>{isFirstReport ? 'Personalised Health Insights' : 'Comparative Health Insights'}</h2>
          </div>
          <button className="ki-close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="ki-badge-row">
          <span className={`ki-badge ${isFirstReport ? 'ki-badge-first' : 'ki-badge-compare'}`}>
            {isFirstReport ? '🌟 First Report — Baseline' : '📊 Comparative — Current vs Previous'}
          </span>
          <span className="ki-badge ki-badge-severity">
            Severity: <strong>{resultData?.report?.severity || resultData?.severity || 'N/A'}</strong>
          </span>
        </div>

        {isFirstReport ? (
          <SinglePurposeInsights reportText={reportText} resultData={resultData} />
        ) : (
          <DoublePurposeInsights
            reportText={reportText}
            resultData={resultData}
            previousRecord={previousRecord}
          />
        )}
      </div>
    </div>
  );
}

// ── Single-Purpose (First Report) ──────────────────────────────────
function SinglePurposeInsights({ reportText, resultData }) {
  const severity = resultData?.report?.severity || resultData?.severity || 'LOW';
  const confidence = resultData?.report?.confidence || resultData?.confidence || 0;
  const validationStatus = resultData?.report?.validation_status || resultData?.rule_validator_result?.status;
  const executionPlan = resultData?.execution_plan;
  const suggestedQuestions = resultData?.suggested_questions || [];

  const severityColor = severity === 'HIGH' ? '#ef4444' : severity === 'MEDIUM' || severity === 'MODERATE' ? '#f59e0b' : '#10b981';

  return (
    <div className="ki-body">
      <div className="ki-section ki-hero-section">
        <div className="ki-severity-ring" style={{ borderColor: severityColor }}>
          <span className="ki-sev-label" style={{ color: severityColor }}>{severity}</span>
          <span className="ki-sev-sub">severity</span>
        </div>
        <div className="ki-hero-details">
          <h3>Your Baseline Health Assessment</h3>
          <p>This is your first health report. It establishes your personal baseline. Future reports will be compared against this to track changes.</p>
          <div className="ki-stat-row">
            <div className="ki-stat">
              <span className="ki-stat-value">{Math.round(confidence * 100)}%</span>
              <span className="ki-stat-label">Confidence</span>
            </div>
            <div className="ki-stat">
              <span className="ki-stat-value">{executionPlan ? (executionPlan.use_rag ? '✓' : '✗') : '—'}</span>
              <span className="ki-stat-label">RAG Evidence</span>
            </div>
            <div className="ki-stat">
              <span className="ki-stat-value">{validationStatus || '—'}</span>
              <span className="ki-stat-label">Validation</span>
            </div>
          </div>
        </div>
      </div>

      {validationStatus === 'override' && (
        <div className="ki-alert ki-alert-red">
          <strong>⚠ Safety Override:</strong> Deterministic clinical rules require HIGH severity. Structured severity is authoritative.
        </div>
      )}
      {validationStatus === 'warning' && (
        <div className="ki-alert ki-alert-amber">
          <strong>⚠ Review Warning:</strong> Minor divergence detected between rule-based and narrative severity.
        </div>
      )}

      <div className="ki-section">
        <h4>📋 Key Findings</h4>
        <KeyFindings resultData={resultData} />
      </div>

      <div className="ki-section">
        <h4>🧭 Plan Summary</h4>
        <PlanChips executionPlan={executionPlan} resultData={resultData} />
      </div>

      {suggestedQuestions.length > 0 && (
        <div className="ki-section">
          <h4>💬 Suggested Follow-up Questions</h4>
          <div className="ki-suggested-questions">
            {suggestedQuestions.map((q, i) => (
              <span key={i} className="ki-sq-chip">{q}</span>
            ))}
          </div>
        </div>
      )}

      <div className="ki-section ki-next-steps">
        <h4>📌 Next Steps</h4>
        <ul>
          <li>Review this report with a qualified healthcare professional</li>
          <li>Submit daily vitals to build your health trend</li>
          <li>Your next report will show a comparative view against this baseline</li>
        </ul>
      </div>
    </div>
  );
}

// ── Double-Purpose (2nd+ Report) ───────────────────────────────────
function DoublePurposeInsights({ reportText, resultData, previousRecord }) {
  const currentSeverity = resultData?.report?.severity || resultData?.severity || 'LOW';
  const prevSeverity = previousRecord?.severity || 'N/A';
  const currentConfidence = resultData?.report?.confidence || resultData?.confidence || 0;
  const prevConfidence = previousRecord?.confidence || 0;
  const validationStatus = resultData?.report?.validation_status || resultData?.rule_validator_result?.status;

  const sevOrder = { LOW: 1, MEDIUM: 2, MODERATE: 2, HIGH: 3 };
  const severityDelta = (sevOrder[currentSeverity] || 0) - (sevOrder[prevSeverity] || 0);
  const deltaLabel = severityDelta > 0 ? '↑ Increased' : severityDelta < 0 ? '↓ Decreased' : '→ Unchanged';
  const deltaColor = severityDelta > 0 ? '#ef4444' : severityDelta < 0 ? '#10b981' : '#64748b';

  return (
    <div className="ki-body">
      <div className="ki-section ki-comparison-hero">
        <div className="ki-compare-col ki-compare-current">
          <span className="ki-compare-tag">Current Report</span>
          <span className="ki-compare-sev" style={{ color: severityColor(currentSeverity) }}>{currentSeverity}</span>
          <span className="ki-compare-conf">Confidence: {Math.round(currentConfidence * 100)}%</span>
        </div>
        <div className="ki-compare-arrow">
          <span className="ki-delta-badge" style={{ color: deltaColor, borderColor: deltaColor }}>
            {deltaLabel}
          </span>
        </div>
        <div className="ki-compare-col ki-compare-previous">
          <span className="ki-compare-tag">Previous Report</span>
          <span className="ki-compare-sev" style={{ color: severityColor(prevSeverity) }}>{prevSeverity}</span>
          <span className="ki-compare-conf">Confidence: {Math.round(prevConfidence * 100)}%</span>
        </div>
      </div>

      {validationStatus === 'override' && (
        <div className="ki-alert ki-alert-red">
          <strong>⚠ Safety Override:</strong> Deterministic clinical rules require HIGH severity. Structured severity is authoritative.
        </div>
      )}

      <div className="ki-section">
        <h4>📋 Current Key Findings</h4>
        <KeyFindings resultData={resultData} />
      </div>

      {previousRecord && (
        <div className="ki-section">
          <h4>📋 Previous Findings</h4>
          <PreviousFindings previousRecord={previousRecord} />
        </div>
      )}

      <div className="ki-section">
        <h4>🔄 What Changed</h4>
        <ChangeSummary
          currentSeverity={currentSeverity}
          prevSeverity={prevSeverity}
          severityDelta={severityDelta}
          resultData={resultData}
          previousRecord={previousRecord}
        />
      </div>

      <div className="ki-section ki-next-steps">
        <h4>📌 Recommended Actions</h4>
        <ul>
          {severityDelta > 0 && <li style={{ color: '#ef4444', fontWeight: 700 }}>Severity has increased — seek clinical review promptly</li>}
          {severityDelta === 0 && <li>Severity is unchanged — continue monitoring and follow-up as advised</li>}
          {severityDelta < 0 && <li style={{ color: '#10b981' }}>Severity has decreased — continue current care plan and monitor</li>}
          <li>Compare your vitals trends between reports on the Dashboard</li>
          <li>Share this comparative report with your healthcare provider</li>
        </ul>
      </div>
    </div>
  );
}

function severityColor(sev) {
  if (sev === 'HIGH') return '#ef4444';
  if (sev === 'MEDIUM' || sev === 'MODERATE') return '#f59e0b';
  return '#10b981';
}

// ── Shared sub-components ──────────────────────────────────────────

function KeyFindings({ resultData }) {
  const items = [];
  if (resultData?.symptom_result?.symptoms?.length) {
    items.push({ label: 'Symptoms', values: resultData.symptom_result.symptoms });
  }
  if (resultData?.lab_result?.abnormal_values?.length) {
    items.push({ label: 'Lab Abnormalities', values: resultData.lab_result.abnormal_values });
  }
  if (resultData?.xray_result?.findings?.length) {
    items.push({ label: 'X-Ray Findings', values: resultData.xray_result.findings });
  }
  if (resultData?.drug_result?.interactions?.length) {
    items.push({ label: 'Drug Interactions', values: resultData.drug_result.interactions.map(i => i.description) });
  }
  if (items.length === 0) {
    return <p className="ki-empty">No structured findings extracted.</p>;
  }
  return (
    <div className="ki-findings-grid">
      {items.map((item) => (
        <div key={item.label} className="ki-finding-group">
          <span className="ki-fg-label">{item.label}</span>
          <div className="ki-fg-values">
            {item.values.slice(0, 5).map((v, i) => (
              <span key={i} className="ki-fg-chip">{typeof v === 'string' ? v : JSON.stringify(v)}</span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function PreviousFindings({ previousRecord }) {
  const prevData = previousRecord?.result_json ? (typeof previousRecord.result_json === 'string' ? JSON.parse(previousRecord.result_json) : previousRecord.result_json) : {};
  const symptoms = previousRecord?.symptoms_text;
  const medications = previousRecord?.medications_json ? (typeof previousRecord.medications_json === 'string' ? JSON.parse(previousRecord.medications_json) : previousRecord.medications_json) : [];
  const xrayFindings = previousRecord?.xray_findings_json ? (typeof previousRecord.xray_findings_json === 'string' ? JSON.parse(previousRecord.xray_findings_json) : previousRecord.xray_findings_json) : [];

  return (
    <div className="ki-findings-grid">
      {symptoms && (
        <div className="ki-finding-group">
          <span className="ki-fg-label">Symptoms</span>
          <span className="ki-fg-chip">{symptoms}</span>
        </div>
      )}
      {medications.length > 0 && (
        <div className="ki-finding-group">
          <span className="ki-fg-label">Medications</span>
          <div className="ki-fg-values">
            {medications.map((m, i) => <span key={i} className="ki-fg-chip">{m}</span>)}
          </div>
        </div>
      )}
      {xrayFindings.length > 0 && (
        <div className="ki-finding-group">
          <span className="ki-fg-label">X-Ray Findings</span>
          <div className="ki-fg-values">
            {xrayFindings.map((f, i) => <span key={i} className="ki-fg-chip">{f}</span>)}
          </div>
        </div>
      )}
    </div>
  );
}

function PlanChips({ executionPlan, resultData }) {
  const plan = executionPlan || resultData?.execution_plan;
  if (!plan) return <p className="ki-empty">No execution plan available.</p>;

  return (
    <div className="ki-plan-chips">
      <div className="ki-plan-row">
        <span className="ki-plan-label">RAG Evidence:</span>
        <span className={`ki-chip ${plan.use_rag ? 'ki-chip-on' : 'ki-chip-off'}`}>{plan.use_rag ? '✓ Enabled' : '✗ Disabled'}</span>
      </div>
      {plan.was_repaired && <span className="ki-chip ki-chip-repair">[REPAIRED] Forced RAG</span>}
      {plan.is_fallback && <span className="ki-chip ki-chip-fallback">[FALLBACK] Planner Fallback</span>}
      {plan.reasoning && <p className="ki-plan-reasoning">{plan.reasoning}</p>}
    </div>
  );
}

function ChangeSummary({ currentSeverity, prevSeverity, severityDelta, resultData, previousRecord }) {
  const changes = [];

  if (severityDelta !== 0) {
    changes.push(`Severity changed from ${prevSeverity} to ${currentSeverity}.`);
  }

  const currentSymptoms = resultData?.symptom_result?.symptoms || [];
  const prevSymptoms = previousRecord?.symptoms_text?.split(', ') || [];
  const newSymptoms = currentSymptoms.filter(s => !prevSymptoms.includes(s));
  if (newSymptoms.length > 0) {
    changes.push(`New symptoms detected: ${newSymptoms.join(', ')}.`);
  }

  if (changes.length === 0) {
    changes.push('No significant changes detected between reports.');
  }

  return (
    <ul className="ki-change-list">
      {changes.map((c, i) => <li key={i}>{c}</li>)}
    </ul>
  );
}
