import React, { useState, useEffect, useRef, useCallback } from "react";
import { useSearchParams, useNavigate, Navigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import {
  deleteJob,
  getJobResult,
  getJobStatus,
  getRecords,
  streamReport,
  synthesizeSpeech,
  synthesizeSpeechStream,
} from "../../services/api";
import "./Report.css";

function openHtmlPreview(html) {
  const iframe = document.createElement("iframe");
  iframe.title = "Aegis Health PDF Generator";
  iframe.setAttribute("aria-hidden", "true");
  iframe.style.position = "fixed";
  iframe.style.left = "-10000px";
  iframe.style.top = "0";
  iframe.style.width = "900px";
  iframe.style.height = "1200px";
  iframe.style.opacity = "0";
  iframe.style.pointerEvents = "none";
  iframe.style.border = "0";

  document.body.appendChild(iframe);

  const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
  if (!iframeDoc) {
    iframe.remove();
    alert("Could not prepare the PDF download. Please try again.");
    return null;
  }

  iframeDoc.open();
  iframeDoc.write(html);
  iframeDoc.close();

  window.setTimeout(() => { if (iframe.parentNode) iframe.remove(); }, 120000);

  return iframe.contentWindow || null;
}

function buildReportHtml(reportText, jobId, resultData) {
  const rendered = renderMarkdown(reportText, {
    heatmapUrl:
      resultData?.xray_result?.heatmap_url ||
      (resultData?.submitted?.xray_image_uploaded ? `/queue/heatmap/${jobId}` : null),
    findingText: resultData?.xray_result?.findings?.[0] || "Detected Finding",
    isPdf: true,
    resultData,
  });
  const safeJobId = (jobId || "report").replace(/[^a-zA-Z0-9_-]/g, "_");
  const fileName = `aegis-health-report-${safeJobId}.pdf`;

  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Aegis Health Report</title>
  <style>
    @page { margin: 15mm; size: A4 portrait; }
    #pdf-download-bar { display: none !important; }
    @media print { #pdf-download-bar { display: none !important; } }
    * { box-sizing: border-box; letter-spacing: normal !important; word-break: normal !important; -webkit-font-smoothing: antialiased; }
    body { font-family: Arial, "Helvetica Neue", Helvetica, sans-serif !important; color: #0d2167; line-height: 1.65; font-size: 14px; width: 700px; max-width: 100%; margin: 0 auto; padding: 0; background: #fff; }
    #report-root { width: 700px; margin: 0 auto; padding: 24px; font-family: Arial, "Helvetica Neue", Helvetica, sans-serif !important; font-size: 14px; line-height: 1.65; }
    .header { border-bottom: 2px solid #2563ff; padding-bottom: 16px; margin-bottom: 28px; }
    .brand { font-size: 26px; font-weight: bold; color: #0d2167; line-height: 1.35; margin-bottom: 8px; }
    .meta { font-size: 13px; color: #425894; line-height: 1.5; }
    main { display: block; }
    .report-section { display: block; margin-bottom: 28px; padding: 0; page-break-inside: auto !important; break-inside: auto !important; }
    .report-section:last-child { margin-bottom: 0; }
    .heading-keep { page-break-inside: avoid !important; break-inside: avoid-page !important; }
    h1, h2, h3, h4 { font-family: Arial, "Helvetica Neue", Helvetica, sans-serif !important; color: #0d2167; font-weight: bold; line-height: 1.4; margin-top: 0; margin-bottom: 12px; page-break-after: avoid !important; break-after: avoid-page !important; }
    h1 { font-size: 22px; }
    h2 { font-size: 18px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }
    h3 { font-size: 16px; }
    h4 { font-size: 14px; }
    p { margin: 10px 0; color: #152b70; font-size: 14px; line-height: 1.65; }
    ul, ol { margin: 6px 0 14px 0; padding-left: 22px; }
    li { margin: 4px 0; color: #2e4378; font-size: 14px; line-height: 1.6; }
    ul > li:first-child, ol > li:first-child { margin-top: 0 !important; }
    ul > li:last-child, ol > li:last-child { margin-bottom: 0 !important; }
    .vital-name { font-weight: 500; color: #0d2167; }
    .vital-value { font-weight: 500; color: #0d2167; font-variant-numeric: tabular-nums; }
    .vital-unit { color: #6b7ba8; font-size: 13px; font-weight: 400; }
    .vital-status { font-weight: 500; color: #4a5b8c; }
    .vital-range { color: #7a8bb5; font-size: 13px; font-weight: 400; }
    .vital-sep { color: #8b9cc0; font-weight: 400; padding: 0 10px; }
    .vital-status-word { font-weight: 600; color: #0d2167; text-transform: uppercase; letter-spacing: 0.3px; font-size: 12.5px; }
    .care-plan-subhead { font-weight: 700; color: #0d2167; font-size: 13.5px; letter-spacing: 0.3px; text-transform: uppercase; }
    strong { color: #0d2167; font-weight: bold; }
    code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 13px; font-family: monospace; }
    .disclaimer { margin-top: 32px; font-size: 12px; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 14px; page-break-inside: avoid !important; break-inside: avoid !important; }
    #pdf-download-bar { position: fixed; top: 0; left: 0; right: 0; background: #0d2167; color: white; padding: 10px 20px; display: flex; align-items: center; justify-content: center; gap: 16px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 13px; z-index: 9999; box-shadow: 0 2px 12px rgba(0,0,0,0.15); }
    #pdf-download-bar button { background: #2563ff; color: white; border: none; padding: 8px 18px; border-radius: 8px; font-weight: 700; font-size: 13px; cursor: pointer; }
    #pdf-download-bar button:hover { background: #1d4ed8; }
    #pdf-download-bar button:disabled { opacity: 0.6; cursor: wait; }
    @media print { body { padding-top: 0 !important; } #pdf-download-bar { display: none !important; } }
  </style>
</head>
<body>
  <div id="pdf-download-bar">
    <span id="pdf-status">HTML report ready – PDF will download automatically…</span>
    <button id="pdf-dl-btn">Download PDF</button>
  </div>
  <div id="report-root">
    <div class="header">
      <div class="brand">AegisHealth Report</div>
      <div class="meta">Job ID: ${String(jobId || "N/A").replace(/</g, "&lt;")} · Generated: ${new Date().toLocaleString()}</div>
    </div>
    <main>${rendered}</main>
    <div class="disclaimer">All processing is done locally. Your data never leaves your device.</div>
  </div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
  <script>
    (function(){
      const btn = document.getElementById('pdf-dl-btn');
      const status = document.getElementById('pdf-status');
      const filename = ${JSON.stringify(fileName)};
      async function doPdf(){
        try {
          if(btn){ btn.disabled = true; btn.textContent = 'Generating PDF…'; }
          if(status) status.textContent = 'Generating PDF, please wait…';
          const source = document.getElementById('report-root');
          const opt = {
            margin: [12, 14, 12, 14], filename: filename, image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2, useCORS: true, letterRendering: true, backgroundColor: '#ffffff', scrollX: 0, scrollY: 0,
              onclone: (clonedDoc) => {
                const bar = clonedDoc.getElementById('pdf-download-bar');
                if (bar) bar.remove();
                clonedDoc.body.style.cssText = 'margin:0;padding:0;background:#fff;font-family:Arial,sans-serif;width:700px;';
                const root = clonedDoc.getElementById('report-root');
                if (root) { root.style.cssText = 'width:700px;padding:24px;box-sizing:border-box;margin:0 auto;float:none;columns:1;font-family:Arial,sans-serif;font-size:14px;line-height:1.65;'; }
              }
            },
            jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
            pagebreak: { mode: ['css', 'legacy'], avoid: ['.heading-keep', 'h1', 'h2', 'h3', 'h4', 'li', '.pdf-heatmap-box'] }
          };
          const pdfBlob = await html2pdf().set(opt).from(source).outputPdf('blob');
          const url = URL.createObjectURL(pdfBlob);
          const a = document.createElement('a');
          a.href = url; a.download = filename;
          document.body.appendChild(a); a.click(); a.remove();
          setTimeout(() => URL.revokeObjectURL(url), 60000);
          if(status) status.textContent = 'PDF downloaded ✓ – closing this tab…';
          if(btn){ btn.textContent = 'PDF Downloaded ✓'; btn.disabled = true; }
          setTimeout(() => window.close(), 1800);
        } catch(e) {
          console.error(e);
          if(status) status.textContent = 'PDF generation failed – try again or use Print → Save as PDF';
          if(btn){ btn.textContent = 'Retry Download'; btn.disabled = false; }
          alert('PDF generation failed: ' + (e.message || e));
        }
      }
      if(btn) btn.onclick = doPdf;
      window.addEventListener('load', () => setTimeout(doPdf, 500));
    })();
  </script>
</body>
</html>`;
}

function openReportPreview(reportText, jobId, resultData) {
  if (!reportText) return null;
  const html = buildReportHtml(reportText, jobId, resultData);
  return openHtmlPreview(html);
}

const PIPELINE_STEPS = [
  { label: "Input Analysis", toolKey: "input" },
  { label: "Voice Transcription", toolKey: "voice" },
  { label: "Lab / X-Ray Analysis", toolKey: "medical" },
  { label: "Evidence & Drug Check", toolKey: "evidence" },
  { label: "Severity Assessment", toolKey: "severity" },
  { label: "Report Generation", toolKey: "report" },
  { label: "Validation", toolKey: "validation" },
];

const TOOL_GROUPS = {
  input: ["ExecutionPlanner", "SymptomExtractor"],
  voice: ["VoiceTranscriber"],
  medical: ["LabReportParser", "XRayProcessor"],
  evidence: ["MedicalRAGSearch", "DrugInteractionChecker"],
  severity: ["SeverityScorer"],
  report: ["ReportGenerator"],
  validation: ["RuleValidator"],
};

function getStepStatus(toolsRun, toolsFailed, currentTool, toolKeys, optionalFailedTools = [], jobStatus = "queued") {
  const anyRunning = toolKeys.some((t) => currentTool === t);
  const anyAttempted = toolKeys.some((t) => toolsRun.includes(t) || toolsFailed.includes(t) || currentTool === t);
  const failedRequiredTools = toolKeys.filter((t) => toolsFailed.includes(t) && !optionalFailedTools.includes(t));
  const anyFailedRequired = failedRequiredTools.length > 0;
  const anyOptionalFailed = toolKeys.some((t) => toolsFailed.includes(t) && optionalFailedTools.includes(t));

  const jobFinished = jobStatus === "completed" || jobStatus === "failed";
  const relevantTools = jobFinished
    ? toolKeys.filter((t) => toolsRun.includes(t) || toolsFailed.includes(t))
    : toolKeys.filter((t) => !optionalFailedTools.includes(t));
  const allRequiredDone = jobFinished
    ? relevantTools.every((t) => toolsRun.includes(t)) && !anyFailedRequired
    : relevantTools.every((t) => toolsRun.includes(t));

  if (anyRunning) return "active";
  if (anyFailedRequired) return "failed";
  if (allRequiredDone || anyOptionalFailed) return "done";
  if (jobFinished && !anyAttempted) return "skipped";
  if (jobFinished) return "done";
  return "pending";
}

function shouldShowStep(toolKeys, toolsRun, toolsFailed, currentTool, optionalFailedTools = []) {
  if (toolKeys.includes("ExecutionPlanner") && !toolKeys.includes("VoiceTranscriber")) return true;
  const visibleFailedTools = toolKeys.filter((t) => toolsFailed.includes(t) && !optionalFailedTools.includes(t));
  return toolKeys.some((t) => toolsRun.includes(t) || currentTool === t) || visibleFailedTools.length > 0;
}

function recordToHistoryItem(record) {
  const createdAt = new Date(record.created_at);
  const validDate = !Number.isNaN(createdAt.getTime());
  const reportText = record.report_text || record.result_data?.report?.text || record.report_data?.text || "";
  return {
    recordId: record.id, jobId: record.job_id,
    name: validDate ? `Report - ${createdAt.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}` : "Report",
    date: validDate ? createdAt.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" }) : "",
    time: validDate ? createdAt.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }) : "",
    status: record.status || "completed", reportText, resultData: record.result_data || null,
  };
}

export default function ReportPage() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [searchParams] = useSearchParams();
  const jobId = searchParams.get("jobId");

  const [statusPayload, setStatusPayload] = useState(null);
  const [reportText, setReportText] = useState("");
  const [resultData, setResultData] = useState(null);
  const [error, setError] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const reportRef = useRef(null);
  const pollingRef = useRef(null);
  const streamAbortRef = useRef(null);
  const streamStartedRef = useRef(false);
  const reportTextRef = useRef("");

  const [openMenuIndex, setOpenMenuIndex] = useState(null);
  const menuRef = useRef(null);
  const dropdownMenuRef = useRef(null);

  const [reportHistoryList, setReportHistoryList] = useState([]);
  const [selectedHistoryItem, setSelectedHistoryItem] = useState(null);
  const [currentHistoryItem, setCurrentHistoryItem] = useState(null);
  const [isReadingOut, setIsReadingOut] = useState(false);
  const [isPreparingReadout, setIsPreparingReadout] = useState(false);
  const [menuCoords, setMenuCoords] = useState(null);
  const ttsAudioRef = useRef(null);
  const ttsObjectUrlRef = useRef(null);
  // Holds the stop() function for the active streaming TTS session.
  // Null when not streaming (non-streaming fallback uses ttsAudioRef).
  const ttsStreamStopRef = useRef(null);
  const [forceDeleteTick, setForceDeleteTick] = useState(0);
  // True once we've resolved whether history has any reports to fall back to.
  // Prevents flashing "No Active Analysis" while history is still loading.
  const [historyLoaded, setHistoryLoaded] = useState(false);

  const loadReportHistory = useCallback(async () => {
    try {
      const data = await getRecords();
      const persistedItems = (data.records || []).map(recordToHistoryItem);
      setReportHistoryList((previousItems) => {
        const currentTransient = previousItems.find((item) => item.jobId === jobId && !persistedItems.some((persisted) => persisted.jobId === item.jobId));
        return currentTransient ? [currentTransient, ...persistedItems] : persistedItems;
      });
    } catch (historyError) {
      console.error("Failed to load report history:", historyError);
      setReportHistoryList([]);
    } finally {
      setHistoryLoaded(true);
    }
  }, [jobId]);

  useEffect(() => { if (user?.id) loadReportHistory(); }, [user?.id, loadReportHistory]);

  // Stop all playback and release resources on unmount so audio does
  // not keep playing after navigating away from the report page.
  useEffect(() => {
    return () => {
      if (ttsStreamStopRef.current) {
        try { ttsStreamStopRef.current(); } catch {}
        ttsStreamStopRef.current = null;
      }
      if (ttsAudioRef.current) {
        ttsAudioRef.current.pause();
        ttsAudioRef.current = null;
      }
      if (ttsObjectUrlRef.current) {
        URL.revokeObjectURL(ttsObjectUrlRef.current);
        ttsObjectUrlRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    function handleClickOutside(event) {
      const clickedHistory = menuRef.current?.contains(event.target);
      const clickedDropdown = dropdownMenuRef.current?.contains(event.target);
      if (!clickedHistory && !clickedDropdown) { setOpenMenuIndex(null); setMenuCoords(null); }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const toggleMenu = (index, event) => {
    if (openMenuIndex === index) {
      setOpenMenuIndex(null);
      setMenuCoords(null);
      return;
    }
    const btn = event?.currentTarget || event?.target;
    const rect = btn?.getBoundingClientRect();
    setOpenMenuIndex(index);
    if (!rect) return;

    const FIRST_ITEM_CENTER_OFFSET = 26;
    const MENU_WIDTH_ESTIMATE = 190;
    const MENU_HEIGHT_ESTIMATE = 4 * 40 + 3 * 4 + 12;

    const dotsCenterY = rect.top + rect.height / 2;
    let top = dotsCenterY - FIRST_ITEM_CENTER_OFFSET;
    let left = rect.right + 8;

    const viewportH = window.innerHeight;
    const viewportW = window.innerWidth;
    if (top + MENU_HEIGHT_ESTIMATE > viewportH - 8) {
      top = Math.max(8, viewportH - MENU_HEIGHT_ESTIMATE - 8);
    }
    if (top < 8) top = 8;
    if (left + MENU_WIDTH_ESTIMATE > viewportW - 8) {
      left = rect.left - MENU_WIDTH_ESTIMATE - 8;
    }
    if (left < 8) left = 8;

    setMenuCoords({ top, left });
  };

  const startStreaming = useCallback(async () => {
    if (!jobId) return;
    setIsStreaming(true);
    try {
      const body = await streamReport(jobId);
      if (!body) { setError("Stream unavailable."); setIsStreaming(false); return; }
      const reader = body.getReader();
      const decoder = new TextDecoder();
      streamAbortRef.current = reader;
      let fullText = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        fullText += chunk;
        reportTextRef.current = fullText;
        setReportText(fullText);
      }
      const finalPayload = await getJobStatus(jobId);
      setStatusPayload(finalPayload);
      if (finalPayload?.status === "completed") {
        const result = await getJobResult(jobId);
        setResultData(result);
        if (result?.report?.text) { reportTextRef.current = result.report.text; setReportText(result.report.text); }
      }
    } catch (err) {
      streamStartedRef.current = false;
      if (err.name !== "AbortError") { console.error("Stream error:", err); setError("Failed to stream report."); }
    } finally { setIsStreaming(false); streamAbortRef.current = null; }
  }, [jobId]);

  const pollStatus = useCallback(async () => {
    if (!jobId) return;
    try {
      const payload = await getJobStatus(jobId);
      if (!payload) { setError("Job not found. It may have expired."); return; }
      setStatusPayload(payload);
      const status = payload.status;
      if (status === "completed") {
        if (pollingRef.current) clearTimeout(pollingRef.current);
        setIsStreaming(false);
        if (streamAbortRef.current) { try { await streamAbortRef.current.cancel(); } catch {} streamAbortRef.current = null; }
        try {
          const result = await getJobResult(jobId);
          setResultData(result);
          if (result?.report?.text) { reportTextRef.current = result.report.text; setReportText(result.report.text); setError(null); }
          else if (!reportTextRef.current) setError("Report completed, but no report text was returned.");
        } catch (resultErr) {
          console.error("Result fetch error:", resultErr);
          if (!reportTextRef.current) setError("Report completed, but the result could not be loaded.");
        }
        return status;
      }
      if (status === "running" && !streamStartedRef.current) { streamStartedRef.current = true; startStreaming(); }
      if (status === "failed") { if (pollingRef.current) clearTimeout(pollingRef.current); setIsStreaming(false); setError(payload.error || "Pipeline failed."); }
      return status;
    } catch (err) { console.error("Poll error:", err); return null; }
  }, [jobId, startStreaming]);

  useEffect(() => {
    if (!jobId) return;
    let isMounted = true;
    let timerId = null;
    async function loop() {
      if (!isMounted) return;
      const currentStatus = await pollStatus();
      if (!isMounted) return;
      if (currentStatus === "completed" || currentStatus === "failed") return;
      const delay = streamStartedRef.current || currentStatus === "running" ? 1500 : 600;
      timerId = setTimeout(loop, delay);
      pollingRef.current = timerId;
    }
    loop();
    return () => {
      isMounted = false;
      streamStartedRef.current = false;
      if (timerId) clearTimeout(timerId);
      if (pollingRef.current) clearTimeout(pollingRef.current);
      if (streamAbortRef.current) { try { streamAbortRef.current.cancel(); } catch {} }
    };
  }, [jobId, pollStatus]);

  useEffect(() => {
    const currentStatus = statusPayload?.status || "queued";
    if (!reportRef.current) return;
    if (currentStatus === "completed") { reportRef.current.scrollTop = 0; return; }
    if (isStreaming) reportRef.current.scrollTop = reportRef.current.scrollHeight;
  }, [reportText, isStreaming, statusPayload?.status]);

  const toolsRun = statusPayload?.tools_run || [];
  const toolsFailed = statusPayload?.tools_failed || [];
  const currentTool = statusPayload?.current_tool || null;
  const jobStatus = jobId ? statusPayload?.status || "queued" : "idle";

  useEffect(() => {
    let indicatorStatus = "idle";
    if (!jobId) indicatorStatus = "idle";
    else if (jobStatus === "queued" || jobStatus === "running") indicatorStatus = "running";
    else if (jobStatus === "completed") indicatorStatus = "completed";
    else if (jobStatus === "failed") indicatorStatus = "failed";
    localStorage.setItem("aegis_pipeline_status", indicatorStatus);
    window.dispatchEvent(new CustomEvent("aegis:pipeline-status", { detail: { status: indicatorStatus } }));
  }, [jobId, jobStatus]);

  const optionalFailedTools = [];
  if (toolsFailed.includes("VoiceTranscriber") && toolsRun.includes("SymptomExtractor")) optionalFailedTools.push("VoiceTranscriber");
  if (toolsFailed.includes("ExecutionPlanner") && toolsRun.includes("SymptomExtractor")) optionalFailedTools.push("ExecutionPlanner");
  if (toolsFailed.includes("MedicalRAGSearch") && toolsRun.includes("SeverityScorer")) optionalFailedTools.push("MedicalRAGSearch");

  const visibleSteps = PIPELINE_STEPS.filter((step) => {
    const keys = TOOL_GROUPS[step.toolKey];
    return shouldShowStep(keys, toolsRun, toolsFailed, currentTool, optionalFailedTools);
  });

  useEffect(() => {
    if (!jobId || !statusPayload) { setCurrentHistoryItem(null); return; }
    const item = {
      jobId: jobId,
      name: `Report - ${new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}`,
      date: new Date().toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" }),
      time: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
      status: statusPayload.status,
    };
    setCurrentHistoryItem((prev) => (prev?.jobId === item.jobId ? prev : item));
    setReportHistoryList((prev) => {
      const filtered = prev.filter((p) => p.jobId !== item.jobId);
      const existing = prev.find((p) => p.jobId === item.jobId);
      const updatedItem = { ...item, reportText: reportText || existing?.reportText || "", resultData: resultData || existing?.resultData || null };
      return [updatedItem, ...filtered].slice(0, 25);
    });
  }, [jobId, statusPayload?.status, reportText, resultData]);

  const fallbackHistoryItem = !jobId && reportHistoryList.length > 0 ? reportHistoryList[0] : null;

  const activeHistoryItem = selectedHistoryItem || currentHistoryItem || fallbackHistoryItem;
  const activeJobId = activeHistoryItem?.jobId || jobId;
  const activeReportText = activeHistoryItem && activeHistoryItem.jobId !== jobId ? activeHistoryItem.reportText || "" : reportText;
  const activeResultData = activeHistoryItem && activeHistoryItem.jobId !== jobId ? activeHistoryItem.resultData : resultData;
  const activeJobStatus = activeHistoryItem && activeHistoryItem.jobId !== jobId ? activeHistoryItem.status : jobStatus;

  // Stop all active TTS playback (streaming or non-streaming).
  const stopReadout = useCallback(() => {
    if (ttsStreamStopRef.current) {
      try { ttsStreamStopRef.current(); } catch {}
      ttsStreamStopRef.current = null;
    }
    if (ttsAudioRef.current) {
      ttsAudioRef.current.pause();
      ttsAudioRef.current.currentTime = 0;
    }
    setIsReadingOut(false);
    setIsPreparingReadout(false);
  }, []);

  const toggleVoiceReadout = async () => {
    const textToSpeak = activeReportText || reportText;
    if (!textToSpeak) return;

    if (isReadingOut || isPreparingReadout) {
      stopReadout();
      return;
    }

    setIsPreparingReadout(true);

    // ── Streaming path (primary) ──────────────────────────────────
    // synthesizeSpeechStream() constructs and resumes its
    // AudioContext synchronously at the very top of the function,
    // before any await, satisfying browser autoplay policy. Playback
    // begins after the first synthesized segment (~100-300 ms) rather
    // than after the full report is synthesized.
    let streamHandle = null;
    try {
      streamHandle = await synthesizeSpeechStream(textToSpeak, activeJobId);

      ttsStreamStopRef.current = streamHandle.stop;
      setIsPreparingReadout(false);
      setIsReadingOut(true);

      // Reset UI when the stream finishes — either naturally, or
      // early via stop(), or without ever scheduling audio (server
      // sent header only). The allEndedPromise contract guarantees
      // exactly one resolution.
      streamHandle.allEndedPromise.then(() => {
        if (ttsStreamStopRef.current === streamHandle.stop) {
          setIsReadingOut(false);
          ttsStreamStopRef.current = null;
        }
      });

      return;
    } catch (streamErr) {
      // Clean up any partial stream state.
      if (ttsStreamStopRef.current) {
        try { ttsStreamStopRef.current(); } catch {}
        ttsStreamStopRef.current = null;
      }
      // Force UI back to idle in case setIsReadingOut(true) already ran.
      setIsReadingOut(false);

      // Hard failures the non-streaming fallback cannot fix either —
      // surface them immediately.
      if (streamErr?.reason === 'model_missing') {
        setIsPreparingReadout(false);
        alert(
          "Voice readout isn't set up on this server yet. " +
          "The text-to-speech voice model needs to be installed " +
          "(see backend/tts.py setup instructions)."
        );
        console.error("Voice readout failed:", streamErr);
        return;
      }
      if (streamErr?.reason === 'text_too_long') {
        setIsPreparingReadout(false);
        alert("This report is too long to read aloud in one go.");
        console.error("Voice readout failed:", streamErr);
        return;
      }
      if (streamErr?.reason === 'autoplay_blocked') {
        setIsPreparingReadout(false);
        alert("Please click the Voice TTS button again to start playback.");
        console.warn(
          "Voice readout blocked by browser autoplay policy:",
          streamErr
        );
        return;
      }

      // Transient or network failure — fall through to the
      // non-streaming path so the user still gets audio.
      console.warn(
        "Streaming TTS failed, falling back to non-streaming:",
        streamErr
      );
    }

    // ── Non-streaming fallback ────────────────────────────────────
    // Behaviour is identical to the original implementation before
    // streaming was added. Used when the /speak/stream endpoint is
    // unavailable or fails transiently.
    try {
      const audioBlob = await synthesizeSpeech(textToSpeak, activeJobId);

      if (ttsObjectUrlRef.current) {
        URL.revokeObjectURL(ttsObjectUrlRef.current);
      }
      const objectUrl = URL.createObjectURL(audioBlob);
      ttsObjectUrlRef.current = objectUrl;

      const audio = new Audio(objectUrl);
      ttsAudioRef.current = audio;
      audio.onended = () => {
        setIsReadingOut(false);
      };
      audio.onerror = () => {
        setIsReadingOut(false);
        setIsPreparingReadout(false);
        alert("Playback failed. Please try again.");
      };

      setIsPreparingReadout(false);
      setIsReadingOut(true);
      try {
        await audio.play();
      } catch (playErr) {
        if (playErr?.name === "NotAllowedError") {
          // Browser autoplay policy: play() must run inside the same
          // trusted user-gesture as the click, with no `await` in
          // between. By the time synthesizeSpeech()'s network call
          // resolves, the browser no longer treats this play() as
          // gesture-triggered and blocks it — even though synthesis
          // itself succeeded and `audio` is fully loaded and ready.
          // Reset to the idle state; the button's next click is a
          // fresh gesture against the already-ready audio, so it
          // plays instantly with no re-fetch.
          setIsReadingOut(false);
          setIsPreparingReadout(false);
          return;
        }
        throw playErr;
      }
    } catch (err) {
      setIsPreparingReadout(false);
      setIsReadingOut(false);
      if (err?.reason === "model_missing") {
        alert(
          "Voice readout isn't set up on this server yet. " +
          "The text-to-speech voice model needs to be installed " +
          "(see backend/tts.py setup instructions)."
        );
      } else if (err?.reason === "text_too_long") {
        alert("This report is too long to read aloud in one go.");
      } else if (err?.reason === "timeout") {
        alert("Voice readout is taking too long and was cancelled. Please try again.");
      } else if (err?.name === "NotAllowedError") {
        // Browser autoplay policy: play() must run inside the same trusted
        // user-gesture as the click, with no `await` in between. By the
        // time synthesizeSpeech()'s network call resolves, the browser no
        // longer treats the resulting play() as gesture-triggered and
        // blocks it. The audio is already fetched and cached in
        // ttsAudioRef at this point, so a second click plays instantly
        // and *is* a fresh gesture — it will succeed.
        alert("Tap the readout button again to start playback.");
      } else {
        alert("Voice readout failed. Please try again.");
      }
      console.error("Voice readout failed:", err);
    }
  };

  const handleKeyInsights = (item) => {
    setOpenMenuIndex(null); setMenuCoords(null);
    sessionStorage.setItem("aegis_dashboard_unlocked", "true");
    sessionStorage.setItem("aegis_dashboard_job_id", item.jobId);
    window.dispatchEvent(new Event("aegis:dashboard-unlocked"));
    const syncQuery = item.jobId === jobId ? "&sync=1" : "";
    navigate(`/dashboard?jobId=${encodeURIComponent(item.jobId)}${syncQuery}`);
  };

  const handleDownloadReport = (item) => {
    setOpenMenuIndex(null); setMenuCoords(null);
    const downloadWindow = window.open(`/export/pdf/${encodeURIComponent(item.jobId)}`, "_blank");
    if (!downloadWindow && item.reportText) openReportPreview(item.reportText, item.jobId, item.resultData || activeResultData);
  };

  const handleFhirExport = (item) => {
    setOpenMenuIndex(null); setMenuCoords(null);
    const identifier = item.recordId || item.jobId;
    window.open(`/export/fhir/${encodeURIComponent(identifier)}`, "_blank");
  };

  const handleDeleteReport = async (item) => {
    const targetJobId = item.jobId;
    setOpenMenuIndex(null); setMenuCoords(null); setError(null);
    try {
      await deleteJob(targetJobId);

      const remainingHistory = reportHistoryList.filter((h) => h.jobId !== targetJobId);
      setReportHistoryList(remainingHistory);

      setSelectedHistoryItem((prev) => (prev?.jobId === targetJobId ? null : prev));
      if (currentHistoryItem?.jobId === targetJobId || jobId === targetJobId) {
        setCurrentHistoryItem(null);
        setReportText("");
        setResultData(null);
        setStatusPayload(null);
        if (jobId === targetJobId) {
          navigate("/report", { replace: true });
        }
      }
      setForceDeleteTick((tick) => tick + 1);
    } catch (deleteError) {
      console.error("Report deletion failed:", deleteError);
      setError(deleteError.message || "Unable to delete this report. Please try again.");
    }
  };

  if (!jobId && !fallbackHistoryItem) {
    if (!historyLoaded) {
      return (
        <div className="form-center-container">
          <div className="report-dashboard-card">
            <div className="report-column" style={{ flex: 1, textAlign: "center", padding: "40px" }}>
              <h2>Loading…</h2>
            </div>
          </div>
        </div>
      );
    }
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <div className="form-center-container report-page-stack">
      <div className="report-dual-cards-wrapper" style={{ display: "flex", gap: "26px", width: "100%", maxWidth: "calc(100vw - 140px)", height: "calc(100vh - 220px)", maxHeight: "none", alignItems: "stretch", justifyContent: "center", position: "relative", zIndex: 5 }}>
        <div className="report-glass-card report-left-card" style={{ flex: "1 1 0%", width: "50%", height: "100%", padding: "26px 30px", background: "rgba(255, 255, 255, 0.12)", backdropFilter: "blur(24px) saturate(1.5)", WebkitBackdropFilter: "blur(24px) saturate(1.5)", border: "1px solid rgba(255, 255, 255, 0.3)", boxShadow: "0 8px 32px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.5)", borderRadius: "38px", display: "flex", flexDirection: "row", gap: "16px", overflow: "hidden" }}>
          <div className="report-column pipeline-col" style={{ flex: "0 0 270px", width: "270px", display: "flex", flexDirection: "column", overflowY: "auto", paddingRight: "8px", paddingLeft: "6px" }}>
            <h2 style={{ fontSize: "19px", marginBottom: "16px", paddingLeft: "6px" }}>Live Pipeline</h2>
            <div className="pipeline-container" style={{ paddingLeft: "12px", paddingTop: "6px", paddingBottom: "6px" }}>
              {visibleSteps.map((step, index) => {
                const rawKeys = TOOL_GROUPS[step.toolKey];
                const attemptedKeys = rawKeys.filter((t) => toolsRun.includes(t) || toolsFailed.includes(t) || currentTool === t);
                const keys = step.toolKey === "medical" && attemptedKeys.length > 0 ? attemptedKeys : rawKeys;
                const stepStatus = getStepStatus(toolsRun, toolsFailed, currentTool, keys, optionalFailedTools, activeJobStatus);
                return (
                  <div key={step.toolKey} className="pipeline-row">
                    <div className="node-wrapper">
                      <div className={`pipeline-node ${stepStatus}`}>
                        {stepStatus === "done" && <span className="check-mark">✓</span>}
                        {stepStatus === "failed" && (<span className="check-mark" style={{ color: "#ef4444" }}>✕</span>)}
                        {stepStatus === "active" && <div className="active-core"></div>}
                      </div>
                      {index < visibleSteps.length - 1 && (<div className={`pipeline-line ${stepStatus === "done" ? "done" : stepStatus === "failed" ? "failed" : ""}`}></div>)}
                    </div>
                    <div className="step-info">
                      <h3>{index + 1}. {step.label}</h3>
                      <span className={`status-label ${stepStatus}`}>
                        {stepStatus === "done" ? "Completed" : stepStatus === "failed" ? "Failed" : stepStatus === "active" ? "In Progress" : stepStatus === "skipped" ? "Skipped" : "Pending"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
            <div style={{ marginTop: "12px", padding: "10px", borderRadius: "10px", background: "rgba(255,255,255,0.1)" }}>
              <div style={{ fontSize: "11px", color: "#425894", fontWeight: 600 }}>
                Status:{" "}
                <strong style={{ color: activeJobStatus === "completed" ? "#10b981" : activeJobStatus === "failed" ? "#ef4444" : "#2563ff" }}>
                  {activeJobStatus?.toUpperCase() || "PENDING"}
                </strong>
              </div>
              {statusPayload?.queue_position && (
                <div style={{ fontSize: "11px", color: "#7182b1", marginTop: "4px" }}>
                  Queue: #{statusPayload.queue_position}
                  {statusPayload.estimated_wait_seconds ? ` (~${Math.round(statusPayload.estimated_wait_seconds)}s)` : ""}
                </div>
              )}
            </div>
          </div>

          <div className="report-column history-col" style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", borderLeft: "1px solid rgba(255, 255, 255, 0.2)", paddingLeft: "16px" }}>
            <h2 className="report-history-heading" style={{ fontSize: "19px", margin: 0, height: "58px", display: "flex", alignItems: "center", paddingBottom: "16px", borderBottom: "1px solid rgba(255, 255, 255, 0.2)", marginBottom: "16px", boxSizing: "border-box" }}>Report History</h2>
            <div className="history-container" ref={menuRef} key={`history-${forceDeleteTick}`} style={{ overflowY: "auto", flex: 1, display: "flex", flexDirection: "column", gap: "10px", paddingRight: "4px" }}>
              {(reportHistoryList.length > 0 ? reportHistoryList : currentHistoryItem ? [currentHistoryItem] : []).map((item, index) => {
                const isSelected = selectedHistoryItem ? selectedHistoryItem.jobId === item.jobId : item.jobId === jobId;
                return (
                  <div key={item.jobId} className={`history-row-item ${openMenuIndex === index ? "active-row" : ""}`}
                    style={{ zIndex: openMenuIndex === index ? 50 : 1, border: isSelected ? "1px solid rgba(37, 99, 255, 0.6)" : undefined, background: isSelected ? "rgba(255, 255, 255, 0.18)" : undefined, cursor: "pointer", height: "auto", minHeight: "66px", padding: "10px 14px", width: "100%" }}
                    onClick={() => setSelectedHistoryItem(item)}>
                    <div className="history-left-item" style={{ gap: "10px", flex: 1, minWidth: 0 }}>
                      <div className="file-doc-icon-wrapper" style={{ flexShrink: 0 }}>
                        <svg className="file-doc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                          <polyline points="14 2 14 8 20 8"></polyline>
                          <line x1="16" y1="13" x2="8" y2="13"></line>
                          <line x1="16" y1="17" x2="8" y2="17"></line>
                          <polyline points="10 9 9 9 8 9"></polyline>
                        </svg>
                      </div>
                      <div className="report-text-stack" style={{ flex: 1, minWidth: 0, paddingRight: "6px" }}>
                        <span className="report-title-text" style={{ fontSize: "13px", fontWeight: 700, display: "block" }}>{item.name}</span>
                        <span className="report-sub-text" style={{ fontSize: "11px", whiteSpace: "normal", lineHeight: "1.35", display: "block" }}>
                          {item.date} • {item.time} •{" "}
                          <span style={{ color: item.status === "completed" ? "#10b981" : item.status === "failed" ? "#ef4444" : "#2563ff", fontWeight: 700 }}>
                            {item.status?.toUpperCase() || "QUEUED"}
                          </span>
                        </span>
                      </div>
                    </div>
                    <div className="action-menu-container" style={{ flexShrink: 0 }} onClick={(e) => e.stopPropagation()}>
                      <button className="dots-btn" onClick={(e) => toggleMenu(index, e)} title="Options">
                        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <circle cx="12" cy="12" r="1"></circle>
                          <circle cx="12" cy="5" r="1"></circle>
                          <circle cx="12" cy="19" r="1"></circle>
                        </svg>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {openMenuIndex !== null && menuCoords && (() => {
          const historyList = reportHistoryList.length > 0 ? reportHistoryList : currentHistoryItem ? [currentHistoryItem] : [];
          const item = historyList[openMenuIndex];
          if (!item) return null;
          return (
            <div ref={dropdownMenuRef} className="dropdown-menu" style={{ position: "fixed", top: `${menuCoords.top}px`, left: `${menuCoords.left}px`, zIndex: 99999, background: "rgba(255, 255, 255, 0.22)", backdropFilter: "blur(28px) saturate(1.8)", WebkitBackdropFilter: "blur(28px) saturate(1.8)", border: "1px solid rgba(255, 255, 255, 0.55)", borderRadius: "16px", boxShadow: "0 12px 36px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.6)", padding: "6px", display: "flex", flexDirection: "column", gap: "4px" }} onClick={(e) => e.stopPropagation()}>
              <button className="dropdown-item" onClick={() => handleKeyInsights(item)} disabled={item.status !== "completed"}>
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2l2.4 7.6 7.6 2.4-7.6 2.4L12 22l-2.4-7.6L2 12l7.6-2.4L12 2z"></path></svg>
                Key Insights
              </button>
              <button className="dropdown-item" onClick={() => handleDownloadReport(item)} disabled={item.status !== "completed"}>
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="7 10 12 15 17 10"></polyline>
                  <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                Download Report
              </button>
              <button className="dropdown-item" onClick={() => handleDeleteReport(item)} style={{ color: "#0d2167" }}>
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="3 6 5 6 21 6"></polyline>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                </svg>
                Delete
              </button>
              <button className="dropdown-item" onClick={() => handleFhirExport(item)} disabled={item.status !== "completed"}>
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                  <path d="M9 13h6M9 17h6"></path>
                </svg>
                FHIR Export
              </button>
            </div>
          );
        })()}

        <div className="report-glass-card report-right-card" style={{ flex: "1 1 0%", width: "50%", height: "100%", padding: "26px 34px", background: "rgba(255, 255, 255, 0.12)", backdropFilter: "blur(24px) saturate(1.5)", WebkitBackdropFilter: "blur(24px) saturate(1.5)", border: "1px solid rgba(255, 255, 255, 0.3)", boxShadow: "0 8px 32px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.5)", borderRadius: "38px", display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div className="right-card-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", height: "58px", boxSizing: "border-box", paddingBottom: "16px", borderBottom: "1px solid rgba(255, 255, 255, 0.2)", marginBottom: "16px", flexShrink: 0 }}>
            <div>
              <h2 style={{ color: "#0d2167", fontSize: "22px", fontWeight: 800, margin: 0 }}>{activeHistoryItem?.name || "Report Overview"}</h2>
              <span style={{ fontSize: "12px", color: "#425894" }}>
                Job ID: {String(activeJobId || "N/A").replace(/</g, "&lt;")} · Status:{" "}
                <strong style={{ color: activeJobStatus === "completed" ? "#10b981" : activeJobStatus === "failed" ? "#ef4444" : "#2563ff" }}>{activeJobStatus?.toUpperCase()}</strong>
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <button
                onClick={() => navigate("/dashboard")}
                style={{
                  background: "rgba(255, 255, 255, 0.2)",
                  border: "1px solid rgba(255, 255, 255, 0.4)",
                  borderRadius: "14px",
                  padding: "10px 14px",
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  cursor: "pointer",
                  color: "#0d2167",
                  fontWeight: 700,
                  fontSize: "13px",
                  transition: "all 0.2s ease",
                }}
                title="Key Insights Dashboard"
              >
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="7" height="9"></rect>
                  <rect x="14" y="3" width="7" height="5"></rect>
                  <rect x="14" y="12" width="7" height="9"></rect>
                  <rect x="3" y="16" width="7" height="5"></rect>
                </svg>
                <span>Key Insights</span>
              </button>
              <button onClick={toggleVoiceReadout} disabled={!activeReportText || isPreparingReadout} style={{ background: isReadingOut ? "rgba(37, 99, 255, 0.25)" : "rgba(255, 255, 255, 0.2)", border: `1px solid ${isReadingOut ? "#2563ff" : "rgba(255, 255, 255, 0.4)"}`, borderRadius: "14px", padding: "10px 14px", display: "flex", alignItems: "center", gap: "8px", cursor: (activeReportText && !isPreparingReadout) ? "pointer" : "not-allowed", color: isReadingOut ? "#2563ff" : "#0d2167", fontWeight: 700, fontSize: "13px", transition: "all 0.2s ease" }} title={isReadingOut ? "Stop Voice Readout" : "Start Voice Readout"}>
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                  {isReadingOut ? (<><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></>) : (<path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>)}
                </svg>
                <span>{isReadingOut ? "Stop Readout" : isPreparingReadout ? "Preparing..." : "Voice TTS"}</span>
              </button>
            </div>
          </div>

          <div className="right-card-fixed-top" style={{ display: "flex", flexDirection: "column", gap: "14px", marginBottom: "20px", flexShrink: 0 }}>
            {(() => {
              const valStatus = activeResultData?.report?.validation_status || activeResultData?.rule_validator_result?.status || statusPayload?.rule_validator_result?.status;
              if (!valStatus) return null;
              const isOverride = valStatus === "override";
              const isWarning = valStatus === "warning";
              const isAgreement = valStatus === "agreement";
              if (!isOverride && !isWarning && !isAgreement) return null;
              const bg = isOverride ? "rgba(239, 68, 68, 0.15)" : isWarning ? "rgba(245, 158, 11, 0.15)" : "rgba(16, 185, 129, 0.15)";
              const border = isOverride ? "rgba(239, 68, 68, 0.4)" : isWarning ? "rgba(245, 158, 11, 0.4)" : "rgba(16, 185, 129, 0.4)";
              const color = isOverride ? "#ef4444" : isWarning ? "#f59e0b" : "#10b981";
              const icon = isOverride ? "⚠" : isWarning ? "⚠" : "✓";
              const title = isOverride ? "SAFETY OVERRIDE: Clinical Rules Overrode AI Severity" : isWarning ? "SAFETY WARNING: Minor Divergence Detected" : "SAFETY VALIDATION: AI & Clinical Rules in Full Agreement";
              const desc = isOverride ? "Deterministic clinical safety rules required HIGH/CRITICAL severity, but the AI narrative expressed a lower tier. Our deterministic rules remain authoritative. Urgent clinician review required." : isWarning ? "Minor discrepancies detected between deterministic rule-based scoring and AI narrative assessment. Clinician review advised before treatment." : "Deterministic safety rules and AI triage evaluation match across all clinical biomarkers and reported symptoms.";
              return (
                <div style={{ background: bg, border: `1px solid ${border}`, borderRadius: "16px", padding: "16px 20px", display: "flex", flexDirection: "column", gap: "6px", backdropFilter: "blur(12px)", flexShrink: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", fontWeight: 800, fontSize: "14px", color: color }}><span style={{ fontSize: "16px" }}>{icon}</span><span>{title}</span></div>
                  <p style={{ margin: 0, fontSize: "13px", color: "#0d2167", lineHeight: "1.45" }}>{desc}</p>
                </div>
              );
            })()}

            {(activeResultData?.execution_plan || statusPayload?.execution_plan || activeResultData?.report?.execution_plan_summary) && (
              <div style={{ background: "rgba(255, 255, 255, 0.12)", border: "1px solid rgba(255, 255, 255, 0.25)", borderRadius: "16px", padding: "16px 20px", display: "flex", flexDirection: "column", gap: "10px", backdropFilter: "blur(12px)", flexShrink: 0 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "8px" }}>
                  <span style={{ fontSize: "12px", fontWeight: 800, color: "#0d2167", textTransform: "uppercase", letterSpacing: "0.5px" }}>Agentic Plan Theater & Tool Audit</span>
                  <div style={{ display: "flex", gap: "6px" }}>
                    {(activeResultData?.execution_plan?.was_repaired || statusPayload?.execution_plan?.was_repaired || activeResultData?.report?.execution_plan_summary?.includes("[REPAIRED]")) && (<span style={{ background: "rgba(245, 158, 11, 0.2)", color: "#d97706", border: "1px solid rgba(245, 158, 11, 0.4)", padding: "2px 10px", borderRadius: "12px", fontSize: "11px", fontWeight: 700 }}>[REPAIRED] Forced RAG</span>)}
                    {(activeResultData?.execution_plan?.is_fallback || statusPayload?.execution_plan?.is_fallback || activeResultData?.report?.execution_plan_summary?.includes("[FALLBACK]")) && (<span style={{ background: "rgba(239, 68, 68, 0.2)", color: "#ef4444", border: "1px solid rgba(239, 68, 68, 0.4)", padding: "2px 10px", borderRadius: "12px", fontSize: "11px", fontWeight: 700 }}>[FALLBACK] Planner Fallback</span>)}
                  </div>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", alignItems: "center" }}>
                  <span style={{ fontSize: "11px", color: "#4b5b8c", fontWeight: 600, marginRight: "4px" }}>Mandatory:</span>
                  {[
                    { label: "VoiceTranscriber", active: Boolean(activeResultData?.submitted?.audio_uploaded || statusPayload?.tools_run?.includes("VoiceTranscriber")) },
                    { label: "SymptomExtractor", active: Boolean(activeResultData?.submitted?.symptoms_text || statusPayload?.tools_run?.includes("SymptomExtractor")) },
                    { label: "LabReportParser", active: Boolean(activeResultData?.submitted?.lab_pdf_uploaded || statusPayload?.tools_run?.includes("LabReportParser")) },
                    { label: "XRayProcessor", active: Boolean(activeResultData?.submitted?.xray_image_uploaded || (activeResultData?.submitted?.xray_findings && activeResultData.submitted.xray_findings.length > 0) || statusPayload?.tools_run?.includes("XRayProcessor")) },
                    { label: "DrugInteractionChecker", active: Boolean((activeResultData?.submitted?.medications && activeResultData.submitted.medications.length > 0) || statusPayload?.tools_run?.includes("DrugInteractionChecker")) },
                  ].map((tool) => (
                    <span key={tool.label} style={{ background: tool.active ? "rgba(16, 185, 129, 0.15)" : "rgba(255, 255, 255, 0.1)", color: tool.active ? "#10b981" : "#64748b", border: `1px solid ${tool.active ? "rgba(16, 185, 129, 0.3)" : "rgba(255, 255, 255, 0.15)"}`, padding: "3px 10px", borderRadius: "12px", fontSize: "11px", fontWeight: 600, display: "flex", alignItems: "center", gap: "4px" }}>
                      <span>{tool.active ? "✓" : "✗"}</span> {tool.label}
                    </span>
                  ))}
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", alignItems: "center" }}>
                  <span style={{ fontSize: "11px", color: "#4b5b8c", fontWeight: 600, marginRight: "4px" }}>Optional:</span>
                  <span style={{ background: (activeResultData?.execution_plan?.use_rag || statusPayload?.execution_plan?.use_rag) ? "rgba(37, 99, 255, 0.15)" : "rgba(255, 255, 255, 0.1)", color: (activeResultData?.execution_plan?.use_rag || statusPayload?.execution_plan?.use_rag) ? "#2563ff" : "#64748b", border: `1px solid ${(activeResultData?.execution_plan?.use_rag || statusPayload?.execution_plan?.use_rag) ? "rgba(37, 99, 255, 0.3)" : "rgba(255, 255, 255, 0.15)"}`, padding: "3px 10px", borderRadius: "12px", fontSize: "11px", fontWeight: 600, display: "flex", alignItems: "center", gap: "4px" }}>
                    <span>{(activeResultData?.execution_plan?.use_rag || statusPayload?.execution_plan?.use_rag) ? "✓" : "✗"}</span> MedicalRAGSearch
                  </span>
                </div>
                {(activeResultData?.execution_plan?.reasoning || statusPayload?.execution_plan?.reasoning) && (
                  <div style={{ marginTop: "4px", paddingTop: "10px", borderTop: "1px dashed rgba(255, 255, 255, 0.2)", fontSize: "12px", color: "#0d2167", fontStyle: "italic", lineHeight: "1.4" }}>
                    <strong style={{ fontStyle: "normal", color: "#4b5b8c" }}>Planner Reasoning:</strong>{" "}
                    {activeResultData?.execution_plan?.reasoning || statusPayload?.execution_plan?.reasoning}
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="right-card-scrollable-preview" style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: "20px", paddingRight: "6px", paddingTop: "4px", paddingBottom: "12px" }}>
            {error && (
              <div className="report-error" style={{ padding: "20px", borderRadius: "16px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: "#ef4444", fontSize: "14px", fontWeight: 600, flexShrink: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="15" y1="9" x2="9" y2="15"></line>
                    <line x1="9" y1="9" x2="15" y2="15"></line>
                  </svg>
                  <span>{error}</span>
                </div>
              </div>
            )}
            {activeJobStatus === "queued" && !error && (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "40px 20px", gap: "12px", flexShrink: 0 }}>
                <div className="rec-spinner" style={{ width: "24px", height: "24px", borderWidth: "2px" }}></div>
                <p style={{ color: "#425894", fontSize: "15px", fontWeight: 600 }}>Your job is queued...</p>
                {statusPayload?.queue_position && (
                  <p style={{ color: "#7182b1", fontSize: "13px" }}>
                    Position: #{statusPayload.queue_position}
                    {statusPayload.estimated_wait_seconds ? ` (~${Math.round(statusPayload.estimated_wait_seconds)}s wait)` : ""}
                  </p>
                )}
              </div>
            )}
            {isStreaming && activeJobStatus !== "completed" && !activeReportText && (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "40px 20px", gap: "16px", flexShrink: 0 }}>
                <div className="rec-spinner" style={{ width: "32px", height: "32px", borderWidth: "3px" }}></div>
                <p style={{ color: "#425894", fontSize: "16px", fontWeight: 600 }}>Generating your health report...</p>
              </div>
            )}
            {(activeResultData?.cached_result || activeResultData?.report?.cached_result || statusPayload?.cached_result) && (
              <div style={{ background: "rgba(16, 185, 129, 0.15)", border: "1px solid rgba(16, 185, 129, 0.35)", borderRadius: "14px", padding: "10px 16px", display: "flex", alignItems: "center", gap: "8px", color: "#10b981", fontSize: "12px", fontWeight: 700, flexShrink: 0 }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
                <span>⚡ INSTANT RESULT CACHE HIT • Rehydrated from 128-entry LRU memory vault (0s SLM inference)</span>
              </div>
            )}
            {activeReportText && (
              <div style={{ fontSize: "12px", fontWeight: 800, color: "#425894", textTransform: "uppercase", letterSpacing: "0.8px", flexShrink: 0 }}>Patient Preview</div>
            )}
            {activeReportText && (
              <div ref={reportRef} className="report-text-container" style={{ padding: "28px 32px", borderRadius: "24px", background: "rgba(255, 255, 255, 0.12)", border: "1px solid rgba(255, 255, 255, 0.25)", color: "#0d2167", fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', fontSize: "15px", lineHeight: "1.7", flexShrink: 0 }}
                dangerouslySetInnerHTML={{
                  __html: renderMarkdown(activeReportText, {
                    heatmapUrl: activeResultData?.xray_result?.heatmap_url || (activeResultData?.submitted?.xray_image_uploaded && activeJobStatus === "completed" ? `/queue/heatmap/${activeJobId}` : null),
                    findingText: activeResultData?.xray_result?.findings?.[0] || "Detected Finding",
                    isPdf: false,
                  }),
                }}
              />
            )}
            {isStreaming && activeReportText && activeJobStatus !== "completed" && (
              <div style={{ fontSize: "12px", color: "#7182b1", fontWeight: 500, textAlign: "center", marginTop: "8px", flexShrink: 0 }}>Streaming report...</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   RENDER MARKDOWN
   ============================================================ */

function renderMarkdown(text, options = {}) {
  if (!text) return "";

  const SPACE = {
    sectionMb: "28px", headingMt: "0px", headingMb: "12px",
    paragraphMy: "10px", listMt: "6px", listMb: "14px",
    listItemMy: "4px", listPl: "22px", heatmapMy: "28px",
  };

  function escapeHtml(value) { return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function inlineFormat(value) {
    return escapeHtml(value)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`(.+?)`/g, '<code class="report-inline-code">$1</code>');
  }

  const RAW_HTML_RE = /<!--RAW_HTML_START-->([\s\S]*?)<!--RAW_HTML_END-->/g;
  const rawBlocks = [];
  text = text.replace(RAW_HTML_RE, (_, html) => {
    rawBlocks.push(html);
    return `\n\n__RAW_HTML_TOKEN_${rawBlocks.length - 1}__\n\n`;
  });

  const lines = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");

  const sections = [];
  let current = null;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line === "-" || line === "*" || line === "•") { if (current) current.lines.push(line); continue; }
    if (line.startsWith("# ") || line.startsWith("## ") || line.startsWith("### ") || line.startsWith("#### ")) {
      if (current) sections.push(current);
      current = { heading: line, lines: [] };
    } else {
      if (!current) current = { heading: null, lines: [] };
      current.lines.push(line);
    }
  }
  if (current) sections.push(current);

  const hasDisclaimer = sections.some((sec) => sec.heading && sec.heading.toLowerCase().includes("disclaimer"));
  if (!hasDisclaimer) {
    sections.push({
      heading: "## Disclaimer",
      lines: ["Clinical decision support only — not a diagnosis. All outputs must be reviewed by a qualified healthcare professional before any clinical action is taken. Do not use in emergency situations."],
    });
  }

  function renderLines(lines, isDisclaimer = false) {
    const cleanLines = [];
    let lastWasBlank = false;
    for (const l of lines) {
      const trimmed = l.trim();
      const isBlank = !trimmed || trimmed === "-" || trimmed === "*" || trimmed === "•";
      if (isBlank) { if (!lastWasBlank) cleanLines.push(l); lastWasBlank = true; }
      else { cleanLines.push(l); lastWasBlank = false; }
    }

    let html = "";
    let listType = null;
    let liIndexInList = 0;

    function closeList() { if (listType) { html += `</${listType}>`; listType = null; liIndexInList = 0; } }

    const textColor = isDisclaimer ? "#64748b" : "#152b70";
    const pStyle = `margin: ${SPACE.paragraphMy} 0; color: ${textColor}; font-size: 15px; line-height: 1.7;`;
    function liStyleFor(isFirst) {
      const topMargin = isFirst ? "0" : SPACE.listItemMy;
      return `margin: ${topMargin} 0 ${SPACE.listItemMy} 0; color: ${textColor}; font-size: 15px; line-height: 1.65;`;
    }

    for (const line of cleanLines) {
      const rawMatch = line.match(/^__RAW_HTML_TOKEN_(\d+)__$/);
      if (rawMatch) { closeList(); html += rawBlocks[parseInt(rawMatch[1], 10)] || ""; continue; }
      const trimmed = line.trim();
      if (!trimmed || trimmed === "-" || trimmed === "*" || trimmed === "•") { closeList(); continue; }
      if (trimmed.startsWith("- ") || trimmed.startsWith("* ") || trimmed.startsWith("• ")) {
        const content = trimmed.replace(/^[-*•]\s+/, "");
        if (listType === "ol") closeList();
        if (!listType) { html += `<ul style="margin: ${SPACE.listMt} 0 ${SPACE.listMb} 0; padding-left: ${SPACE.listPl};">`; listType = "ul"; liIndexInList = 0; }
        html += `<li style="${liStyleFor(liIndexInList === 0)}">${inlineFormat(content)}</li>`;
        liIndexInList++;
        continue;
      }
      const numberedMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
      if (numberedMatch) {
        if (listType === "ul") closeList();
        if (!listType) { html += `<ol style="margin: ${SPACE.listMt} 0 ${SPACE.listMb} 0; padding-left: ${SPACE.listPl};">`; listType = "ol"; liIndexInList = 0; }
        html += `<li style="${liStyleFor(liIndexInList === 0)}">${inlineFormat(numberedMatch[2])}</li>`;
        liIndexInList++;
        continue;
      }
      closeList();
      html += `<p style="${pStyle}">${inlineFormat(trimmed)}</p>`;
    }
    closeList();
    return html;
  }

  let insertIdx = sections.findIndex((sec) => sec.heading && sec.heading.toLowerCase().includes("finding"));
  if (insertIdx === -1) insertIdx = Math.floor(sections.length / 2);
  if (insertIdx === 0 && sections.length > 1) insertIdx = 1;

  let result = "";

  for (let sIdx = 0; sIdx < sections.length; sIdx++) {
    const section = sections[sIdx];
    const isDisclaimer = Boolean(section.heading && section.heading.toLowerCase().includes("disclaimer"));
    const isLast = sIdx === sections.length - 1;
    const sectionMb = isLast ? "0px" : SPACE.sectionMb;

    if (options.heatmapUrl && sIdx === insertIdx) {
      if (options.isPdf) {
        result += `<div class="pdf-heatmap-box" style="margin: ${SPACE.heatmapMy} auto; max-width: 580px; padding: 16px; border: 1px solid #cbdcf7; border-radius: 12px; background: #f8fafc; text-align: center; page-break-inside: avoid !important; break-inside: avoid !important;">
          <div style="font-size: 14px; font-weight: bold; color: #0d2167; margin-bottom: 12px; text-transform: uppercase;">Grad-CAM X-Ray Explainability Heatmap</div>
          <img src="${options.heatmapUrl}" style="max-width: 280px; max-height: 180px; border-radius: 8px; object-fit: contain; margin: 0 auto 10px auto; display: block;" onerror="this.style.display='none';" />
          <p style="font-size: 12px; color: #425894; line-height: 1.5; margin: 0;"><strong>Saliency Overlay:</strong> Highlights structural regions and contrast gradients associated with the AI classification of ${options.findingText || "Detected Finding"}.</p>
        </div>`;
      } else {
        result += `<div class="ui-heatmap-box" style="margin: ${SPACE.heatmapMy} 0; padding: 24px; border: 1px solid rgba(255, 255, 255, 0.25); border-radius: 16px; background: rgba(0, 0, 0, 0.08); text-align: center; display: flex; flex-direction: column; align-items: center; gap: 14px;">
          <div style="width: 100%; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <span style="font-size: 13px; font-weight: 800; color: #0d2167; text-transform: uppercase; letter-spacing: 0.5px;">Grad-CAM X-Ray Explainability Heatmap</span>
            <span style="background: rgba(37, 99, 255, 0.15); color: #2563ff; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 700;">${options.findingText || "Detected Finding"}</span>
          </div>
          <div style="width: 100%; max-width: 460px; overflow: hidden; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.3); background: rgba(0,0,0,0.05); padding: 8px; display: flex; justify-content: center;">
            <img src="${options.heatmapUrl}" alt="Grad-CAM X-ray Heatmap Overlay" style="max-width: 100%; max-height: 350px; border-radius: 8px; object-fit: contain;" onerror="this.style.display='none';" />
          </div>
          <p style="font-size: 13px; color: #425894; margin: 0; max-width: 580px; line-height: 1.5;"><strong style="color: #0d2167;">Saliency Overlay:</strong> Highlights structural regions and contrast gradients most strongly associated with the AI classification of <em style="color: #2563ff; font-weight: 600;">${options.findingText || "Detected Finding"}</em>. This is a best-effort, non-fatal explainability artifact.</p>
        </div>`;
      }
    }

    if (section.heading) {
      const level = section.heading.startsWith("#### ") ? 4 : section.heading.startsWith("### ") ? 3 : section.heading.startsWith("## ") ? 2 : 1;
      const headingText = inlineFormat(section.heading.replace(/^#{1,4}\s/, ""));
      const hSize = level === 1 ? "22px" : level === 2 ? "18px" : level === 3 ? "16px" : "14px";
      const hBorder = level <= 2 ? "border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 8px;" : "";
      const contentHtml = renderLines(section.lines, isDisclaimer);
      result += `<section class="report-section" style="margin: 0 0 ${sectionMb} 0; padding: 0;">
        <div class="heading-keep">
          <h${level} style="margin: ${SPACE.headingMt} 0 ${SPACE.headingMb} 0; font-size: ${hSize}; color: #0d2167; font-weight: 800; line-height: 1.4; ${hBorder}">${headingText}</h${level}>
        </div>
        ${contentHtml}
      </section>`;
    } else {
      result += `<section class="report-section" style="margin: 0 0 ${sectionMb} 0; padding: 0;">${renderLines(section.lines, isDisclaimer)}</section>`;
    }
  }

  return result;
}