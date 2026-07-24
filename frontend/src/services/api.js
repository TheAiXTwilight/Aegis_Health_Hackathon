/**
 * frontend/src/services/api.js
 * AegisHealth Backend API Service
 *
 * Single point of interaction with the FastAPI backend.
 * All API URLs are relative (proxy handles forwarding in dev).
 *
 * Usage:
 *   import api from '../services/api';
 *   const job = await api.submitForm(formData);
 *   const status = await api.getJobStatus(jobId);
 *   const stream = api.streamReport(jobId);
 *   const init = await api.getChatInit(jobId);
 *   const reply = await api.sendChatMessage({ jobId, message });
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

// Keep the access token in memory only. The backend also sets an httpOnly
// access cookie, so existing same-origin fetches remain authenticated while
// JavaScript never persists a JWT in localStorage.
let accessToken = null;
let refreshPromise = null;

export function setAccessToken(token) {
  accessToken = token || null;
}

export function clearAccessToken() {
  accessToken = null;
}

/**
 * Build full API URL, stripping double slashes.
 */
function url(path) {
  const base = BASE_URL.replace(/\/+$/, '');
  const p = path.replace(/^\/+/, '');
  return `${base}/${p}`;
}

/**
 * Return the in-memory access token, if one has been issued this page load.
 */
export function getAccessToken() {
  return accessToken;
}

/**
 * Default request headers for JSON endpoints.
 */
function jsonHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  const token = getAccessToken();
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

/**
 * Fetch helper that attaches the access token and retries once with a
 * refreshed token when the backend returns 401.
 */
async function fetchWithAuth(input, init = {}) {
  const token = getAccessToken();
  const initWithAuth = {
    ...init,
    credentials: 'include',
    headers: {
      ...(init.headers || {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  };
  let resp = await fetch(input, initWithAuth);
  // If either the in-memory token or httpOnly access cookie expired,
  // rotate the refresh cookie once and retry the original request.
  if (resp.status === 401) {
    const refreshed = await refreshAccessToken();
    if (refreshed) {
      const refreshedToken = getAccessToken();
      if (refreshedToken) {
        initWithAuth.headers.Authorization = `Bearer ${refreshedToken}`;
      }
      resp = await fetch(input, initWithAuth);
    }
  }
  return resp;
}

/**
 * Extract a human-readable error message from a FastAPI / Pydantic response.
 */
function extractErrorMessage(data, fallback) {
  if (!data) return fallback;
  if (typeof data === 'string') return data;
  if (data.detail) {
    if (typeof data.detail === 'string') return data.detail;
    if (Array.isArray(data.detail) && data.detail.length > 0) {
      const first = data.detail[0];
      if (typeof first === 'string') return first;
      if (first && typeof first.msg === 'string') return first.msg;
      if (first && typeof first.message === 'string') return first.message;
    }
  }
  if (data.message && typeof data.message === 'string') return data.message;
  if (data.reason && typeof data.reason === 'string') return data.reason;
  return fallback;
}

/* ── Auth endpoints ──────────────────────────────────────────────── */

/**
 * POST /auth/register — create a new account.
 */
export async function register({
  email,
  password,
  displayName,
  username,
  phone,
  securityQuestion,
  securityAnswer,
}) {
  const resp = await fetch(url('auth/register'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({
      email,
      password,
      display_name: displayName,
      username: username || undefined,
      phone: phone || undefined,
      security_question: securityQuestion,
      security_answer: securityAnswer,
    }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    const err = new Error(
      extractErrorMessage(data, `Registration failed: HTTP ${resp.status}`)
    );
    err.status = resp.status;
    throw err;
  }
  return data;
}

/**
 * POST /auth/login — authenticate and store the access token.
 * The refresh token is set as an httpOnly cookie by the backend.
 */
export async function login({ email, password }) {
  const resp = await fetch(url('auth/login'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ email, password }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    const err = new Error(
      extractErrorMessage(data, `Login failed: HTTP ${resp.status}`)
    );
    err.status = resp.status;
    throw err;
  }
  if (data.access_token) {
    setAccessToken(data.access_token);
  }
  return data;
}

/**
 * POST /auth/forgot-password — request a password reset link.
 * Always returns a success-like response regardless of whether the email
 * exists, to prevent email enumeration.
 */
export async function forgotPassword({ email }) {
  const resp = await fetch(url('auth/forgot-password'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ email }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(
      extractErrorMessage(data, `Request failed: HTTP ${resp.status}`)
    );
  }
  return data;
}

/**
 * POST /auth/verify-security-answer — verify answer and get reset link.
 */
export async function verifySecurityAnswer({ email, securityAnswer }) {
  const resp = await fetch(url('auth/verify-security-answer'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ email, security_answer: securityAnswer }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(
      extractErrorMessage(data, `Verification failed: HTTP ${resp.status}`)
    );
  }
  return data;
}

/**
 * POST /auth/reset-password — reset password using a valid token.
 */
export async function resetPassword({ token, newPassword }) {
  const resp = await fetch(url('auth/reset-password'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ token, new_password: newPassword }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(
      extractErrorMessage(data, `Reset failed: HTTP ${resp.status}`)
    );
  }
  return data;
}

/**
 * POST /auth/refresh — rotate refresh cookie and get a new access token.
 * Concurrent 401 responses share one refresh request so a rotated refresh
 * token is never replayed accidentally.
 */
export async function refreshAccessToken() {
  if (refreshPromise) return refreshPromise;
  refreshPromise = (async () => {
    try {
      const resp = await fetch(url('auth/refresh'), {
        method: 'POST',
        credentials: 'include',
      });
      if (!resp.ok) {
        clearAccessToken();
        return false;
      }
      const data = await resp.json();
      if (!data.access_token) {
        clearAccessToken();
        return false;
      }
      setAccessToken(data.access_token);
      return true;
    } catch {
      clearAccessToken();
      return false;
    }
  })();
  try {
    return await refreshPromise;
  } finally {
    refreshPromise = null;
  }
}

/**
 * POST /auth/logout — revoke auth cookies and clear the in-memory token.
 */
export async function logout() {
  try {
    await fetch(url('auth/logout'), {
      method: 'POST',
      credentials: 'include',
    });
  } finally {
    clearAccessToken();
  }
}

/**
 * GET /auth/me — return the current user's profile.
 */
export async function getMe() {
  const resp = await fetchWithAuth(url('auth/me'));
  if (!resp.ok) return null;
  return resp.json();
}

/* ── Account endpoints ───────────────────────────────────────────── */

/**
 * GET /account/profile — return the authenticated user's health profile.
 */
export async function getProfile() {
  const resp = await fetchWithAuth(url('account/profile'));
  if (!resp.ok) throw new Error(`Profile fetch failed: HTTP ${resp.status}`);
  return resp.json();
}

/**
 * PUT /account/profile — create or update the authenticated health profile.
 */
export async function updateProfile(profile) {
  const resp = await fetchWithAuth(url('account/profile'), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(profile),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(
      extractErrorMessage(data, `Profile update failed: HTTP ${resp.status}`)
    );
  }
  return data;
}

/**
 * PUT /account/email — change the authenticated user's email.
 * Requires password confirmation.
 */
export async function changeEmail({ newEmail, password }) {
  const resp = await fetchWithAuth(url('account/email'), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ new_email: newEmail, password }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(
      extractErrorMessage(data, `Email change failed: HTTP ${resp.status}`)
    );
  }
  return data;
}

/* ── Health / System endpoints ────────────────────────────────────── */

/**
 * Health check — returns backend system status.
 */
export async function getHealth() {
  const resp = await fetch(url('health'));
  if (!resp.ok) throw new Error(`Health check failed: HTTP ${resp.status}`);
  return resp.json();
}

/* ── Dashboard / Records endpoints ───────────────────────────────── */

/**
 * GET /dashboard — return the authenticated user's dashboard summary.
 */
export async function getDashboard() {
  const resp = await fetchWithAuth(url('/api/dashboard'));
  if (!resp.ok) throw new Error(`Dashboard fetch failed: HTTP ${resp.status}`);
  return resp.json();
}

/**
 * GET /records — return only the authenticated user's persisted reports.
 */
export async function getRecords() {
  const resp = await fetchWithAuth(url('records'));
  if (!resp.ok)
    throw new Error(`Report history fetch failed: HTTP ${resp.status}`);
  return resp.json();
}

/**
 * GET /records/{record_id} — return one authenticated-user-owned report.
 */
export async function getRecord(recordId) {
  const resp = await fetchWithAuth(
    url(`records/${encodeURIComponent(recordId)}`)
  );
  if (resp.status === 404) return null;
  if (!resp.ok) throw new Error(`Report fetch failed: HTTP ${resp.status}`);
  return resp.json();
}

/* ── Pipeline / Queue endpoints ──────────────────────────────────── */

/**
 * Submit medical form to the backend pipeline.
 * Accepts a plain object with the following keys:
 *   patientName — string
 *   patientDob — string
 *   patientSex — string
 *   patientBloodGroup — string
 *   patientAllergies — string
 *   symptomsText — string, patient's symptom description
 *   medications — string[], list of medication names
 *   xrayFindings — string[], selected X-ray finding checkboxes
 *   xrayFreeText — string, optional clinician note
 *   labPdfs — File[], multiple lab report PDFs
 *   xrayImages — File[], multiple X-ray image files
 *   audioBlob — Blob | null, recorded audio
 *
 * Returns the PipelineJob JSON on success.
 * Throws on network or server error.
 */
export async function submitForm({
  patientName = '',
  patientDob = '',
  patientSex = '',
  patientBloodGroup = '',
  patientWeightKg = null,
  patientHeightCm = null,
  patientAllergies = '',
  patientMedicalConditions = [],
  symptomsText = '',
  medications = [],
  xrayFindings = [],
  xrayFreeText = '',
  labPdfs = [],
  xrayImages = [],
  audioBlob = null,
} = {}) {
  const fd = new FormData();
  const normalizedSymptomsText = symptomsText.trim();
  if (patientName.trim()) fd.append('patient_name', patientName.trim());
  if (patientDob.trim()) fd.append('patient_dob', patientDob.trim());
  if (patientSex.trim()) fd.append('patient_sex', patientSex.trim());
  if (patientBloodGroup.trim())
    fd.append('patient_blood_group', patientBloodGroup.trim());
  if (patientWeightKg !== null && patientWeightKg !== '')
    fd.append('patient_weight_kg', String(patientWeightKg));
  if (patientHeightCm !== null && patientHeightCm !== '')
    fd.append('patient_height_cm', String(patientHeightCm));
  if (patientAllergies.trim())
    fd.append('patient_allergies', patientAllergies.trim());
  fd.append(
    'patient_medical_conditions',
    JSON.stringify(patientMedicalConditions || [])
  );
  if (normalizedSymptomsText) fd.append('symptoms_text', normalizedSymptomsText);
  fd.append('medications', JSON.stringify(medications));
  fd.append('xray_findings', JSON.stringify(xrayFindings));
  if (xrayFreeText) fd.append('xray_free_text', xrayFreeText);
  // Append multiple lab PDFs
  if (labPdfs && labPdfs.length > 0) {
    labPdfs.forEach((file) => {
      fd.append('lab_pdf', file, file.name || 'lab_report.pdf');
    });
  }
  // Append multiple X-ray images
  if (xrayImages && xrayImages.length > 0) {
    xrayImages.forEach((file) => {
      fd.append('xray_image', file, file.name || 'xray_image.png');
    });
  }
  // Always upload the recording when present so the backend
  // VoiceTranscriber step is actually exercised. If browser speech
  // recognition also produced symptoms_text, the backend can still use
  // that text as a fallback if audio transcription fails.
  if (audioBlob) {
    const ext = audioBlob.type.includes('webm') ? '.webm' : '.wav';
    fd.append('audio', audioBlob, `recording${ext}`);
  }
  const resp = await fetchWithAuth(url('queue/submit'), {
    method: 'POST',
    body: fd,
  });
  const data = await resp.json();
  if (!resp.ok) {
    const err = new Error(
      data?.reason || data?.detail || `HTTP ${resp.status}`
    );
    err.status = resp.status;
    err.code = data?.code;
    err.data = data;
    throw err;
  }
  return data;
}

/**
 * Poll job status. Returns the full status payload or null if unknown.
 */
export async function getJobStatus(jobId) {
  const resp = await fetchWithAuth(url(`queue/status/${jobId}`));
  if (resp.status === 404) return null;
  if (!resp.ok) throw new Error(`Status poll failed: HTTP ${resp.status}`);
  return resp.json();
}

/**
 * Stream report tokens from the backend.
 * Returns a ReadableStream that yields plain text tokens.
 * The stream ends when the backend sends the None sentinel.
 *
 * Usage:
 *   const stream = api.streamReport(jobId);
 *   const reader = stream.getReader();
 *   while (true) {
 *     const { done, value } = await reader.read();
 *     if (done) break;
 *     setReportText(prev => prev + value);
 *   }
 */
export function streamReport(jobId) {
  const token = getAccessToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  return fetch(url(`queue/stream/${jobId}`), {
    headers,
    credentials: 'include',
  }).then((resp) => {
    if (!resp.ok) throw new Error(`Stream failed: HTTP ${resp.status}`);
    return resp.body;
  });
}

/**
 * GET /queue/result/{jobId} — fetch the canonical result for a completed job.
 */
export async function getJobResult(jobId) {
  const resp = await fetchWithAuth(url(`queue/result/${jobId}`));
  if (resp.status === 404) {
    return null;
  }
  if (!resp.ok) {
    throw new Error(`Result fetch failed: HTTP ${resp.status}`);
  }
  return resp.json();
}

/**
 * DELETE /queue/{job_id} — permanently delete a job and all its data.
 */
export async function deleteJob(jobId) {
  const resp = await fetchWithAuth(
    url(`queue/${encodeURIComponent(jobId)}`),
    { method: 'DELETE' }
  );
  if (!resp.ok && resp.status !== 204) {
    let data = null;
    try {
      data = await resp.json();
    } catch {
      // The response may not contain JSON.
    }
    throw new Error(
      extractErrorMessage(data, `Delete report failed: HTTP ${resp.status}`)
    );
  }
  return true;
}

/* ── Chat endpoints ──────────────────────────────────────────────── */

/**
 * GET /queue/chat/{jobId}/init
 *
 * Load initial suggested questions and restore any existing conversation
 * for this report. Does NOT cost the user a turn.
 *
 * Suggested questions are generated dynamically from the actual report
 * findings — lab values with real numbers, specific symptoms, severity
 * reasons, drug interactions, X-ray findings — not a static list.
 *
 * Returns:
 *   {
 *     job_id: string,
 *     turn: number,
 *     turns_remaining: number,
 *     messages: Array<{ role: string, content: string }>,
 *     suggested_questions: string[],
 *     limit_reached: boolean,
 *   }
 */
export async function getChatInit(jobId) {
  const resp = await fetchWithAuth(
    url(`queue/chat/${encodeURIComponent(jobId)}/init`)
  );
  if (!resp.ok) {
    throw new Error(`Chat init failed: HTTP ${resp.status}`);
  }
  return resp.json();
}

/**
 * POST /queue/chat
 *
 * Send a follow-up question about a completed report.
 *
 * Answer pipeline (backend):
 *   1. Deterministic answer — reads actual ReportIntelligence fields,
 *      always correct, no model involved.
 *   2. Optional +1 sentence — idle model adds a connecting sentence ONLY
 *      for cross-domain questions (e.g. "how are my labs connected to my
 *      symptoms?"). Silently skipped on any failure. Base answer always shown.
 *
 * Args:
 *   jobId   — the completed report's job ID
 *   message — the user's question (1-1000 chars)
 *
 * Returns:
 *   {
 *     job_id: string,
 *     turn: number,
 *     answer: string,
 *     severity_delta: "increased" | "decreased" | "unchanged" | null,
 *     suggested_questions: string[],
 *     enriched: boolean,
 *   }
 *
 * enriched: true when the idle model added a connecting sentence.
 * The deterministic answer is always shown regardless.
 */
export async function sendChatMessage({ jobId, message }) {
  const resp = await fetchWithAuth(url('queue/chat'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId, message }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(
      extractErrorMessage(data, `Chat failed: HTTP ${resp.status}`)
    );
  }
  return data;
}

/* ── Vitals endpoints ─────────────────────────────────────────────── */

/**
 * POST /vitals/checkin — save daily vitals check-in.
 */
export async function submitVitalsCheckin(data) {
  const resp = await fetchWithAuth(url('vitals/checkin'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!resp.ok) throw new Error(`Vitals checkin failed: HTTP ${resp.status}`);
  return resp.json();
}

/**
 * GET /vitals/trends — return historical vitals with baseline z-scores.
 */
export async function getVitalsTrends() {
  const resp = await fetchWithAuth(url('vitals/trends'));
  if (!resp.ok) throw new Error(`Vitals trends failed: HTTP ${resp.status}`);
  return resp.json();
}

/**
 * GET /metrics — return Prometheus-style metrics (plain text).
 */
export async function getMetrics() {
  const resp = await fetch(url('metrics'));
  return resp.text();
}

/* ── Text-to-speech (local, no external API) ──────────────────────── */

/**
 * POST /tts/speak — synthesize text to speech entirely on-server via
 * Piper. Returns a Blob (audio/wav) for playback via an <audio>
 * element or URL.createObjectURL. Replaces the previous client-side
 * window.speechSynthesis approach, which depended on OS-installed
 * voices and silently produced no audio when none were present.
 *
 * Throws an Error with `.reason` set to the backend's machine-readable
 * reason code ('model_missing', 'text_too_long', 'synthesis_failed',
 * or 'timeout') when available, so callers can show a specific
 * message.
 *
 * First-request latency: on the server's first call after startup,
 * the Piper voice model has to be loaded from disk, which combined
 * with synthesis itself for a long report can take a while. A
 * client-side timeout still applies so the UI never hangs on
 * "Preparing..." indefinitely if the server is unresponsive.
 */
const TTS_TIMEOUT_MS = 90_000;

export async function synthesizeSpeech(text, jobId) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TTS_TIMEOUT_MS);
  let resp;
  try {
    resp = await fetchWithAuth(url('tts/speak'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(jobId ? { text, job_id: jobId } : { text }),
      signal: controller.signal,
    });
  } catch (err) {
    if (err.name === 'AbortError') {
      const timeoutErr = new Error('Speech synthesis timed out. Please try again.');
      timeoutErr.reason = 'timeout';
      throw timeoutErr;
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
  if (!resp.ok) {
    let detail = `Speech synthesis failed: HTTP ${resp.status}`;
    try {
      const data = await resp.json();
      detail = extractErrorMessage(data, detail);
    } catch {
      // Response wasn't JSON — keep the generic message.
    }
    const err = new Error(detail);
    err.status = resp.status;
    // 503 = voice model not installed on server; 413 = text too long.
    if (resp.status === 503) err.reason = 'model_missing';
    else if (resp.status === 413) err.reason = 'text_too_long';
    else err.reason = 'synthesis_failed';
    throw err;
  }
  return resp.blob();
}

/**
 * POST /tts/speak/stream — streaming variant of synthesizeSpeech().
 *
 * Sends the same request but receives a chunked WAV stream. Uses the
 * Web Audio API to decode and schedule each PCM chunk for gapless
 * playback as it arrives, so audio starts after the first segment
 * (~100-300 ms) instead of waiting for the full report (~2-5 s).
 *
 * IMPORTANT — autoplay policy: The AudioContext is constructed
 * synchronously at the very top of this function, before any await,
 * so the browser treats it as directly gesture-initiated. The caller
 * MUST invoke this function directly from a click/tap handler with no
 * async gap before the call site. Any await before calling this
 * function will break AudioContext creation on Safari and Chrome.
 *
 * The AudioContext is also explicitly resumed after construction —
 * on Chrome/Safari, an AudioContext created inside async flow can
 * start in the "suspended" state, silently accepting scheduled audio
 * that never actually plays. resume() must run inside the gesture
 * turn to succeed.
 *
 * Returns { stop, allEndedPromise }:
 *   stop()           — immediately halt playback and release resources.
 *   allEndedPromise  — Promise that resolves when all scheduled audio
 *                      has finished playing (use to reset UI state).
 *
 * Throws TTSError-shaped errors (with .reason) on network or server
 * failure, matching the error contract of synthesizeSpeech(). May
 * throw with reason='autoplay_blocked' if the browser refused to run
 * the AudioContext despite the gesture — the caller should prompt the
 * user to click again.
 */
const TTS_STREAM_TIMEOUT_MS = 120_000;

export async function synthesizeSpeechStream(text, jobId) {
  // Construct AudioContext synchronously — MUST be the first statement,
  // before any await — so browsers allow it under autoplay policy.
  // Do NOT force a sampleRate here — passing one the OS can't provide
  // natively causes silent resampling glitches on some macOS/Safari
  // setups. Let the browser pick the device rate; we resample when
  // we build each AudioBuffer below.
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioCtx();

  // Kick the context into the "running" state immediately, while we
  // are still inside the click gesture. If we wait until after the
  // fetch resolves, resume() no longer counts as gesture-initiated
  // on Chrome/Safari and the context stays "suspended" forever —
  // scheduled sources fire onended but no sound comes out.
  const resumePromise = audioContext.resume().catch(() => {});

  // Piper always emits 22050 Hz mono int16-LE PCM. We create the
  // AudioBuffer at that rate and let Web Audio resample to the
  // device rate at playback time, preserving pitch and duration.
  const PIPER_SAMPLE_RATE = 22050;

  let stopped = false;
  const abortController = new AbortController();

  function stop() {
    if (stopped) return;
    stopped = true;
    abortController.abort();
    try { audioContext.close(); } catch {}
  }

  // Tracks the scheduled end-time of the last queued audio chunk so
  // each new chunk is scheduled to start exactly where the previous
  // one ends (gapless playback). Initialised on the first chunk.
  let nextStartTime = 0;

  // Track in-flight AudioBufferSourceNodes so allEndedPromise
  // resolves only after the last chunk has actually finished playing.
  let pendingSources = 0;
  let readerDone = false;

  // Guard so allEndedPromise resolves at most once. Without this, a
  // reader that finishes before any chunk is scheduled could resolve
  // via both the "no chunks scheduled" fallback and a later onended
  // event, causing double-resolution and confusing state tracking.
  let allEndedResolved = false;

  // True once at least one chunk has been scheduled. Prevents
  // checkAllEnded() from resolving the promise before any audio has
  // even started — e.g. if readerDone flips true from the initial
  // header-only read before the first PCM chunk is ready.
  let anyChunkScheduled = false;

  let resolveAllEnded;
  const allEndedPromise = new Promise((resolve) => {
    resolveAllEnded = resolve;
  });

  function checkAllEnded() {
    if (allEndedResolved) return;
    if (!anyChunkScheduled) return;
    if (!readerDone) return;
    if (pendingSources > 0) return;
    allEndedResolved = true;
    resolveAllEnded();
  }

  function scheduleChunk(int16Bytes) {
    if (stopped || int16Bytes.byteLength < 2) return;

    // Enforce even byte count (each int16 sample = 2 bytes).
    const byteLen = int16Bytes.byteLength - (int16Bytes.byteLength % 2);
    if (byteLen === 0) return;

    // Copy the bytes into a fresh, aligned ArrayBuffer. The Uint8Array
    // we received from fetch may not sit on a 2-byte boundary within
    // its underlying ArrayBuffer, which throws RangeError when passed
    // to new Int16Array(buffer, offset, len). Copying costs an O(n)
    // walk but guarantees alignment on every browser.
    const aligned = new ArrayBuffer(byteLen);
    new Uint8Array(aligned).set(
      new Uint8Array(int16Bytes.buffer, int16Bytes.byteOffset, byteLen)
    );
    const int16 = new Int16Array(aligned);

    // Convert int16 [-32768, 32767] to float32 [-1.0, 1.0] for Web Audio.
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }

    // Build the AudioBuffer at Piper's native rate. The Web Audio
    // graph automatically resamples to the device rate on playback.
    const audioBuffer = audioContext.createBuffer(
      1,                    // mono
      float32.length,
      PIPER_SAMPLE_RATE
    );
    audioBuffer.getChannelData(0).set(float32);

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    // Initialise nextStartTime on the first chunk. Using
    // audioContext.currentTime + a tiny lead-in gives the browser a
    // moment to actually begin playback rather than scheduling the
    // start exactly at "now" (which some browsers treat as already
    // past and skip).
    if (!anyChunkScheduled) {
      nextStartTime = audioContext.currentTime + 0.05;
    }

    // Schedule gaplessly after the previous chunk.
    const startAt = Math.max(nextStartTime, audioContext.currentTime);
    source.start(startAt);
    nextStartTime = startAt + audioBuffer.duration;

    pendingSources++;
    anyChunkScheduled = true;
    source.onended = () => {
      pendingSources--;
      checkAllEnded();
    };
  }

  // ── Open the stream ─────────────────────────────────────────────
  const timeoutId = setTimeout(
    () => abortController.abort(),
    TTS_STREAM_TIMEOUT_MS
  );
  let resp;
  try {
    // Wait for resume() to complete before starting the fetch so
    // we don't miss the gesture window.
    await resumePromise;
    resp = await fetchWithAuth(url('tts/speak/stream'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(jobId ? { text, job_id: jobId } : { text }),
      signal: abortController.signal,
    });
  } catch (err) {
    clearTimeout(timeoutId);
    stop();
    if (err.name === 'AbortError') {
      const e = new Error('Speech synthesis timed out. Please try again.');
      e.reason = 'timeout';
      throw e;
    }
    throw err;
  }
  clearTimeout(timeoutId);

  if (!resp.ok) {
    stop();
    let detail = `Speech synthesis failed: HTTP ${resp.status}`;
    try {
      const data = await resp.json();
      detail = extractErrorMessage(data, detail);
    } catch {
      // Response was not JSON — keep the generic message.
    }
    const err = new Error(detail);
    err.status = resp.status;
    if (resp.status === 503) err.reason = 'model_missing';
    else if (resp.status === 413) err.reason = 'text_too_long';
    else err.reason = 'synthesis_failed';
    throw err;
  }

  // If the AudioContext failed to enter the "running" state, tell
  // the caller so it can prompt the user for another click. This
  // happens on Safari/Chrome when the click gesture was consumed by
  // something else before we got here.
  if (audioContext.state !== 'running') {
    stop();
    const err = new Error(
      'Browser blocked audio playback (AudioContext suspended). ' +
      'Please click the readout button again.'
    );
    err.reason = 'autoplay_blocked';
    throw err;
  }

  // ── Consume the chunked WAV stream in the background ────────────
  //
  // Server sends: [44-byte WAV header] [PCM chunk...] [PCM chunk...]
  // We skip the header bytes and schedule each PCM chunk for Web
  // Audio playback as it arrives. TCP may split or merge chunks
  // arbitrarily, so we buffer across reads to:
  //   (a) accumulate the full 44-byte header before skipping it, and
  //   (b) maintain int16 alignment (2 bytes per sample) across reads.

  const reader = resp.body.getReader();
  let headerConsumed = false;
  let leftovers = new Uint8Array(0); // carries the odd byte between reads

  (async () => {
    try {
      while (!stopped) {
        const { done, value } = await reader.read();
        if (done || stopped) break;

        // Prepend any leftover byte from the previous read.
        let incoming;
        if (leftovers.length > 0) {
          incoming = new Uint8Array(leftovers.length + value.length);
          incoming.set(leftovers);
          incoming.set(value, leftovers.length);
          leftovers = new Uint8Array(0);
        } else {
          incoming = value;
        }

        // Accumulate until we have the full 44-byte WAV header.
        if (!headerConsumed) {
          if (incoming.length < 44) {
            leftovers = incoming;
            continue;
          }
          incoming = incoming.slice(44);
          headerConsumed = true;
        }

        if (incoming.length === 0) continue;

        // Save the trailing odd byte for the next iteration so we
        // never pass a non-integer number of int16 samples to
        // scheduleChunk.
        if (incoming.length % 2 !== 0) {
          leftovers = incoming.slice(incoming.length - 1);
          incoming = incoming.slice(0, incoming.length - 1);
        }

        if (incoming.length > 0) {
          scheduleChunk(incoming);
        }
      }
    } catch (err) {
      if (!stopped && err.name !== 'AbortError') {
        console.error('TTS stream read error:', err);
      }
    } finally {
      readerDone = true;
      // If the reader finished without ever scheduling a chunk
      // (e.g. server sent header only, or errored immediately),
      // resolve the promise so the caller doesn't hang on the
      // "Stop Readout" state forever.
      if (!anyChunkScheduled && !allEndedResolved) {
        allEndedResolved = true;
        resolveAllEnded();
      } else {
        checkAllEnded();
      }
    }
  })();

  return { stop, allEndedPromise };
}

/* ── Default export (object with all functions) ──────────────────── */

export default {
  // Auth
  login,
  logout,
  register,
  forgotPassword,
  verifySecurityAnswer,
  resetPassword,
  refreshAccessToken,
  getMe,
  setAccessToken,
  clearAccessToken,
  getAccessToken,

  // Account
  getProfile,
  updateProfile,
  changeEmail,

  // System
  getHealth,
  getMetrics,

  // Dashboard & Records
  getDashboard,
  getRecords,
  getRecord,

  // Pipeline / Queue
  submitForm,
  getJobStatus,
  streamReport,
  getJobResult,
  deleteJob,

  // Chat
  getChatInit,
  sendChatMessage,

  // Vitals
  submitVitalsCheckin,
  getVitalsTrends,

  // Text-to-speech
  synthesizeSpeech,
  synthesizeSpeechStream,
};