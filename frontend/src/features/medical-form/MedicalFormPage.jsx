import { useLayoutEffect, useRef, useState, useCallback, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import VoiceRecorder, { Waveform } from "../voice/VoiceRecorder/VoiceRecorder";
import { getProfile, submitForm } from "../../services/api";
import { validateFile } from "./validation";
import "./MedicalForm.css";

export default function MedicalFormPage() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const draftKey = `aegis_form_draft:${user?.id || "anonymous"}`;
  const [expandedSection, setExpandedSection] = useState(1);

  const [formData, setFormData] = useState(() => {
    const saved = localStorage.getItem(draftKey);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        return {
          fullName: parsed.fullName || "",
          dob: parsed.dob || "",
          sex: parsed.sex || "",
          medicalHistory: parsed.medicalHistory || "",
          bloodGroup: parsed.bloodGroup || "",
                    
          allergies: parsed.allergies || "",
          medicalConditions: parsed.medicalConditions || "",
          medications: parsed.medications || "",
          labReports: [],
          xrayImages: [],
          xrayFindings: parsed.xrayFindings || [],
          xrayFreeText: parsed.xrayFreeText || "",
        };
      } catch (e) {}
    }
    return {
      fullName: "",
      dob: "",
      sex: "",
      medicalHistory: "",
      bloodGroup: "",
            
      allergies: "",
      medicalConditions: "",
      medications: "",
      labReports: [],
      xrayImages: [],
      xrayFindings: [],
      xrayFreeText: "",
    };
  });

  useEffect(() => {
    const draft = {
      fullName: formData.fullName,
      dob: formData.dob,
      sex: formData.sex,
      medicalHistory: formData.medicalHistory,
      bloodGroup: formData.bloodGroup,
      allergies: formData.allergies,
      medicalConditions: formData.medicalConditions,
      medications: formData.medications,
      xrayFindings: formData.xrayFindings,
      xrayFreeText: formData.xrayFreeText,
    };
    localStorage.setItem(draftKey, JSON.stringify(draft));
  }, [formData, draftKey]);

  useEffect(() => {
    let cancelled = false;

    getProfile()
      .then((profile) => {
        if (cancelled || !profile?.profile_complete) return;
        setFormData((current) => ({
          ...current,
          fullName: current.fullName || profile.full_name || "",
          dob: current.dob || profile.date_of_birth || "",
          sex: current.sex || profile.sex || "",
          bloodGroup: current.bloodGroup || profile.blood_group || "",
                    
          allergies: current.allergies || (profile.allergies || []).join(", "),
          medicalConditions: current.medicalConditions || (profile.medical_conditions || []).join(", "),
          medications: current.medications || (profile.current_medications || []).join(", "),
        }));
      })
      .catch(() => {
        // Profile prefill is best-effort; the Medical Form remains usable.
      });

    return () => {
      cancelled = true;
    };
  }, [user?.id]);

  const [isRecording, setIsRecording] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState("idle");
  const [audioBlob, setAudioBlob] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const medicalHistoryRef = useRef(null);
  const radiologistNotesRef = useRef(null);
  const [dobError, setDobError] = useState("");

  const autoResizeTextarea = useCallback((el) => {
    if (!el) return;
    el.style.setProperty("height", "auto", "important");
    el.style.setProperty("height", `${el.scrollHeight}px`, "important");
  }, []);

  useLayoutEffect(() => {
    autoResizeTextarea(medicalHistoryRef.current);
  }, [formData.medicalHistory, autoResizeTextarea]);

  useLayoutEffect(() => {
    autoResizeTextarea(radiologistNotesRef.current);
  }, [formData.xrayFreeText, autoResizeTextarea]);

  // ── Full Name: letters and spaces only ──────────────────────────
  const handleNameChange = useCallback((e) => {
    const raw = e.target.value;
    // Allow only letters (including accented), spaces, hyphens, periods
    const cleaned = raw.replace(/[^a-zA-ZÀ-ÿ\s.\-']/g, "");
    setFormData((prev) => ({ ...prev, fullName: cleaned }));
  }, []);

  // ── DOB: auto-format DD/MM/YYYY, numbers only, no future dates ──
  const handleDobChange = useCallback((e) => {
    const raw = e.target.value;
    // Remove all non-digit characters
    const digits = raw.replace(/\D/g, "");

    // Format: DD/MM/YYYY
    let formatted = "";
    if (digits.length > 0) {
      formatted = digits.substring(0, 2);
      if (digits.length >= 3) {
        formatted += "/" + digits.substring(2, 4);
      }
      if (digits.length >= 5) {
        formatted += "/" + digits.substring(4, 8);
      }
    }

    setFormData((prev) => ({ ...prev, dob: formatted }));

    // Validate date
    if (digits.length === 8) {
      const day = parseInt(digits.substring(0, 2), 10);
      const month = parseInt(digits.substring(2, 4), 10);
      const year = parseInt(digits.substring(4, 8), 10);

      if (day < 1 || day > 31) {
        setDobError("Day must be between 01 and 31");
      } else if (month < 1 || month > 12) {
        setDobError("Month must be between 01 and 12");
      } else if (year < 1900 || year > new Date().getFullYear()) {
        setDobError(`Year must be between 1900 and ${new Date().getFullYear()}`);
      } else {
        // Check if date is in the future
        const entered = new Date(year, month - 1, day);
        const today = new Date();
        today.setHours(23, 59, 59, 999); // End of today
        if (entered > today) {
          setDobError("Date of birth cannot be in the future");
        } else {
          setDobError("");
        }
      }
    } else if (digits.length > 0 && digits.length < 8) {
      setDobError("");
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validate required fields
    if (!formData.fullName.trim()) {
      setSubmitError("Please enter your full name.");
      return;
    }
    if (formData.dob.length !== 10) {
      setSubmitError("Please enter a valid date of birth (DD/MM/YYYY).");
      return;
    }
    if (dobError) {
      setSubmitError(dobError);
      return;
    }
    if (!formData.sex) {
      setSubmitError("Please select your sex.");
      return;
    }
    if (isRecording || voiceStatus === "uploading") {
      setSubmitError("Please stop the voice recording and wait until it says Voice ready before submitting.");
      return;
    }
    if (!formData.medicalHistory.trim() && !audioBlob) {
      setSubmitError("Please describe your symptoms or record your symptoms using the microphone.");
      return;
    }
    if (!formData.bloodGroup) {
      setSubmitError("Please select your blood group.");
      return;
    }

    setIsSubmitting(true);
    setSubmitError(null);

    try {
      const medicationsList = formData.medications
        .split(",")
        .map((m) => m.trim())
        .filter(Boolean);
      const medicalConditionsList = formData.medicalConditions
        .split(",")
        .map((condition) => condition.trim())
        .filter(Boolean);

      const result = await submitForm({
        patientName: formData.fullName,
        patientDob: formData.dob,
        patientSex: formData.sex,
        patientBloodGroup: formData.bloodGroup,
        patientWeightKg: null,
        patientHeightCm: null,
        patientAllergies: formData.allergies,
        patientMedicalConditions: medicalConditionsList,

        symptomsText: formData.medicalHistory,
        medications: medicationsList,
        xrayFindings: formData.xrayFindings,
        xrayFreeText: formData.xrayFreeText,
        labPdfs: formData.labReports,  // ← Passing array of files
        xrayImages: formData.xrayImages, // ← Passing array of files
        audioBlob: audioBlob,
      });

      localStorage.removeItem(draftKey);
      navigate(`/report?jobId=${result.job_id}`);
    } catch (err) {
      console.error("Submission failed:", err);
      const msg = err.data?.reason || err.message || "Submission failed. Please try again.";
      if (err.status === 429) {
        setSubmitError("Rate limit exceeded: Please wait a moment before submitting another case.");
      } else if (err.status === 503 || err.code === "queue_full") {
        setSubmitError("Queue capacity full: The triage queue is temporarily busy. Your draft is preserved—please try again shortly.");
      } else {
        setSubmitError(msg);
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const toggleSection = (sectionIndex) => {
    setExpandedSection(expandedSection === sectionIndex ? null : sectionIndex);
  };

  // ── MULTIPLE FILES HANDLERS ──────────────────────────────────────
  const handleFilesChange = (e, field) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files);
      const kind = field === "labReports" ? "pdf" : "xray";
      const maxMb = field === "labReports" ? 10 : 25;
      
      const validFiles = [];
      for (const file of newFiles) {
        const err = validateFile(file, kind, maxMb);
        if (err) {
          setSubmitError(err);
          return;
        }
        validFiles.push(file);
      }

      setFormData((prev) => ({
        ...prev,
        [field]: [...prev[field], ...validFiles],
      }));
      setSubmitError(null);
      e.target.value = null;
    }
  };

  const removeFile = (e, field, indexToRemove) => {
    e.stopPropagation();
    e.preventDefault();
    setFormData((prev) => ({
      ...prev,
      [field]: prev[field].filter((_, i) => i !== indexToRemove),
    }));
  };

  const toggleXrayFinding = (finding) => {
    setFormData((prev) => {
      const exists = prev.xrayFindings.includes(finding);
      return {
        ...prev,
        xrayFindings: exists
          ? prev.xrayFindings.filter((f) => f !== finding)
          : [...prev.xrayFindings, finding],
      };
    });
  };

  const handleVoiceComplete = (text, replace = false) => {
    if (!text?.trim()) return;
    setFormData((prev) => ({
      ...prev,
      medicalHistory: replace
        ? text.trim()
        : prev.medicalHistory
        ? prev.medicalHistory + " " + text.trim()
        : text.trim(),
    }));
  };

  const handleVoiceStatusChange = (status) => {
    setVoiceStatus(status);
    setIsRecording(status === "recording");
  };

  const handleAudioCapture = (blob) => {
    setAudioBlob(blob);
  };

  return (
    <div className="form-center-container">
      <div className="medical-form-card">
        <h2>Medical Form</h2>

        {/* Error Banner */}
        {submitError && (
          <div className="error-banner">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="8" x2="12" y2="12"></line>
              <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            <span>{submitError}</span>
            <button className="error-dismiss" onClick={() => setSubmitError(null)}>✕</button>
          </div>
        )}

        <div className="accordion-wrapper">
          {/* --- SECTION 1: PERSONAL DETAILS --- */}
          <div className={`accordion-section ${expandedSection === 1 ? "open" : ""}`}>
            <div className="accordion-header" onClick={() => toggleSection(1)}>
              <div className="header-left">
                <div className="icon-badge blue-badge">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
                </div>
                <h3>1. Personal Details</h3>
              </div>
              <div className="toggle-indicator" style={{ background: 'none', border: 'none', boxShadow: 'none', borderRadius: 0, padding: 0, width: 'auto', height: 'auto', minWidth: 'auto', minHeight: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {expandedSection === 1 ? (
                  <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8ea1d6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="8" y1="12" x2="16" y2="12"></line></svg>
                ) : (
                  <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8ea1d6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>
                )}
              </div>
            </div>

            {expandedSection === 1 && (
              <div className="accordion-content animate-slide-down">
                <div className="form-grid">
                  <div className="input-group">
                    <label>Full Name <span className="required">*</span></label>
                    <div className="input-with-icon">
                      <span className="field-icon">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#7182b1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
                      </span>
                      <input
                        type="text"
                        placeholder="Enter your full name"
                        value={formData.fullName}
                        onChange={handleNameChange}
                        maxLength={60}
                      />
                    </div>
                  </div>

                  <div className="input-group">
                    <label>Date of Birth <span className="required">*</span></label>
                    <div className="input-with-icon">
                      <span className="field-icon">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#7182b1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
                      </span>
                      <input
                        type="text"
                        placeholder="DD/MM/YYYY"
                        value={formData.dob}
                        onChange={handleDobChange}
                        maxLength={10}
                      />
                    </div>
                    {dobError && (
                      <span style={{ color: "#ef4444", fontSize: "11px", fontWeight: 600, marginTop: "4px" }}>
                        {dobError}
                      </span>
                    )}
                  </div>

                  <div className="input-group">
                    <label>Sex <span className="required">*</span></label>
                    <div className="radio-group">
                      <label className={`radio-label ${formData.sex === "Male" ? "selected" : ""}`}>
                        <input type="radio" name="sex" value="Male" checked={formData.sex === "Male"} onChange={(e) => setFormData({ ...formData, sex: e.target.value })} />
                        <span className="custom-radio"></span>Male
                      </label>
                      <label className={`radio-label ${formData.sex === "Female" ? "selected" : ""}`}>
                        <input type="radio" name="sex" value="Female" checked={formData.sex === "Female"} onChange={(e) => setFormData({ ...formData, sex: e.target.value })} />
                        <span className="custom-radio"></span>Female
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* --- SECTION 2: MEDICAL INFORMATION --- */}
          <div className={`accordion-section ${expandedSection === 2 ? "open" : ""}`}>
            <div className="accordion-header" onClick={() => toggleSection(2)}>
              <div className="header-left">
                <div className="icon-badge pink-badge">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20.42 4.58a5.4 5.4 0 0 0-7.65 0l-.77.78-.77-.78a5.4 5.4 0 0 0-7.65 0C1.46 6.7 1.33 10.28 4 13l8 8 8-8c2.67-2.72 2.54-6.3.42-8.42z"></path><polyline points="3 12 7 12 10 7 14 17 17 12 21 12"></polyline></svg>
                </div>
                <h3>2. Medical Information</h3>
              </div>
              <div className="toggle-indicator" style={{ background: 'none', border: 'none', boxShadow: 'none', borderRadius: 0, padding: 0, width: 'auto', height: 'auto', minWidth: 'auto', minHeight: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {expandedSection === 2 ? (
                  <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8ea1d6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="8" y1="12" x2="16" y2="12"></line></svg>
                ) : (
                  <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8ea1d6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>
                )}
              </div>
            </div>

            {expandedSection === 2 && (
              <div className="accordion-content animate-slide-down">
                <div className="input-group margin-bottom-20">
                  <label>Medical History / Symptoms <span className="required">*</span></label>
                  <div className={`input-with-icon medical-history-input-wrap ${isRecording ? "is-recording" : ""}`}>
                    <span className="field-icon textarea-icon">
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#7182b1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                    </span>
                    <textarea
                      ref={medicalHistoryRef}
                      rows={1}
                      placeholder="Describe your symptoms, previous illnesses, surgeries or chronic diseases..."
                      value={formData.medicalHistory}
                      onChange={(e) => setFormData((prev) => ({ ...prev, medicalHistory: e.target.value }))}
                      className="glass-textarea medical-history-auto-textarea"
                    />
                    {isRecording && <Waveform />}
                    <div className="medical-history-voice-corner">
                      <VoiceRecorder
                        onStatusChange={handleVoiceStatusChange}
                        onComplete={handleVoiceComplete}
                        onAudioCapture={handleAudioCapture}
                      />
                    </div>
                  </div>
                </div>

                <div className="form-grid">
                  <div className="input-group">
                    <label>Blood Group <span className="required">*</span></label>
                    <div className="input-with-icon custom-select-wrapper">
                      <span className="field-icon">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#7182b1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path></svg>
                      </span>
                      <select value={formData.bloodGroup} onChange={(e) => setFormData({ ...formData, bloodGroup: e.target.value })} className="glass-select">
                        <option value="" disabled hidden>Select Blood Group</option>
                        <option value="A+">A+</option>
                        <option value="A-">A-</option>
                        <option value="B+">B+</option>
                        <option value="B-">B-</option>
                        <option value="AB+">AB+</option>
                        <option value="AB-">AB-</option>
                        <option value="O+">O+</option>
                        <option value="O-">O-</option>
                      </select>
                      <span className="select-arrow">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7182b1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
                      </span>
                    </div>
                  </div>

                  <div className="input-group">
                    <label>Allergies <span className="optional">(optional)</span></label>
                    <div className="input-with-icon">
                      <span className="field-icon">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#7182b1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                      </span>
                      <input type="text" placeholder="Enter allergies if any" value={formData.allergies} onChange={(e) => setFormData({ ...formData, allergies: e.target.value })} />
                    </div>
                  </div>
                </div>

                <div className="form-grid" style={{ marginTop: "16px" }}>
                  
                </div>

                <div className="input-group margin-bottom-20" style={{ marginTop: "16px" }}>
                  <label>Existing Medical Conditions <span className="optional">(comma-separated, optional)</span></label>
                  <div className="input-with-icon">
                    <span className="field-icon">✚</span>
                    <input
                      type="text"
                      placeholder="e.g. Asthma, diabetes"
                      value={formData.medicalConditions}
                      onChange={(e) => setFormData({ ...formData, medicalConditions: e.target.value })}
                    />
                  </div>
                </div>

                {/* Medications */}
                <div className="input-group margin-bottom-20" style={{ marginTop: "16px" }}>
                  <label>Current Medications <span className="optional">(comma-separated)</span></label>
                  <div className="input-with-icon">
                    <span className="field-icon">
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#7182b1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.5 4.5L3 12l7.5 7.5"></path><path d="M13.5 4.5L21 12l-7.5 7.5"></path></svg>
                    </span>
                    <input
                      type="text"
                      placeholder="e.g. Warfarin, Aspirin, Metformin"
                      value={formData.medications}
                      onChange={(e) => setFormData({ ...formData, medications: e.target.value })}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* --- SECTION 3: LAB REPORT --- */}
          <div className={`accordion-section ${expandedSection === 3 ? "open" : ""}`}>
            <div className="accordion-header" onClick={() => toggleSection(3)}>
              <div className="header-left">
                <div className="icon-badge navy-badge">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="12" y1="18" x2="12" y2="12"></line><line x1="9" y1="15" x2="15" y2="15"></line></svg>
                </div>
                <h3>3. Lab Reports (PDF) <span className="optional">(Optional)</span></h3>
              </div>
              <div className="toggle-indicator" style={{ background: 'none', border: 'none', boxShadow: 'none', borderRadius: 0, padding: 0, width: 'auto', height: 'auto', minWidth: 'auto', minHeight: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {expandedSection === 3 ? (
                  <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8ea1d6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="8" y1="12" x2="16" y2="12"></line></svg>
                ) : (
                  <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8ea1d6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>
                )}
              </div>
            </div>

            {expandedSection === 3 && (
              <div className="accordion-content animate-slide-down">
                <div className="file-upload-row">
                  {/* Selected Lab Reports Preview */}
                  {formData.labReports.length > 0 && (
                    <div className="selected-files-container selected-files-container-top">
                      {formData.labReports.map((file, index) => (
                        <div key={index} className="file-badge">
                          <span className="file-name" title={file.name}>{file.name}</span>
                          <button
                            type="button"
                            className="remove-file-btn"
                            onClick={(e) => removeFile(e, "labReports", index)}
                            title="Remove file"
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  <label className="glass-file-dropzone">
                    <input type="file" accept=".pdf" multiple onChange={(e) => handleFilesChange(e, "labReports")} />
                    <span className="upload-cloud-icon">
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#2563ff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                    </span>
                    <div className="upload-text-group">
                      <span className="primary-upload-text">
                        Click to upload or drag Lab Reports (Multiple PDFs allowed)
                      </span>
                      <span className="secondary-upload-text">Supports PDF up to 25MB per file</span>
                    </div>
                  </label>
                </div>
              </div>
            )}
          </div>

          {/* --- SECTION 4: X-RAY IMAGE --- */}
          <div className={`accordion-section ${expandedSection === 4 ? "open" : ""}`}>
            <div className="accordion-header" onClick={() => toggleSection(4)}>
              <div className="header-left">
                <div className="icon-badge purple-badge">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#a855f7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><path d="M12 7v10"></path><path d="M9 10h6"></path><path d="M9 14h6"></path></svg>
                </div>
                <h3>4. X-Ray Images <span className="optional">(Optional)</span></h3>
              </div>
              <div className="toggle-indicator" style={{ background: 'none', border: 'none', boxShadow: 'none', borderRadius: 0, padding: 0, width: 'auto', height: 'auto', minWidth: 'auto', minHeight: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {expandedSection === 4 ? (
                  <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8ea1d6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="8" y1="12" x2="16" y2="12"></line></svg>
                ) : (
                  <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8ea1d6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>
                )}
              </div>
            </div>

            {expandedSection === 4 && (
              <div className="accordion-content animate-slide-down">
                <div className="file-upload-row">
                  {/* Selected X-Ray Images Preview */}
                  {formData.xrayImages.length > 0 && (
                    <div className="selected-files-container selected-files-container-top">
                      {formData.xrayImages.map((file, index) => (
                        <div key={index} className="file-badge">
                          <span className="file-name" title={file.name}>{file.name}</span>
                          <button
                            type="button"
                            className="remove-file-btn"
                            onClick={(e) => removeFile(e, "xrayImages", index)}
                            title="Remove file"
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  <label className="glass-file-dropzone">
                    <input type="file" accept="image/*,.dcm" multiple onChange={(e) => handleFilesChange(e, "xrayImages")} />
                    <span className="upload-cloud-icon">
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#a855f7" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                    </span>
                    <div className="upload-text-group">
                      <span className="primary-upload-text">
                        Click to upload or drag X-Ray Images (Multiple images allowed)
                      </span>
                      <span className="secondary-upload-text">Supports PNG, JPEG, or DICOM files</span>
                    </div>
                  </label>
                </div>

                {/* X-ray Findings Checklist */}
                <div className="xray-findings-section" style={{ marginTop: "16px" }}>
                  <label>X-Ray Findings <span className="optional">(select all that apply)</span></label>
                  <div className="findings-checklist">
                    {[
                      "Cardiomegaly",
                      "Pleural Effusion",
                      "Pneumonia",
                      "Pneumothorax",
                      "Consolidation",
                      "Atelectasis",
                      "Infiltrates",
                      "Pulmonary Edema",
                      "Nodule / Mass",
                      "Fracture",
                    ].map((finding) => (
                      <label
                        key={finding}
                        className={`finding-chip ${formData.xrayFindings.includes(finding) ? "selected" : ""}`}
                      >
                        <input
                          type="checkbox"
                          checked={formData.xrayFindings.includes(finding)}
                          onChange={() => toggleXrayFinding(finding)}
                        />
                        {finding}
                      </label>
                    ))}
                  </div>
                </div>

                {/* X-ray Free Text */}
                <div className="input-group" style={{ marginTop: "12px" }}>
                  <label>Radiologist Notes <span className="optional">(optional)</span></label>
                  <textarea
                    ref={radiologistNotesRef}
                    rows={1}
                    placeholder="Any additional observations or findings..."
                    value={formData.xrayFreeText}
                    onChange={(e) => setFormData({ ...formData, xrayFreeText: e.target.value })}
                    className="glass-textarea medical-history-auto-textarea"
                    style={{ resize: "none", overflow: "hidden", padding: "12px 16px", borderRadius: "12px" }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        <button
          className="form-submit-btn"
          onClick={handleSubmit}
          disabled={isSubmitting || isRecording || voiceStatus === "uploading"}
        >
          {isSubmitting ? (
            <>
              <span className="submit-spinner"></span>
              <span>Submitting...</span>
            </>
          ) : isRecording || voiceStatus === "uploading" ? (
            <>
              <span className="submit-spinner"></span>
              <span>Finish voice recording...</span>
            </>
          ) : (
            <>
              <span className="submit-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><polyline points="9 15 12 18 16 12"></polyline></svg>
              </span>
              <span>Submit</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}