import { useEffect, useState } from "react";
import { useAuth } from "../../context/AuthContext";
import { getProfile, updateProfile, changeEmail } from "../../services/api";
import "./Profile.css";

const SECURITY_QUESTIONS = [
  "What is your favorite pet's name?",
  "What is your favorite food?",
  "What city were you born in?",
  "What is your favorite movie?",
];

const SECURITY_QUESTION_KEYS = ["pet_name", "favorite_food", "birth_city", "favorite_movie"];
const KEY_TO_QUESTION = Object.fromEntries(SECURITY_QUESTION_KEYS.map((k, i) => [k, SECURITY_QUESTIONS[i]]));
const QUESTION_TO_KEY = Object.fromEntries(SECURITY_QUESTIONS.map((q, i) => [q, SECURITY_QUESTION_KEYS[i]]));

const EMPTY_FORM = {
  fullName: "",
  dateOfBirth: "",
  sex: "",
  bloodGroup: "",
  allergies: "",
};

function joinList(value) {
  return Array.isArray(value) ? value.join(", ") : "";
}

function splitList(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export default function ProfileForm({
  onSaved,
  onCancel,
  submitLabel = "Save profile",
  allowCancel = false,
}) {
  const { user, updateUser } = useAuth();
  const [form, setForm] = useState(EMPTY_FORM);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Email change state
  const [showEmailChange, setShowEmailChange] = useState(false);
  const [newEmail, setNewEmail] = useState("");
  const [emailPassword, setEmailPassword] = useState("");
  const [emailLoading, setEmailLoading] = useState(false);
  const [emailError, setEmailError] = useState("");

  // Security question change state
  const [showSecurityChange, setShowSecurityChange] = useState(false);
  const [newSecurityQuestion, setNewSecurityQuestion] = useState("");
  const [newSecurityAnswer, setNewSecurityAnswer] = useState("");
  const [securityLoading, setSecurityLoading] = useState(false);
  const [securityError, setSecurityError] = useState("");

  useEffect(() => {
    let cancelled = false;
    getProfile()
      .then((profile) => {
        if (cancelled) return;
        setForm({
          fullName: profile.full_name || "",
          dateOfBirth: profile.date_of_birth || "",
          sex: profile.sex || "",
          bloodGroup: profile.blood_group || "",
          allergies: joinList(profile.allergies),
        });
      })
      .catch((requestError) => {
        if (!cancelled) setError(requestError.message || "Unable to load your profile.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const updateField = (field) => (event) => {
    setForm((current) => ({ ...current, [field]: event.target.value }));
  };

  const handleDobChange = (event) => {
    const digits = event.target.value.replace(/\D/g, "").slice(0, 8);
    let formatted = digits.slice(0, 2);
    if (digits.length > 2) formatted += `/${digits.slice(2, 4)}`;
    if (digits.length > 4) formatted += `/${digits.slice(4, 8)}`;
    setForm((current) => ({ ...current, dateOfBirth: formatted }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setSuccess("");

    if (!form.fullName.trim()) {
      setError("Please enter your full name.");
      return;
    }
    if (!/^\d{2}\/\d{2}\/\d{4}$/.test(form.dateOfBirth)) {
      setError("Date of birth must use DD/MM/YYYY.");
      return;
    }
    if (!form.sex) {
      setError("Please select your sex.");
      return;
    }
    if (!form.bloodGroup) {
      setError("Please select your blood group.");
      return;
    }

    setSaving(true);
    try {
      const saved = await updateProfile({
        full_name: form.fullName.trim(),
        date_of_birth: form.dateOfBirth,
        sex: form.sex,
        blood_group: form.bloodGroup,
        weight_kg: null,
        height_cm: null,
        allergies: splitList(form.allergies),
        medical_conditions: [],
        current_medications: [],
      });
      updateUser({
        display_name: saved.full_name,
        profile_complete: true,
      });
      setSuccess("Profile saved successfully.");
      onSaved?.(saved);
    } catch (saveError) {
      setError(saveError.message || "Unable to save your profile.");
    } finally {
      setSaving(false);
    }
  };

  const handleEmailChange = async (e) => {
    e.preventDefault();
    setEmailError("");

    if (!newEmail.trim()) {
      setEmailError("Please enter a new email address.");
      return;
    }
    if (!emailPassword) {
      setEmailError("Please enter your password to confirm.");
      return;
    }

    setEmailLoading(true);
    try {
      const result = await changeEmail({ newEmail: newEmail.trim(), password: emailPassword });
      updateUser({ email: result.email });
      setNewEmail("");
      setEmailPassword("");
      setShowEmailChange(false);
      setSuccess("Email updated successfully.");
    } catch (err) {
      setEmailError(err.message || "Failed to change email.");
    } finally {
      setEmailLoading(false);
    }
  };

  const handleSecurityChange = async (e) => {
    e.preventDefault();
    setSecurityError("");

    if (!newSecurityQuestion) {
      setSecurityError("Please select a security question.");
      return;
    }
    if (!newSecurityAnswer.trim()) {
      setSecurityError("Please answer the security question.");
      return;
    }

    setSecurityLoading(true);
    try {
      const token = localStorage.getItem("aegis_access") || "";
      const headers = { "Content-Type": "application/json" };
      if (token) headers.Authorization = `Bearer ${token}`;
      
      const resp = await fetch("/account/security-question", {
        method: "PUT",
        headers,
        credentials: "include",
        body: JSON.stringify({
          security_question: newSecurityQuestion,
          security_answer: newSecurityAnswer.trim(),
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || "Failed to update security question.");
      }
      updateUser({ security_question: KEY_TO_QUESTION[newSecurityQuestion] || newSecurityQuestion });
      setNewSecurityQuestion("");
      setNewSecurityAnswer("");
      setShowSecurityChange(false);
      setSuccess("Security question updated successfully.");
    } catch (err) {
      setSecurityError(err.message || "Failed to update security question.");
    } finally {
      setSecurityLoading(false);
    }
  };

  if (loading) {
    return <div className="profile-form-loading">Loading your profile…</div>;
  }

  // Map user's security question to display text
  const userQuestionKey = user?.security_question ? (QUESTION_TO_KEY[user.security_question] || user.security_question) : null;

  return (
    <form className="profile-form" onSubmit={handleSubmit} noValidate>
      {error && <div className="profile-form-error" role="alert">{error}</div>}
      {success && <div className="profile-success" role="status">{success}</div>}

      <div className="profile-form-grid">
        <label className="profile-field profile-field-wide">
          <span>Full Name *</span>
          <div className="profile-input-with-icon">
            <span className="profile-field-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7182b1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
            </span>
            <input
              type="text"
              value={form.fullName}
              onChange={updateField("fullName")}
              autoComplete="name"
              disabled={saving}
            />
          </div>
        </label>
        <label className="profile-field">
          <span>Date of Birth *</span>
          <div className="profile-input-with-icon">
            <span className="profile-field-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7182b1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
            </span>
            <input
              type="text"
              value={form.dateOfBirth}
              onChange={handleDobChange}
              placeholder="DD/MM/YYYY"
              inputMode="numeric"
              maxLength={10}
              disabled={saving}
            />
          </div>
        </label>
        <label className="profile-field">
          <span>Sex *</span>
          <div className="profile-input-with-icon">
            <span className="profile-field-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7182b1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="10" cy="14" r="6"></circle><path d="M21 3l-6.35 6.35"></path><path d="M15 3h6v6"></path></svg>
            </span>
            <select value={form.sex} onChange={updateField("sex")} disabled={saving}>
              <option value="">Select</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
              <option value="Prefer not to say">Prefer not to say</option>
            </select>
          </div>
        </label>
        <label className="profile-field">
          <span>Blood Group *</span>
          <div className="profile-input-with-icon">
            <span className="profile-field-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7182b1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2s7 8.5 7 13a7 7 0 0 1-14 0c0-4.5 7-13 7-13z"></path></svg>
            </span>
            <select value={form.bloodGroup} onChange={updateField("bloodGroup")} disabled={saving}>
              <option value="">Select</option>
              {['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'].map((group) => (
                <option value={group} key={group}>{group}</option>
              ))}
            </select>
          </div>
        </label>
        <label className="profile-field profile-field-wide">
          <span>Allergies</span>
          <div className="profile-input-with-icon">
            <span className="profile-field-icon profile-textarea-icon">✚</span>
            <textarea
              value={form.allergies}
              onChange={updateField("allergies")}
              placeholder="Comma-separated, e.g. Penicillin, peanuts"
              rows="2"
              disabled={saving}
            />
          </div>
        </label>
      </div>

      {/* ── Account details section (only in Edit Profile) ── */}
      {allowCancel && (
        <div className="profile-email-section">
          {/* Email */}
          <div className="profile-email-row">
            <span className="profile-email-label">Email</span>
            <span className="profile-email-value">{user?.email}</span>
            {!showEmailChange && (
              <button
                type="button"
                className="profile-email-change-btn"
                onClick={() => {
                  setShowEmailChange(true);
                  setEmailError("");
                  setNewEmail("");
                  setEmailPassword("");
                }}
              >
                Change
              </button>
            )}
          </div>

          {showEmailChange && (
            <div className="profile-email-form">
              {emailError && <div className="profile-form-error" role="alert">{emailError}</div>}
              <label className="profile-field">
                <span>New Email</span>
                <input
                  type="email"
                  inputMode="email"
                  autoComplete="off"
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
                  placeholder="new@email.com"
                  disabled={emailLoading}
                />
              </label>
              <label className="profile-field">
                <span>Confirm Password</span>
                <input
                  type="password"
                  autoComplete="current-password"
                  value={emailPassword}
                  onChange={(e) => setEmailPassword(e.target.value)}
                  placeholder="Enter your password"
                  disabled={emailLoading}
                />
              </label>
              <div className="profile-email-actions">
                <button
                  type="button"
                  className="profile-secondary-button"
                  onClick={() => {
                    setShowEmailChange(false);
                    setEmailError("");
                  }}
                  disabled={emailLoading}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="profile-primary-button"
                  onClick={handleEmailChange}
                  disabled={emailLoading}
                >
                  {emailLoading ? "Updating…" : "Update Email"}
                </button>
              </div>
            </div>
          )}

          {/* Security Question */}
          <div className="profile-email-row">
            <span className="profile-email-label">Security Q</span>
            <span className="profile-email-value">
              {user?.security_question || "Not set"}
            </span>
            {!showSecurityChange && (
              <button
                type="button"
                className="profile-email-change-btn"
                onClick={() => {
                  setShowSecurityChange(true);
                  setSecurityError("");
                  setNewSecurityQuestion(userQuestionKey || "");
                  setNewSecurityAnswer("");
                }}
              >
                {user?.security_question ? "Change" : "Set"}
              </button>
            )}
          </div>

          {showSecurityChange && (
            <div className="profile-email-form">
              {securityError && <div className="profile-form-error" role="alert">{securityError}</div>}
              <label className="profile-field">
                <span>Security Question</span>
                <select
                  value={newSecurityQuestion}
                  onChange={(e) => setNewSecurityQuestion(e.target.value)}
                  disabled={securityLoading}
                >
                  <option value="">Select a question</option>
                  {SECURITY_QUESTIONS.map((q, i) => (
                    <option key={SECURITY_QUESTION_KEYS[i]} value={SECURITY_QUESTION_KEYS[i]}>
                      {q}
                    </option>
                  ))}
                </select>
              </label>
              <label className="profile-field">
                <span>Your Answer</span>
                <input
                  type="text"
                  value={newSecurityAnswer}
                  onChange={(e) => setNewSecurityAnswer(e.target.value)}
                  placeholder="Enter your answer"
                  disabled={securityLoading}
                />
              </label>
              <div className="profile-email-actions">
                <button
                  type="button"
                  className="profile-secondary-button"
                  onClick={() => {
                    setShowSecurityChange(false);
                    setSecurityError("");
                  }}
                  disabled={securityLoading}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="profile-primary-button"
                  onClick={handleSecurityChange}
                  disabled={securityLoading}
                >
                  {securityLoading ? "Updating…" : "Update Question"}
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {!showEmailChange && !showSecurityChange && (
        <div className="profile-form-actions">
          {allowCancel && (
            <button className="profile-secondary-button" type="button" onClick={onCancel} disabled={saving}>
              Cancel
            </button>
          )}
          <button className="profile-primary-button" type="submit" disabled={saving}>
            {saving ? "Saving…" : submitLabel}
          </button>
        </div>
      )}
    </form>
  );
}