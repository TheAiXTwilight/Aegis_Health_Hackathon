import { useEffect } from "react";
import ProfileForm from "./ProfileForm";
import "./Profile.css";
import "./ProfileGlassOverride.css";

export default function ProfileModal({ onClose }) {
  useEffect(() => {
    function handleEscape(event) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [onClose]);

  return (
    <div className="profile-modal-overlay" onMouseDown={onClose} role="presentation">
      <section
        className="profile-modal-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="profile-modal-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="profile-modal-header">
          <div>
            <h2 id="profile-modal-title">Your Profile</h2>
            <p>Review or update the information used to pre-fill assessments.</p>
          </div>
          <button className="profile-modal-close" type="button" onClick={onClose} aria-label="Close profile">
            ✕
          </button>
        </div>
        <div className="profile-modal-scroll">
          <ProfileForm
            allowCancel
            onCancel={onClose}
            onSaved={onClose}
            submitLabel="Save changes"
          />
        </div>
      </section>
    </div>
  );
}
