import { useNavigate } from "react-router-dom";
import ProfileForm from "../../components/Profile/ProfileForm";
import "../../components/Profile/Profile.css";
import "../../components/Profile/ProfileGlassOverride.css";

export default function ProfileSetupPage() {
  const navigate = useNavigate();

  return (
    <div className="profile-setup-container">
      <section className="profile-setup-card" aria-labelledby="profile-setup-title">
        <div className="profile-setup-heading">
          <span className="profile-setup-step">Profile setup</span>
          <h1 id="profile-setup-title">Tell us about yourself</h1>
          <p>
            Save your basic information once. Aegis Health will use it to pre-fill future medical assessments.
          </p>
        </div>
        <ProfileForm
          submitLabel="Save and continue"
          onSaved={() => navigate("/dashboard", { replace: true })}
        />
      </section>
    </div>
  );
}
