import { Link } from "react-router-dom";
import "../auth/Auth.css";
import "../health-scan/HealthScan.css";
import "./Dashboard.css";

export default function EmptyDashboard() {
  return (
    <div className="scan-center-container">
      <div className="scan-glass-card">
        <h2>Health Scan</h2>
        <p className="scan-subtitle">
          Complete the health form to begin your health analysis.
        </p>
        <p className="scan-desc">
          Our multi-agent AI system analyzes everything locally on your device.
        </p>
        <Link to="/medical-form" style={{ textDecoration: "none" }}>
          <button className="form-action-btn">
            <span className="form-icon">📄</span>
            <span>Fill the Form</span>
          </button>
        </Link>
        <div className="time-estimate">
          <span className="clock-icon">🕒</span>
          <span>Takes approximately 5–10 minutes to complete</span>
        </div>
      </div>
    </div>
  );
}
