import { useNavigate } from "react-router-dom";
import "./Home.css";

export default function HomePage() {
  const navigate = useNavigate();

  const startHealthScan = () => {
    navigate("/medical-form");
  };

  return (
    <div className="hero-container">
      <div className="hero-card">
        <div className="tag">
          <span className="tag-icon">💙</span> AI POWERED HEALTHCARE
        </div>

        <h3>
          AI-Powered Healthcare.
          <br />
          Private. Local. Intelligent.
        </h3>

        <p>
          AegisHealth is your privacy-first AI assistant for
          intelligent medical analysis. All processing
          happens locally on your device to keep your data
          secure and confidential.
        </p>

        <h1>
          Analyze. Understand. Act with Confidence.
        </h1>

        <button className="scan-btn" onClick={startHealthScan}>
          <span>Start Your Health Scan</span>
          <span className="arrow">→</span>
        </button>
      </div>
    </div>
  );
}
