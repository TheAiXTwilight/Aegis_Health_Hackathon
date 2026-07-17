import { useLocation } from "react-router-dom";
import "./Footer.css";

export default function Footer() {
  const location = useLocation();
  const isDashboard = location.pathname === "/dashboard";

  if (isDashboard) return null;

  return (
    <div className="footer-note">
      <span className="footer-lock-icon" aria-hidden="true">🔒</span>
      <span>All processing is done locally. Your data never leaves your device.</span>
    </div>
  );
}
