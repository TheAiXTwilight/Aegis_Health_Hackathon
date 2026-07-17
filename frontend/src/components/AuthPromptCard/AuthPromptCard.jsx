import { Link } from 'react-router-dom';
import './AuthPromptCard.css';

export default function AuthPromptCard({ title = 'Access Required' }) {
  return (
    <div className="prompt-center-container">
      <div className="prompt-glass-card">
        <h2>{title}</h2>
        <p className="prompt-subtitle">Please log in or sign up first to continue.</p>
        <div className="prompt-actions">
          <Link to="/login">
            <button className="prompt-action-btn" type="button">Log in</button>
          </Link>
          <Link to="/register">
            <button className="prompt-action-btn secondary" type="button">Sign up</button>
          </Link>
        </div>
      </div>
    </div>
  );
}
