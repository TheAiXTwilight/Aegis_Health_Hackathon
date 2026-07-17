import { useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { register as apiRegister } from '../../services/api';
import './Auth.css';

const SECURITY_QUESTIONS = [
  "What is your favorite pet's name?",
  "What is your favorite food?",
  "What city were you born in?",
  "What is your favorite movie?",
];

const SECURITY_QUESTION_KEYS = ["pet_name", "favorite_food", "birth_city", "favorite_movie"];

export default function RegisterPage() {
  const navigate = useNavigate();
  const { register, login, isAuthenticated } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [displayName, setDisplayName] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [securityQuestion, setSecurityQuestion] = useState('');
  const [securityAnswer, setSecurityAnswer] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [profileSetupPending, setProfileSetupPending] = useState(false);

  useEffect(() => {
    if (isAuthenticated && !profileSetupPending) {
      navigate('/dashboard', { replace: true });
    }
  }, [isAuthenticated, navigate, profileSetupPending]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    const trimmedEmail = email.trim();
    const trimmedDisplayName = displayName.trim();
    const trimmedPhone = phoneNumber.trim();

    if (!trimmedDisplayName) {
      setError('Please enter your full name.');
      return;
    }
    if (!trimmedEmail) {
      setError('Please enter your email address.');
      return;
    }
    if (!trimmedPhone) {
      setError('Please enter your phone number.');
      return;
    }
    if (!password || password.length < 8) {
      setError('Password must be at least 8 characters.');
      return;
    }
    if (!securityQuestion) {
      setError('Please select a security question.');
      return;
    }
    if (!securityAnswer.trim()) {
      setError('Please answer the security question.');
      return;
    }

    setLoading(true);
    setProfileSetupPending(true);

    try {
      const regResult = await apiRegister({
        email: trimmedEmail,
        password,
        displayName: trimmedDisplayName,
        phone: trimmedPhone,
        securityQuestion,
        securityAnswer: securityAnswer.trim(),
      });

      if (regResult && regResult.id) {
        const loginResult = await login({ email: trimmedEmail, password });
        setLoading(false);
        if (loginResult.success) {
          navigate('/profile/setup', { replace: true });
        } else {
          navigate('/login', {
            replace: true,
            state: { profileSetup: true },
          });
        }
      } else {
        setLoading(false);
        setProfileSetupPending(false);
        setError(regResult?.detail || 'Registration failed.');
      }
    } catch (err) {
      setLoading(false);
      setProfileSetupPending(false);
      setError(err.message || 'Registration failed. Please try again.');
    }
  };

  if (isAuthenticated) return null;

  return (
    <div className="auth-center-container">
      <div className="auth-glass-card">
        <h1>Create account</h1>
        <p className="auth-subtitle">Join Aegis Health to track your health</p>

        <form className="auth-form" onSubmit={handleSubmit} noValidate>
          {error && <div className="auth-error">{error}</div>}

          <div className="auth-field">
            <label htmlFor="displayName">Full name</label>
            <div className="auth-input-wrapper">
              <input
                id="displayName"
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="Your full name"
                disabled={loading}
              />
            </div>
          </div>

          <div className="auth-field">
            <label htmlFor="email">Email</label>
            <div className="auth-input-wrapper">
              <input
                id="email"
                type="email"
                inputMode="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@gmail.com"
                disabled={loading}
              />
            </div>
          </div>

          <div className="auth-field">
            <label htmlFor="phoneNumber">Phone Number</label>
            <div className="auth-input-wrapper">
              <input
                id="phoneNumber"
                type="tel"
                inputMode="tel"
                autoComplete="tel"
                value={phoneNumber}
                onChange={(e) => setPhoneNumber(e.target.value)}
                placeholder="+91 98765 43210"
                disabled={loading}
              />
            </div>
          </div>

          <div className="auth-field">
            <label htmlFor="password">Password</label>
            <div className="auth-input-wrapper">
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="At least 8 characters"
                disabled={loading}
                className="password-input"
              />
              <button
                type="button"
                className="auth-eye-btn"
                onClick={() => setShowPassword((v) => !v)}
                tabIndex={-1}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
                    <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
                    <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
                    <line x1="1" y1="1" x2="23" y2="23" />
                  </svg>
                ) : (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                )}
              </button>
            </div>
          </div>

          <div className="auth-field">
            <div className="auth-field-row">
              <label htmlFor="securityQuestion">Security Question</label>
              <span style={{ fontSize: '11px', color: '#4c5d8f', fontWeight: 500 }}>Helps recover password</span>
            </div>
            <div className="auth-input-wrapper">
              <select
                id="securityQuestion"
                value={securityQuestion}
                onChange={(e) => setSecurityQuestion(e.target.value)}
                disabled={loading}
                style={{
                  width: '100%',
                  boxSizing: 'border-box',
                  padding: '14px 16px',
                  borderRadius: '16px',
                  border: '1px solid rgba(255, 255, 255, 0.5)',
                  background: 'rgba(255, 255, 255, 0.2)',
                  fontSize: '15px',
                  color: securityQuestion ? '#0d2167' : 'rgba(13, 33, 103, 0.45)',
                  outline: 'none',
                  appearance: 'none',
                  cursor: 'pointer',
                }}
              >
                <option value="" disabled>Select a question</option>
                {SECURITY_QUESTIONS.map((q, i) => (
                  <option key={SECURITY_QUESTION_KEYS[i]} value={SECURITY_QUESTION_KEYS[i]}>{q}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="auth-field">
            <label htmlFor="securityAnswer">Your Answer</label>
            <div className="auth-input-wrapper">
              <input
                id="securityAnswer"
                type="text"
                value={securityAnswer}
                onChange={(e) => setSecurityAnswer(e.target.value)}
                placeholder="Remember this — you'll need it to reset your password"
                disabled={loading}
              />
            </div>
          </div>

          <button className="auth-button" type="submit" disabled={loading}>
            {loading ? 'Creating account…' : 'Create account'}
          </button>
        </form>

        <p className="auth-footer">
          Already have an account?
          <Link to="/login"><button type="button">Sign in</button></Link>
        </p>
      </div>
    </div>
  );
}
