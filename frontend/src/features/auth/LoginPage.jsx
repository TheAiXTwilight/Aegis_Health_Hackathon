import { useEffect, useState } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { forgotPassword, verifySecurityAnswer } from '../../services/api';
import './Auth.css';

export default function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, isAuthenticated } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // Forgot password state
  const [showForgot, setShowForgot] = useState(false);
  const [forgotEmail, setForgotEmail] = useState('');
  const [forgotLoading, setForgotLoading] = useState(false);
  const [forgotSuccess, setForgotSuccess] = useState('');
  const [forgotError, setForgotError] = useState('');
  const [resetLink, setResetLink] = useState('');

  // Step 2: security question & answer
  const [securityQuestion, setSecurityQuestion] = useState('');
  const [securityAnswer, setSecurityAnswer] = useState('');
  const [answerLoading, setAnswerLoading] = useState(false);
  const [forgotStep, setForgotStep] = useState(1); // 1 = enter email, 2 = answer question

  const requestedLocation = location.state?.from;
  const redirectTo = location.state?.profileSetup
    ? '/profile/setup'
    : requestedLocation?.pathname
      ? `${requestedLocation.pathname}${requestedLocation.search || ''}${requestedLocation.hash || ''}`
      : '/dashboard';

  useEffect(() => {
    if (isAuthenticated) {
      navigate(redirectTo, { replace: true });
    }
  }, [isAuthenticated, navigate, redirectTo]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    const trimmedEmail = email.trim();
    if (!trimmedEmail) {
      setError('Please enter your email address.');
      return;
    }
    if (!password) {
      setError('Please enter your password.');
      return;
    }

    setLoading(true);
    const result = await login({ email: trimmedEmail, password });
    setLoading(false);
    if (result.success) {
      navigate(redirectTo, { replace: true });
    } else {
      console.error('Login failed:', result.error);
      setError(result.error);
    }
  };

  // Step 1: Enter email → get security question
  const handleForgotSubmit = async (e) => {
    e.preventDefault();
    setForgotError('');
    setForgotSuccess('');
    setResetLink('');
    setSecurityQuestion('');

    const trimmedEmail = forgotEmail.trim();
    if (!trimmedEmail) {
      setForgotError('Please enter your email address.');
      return;
    }

    setForgotLoading(true);
    try {
      const data = await forgotPassword({ email: trimmedEmail });
      if (data.account_exists === false) {
        setForgotError(data.detail || 'No account found with this email address.');
      } else if (data.security_question) {
        setSecurityQuestion(data.security_question);
        setForgotStep(2);
      } else {
        setForgotError(data.detail || 'Security question not set for this account.');
      }
    } catch (err) {
      setForgotError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setForgotLoading(false);
    }
  };

  // Step 2: Answer security question → get reset link
  const handleAnswerSubmit = async (e) => {
    e.preventDefault();
    setForgotError('');
    setForgotSuccess('');
    setResetLink('');

    if (!securityAnswer.trim()) {
      setForgotError('Please answer the security question.');
      return;
    }

    setAnswerLoading(true);
    try {
      const data = await verifySecurityAnswer({
        email: forgotEmail.trim(),
        securityAnswer: securityAnswer.trim(),
      });
      if (data.verified && data.reset_link) {
        setForgotSuccess(data.detail || 'Answer verified! Click below to reset your password.');
        setResetLink(data.reset_link);
      } else {
        setForgotError(data.detail || 'Incorrect answer. Please try again.');
      }
    } catch (err) {
      setForgotError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setAnswerLoading(false);
    }
  };

  const resetForgotState = () => {
    setShowForgot(false);
    setForgotStep(1);
    setSecurityQuestion('');
    setSecurityAnswer('');
    setResetLink('');
    setForgotError('');
    setForgotSuccess('');
  };

  if (isAuthenticated) return null;

  return (
    <div className="auth-center-container">
      <div className="auth-glass-card">
        {!showForgot ? (
          <>
            <h1>Welcome back</h1>
            <p className="auth-subtitle">Sign in to continue to Aegis Health</p>

            <form className="auth-form" onSubmit={handleSubmit} noValidate>
              {error && <div className="auth-error">{error}</div>}

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
                    placeholder="you@aegis.health"
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
                    placeholder="••••••••"
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

              <button className="auth-button" type="submit" disabled={loading}>
                {loading ? 'Signing in…' : 'Sign in'}
              </button>

              <button
                type="button"
                className="auth-forgot-link"
                onClick={() => {
                  setShowForgot(true);
                  setForgotEmail(email);
                  resetForgotState();
                  setShowForgot(true);
                }}
                style={{
                  display: 'block',
                  width: '100%',
                  textAlign: 'center',
                  marginTop: '8px',
                  fontSize: '14px',
                }}
                tabIndex={-1}
              >
                Forgot Password?
              </button>
            </form>

            <p className="auth-footer">
              Don't have an account?
              <Link to="/register"><button type="button">Create one</button></Link>
            </p>
          </>
        ) : (
          /* ── Forgot Password Panel ── */
          <>
            <h1>Reset Password</h1>
            <p className="auth-subtitle">
              {forgotStep === 1
                ? 'Enter your email to find your account'
                : 'Answer your security question to continue'}
            </p>

            {forgotStep === 1 ? (
              /* Step 1: Enter email */
              <form className="auth-form" onSubmit={handleForgotSubmit} noValidate>
                {forgotError && <div className="auth-error">{forgotError}</div>}

                <div className="auth-field">
                  <label htmlFor="forgot-email">Email</label>
                  <div className="auth-input-wrapper">
                    <input
                      id="forgot-email"
                      type="email"
                      inputMode="email"
                      autoComplete="email"
                      value={forgotEmail}
                      onChange={(e) => setForgotEmail(e.target.value)}
                      placeholder="you@gmail.com"
                      disabled={forgotLoading}
                    />
                  </div>
                </div>

                <button className="auth-button" type="submit" disabled={forgotLoading}>
                  {forgotLoading ? 'Finding account…' : 'Continue'}
                </button>
              </form>
            ) : (
              /* Step 2: Answer security question */
              <form className="auth-form" onSubmit={handleAnswerSubmit} noValidate>
                {forgotError && <div className="auth-error">{forgotError}</div>}
                {forgotSuccess && <div className="auth-success">{forgotSuccess}</div>}

                <div className="auth-field">
                  <label>Security Question</label>
                  <div style={{
                    padding: '14px 16px',
                    borderRadius: '16px',
                    border: '1px solid rgba(255, 255, 255, 0.3)',
                    background: 'rgba(255, 255, 255, 0.1)',
                    fontSize: '15px',
                    color: '#0d2167',
                    fontWeight: 600,
                  }}>
                    {securityQuestion}
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
                      placeholder="Enter your answer"
                      disabled={answerLoading}
                      autoFocus
                    />
                  </div>
                </div>

                {resetLink && (
                  <button
                    type="button"
                    onClick={() => {
                      const token = new URL(resetLink).searchParams.get('token');
                      if (token) navigate(`/reset-password?token=${token}`);
                    }}
                    style={{
                      display: 'block',
                      width: '100%',
                      boxSizing: 'border-box',
                      padding: '14px 24px',
                      borderRadius: '16px',
                      border: '1px solid rgba(255, 255, 255, 0.35)',
                      background: 'rgba(37, 99, 255, 0.15)',
                      backdropFilter: 'blur(16px) saturate(1.3)',
                      color: '#2563ff',
                      fontSize: '16px',
                      fontWeight: 700,
                      textAlign: 'center',
                      textDecoration: 'none',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.5)',
                    }}
                  >
                    Reset Password →
                  </button>
                )}

                {!resetLink && (
                  <button className="auth-button" type="submit" disabled={answerLoading}>
                    {answerLoading ? 'Verifying…' : 'Verify Answer'}
                  </button>
                )}
              </form>
            )}

            <p className="auth-footer">
              Remember your password?
              <button type="button" onClick={resetForgotState}>Back to Sign in</button>
            </p>
          </>
        )}
      </div>
    </div>
  );
}
