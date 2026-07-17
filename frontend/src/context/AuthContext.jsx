import { createContext, useContext, useEffect, useState, useCallback } from 'react';
import {
  clearAccessToken,
  getMe,
  login as apiLogin,
  logout as apiLogout,
  refreshAccessToken,
  register as apiRegister,
} from '../services/api';

const AuthContext = createContext(null);

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return ctx;
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const clearSession = useCallback(() => {
    clearAccessToken();
    setUser(null);
  }, []);

  const hydrate = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // getMe authenticates with the httpOnly access cookie and performs one
      // silent refresh when required. No dummy/localStorage user is hydrated.
      const profile = await getMe();
      if (profile) {
        setUser(profile);
      } else {
        clearSession();
      }
    } catch {
      clearSession();
    } finally {
      setLoading(false);
    }
  }, [clearSession]);

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  useEffect(() => {
    if (!user) return undefined;

    const interval = window.setInterval(() => {
      refreshAccessToken().then((ok) => {
        if (!ok) clearSession();
      });
    }, 25 * 60 * 1000);

    return () => window.clearInterval(interval);
  }, [user, clearSession]);

  const login = useCallback(async (credentials) => {
    setError(null);
    try {
      const data = await apiLogin(credentials);
      const profile = await getMe();
      if (!profile) {
        clearSession();
        throw new Error('Could not retrieve your profile. Please try again.');
      }
      setUser(profile);
      return { success: true, data };
    } catch (err) {
      setError(err.message || 'Login failed');
      return { success: false, error: err.message || 'Login failed' };
    }
  }, [clearSession]);

  const register = useCallback(async (payload) => {
    setError(null);
    try {
      const data = await apiRegister(payload);
      return { success: true, data };
    } catch (err) {
      setError(err.message || 'Registration failed');
      return { success: false, error: err.message || 'Registration failed' };
    }
  }, []);

  const updateUser = useCallback((updates) => {
    setUser((current) => (current ? { ...current, ...updates } : current));
  }, []);

  const logout = useCallback(async () => {
    try {
      await apiLogout();
    } catch {
      // best-effort: still clear local state even if the backend call fails
    }
    clearSession();
  }, [clearSession]);

  const value = {
    user,
    loading,
    error,
    isAuthenticated: Boolean(user),
    login,
    register,
    logout,
    updateUser,
    refresh: hydrate,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export default AuthContext;

