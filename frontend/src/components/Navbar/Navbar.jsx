import { useEffect, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import SystemIndicator from "../SystemIndicator/SystemIndicator";
import ProfileModal from "../Profile/ProfileModal";
import { useAuth } from "../../context/AuthContext";
import "./Navbar.css";

export default function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, isAuthenticated, logout } = useAuth();
  const [profileOpen, setProfileOpen] = useState(false);
  const [profileModalOpen, setProfileModalOpen] = useState(false);
  const profileRef = useRef(null);

  const isActive = (path) => location.pathname === path;
  const isAuthPage = location.pathname === "/login" || location.pathname === "/register";

  useEffect(() => {
    function handlePointerDown(event) {
      if (profileRef.current && !profileRef.current.contains(event.target)) {
        setProfileOpen(false);
      }
    }

    function handleEscape(event) {
      if (event.key === "Escape") setProfileOpen(false);
    }

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleEscape);
    };
  }, []);

  useEffect(() => {
    setProfileOpen(false);
    setProfileModalOpen(false);
  }, [location.pathname, location.search]);

  const handleLogout = async () => {
    setProfileOpen(false);
    await logout();
    navigate("/login", { replace: true });
  };

  const handleDashboardClick = () => {
    window.dispatchEvent(new Event("aegis:dashboard-refresh"));
  };

  const initials = (user?.display_name || user?.email || "U")
    .split(/\s+/)
    .map((part) => part[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);

  const memberSince = user?.created_at
    ? new Date(user.created_at).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      })
    : "—";

  return (
    <nav className="navbar">
      <div className="logo">
        Aegis<span>Health</span>
      </div>

      <div className="nav-links">
        <Link to="/">
          <button type="button" className={isActive("/") ? "active-nav" : ""}>
            Home
          </button>
        </Link>

        <Link to="/dashboard" onClick={handleDashboardClick}>
          <button
            type="button"
            className={
              isActive("/dashboard") || isActive("/medical-form") ? "active-nav" : ""
            }
          >
            Dashboard
          </button>
        </Link>

        <Link to="/report">
          <button type="button" className={isActive("/report") ? "active-nav" : ""}>
            Report
          </button>
        </Link>

        <Link to="/about">
          <button type="button" className={isActive("/about") ? "active-nav" : ""}>
            About
          </button>
        </Link>
      </div>

      <div className="nav-right">
        <SystemIndicator />

        <div className="nav-user">
          {isAuthenticated && (
            <div className="nav-profile" ref={profileRef}>
              <button
                className={`nav-profile-button${profileOpen ? " nav-profile-button-open" : ""}`}
                type="button"
                onClick={() => setProfileOpen((open) => !open)}
                aria-label="Open profile menu"
                aria-haspopup="menu"
                aria-expanded={profileOpen}
                title="Profile"
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <circle cx="12" cy="8" r="4" />
                  <path d="M4 21a8 8 0 0 1 16 0" />
                </svg>
              </button>

              {profileOpen && (
                <div className="nav-profile-menu" role="menu" aria-label="Profile">
                  <div className="nav-profile-header">
                    <div className="nav-profile-avatar">{initials}</div>
                    <div className="nav-profile-heading">
                      <strong>{user?.display_name || "User"}</strong>
                      <span>{user?.email}</span>
                    </div>
                  </div>

                  <div className="nav-profile-details">
                    <div className="nav-profile-detail">
                      <span>Full name</span>
                      <strong>{user?.display_name || "—"}</strong>
                    </div>
                    <div className="nav-profile-detail">
                      <span>Email</span>
                      <strong>{user?.email || "—"}</strong>
                    </div>
                    {user?.phone && (
                      <div className="nav-profile-detail">
                        <span>Phone</span>
                        <strong>{user.phone}</strong>
                      </div>
                    )}
                    {user?.security_question && (
                      <div className="nav-profile-detail">
                        <span>Security Q Ans</span>
                        <strong>{user.security_answer || "—"}</strong>
                      </div>
                    )}
                    <div className="nav-profile-detail">
                      <span>Member since</span>
                      <strong>{memberSince}</strong>
                    </div>
                  </div>

                  <button
                    className="nav-profile-edit"
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setProfileOpen(false);
                      setProfileModalOpen(true);
                    }}
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M12 20h9" />
                      <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4z" />
                    </svg>
                    Edit Profile
                  </button>

                  <button
                    className="nav-profile-logout"
                    type="button"
                    role="menuitem"
                    onClick={handleLogout}
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M10 17l5-5-5-5" />
                      <path d="M15 12H3" />
                      <path d="M14 3h5a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-5" />
                    </svg>
                    Logout
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {profileModalOpen && (
        <ProfileModal onClose={() => setProfileModalOpen(false)} />
      )}
    </nav>
  );
}
