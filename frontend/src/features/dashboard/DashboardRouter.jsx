import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { getDashboard } from "../../services/api";
import DashboardPage from "./DashboardPage";
import EmptyDashboard from "./EmptyDashboard";
import "../auth/Auth.css";
import "./Dashboard.css";

const KEY_INSIGHTS_SYNC_DELAYS_MS = [0, 150, 300, 500, 750, 1000];

function wait(delayMs) {
  return new Promise((resolve) => window.setTimeout(resolve, delayMs));
}

function getReportSnapshot(data) {
  const recentRecords = Array.isArray(data?.recent_records)
    ? data.recent_records
    : [];
  const reportedTotal = Number.isFinite(data?.total_records)
    ? data.total_records
    : 0;

  return {
    recentRecords,
    totalRecords: Math.max(reportedTotal, recentRecords.length),
  };
}

export default function DashboardRouter() {
  const [searchParams] = useSearchParams();
  const requestedJobId = searchParams.get("jobId");
  const shouldSyncRequestedReport = searchParams.get("sync") === "1";
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [requestVersion, setRequestVersion] = useState(0);

  const retry = useCallback(() => {
    setRequestVersion((version) => version + 1);
  }, []);

  useEffect(() => {
    window.addEventListener("aegis:dashboard-refresh", retry);
    return () => window.removeEventListener("aegis:dashboard-refresh", retry);
  }, [retry]);

  useEffect(() => {
    let cancelled = false;

    async function loadDashboard() {
      setLoading(true);
      setError("");
      setDashboardData(null);

      let lastRequestError = null;
      let lastSnapshot = null;
      const delays = shouldSyncRequestedReport
        ? KEY_INSIGHTS_SYNC_DELAYS_MS
        : [0];

      try {
        for (const delayMs of delays) {
          if (delayMs > 0) await wait(delayMs);
          if (cancelled) return;

          try {
            const data = await getDashboard();
            const snapshot = getReportSnapshot(data);
            const requestedReportIsReady = snapshot.recentRecords.some(
              (record) => record.job_id === requestedJobId
            );

            lastSnapshot = snapshot;
            lastRequestError = null;

            // The queue marks a job completed immediately before its
            // HealthRecord transaction finishes. When Key Insights is opened
            // for the active report, wait until that job appears so the first
            // report cannot open EmptyDashboard and later reports cannot open
            // a comparison that is missing the newly completed report.
            if (!shouldSyncRequestedReport || requestedReportIsReady) {
              if (!cancelled) setDashboardData(data);
              return;
            }
          } catch (err) {
            lastRequestError = err;
          }
        }

        if (!cancelled) {
          setError(
            lastRequestError?.message ||
              (lastSnapshot
                ? "Your completed report is still syncing with the dashboard. Please try again."
                : "Unable to load your dashboard.")
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadDashboard();

    return () => {
      cancelled = true;
    };
  }, [requestVersion, requestedJobId, shouldSyncRequestedReport]);

  if (loading) {
    return (
      <div className="dashv2-page">
        <div className="dashv2-loading" role="status" aria-live="polite">
          <div className="dashv2-spinner" />
          <span className="dashv2-loading-text">Loading your dashboard...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="auth-center-container">
        <div className="auth-glass-card" role="alert">
          <h1>Dashboard unavailable</h1>
          <p className="auth-subtitle">{error}</p>

          <div className="auth-form">
            <button className="auth-button" type="button" onClick={retry}>
              Try again
            </button>
          </div>
        </div>
      </div>
    );
  }

  const { totalRecords } = getReportSnapshot(dashboardData);

  if (totalRecords === 0) {
    return <EmptyDashboard />;
  }

  // DashboardPage already contains both existing dashboard variants.
  // The key documents and enforces the route decision without changing it:
  // 1 report = single-purpose; 2+ reports = dual-purpose comparison.
  const dashboardVariant = totalRecords === 1 ? "single" : "dual";
  return <DashboardPage key={`${dashboardVariant}-${requestVersion}`} />;
}
