import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { getHealth } from "../../services/api";
import "./SystemIndicator.css";

const FALLBACK_STATS = {
  ramUsedGb: 0,
  ramTotalGb: 0,
  cpuPercent: null,
  gpuAvailable: false,
  backendOnline: false,
};

const clamp = (value) => Math.max(0, Math.min(100, value));

function normalizeHealth(payload) {
  const usedMb = Number(payload?.memory_used_mb || 0);
  const totalMb = Number(payload?.memory_total_mb || 0);
  const cpuValue = payload?.cpu_percent;

  return {
    ramUsedGb: usedMb > 0 ? usedMb / 1024 : 0,
    ramTotalGb: totalMb > 0 ? totalMb / 1024 : 0,
    cpuPercent: cpuValue === null || cpuValue === undefined
      ? null
      : clamp(Math.round(Number(cpuValue))),
    gpuAvailable: Boolean(payload?.gpu_available),
    backendOnline: true,
  };
}

export default function SystemIndicator({ isProcessing = false }) {
  const location = useLocation();
  const [stats, setStats] = useState(FALLBACK_STATS);

  const [pipelineStatus, setPipelineStatus] = useState(() => {
    return localStorage.getItem("aegis_pipeline_status") || "idle";
  });

  useEffect(() => {
    let cancelled = false;

    async function pollHealth() {
      try {
        const payload = await getHealth();
        if (!cancelled) setStats(normalizeHealth(payload));
      } catch (error) {
        if (!cancelled) {
          setStats((prev) => ({ ...prev, backendOnline: false }));
        }
      }
    }

    pollHealth();
    const intervalId = window.setInterval(pollHealth, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    function handlePipelineStatus(event) {
      const nextStatus = event?.detail?.status || "idle";
      setPipelineStatus(nextStatus);
    }

    function handleStorage(event) {
      if (event.key === "aegis_pipeline_status") {
        setPipelineStatus(event.newValue || "idle");
      }
    }

    window.addEventListener("aegis:pipeline-status", handlePipelineStatus);
    window.addEventListener("storage", handleStorage);

    return () => {
      window.removeEventListener("aegis:pipeline-status", handlePipelineStatus);
      window.removeEventListener("storage", handleStorage);
    };
  }, []);

  const ramPercent = stats.ramTotalGb > 0
    ? clamp(Math.round((stats.ramUsedGb / stats.ramTotalGb) * 100))
    : 0;
  const cpuPercent = stats.cpuPercent === null ? 0 : clamp(stats.cpuPercent);

  const searchParams = new URLSearchParams(location.search);
  const hasReportJobId = Boolean(searchParams.get("jobId"));

  // UPDATED: Now active for BOTH Report and Dashboard pages so the indicator stays green
  const isActiveJobPage =
    (location.pathname.startsWith("/report") || location.pathname.startsWith("/dashboard")) && hasReportJobId;

  let effectiveStatus = "idle";

  if (isActiveJobPage) {
    effectiveStatus = isProcessing ? "running" : pipelineStatus;
  } else {
    effectiveStatus = "idle";
  }

  const dotClass =
    !stats.backendOnline
      ? "failed"
      : effectiveStatus === "running"
      ? "running"
      : effectiveStatus === "completed"
      ? "completed"
      : effectiveStatus === "failed"
      ? "failed"
      : "idle";

  const dotTitle =
    !stats.backendOnline
      ? "Backend offline"
      : effectiveStatus === "running"
      ? "Processing..."
      : effectiveStatus === "completed"
      ? "Processing completed"
      : effectiveStatus === "failed"
      ? "Processing failed"
      : "Idle";

  return (
    <div className="system-indicator">
      <div
        className={`status-dot ${dotClass}`}
        title={dotTitle}
        aria-label={dotTitle}
      />

      <div className="resource-group">
        <div className="resource-item">
          <span className="resource-label">RAM</span>

          <div
            className="resource-bar"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={ramPercent}
          >
            <div
              className="resource-fill ram-fill"
              style={{ width: `${ramPercent}%` }}
            />
          </div>

          <span className="resource-value">
            {stats.ramTotalGb > 0
              ? `${stats.ramUsedGb.toFixed(1)} / ${stats.ramTotalGb.toFixed(1)} GB`
              : "-- / -- GB"}
          </span>
        </div>

        <div className="resource-item">
          <span className="resource-label">CPU</span>

          <div
            className="resource-bar"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={cpuPercent}
          >
            <div
              className="resource-fill cpu-fill"
              style={{ width: `${cpuPercent}%` }}
            />
          </div>

          <span className="resource-value">
            {stats.backendOnline && stats.cpuPercent !== null ? `${cpuPercent}%` : "--%"}
          </span>
        </div>

        {stats.backendOnline && stats.gpuAvailable && (
          <div className="resource-item compact-resource-note">
            <span className="resource-label">GPU</span>
            <span className="resource-value">Jetson/NVIDIA ready</span>
          </div>
        )}
      </div>
    </div>
  );
}