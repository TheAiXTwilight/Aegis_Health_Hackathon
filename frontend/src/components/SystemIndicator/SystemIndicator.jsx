import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { getHealth } from "../../services/api";
import "./SystemIndicator.css";

const FALLBACK_STATS = {
  ramUsedMb: 0,
  cpuPercent: null,
  gpuAvailable: false,
  backendOnline: false,
};

const clamp = (value) => Math.max(0, Math.min(100, value));

function normalizeHealth(payload) {
  // Was reading memory_used_mb / memory_total_mb / cpu_percent — those
  // are SYSTEM-WIDE (whole machine, from /proc/meminfo and /proc/stat).
  // Switched to app_memory_used_mb / app_cpu_percent, which are scoped
  // to this backend process only (via /proc/self/*), so the indicator
  // reflects what AegisHealth itself is using, not the whole device.
  // Note: app_cpu_percent is a percent of ONE core (matches `top`
  // convention) and can exceed 100% under multi-threaded load — it is
  // NOT normalized against total core count like system cpu_percent was.
  const appUsedMb = Number(payload?.app_memory_used_mb || 0);
  const cpuValue = payload?.app_cpu_percent;

  return {
    ramUsedMb: appUsedMb > 0 ? appUsedMb : 0,
    cpuPercent: cpuValue === null || cpuValue === undefined
      ? null
      : Math.round(Number(cpuValue)),
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

  // App RAM has no natural "total" the way system RAM does (there's no
  // single ceiling for "how much RAM could this process use"), so the
  // bar fill uses a soft reference ceiling just to give the bar visual
  // motion — the displayed number is always the real MB/GB value.
  const RAM_BAR_REFERENCE_MB = 1024; // 1 GB — adjust if typical footprint changes
  const ramPercent = clamp(Math.round((stats.ramUsedMb / RAM_BAR_REFERENCE_MB) * 100));
  // App CPU is a percent of ONE core and can legitimately exceed 100%
  // under multi-threaded work — clamp only the bar's visual fill, not
  // the displayed number.
  const cpuDisplay = stats.cpuPercent === null ? 0 : Math.max(0, stats.cpuPercent);
  const cpuBarPercent = clamp(cpuDisplay);

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
            {stats.ramUsedMb > 0
              ? stats.ramUsedMb >= 1024
                ? `${(stats.ramUsedMb / 1024).toFixed(1)} GB`
                : `${Math.round(stats.ramUsedMb)} MB`
              : "-- MB"}
          </span>
        </div>

        <div className="resource-item">
          <span className="resource-label">CPU</span>

          <div
            className="resource-bar"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={cpuBarPercent}
          >
            <div
              className="resource-fill cpu-fill"
              style={{ width: `${cpuBarPercent}%` }}
            />
          </div>

          <span className="resource-value">
            {stats.backendOnline && stats.cpuPercent !== null ? `${cpuDisplay}%` : "--%"}
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