"""
app/settings.py — Centralised application settings.
Uses pydantic-settings to read from environment / .env.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    AEGIS_DB_URL: str = f"sqlite:///{_PROJECT_ROOT / 'data' / 'aegis.db'}"

    # Development fallback only. Set a unique random value in .env for every
    # deployed installation (for example: `openssl rand -hex 32`).
    AEGIS_SECRET_KEY: str = "dev-only-change-me-aegis-health-2026"

    AEGIS_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    AEGIS_REFRESH_TOKEN_EXPIRE_HOURS: int = 72
    AEGIS_CORS_ORIGINS: list[str] = ["http://localhost:5173"]
    AEGIS_UPLOAD_ROOT: str = "/tmp/aegis_uploads"
    AEGIS_CHECKPOINT_DIR: str = "/tmp/aegis_checkpoint"

    # — Auth (JWT) —
    AEGIS_JWT_ALGORITHM: str = "HS256"
    AEGIS_ACCESS_COOKIE_NAME: str = "aegis_access"
    AEGIS_REFRESH_COOKIE_NAME: str = "aegis_refresh"
    AEGIS_LOGIN_RATE_LIMIT_ATTEMPTS: int = 5
    AEGIS_LOGIN_RATE_LIMIT_WINDOW_S: int = 60

    # Real accounts are created through /auth/register. Demo users remain
    # available as an explicit opt-in for presentations and development only.
    AEGIS_SEED_DEMO_USERS: bool = False

    COOKIE_SECURE: bool = False  # set True behind HTTPS in production

    # — Queue —
    AEGIS_QUEUE_MIN_SIZE: int = 5
    AEGIS_QUEUE_MAX_SIZE: int = 20
    AEGIS_QUEUE_RATE_LIMIT_ATTEMPTS: int = 5
    AEGIS_QUEUE_RATE_LIMIT_WINDOW_S: int = 60
    AEGIS_QUEUE_TARGET_WAIT_S: float = 60.0

    # — Ollama —
    AEGIS_OLLAMA_BASE_URL: str = "http://localhost:11434"
    AEGIS_OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_NUM_CTX: int = 3072        # tuned down from 4096 for Jetson
    OLLAMA_NUM_PREDICT: int = 768     # tuned down from 1024
    OLLAMA_TEMPERATURE: float = 0.2
    OLLAMA_MEMORY_FLOOR_MB: int = 900
    OLLAMA_UI_STEP_SECONDS: float = 0.0

    AEGIS_ENV: str = "development"

    # Text-to-Speech (Piper, fully local, no external API).
    AEGIS_TTS_MODEL_DIR: str = "data/audio/piper-en-gb-jenny-medium"
    AEGIS_TTS_VOICE_NAME: str = "en_GB-jenny_dioco-medium"
    AEGIS_TTS_CACHE_DIR: str = "/tmp/aegis_tts_cache"
    AEGIS_TTS_MAX_CHARS: int = 20000

    # Idle-eviction window for the Piper voice model. The voice is
    # loaded on-demand (triggered when a report job starts running or
    # when the frontend polls /tts/status/{job_id}) — it is NOT preloaded
    # at server boot, so an idle server holds zero TTS memory. Once
    # loaded, the voice occupies ~180MB of RAM (ONNX weights +
    # activation buffers). After AEGIS_TTS_IDLE_EVICT_SECS with no TTS
    # activity of any kind, a background monitor task unloads the voice
    # and reclaims that memory. Reload on next use is transparent — the
    # next request just pays the ~500ms cold-load cost once. Set to 0
    # (or any non-positive value) to disable eviction entirely and keep
    # the voice loaded until process exit once first used.
    AEGIS_TTS_IDLE_EVICT_SECS: int = 600  # 10 minutes

    # How often the idle-monitor task wakes up to check whether the
    # voice should be evicted. 60s check granularity is fine for a
    # 10-minute eviction window — no reason to tune this unless the
    # eviction window itself is being changed to something very short.
    AEGIS_TTS_IDLE_CHECK_SECS: int = 60

    # ── Email (Resend API — recommended) ──────────────────────────
    # Get your free API key at https://resend.com (100 emails/day free tier)
    # When set, Resend is used; falls back to SMTP if not set.
    AEGIS_RESEND_API_KEY: str = ""

    # ── SMTP / Email (fallback) ──────────────────────────────────
    # All default to empty/off. When SMTP_HOST is set, the forgot-password
    # flow will send real emails; otherwise the token is logged only.
    AEGIS_SMTP_HOST: str = ""
    AEGIS_SMTP_PORT: int = 587
    AEGIS_SMTP_USER: str = ""
    AEGIS_SMTP_PASSWORD: str = ""
    AEGIS_SMTP_FROM: str = ""  # e.g. "Aegis Health <noreply@aegis.health>"
    AEGIS_SMTP_USE_TLS: bool = True
    AEGIS_SMTP_SSL: bool = False  # Use SSL (port 465) instead of STARTTLS
    # The base URL for constructing reset links. e.g. "http://localhost:5173"
    AEGIS_PUBLIC_URL: str = "http://localhost:5173"


settings = Settings()