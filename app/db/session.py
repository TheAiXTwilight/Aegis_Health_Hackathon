"""
app/db/session.py — SQLAlchemy engine, session factory, and Base.

Provides:
    Base          — DeclarativeBase for all models
    engine        — SQLAlchemy engine (SQLite with WAL mode)
    SessionLocal  — sessionmaker for per-request sessions
    get_db()      — FastAPI dependency yielding a session
    init_db()     — create all tables (called at startup)
"""
from __future__ import annotations

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.settings import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


engine = create_engine(
    settings.AEGIS_DB_URL,
    connect_args=(
        {"check_same_thread": False}
        if settings.AEGIS_DB_URL.startswith("sqlite")
        else {}
    ),
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_connection, _connection_record):
    """Enable foreign keys and WAL journal mode on every SQLite connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create all tables. Import models first so they register on Base.

    Also creates the parent directory for SQLite databases —
    without this, sqlite:///./data/aegis.db fails if ./data/ doesn't exist.
    In Docker, /app/data/ is already mounted as a volume, so this is a no-op.
    """
    import os
    from pathlib import Path

    # Extract the file path from the SQLite URL and ensure parent dir exists
    db_url = settings.AEGIS_DB_URL
    if db_url.startswith("sqlite:///"):
        # sqlite:///./data/aegis.db  -> ./data/aegis.db
        # sqlite:////app/data/aegis.db -> /app/data/aegis.db
        file_path = db_url.removeprefix("sqlite:///")
        parent = Path(file_path).parent
        parent.mkdir(parents=True, exist_ok=True)

    import app.db.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _migrate_sqlite_users_phone_number()


def _migrate_sqlite_users_phone_number() -> None:
    """Add the Step-1 phone column to an existing SQLite database.

    ``create_all`` creates new tables but does not alter existing ones. This
    small idempotent migration preserves current user/report data while adding
    the nullable column required by new registrations.
    """
    if not settings.AEGIS_DB_URL.startswith("sqlite"):
        return

    columns = {column["name"] for column in inspect(engine).get_columns("users")}
    with engine.begin() as connection:
        if "phone_number" not in columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN phone_number VARCHAR"))
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_phone_number "
                "ON users (phone_number) WHERE phone_number IS NOT NULL"
            )
        )


def get_db():
    """FastAPI dependency: yields a DB session, closed on teardown."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
