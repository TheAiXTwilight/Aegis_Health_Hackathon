"""Shared helpers for the refreshed integration tests.

These helpers deliberately create an isolated SQLite database for every test
module. They keep endpoint tests independent of the developer's aegis.db,
Ollama, Piper, and model artifacts.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth import hash_password
from app.db import models  # noqa: F401 - registers ORM models on Base
from app.db.models import User
from app.db.session import Base, get_db


def make_session_factory() -> sessionmaker:
    """Return a StaticPool-backed, isolated in-memory SQLite factory."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def install_db_override(app: FastAPI, factory: sessionmaker) -> None:
    """Install a fresh SQLAlchemy session per request."""
    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db


def seed_user(
    factory: sessionmaker,
    *,
    user_id: str,
    email: str,
    password: str = "Password123!",
    display_name: str = "Test User",
    role: str = "user",
    security_question: str | None = "What is your favorite pet's name?",
    security_answer: str | None = "Milo",
) -> User:
    """Create one real bcrypt-backed user and return a detached copy."""
    with factory() as db:
        user = User(
            id=user_id,
            email=email.lower(),
            display_name=display_name,
            password_hash=hash_password(password),
            role=role,
            security_question=security_question,
            security_answer_hash=(
                hash_password(security_answer.lower()) if security_answer else None
            ),
            # Kept only because the CURRENT schema requires/permits it.
            # Security-gate tests assert this field is removed/redacted.
            security_answer=security_answer,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        db.expunge(user)
        return user


def mutable_user_override(
    app: FastAPI,
    factory: sessionmaker,
    current_user_id: dict[str, str],
    dependency: Callable[..., Any],
) -> None:
    """Override an auth dependency with a mutable user selector.

    Tests can switch identity by assigning current_user_id["value"].
    """
    def override_user() -> User:
        with factory() as db:
            user = db.get(User, current_user_id["value"])
            assert user is not None, "test fixture selected a user that does not exist"
            db.expunge(user)
            return user

    app.dependency_overrides[dependency] = override_user


def registration_payload(**overrides: Any) -> dict[str, Any]:
    """A valid payload for the current /auth/register contract."""
    payload: dict[str, Any] = {
        "email": "alice@example.com",
        "password": "Password123!",
        "display_name": "Alice Example",
        "username": "alice",
        "phone": "+919876543210",
        "security_question": "pet_name",
        "security_answer": "Milo",
    }
    payload.update(overrides)
    return payload


def login(client: Any, email: str = "alice@example.com", password: str = "Password123!") -> dict[str, Any]:
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return response.json()
