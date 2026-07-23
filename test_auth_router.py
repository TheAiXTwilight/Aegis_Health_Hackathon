#!/usr/bin/env python3
"""Minimal smoke test for the auth router (DB + endpoints)."""
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.db.session import init_db, get_db
from backend.auth import router as auth_router

app = FastAPI()
app.include_router(auth_router)

init_db()
client = TestClient(app)

# Register
resp = client.post("/auth/register", json={
    "email": "test@example.com",
    "password": "Password123",
    "display_name": "Test User",
})
print("register", resp.status_code, resp.json())

# Login (no refresh cookie in TestClient by default, but endpoint should work)
resp = client.post("/auth/login", json={
    "email": "test@example.com",
    "password": "Password123",
})
print("login", resp.status_code, resp.json())
assert resp.status_code == 200, resp.text
access_token = resp.json()["access_token"]

# Me
resp = client.get("/auth/me", headers={"Authorization": f"Bearer {access_token}"})
print("me", resp.status_code, resp.json())

print("auth router OK")
