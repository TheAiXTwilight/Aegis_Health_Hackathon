#!/usr/bin/env python3
"""
CLI script to seed demo users into the Aegis Health database.

Usage:
    python scripts/seed_demo.py
    python scripts/seed_demo.py --clear
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so `app` imports work when run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.models import User
from app.db.seed import clear_demo_users, seed_demo_users
from app.db.session import SessionLocal, init_db


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Aegis Health demo users")
    parser.add_argument("--clear", action="store_true", help="Remove demo users instead of seeding")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        if args.clear:
            count = clear_demo_users(db)
            print(f"Cleared {count} demo users")
        else:
            users = seed_demo_users(db)
            print("Seeded demo users:")
            for user in users:
                print(f"  - {user.display_name} <{user.email}>  role={user.role}")
    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
