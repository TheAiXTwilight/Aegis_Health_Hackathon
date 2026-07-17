"""
Migration: Add `phone` column to the `users` table.

Run this script once to add the phone column:
    python -m migrations.add_phone_column

This is idempotent — it checks if the column already exists before adding.
"""

import sqlite3
import os


def migrate(db_path: str | None = None) -> None:
    if db_path is None:
        # Default path matching app/settings.py
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_root, "data", "aegis.db")

    if not os.path.exists(db_path):
        print(f"Database not found at {db_path} — nothing to migrate.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if column already exists
    cursor.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cursor.fetchall()]

    if "phone" in columns:
        print("Column 'phone' already exists in 'users' table — skipping.")
    else:
        cursor.execute("ALTER TABLE users ADD COLUMN phone VARCHAR(20)")
        conn.commit()
        print("Column 'phone' added to 'users' table successfully.")

    conn.close()


if __name__ == "__main__":
    migrate()
