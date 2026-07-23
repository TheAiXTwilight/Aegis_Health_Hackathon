"""
Migration: Add `cache_key` column to the `health_records` table.

Lets report deletion evict the exact result_cache entry a report was
served from, so deleting a report can no longer be silently undone by a
later cache hit on the same symptoms/medications/xray/lab combination.

Run this script once to add the column:
    python -m migrations.add_cache_key_column

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

    cursor.execute("PRAGMA table_info(health_records)")
    columns = [row[1] for row in cursor.fetchall()]

    if "cache_key" in columns:
        print("Column 'cache_key' already exists in 'health_records' table — skipping.")
    else:
        cursor.execute("ALTER TABLE health_records ADD COLUMN cache_key VARCHAR")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_health_records_cache_key "
            "ON health_records (cache_key)"
        )
        conn.commit()
        print("Column 'cache_key' added to 'health_records' table successfully.")

    conn.close()


if __name__ == "__main__":
    migrate()