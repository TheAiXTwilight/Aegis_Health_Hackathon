"""
Migration: Add security_answer plain text column to users table.

Run the SQL directly:
    sqlite3 data/aegis.db "ALTER TABLE users ADD COLUMN security_answer VARCHAR(200);"
"""
