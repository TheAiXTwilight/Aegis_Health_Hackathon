#!/bin/bash
set -e

# Create necessary directories
mkdir -p /app/data/audio/whisper-tiny-en
mkdir -p /app/data/xray
mkdir -p /app/data/knowledge/chroma
mkdir -p /tmp/aegis_uploads

# Start uvicorn
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
