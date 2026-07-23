# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (ffmpeg is critical for voice transcription)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV AEGIS_WHISPER_DIR=/app/data/audio/whisper-tiny-en
ENV AEGIS_XRAY_MODEL_DIR=/app/data/xray

# Entrypoint script (ensure it has executable permissions)
RUN chmod +x docker/entrypoint.sh

# Start the application using the entrypoint script
ENTRYPOINT ["docker/entrypoint.sh"]
