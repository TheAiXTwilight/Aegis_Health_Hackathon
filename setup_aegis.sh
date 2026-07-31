#!/bin/bash
# One-shot setup/recovery script for Aegis Health on the EdgeMinds Jetson board.
# Run this at the start of every fresh session: bash setup_aegis.sh
set -e

echo "=== 1. Fixing missing tzdata (common on fresh containers) ==="
if [ ! -f /usr/share/zoneinfo/tzdata.zi ]; then
    sudo apt update -qq && sudo apt install -y tzdata || pip install --user tzdata
else
    echo "tzdata already present, skipping."
fi

echo "=== 2. Cloning or updating the repo ==="
cd ~
if [ ! -d "Aegis_Health" ]; then
    echo "No local copy found — cloning fresh."
    git clone https://github.com/MkSachdev/Aegis_Health.git
else
    echo "Local copy found — checking if patches are present."
fi
cd ~/Aegis_Health

echo "=== 3. Verifying TTS auto-trigger patches ==="
PATCH_COUNT=$(grep -c "DISABLED on Jetson board" backend/queue.py || true)
if [ "$PATCH_COUNT" != "2" ]; then
    echo "Patches missing or incomplete ($PATCH_COUNT/2) — pulling latest from GitHub."
    git pull origin main || echo "git pull failed (uncommitted local changes?) — continuing anyway."
    PATCH_COUNT=$(grep -c "DISABLED on Jetson board" backend/queue.py || true)
    echo "Patch count after pull: $PATCH_COUNT/2"
else
    echo "Both TTS patches confirmed present."
fi

echo "=== 4. Checking Ollama reachability ==="
if curl -sf http://172.17.0.1:11434/api/tags > /dev/null; then
    echo "Ollama reachable at 172.17.0.1:11434"
else
    echo "WARNING: Ollama not reachable — check board status manually."
fi

echo "=== 5. Checking .env ==="
if [ ! -f .env ]; then
    echo "No .env found — creating one. Generating secret key..."
    SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    cat > .env << EOF
AEGIS_SECRET_KEY=${SECRET}
AEGIS_OLLAMA_BASE_URL=http://172.17.0.1:11434
AEGIS_OLLAMA_MODEL=llama3.2:1b
OLLAMA_NUM_CTX=1024
OLLAMA_NUM_PREDICT=768
OLLAMA_TEMPERATURE=0.2
AEGIS_ENV=production
COOKIE_SECURE=true
AEGIS_CORS_ORIGINS=["https://scrounger-headstone-entrench.ngrok-free.dev"]
EOF
    echo ".env created."
else
    echo ".env already exists - fixing Ollama URL to host gateway."
    sed -i 's#AEGIS_OLLAMA_BASE_URL=http://localhost:11434#AEGIS_OLLAMA_BASE_URL=http://172.17.0.1:11434#' .env
    sed -i 's#AEGIS_OLLAMA_BASE_URL=http://127.0.0.1:11434#AEGIS_OLLAMA_BASE_URL=http://172.17.0.1:11434#' .env
fi

echo "=== 6. Killing any stray processes ==="
pkill -f uvicorn 2>/dev/null || true
pkill -f "tools.piper_server" 2>/dev/null || true   # stray Piper TTS worker from a previous run
pkill -f ngrok 2>/dev/null || true
sleep 2

echo "=== 7. Starting backend ==="
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
echo "Waiting for backend to become healthy (up to 40s)..."
READY=0
for i in $(seq 1 20); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 2
done
if [ "$READY" = "1" ]; then
    echo "Backend healthy after $((i*2))s."
else
    echo "ERROR: Backend did not become healthy within 40s. Last 30 log lines:"
    tail -30 server.log
    exit 1
fi

echo "=== 8. Starting ngrok ==="
cd ~
if [ ! -f ngrok ]; then
    echo "ngrok binary not found — downloading."
    curl -L -o ngrok.tgz https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.tgz
    tar -xzf ngrok.tgz
    chmod +x ngrok
fi
nohup ./ngrok http 127.0.0.1:8000 > ngrok.log 2>&1 &
sleep 3
PUBLIC_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | grep -o '"public_url":"[^"]*"' | head -1)
echo ""
echo "=== DONE ==="
echo "Public URL: $PUBLIC_URL"
echo "Test it: curl $PUBLIC_URL/health"