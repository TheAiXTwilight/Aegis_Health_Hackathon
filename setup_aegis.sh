#!/bin/bash
#
# Aegis Health — one-shot startup for the Jetson Cloud-Lab container.
#
# REALITY: this Cloud-Lab tier hands you an EPHEMERAL container that is recycled
# roughly every ~10-15 min (the container ID changes each session, e.g.
# a2d9468e8c90 -> ... -> 7e976cd02417, and `uptime` shows the HOST kernel, not
# the container). No process survives a recycle, so this script is meant to be
# re-run on EVERY fresh container / reconnect to bring the app back up.
# For hands-free restart, call it from ~/.bashrc (autostart snippet).
#
#   bash setup_aegis.sh            # start if not already running
#   bash setup_aegis.sh --restart  # force restart
set -e

# --- Already up? nothing to do (re-run inside the same container). ---
if [ "$1" != "--restart" ] && curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    PUBLIC_URL=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | grep -o '"public_url":"[^"]*"' | head -1 | sed 's/.*"public_url":"//; s/"//')
    echo "Backend already running. Public URL: $PUBLIC_URL"
    exit 0
fi

echo "=== 1. tzdata ==="
python3 -c "import tzdata" 2>/dev/null || pip install --user -q tzdata || sudo apt install -y tzdata 2>/dev/null || true

echo "=== 2. Repo ==="
cd ~
[ -d Aegis_Health ] || git clone --depth 1 https://github.com/MkSachdev/Aegis_Health.git
cd ~/Aegis_Health

echo "=== 3. Ollama ==="
curl -sf http://172.17.0.1:11434/api/tags >/dev/null && echo "Ollama OK @ 172.17.0.1:11434" || echo "WARN: Ollama not reachable"

echo "=== 4. .env (ensure exists + Ollama URL) ==="
if [ ! -f .env ]; then
    SECRET=$(python3 -c "import secrets;print(secrets.token_hex(32))")
    cat > .env << EOF
AEGIS_SECRET_KEY=${SECRET}
AEGIS_OLLAMA_BASE_URL=http://172.17.0.1:11434
AEGIS_OLLAMA_MODEL=llama3.2:1b
OLLAMA_NUM_CTX=3072
OLLAMA_NUM_PREDICT=768
OLLAMA_TEMPERATURE=0.2
AEGIS_ENV=production
COOKIE_SECURE=true
AEGIS_CORS_ORIGINS=["https://placeholder.ngrok-free.dev"]
AEGIS_PUBLIC_URL=https://placeholder.ngrok-free.dev
EOF
fi
sed -i 's#AEGIS_OLLAMA_BASE_URL=http://localhost:11434#AEGIS_OLLAMA_BASE_URL=http://172.17.0.1:11434#' .env
sed -i 's#AEGIS_OLLAMA_BASE_URL=http://127.0.0.1:11434#AEGIS_OLLAMA_BASE_URL=http://172.17.0.1:11434#' .env

echo "=== 5. Stop stale processes ==="
pkill -f uvicorn 2>/dev/null || true
pkill -f "tools.piper_server" 2>/dev/null || true
pkill -f ngrok 2>/dev/null || true
sleep 2

echo "=== 6. ngrok (started first so CORS can match its URL) ==="
cd ~
if pgrep -f "ngrok http" >/dev/null 2>&1; then
    echo "ngrok already running."
else
    [ -f ngrok ] || { curl -L -o ngrok.tgz https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.tgz && tar -xzf ngrok.tgz && chmod +x ngrok; }
    setsid ./ngrok http 127.0.0.1:8000 > ngrok.log 2>&1 < /dev/null &
    sleep 4
fi
PUBLIC_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | grep -o '"public_url":"[^"]*"' | head -1 | sed 's/.*"public_url":"//; s/"//')
echo "ngrok URL: $PUBLIC_URL"

echo "=== 7. Point .env CORS / PUBLIC_URL at the current ngrok URL ==="
cd ~/Aegis_Health
if [ -n "$PUBLIC_URL" ]; then
    sed -i "s#AEGIS_CORS_ORIGINS=.*#AEGIS_CORS_ORIGINS=[\"$PUBLIC_URL\"]#" .env
    if grep -q '^AEGIS_PUBLIC_URL=' .env; then
        sed -i "s#AEGIS_PUBLIC_URL=.*#AEGIS_PUBLIC_URL=$PUBLIC_URL#" .env
    else
        echo "AEGIS_PUBLIC_URL=$PUBLIC_URL" >> .env
    fi
fi

echo "=== 8. Start backend (detached via setsid) ==="
cd ~/Aegis_Health
setsid python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 < /dev/null &
disown
echo "Waiting for health (up to 40s)..."
READY=0
for i in $(seq 1 20); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { READY=1; break; }
    sleep 2
done
if [ "$READY" = 1 ]; then
    echo "Backend healthy after $((i*2))s."
else
    echo "ERROR: backend not healthy. Last 30 log lines:"
    tail -30 server.log
    exit 1
fi

echo ""
echo "=== DONE ==="
echo "Public URL: $PUBLIC_URL"
echo "Test:  curl $PUBLIC_URL/health"
echo "Logs:  tail -f ~/Aegis_Health/server.log"
echo "Force: bash setup_aegis.sh --restart"
echo ""
echo "Note: this container is recycled ~every 10-15 min. On disconnect, reconnect"
echo "and re-run (or use the ~/.bashrc autostart to do it automatically)."
