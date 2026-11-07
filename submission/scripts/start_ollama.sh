#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "[*] Checking Ollama service..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[*] Starting background Ollama daemon..."
    ollama serve > /tmp/ollama_daemon.log 2>&1 &
    sleep 3
fi

if ! ollama list | grep -q "gemma2-infiller"; then
    echo "[*] Registering gemma2-infiller from Modelfile..."
    cd "$BASE_DIR/models"
    ollama create gemma2-infiller -f Modelfile
fi

echo "[✓] Ollama backend verified and ready."
