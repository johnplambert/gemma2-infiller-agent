#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# 1. Ensure backend service is running
"$SCRIPT_DIR/start_ollama.sh"

# 2. Parse arguments
TARGET_FILE="${1:-solution.py}"
INSTRUCTION="${2:-Resolve the stub implementation.}"
WORKSPACE="${3:-$(pwd)}"

export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

python3 - << PYEOF
import sys
from pathlib import Path
from agent.harness import run_infill_task

workspace = Path("$WORKSPACE").resolve()
target = "$TARGET_FILE"
inst = "$INSTRUCTION"
cfg_path = Path("$BASE_DIR/config/agent_config.yaml")

print(f"[*] Starting Agent for target: {target}")
success = run_infill_task(workspace, target, inst, cfg_path)
if success:
    print(f"[✓] Task completed successfully.")
    sys.exit(0)
else:
    print(f"[!] Task failed or max turns exceeded.")
    sys.exit(1)
PYEOF
