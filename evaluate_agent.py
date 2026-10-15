import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path.home() / "kaggle/gemma4-agent"
DATA_DIR = BASE_DIR / "data"
TASKS_FILE = DATA_DIR / "tasks.jsonl"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
RESULTS_DIR = BASE_DIR / "results/agent_baseline"

DOCKER_IMAGE = "python:3.10-slim"

def extract_test_targets(test_patch):
    targets = []
    for line in test_patch.splitlines():
        if line.startswith("+++ b/") and "test" in line:
            clean_path = line.replace("+++ b/", "").strip()
            targets.append(clean_path)
    return list(set(targets))

def evaluate_task(task):
    instance_id = task["instance_id"]
    repo = task["repo"]
    test_patch = (task.get("test_patch") or "").strip() + "\n"

    print(f"\n==========================================")
    print(f"Agent Eval: {instance_id} | Repo: {repo}")
    print(f"==========================================")

    archive_path = (SNAPSHOTS_DIR / f"{instance_id}.tgz").resolve()
    task_workdir = (RESULTS_DIR / instance_id).resolve()
    task_workdir.mkdir(parents=True, exist_ok=True)

    task_json_file = task_workdir / "task.json"
    task_json_file.write_text(json.dumps(task, indent=2))

    agent_patch_file = task_workdir / "agent.patch"

    # Step A: Run Agent Inference
    print("[*] Generating patch via Gemma 4...")
    subprocess.run([
        sys.executable,
        str(BASE_DIR / "agent_infer.py"),
        str(task_json_file),
        str(agent_patch_file)
    ], check=True)

    test_patch_file = task_workdir / "test.patch"
    test_patch_file.write_text(test_patch, encoding="utf-8")

    test_files = extract_test_targets(test_patch)
    test_args = " ".join(test_files) if test_files else "tests/"

    # Step B: Run Sandbox Evaluation
    container_script = f"""#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

if ! command -v git &> /dev/null || ! command -v patch &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq git patch > /dev/null
fi

mkdir -p /workspace/repo
tar -xzf /archive.tgz -C /workspace/repo
cd /workspace/repo

git config --global --add safe.directory /workspace/repo

pip install -q --upgrade pip
pip install -q pytest pytest-timeout pytest-asyncio anyio dirty-equals typer[all] httpx inline-snapshot sqlalchemy orjson pydantic-settings ujson
pip install -q -e . 2>/dev/null || true

apply_patch() {{
    local patch_path="$1"
    if [ ! -s "$patch_path" ]; then
        return 1
    fi
    git apply -v --recount --whitespace=nowarn "$patch_path" 2>/dev/null || \
    patch -p1 --fuzz=3 --ignore-whitespace -N --binary < "$patch_path" 2>/dev/null || patch -p0 --fuzz=3 --ignore-whitespace -N --binary < "$patch_path" 2>/dev/null || \
    git apply --reject "$patch_path" 2>/dev/null || return 1
    return 0
}}

# Apply reproduction test
apply_patch /test.patch

# Apply agent patch
if apply_patch /agent.patch; then
    echo "[+] Agent patch applied successfully."
else
    echo "[-] Agent patch failed to apply or was empty."
    exit 2
fi

# Run verification
pytest -W default {test_args} > /task_output/agent_test.log 2>&1
AGENT_EC=$?
echo "Agent Test Exit Code: $AGENT_EC"

if [ $AGENT_EC -eq 0 ]; then
    exit 0
else
    exit 1
fi
"""
    entrypoint_file = task_workdir / "entrypoint.sh"
    entrypoint_file.write_text(container_script, encoding="utf-8")
    os.chmod(entrypoint_file, 0o755)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{archive_path}:/archive.tgz:ro",
        "-v", f"{test_patch_file}:/test.patch:ro",
        "-v", f"{agent_patch_file}:/agent.patch:ro",
        "-v", f"{task_workdir}:/task_output",
        "-v", f"{entrypoint_file}:/entrypoint.sh:ro",
        DOCKER_IMAGE,
        "/bin/bash", "/entrypoint.sh"
    ]

    start_time = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = round(time.time() - start_time, 2)
    (task_workdir / "stdout.log").write_text(res.stdout)
    (task_workdir / "stderr.log").write_text(res.stderr)

    if res.returncode == 0:
        status = "SOLVED"
    elif res.returncode == 2:
        status = "PATCH_FAILED"
    else:
        status = "TEST_FAILED"

    print(f"[+] Result: {status} ({elapsed}s)")
    return {"instance_id": instance_id, "status": status, "duration_sec": elapsed}

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TASKS_FILE, "r") as f:
        tasks = [json.loads(line) for line in f]

    dev_subset = tasks[:10]
    print(f"Evaluating agent on {len(dev_subset)} dev tasks...")

    results = []
    for task in dev_subset:
        results.append(evaluate_task(task))

    summary_file = RESULTS_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[✓] Agent run complete. Summary saved to {summary_file}")

if __name__ == "__main__":
    main()
