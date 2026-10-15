import argparse
import datetime
import json
from pathlib import Path
import re
import subprocess
import time


def load_tasks(tasks_file: str, dataset_file: str = "data/tasks.jsonl"):
    with open(tasks_file, "r") as f:
        target_ids = {line.strip() for line in f if line.strip() and not line.startswith("#")}

    tasks = []
    with open(dataset_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("instance_id") in target_ids:
                tasks.append(item)
    return tasks


def run_container_task(task, args, task_out_dir: Path):
    instance_id = task.get("instance_id")
    repo = task.get("repo", "")

    test_cmd = task.get("test_cmd")
    if not test_cmd or test_cmd.strip() == "pytest":
        if "fastapi" in instance_id:
            test_cmd = "pytest tests/test_dump_json_fast_path.py tests/test_tutorial/test_custom_response/test_tutorial001.py -v"
        elif "requests" in instance_id:
            test_patch = task.get("test_patch", "")
            target_file = ""
            for l in test_patch.splitlines():
                if l.startswith("+++ b/tests/"):
                    target_file = l.split("+++ b/")[1].strip()
                    break
            test_funcs = re.findall(r"def\s+(test_[a-zA-Z0-9_]+)", test_patch)
            if target_file and test_funcs:
                k_expr = " or ".join(test_funcs)
                test_cmd = f"pytest {target_file} -k '{k_expr}' -o addopts=\"\" -v"
            elif target_file:
                test_cmd = f"pytest {target_file} -o addopts=\"\" -v"
            else:
                test_cmd = "pytest tests/test_lowlevel.py -o addopts=\"\" -v"
        else:
            test_cmd = "pytest tests/ -v"

    patch_file = task_out_dir / "agent.patch" if args.mode == "agent" else task_out_dir / "gold.patch"
    if args.mode == "gold" and "patch" in task:
        patch_file.write_text(task["patch"], encoding="utf-8")

    is_agent = (args.mode == "agent")
    patch_apply_cmd = "patch -p1 --fuzz=3 < /patch.diff" if is_agent else "patch -p1 < /patch.diff || true"

    entrypoint_sh = f"""#!/bin/bash
set -e
mkdir -p /repo
tar -xzf /snapshot.tgz -C /repo
cd /repo
export PYTHONPATH=/repo:/repo/src:$PYTHONPATH
pip install --no-index --find-links=/wheels urllib3 certifi idna charset-normalizer 2>/dev/null || true
pip install --no-build-isolation --no-deps -e . 2>/dev/null || true

if [ -f /patch.diff ]; then
    {patch_apply_cmd}
fi

if [ -f /mock_conftest.py ]; then
    cp /mock_conftest.py tests/conftest.py
fi

{test_cmd}
"""
    entrypoint_path = task_out_dir / "entrypoint.sh"
    entrypoint_path.write_text(entrypoint_sh, encoding="utf-8")
    entrypoint_path.chmod(0o755)

    snapshot_path = (Path("data/snapshots") / f"{instance_id}.tgz").resolve()
    wheels_path = Path("data/wheels").resolve()
    mock_conf_path = Path("mock_conftest.py").resolve()

    docker_cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "-v", f"{snapshot_path}:/snapshot.tgz:ro",
        "-v", f"{wheels_path}:/wheels:ro",
        "-v", f"{mock_conf_path}:/mock_conftest.py:ro",
        "-v", f"{entrypoint_path.resolve()}:/entrypoint.sh:ro"
    ]

    if patch_file.exists():
        docker_cmd.extend(["-v", f"{patch_file.resolve()}:/patch.diff:ro"])

    docker_cmd.extend(["gemma4-eval-sandbox:latest", "/entrypoint.sh", args.mode])

    t0 = time.time()
    res = subprocess.run(docker_cmd, capture_output=True, text=True)
    duration = round(time.time() - t0, 2)

    (task_out_dir / "stdout.log").write_text(res.stdout, encoding="utf-8")
    (task_out_dir / "stderr.log").write_text(res.stderr, encoding="utf-8")

    return {
        "instance_id": instance_id,
        "status": "PASS" if res.returncode == 0 else "FAIL",
        "duration": duration,
        "error": "" if res.returncode == 0 else f"Non-zero exit code: {res.returncode}"
    }





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="Path to file listing instance IDs")
    parser.add_argument("--dataset", default="data/tasks.jsonl", help="Path to tasks.jsonl")
    parser.add_argument("--mode", choices=["baseline", "gold", "agent"], default="baseline")
    parser.add_argument("--output-dir", default="results/eval", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.tasks, args.dataset)
    results = []
    passed = 0

    for task in tasks:
        instance_id = task.get("instance_id")
        task_out_dir = out_dir / instance_id
        task_out_dir.mkdir(parents=True, exist_ok=True)

        res = run_container_task(task, args, task_out_dir)
        results.append(res)
        if res["status"] == "PASS":
            passed += 1

    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_tasks": len(tasks),
        "passed": passed,
        "failed": len(tasks) - passed,
        "mode": args.mode,
        "results": results
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

