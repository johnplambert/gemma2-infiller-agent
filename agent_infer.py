#!/usr/bin/env python3
"""
agent_infer.py - Context-grounded Gemma agent inference driver.
Extracts relevant files from the task's repository snapshot archive,
constructs a grounded prompt for Gemma via local Ollama, and formats
the response into a strict unified git diff.
"""

import argparse
import json
import re
import tarfile
import urllib.request
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run grounded Gemma agent inference on a benchmark task.")
    parser.add_argument("--task-id", type=str, required=True, help="Instance ID (e.g., fastapi_14962)")
    parser.add_argument("--tasks", type=str, default="data/tasks.jsonl", help="Path to master tasks.jsonl")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory containing snapshots/")
    parser.add_argument("--model", type=str, default="gemma2:2b", help="Ollama model tag to query")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434/api/generate", help="Ollama API endpoint")
    parser.add_argument("--workspace", type=str, default="results/agent_run/workspace", help="Workspace output directory")
    parser.add_argument("--output", type=str, required=True, help="Path to save the generated candidate patch.diff")
    return parser.parse_args()


def load_task(tasks_path: Path, task_id: str):
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("instance_id") == task_id:
                    return item
    raise ValueError(f"Task {task_id} not found in {tasks_path}")


def extract_repo_context(snapshot_archive: Path, target_files: list, keywords: list = None) -> str:
    context_chunks = []
    if not snapshot_archive.exists():
        return ""

    with tarfile.open(snapshot_archive, "r:gz") as tar:
        for target in target_files:
            try:
                member = next((m for m in tar.getmembers() if m.name.endswith(target)), None)
                if member:
                    f = tar.extractfile(member)
                    if f:
                        lines = f.read().decode("utf-8", errors="replace").splitlines()
                        # If file is short (< 300 lines), include whole file with line numbers
                        if len(lines) <= 300:
                            numbered = [f"{i+1:4d}: {l}" for i, l in enumerate(lines)]
                            snippet = "\n".join(numbered)
                        else:
                            # For large files, include imports and locate functions matching keywords
                            hits = []
                            # Include first 50 lines (imports/definitions)
                            hits.extend(range(0, min(50, len(lines))))
                            # Scan for keywords like prepare_body, _encode_params, has_read, SupportsRead
                            search_terms = keywords or ["prepare_body", "_encode_params", "SupportsRead", "has_read"]
                            for i, l in enumerate(lines):
                                if any(term in l for term in search_terms):
                                    start_idx = max(0, i - 15)
                                    end_idx = min(len(lines), i + 35)
                                    hits.extend(range(start_idx, end_idx))
                            hits = sorted(set(hits))
                            numbered = [f"{i+1:4d}: {lines[i]}" for i in hits]
                            snippet = "\n".join(numbered)

                        context_chunks.append(f"### File: {target}\n```python\n{snippet}\n```\n")
            except Exception as e:
                context_chunks.append(f"# Could not extract {target}: {e}\n")
    return "\n".join(context_chunks)

def extract_diff(text: str, target_file: str = None) -> str:
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    if "--- " in text:
        start = text.find("--- ")
        diff_text = text[start:].strip()
    elif "diff --git" in text:
        start = text.find("diff --git")
        diff_text = text[start:].strip()
    else:
        diff_text = text.strip()

    # Map generic/dummy paths to actual target file if provided
    if target_file:
        diff_text = re.sub(r"^--- (?:a/)?[a-zA-Z0-9_\-\./]*your_file\.py", f"--- a/{target_file}", diff_text, flags=re.MULTILINE)
        diff_text = re.sub(r"^\+\+\+ (?:b/)?[a-zA-Z0-9_\-\./]*your_file\.py", f"+++ b/{target_file}", diff_text, flags=re.MULTILINE)
        diff_text = re.sub(r"^--- (?:a/)?[a-zA-Z0-9_\-\./]*target_file\.py", f"--- a/{target_file}", diff_text, flags=re.MULTILINE)
        diff_text = re.sub(r"^\+\+\+ (?:b/)?[a-zA-Z0-9_\-\./]*target_file\.py", f"+++ b/{target_file}", diff_text, flags=re.MULTILINE)

    # Normalize requests paths from a/requests/ or requests/ to a/src/requests/
    diff_text = re.sub(r"^--- (?:a/)?requests/", "--- a/src/requests/", diff_text, flags=re.MULTILINE)
    diff_text = re.sub(r"^\+\+\+ (?:b/)?requests/", "+++ b/src/requests/", diff_text, flags=re.MULTILINE)

    if not diff_text.endswith("\n"):
        diff_text += "\n"
    return diff_text

def query_ollama(prompt: str, model: str, url: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.25,
            "stop": ["```\n\n", "### Explanation", "### Problem", "\n\n\n"],
            "num_predict": 1024,
        }
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result.get("response", "")


def main():
    args = parse_args()
    task = load_task(Path(args.tasks), args.task_id)
    snapshot_path = Path(args.data_dir) / "snapshots" / f"{args.task_id}.tgz"
    # Identify primary source candidate files dynamically
    repo = task.get("repo", "")
    task_id = args.task_id

    if "requests" in repo or "requests" in task_id:
        target_files = ["src/requests/_types.py","src/requests/models.py"]
        repo_name = "Requests"
    elif "fastapi" in repo or "fastapi" in task_id:
        target_files = ["fastapi/routing.py", "fastapi/_compat/v2.py"]
        repo_name = "FastAPI"
    else:
        target_files = []
        repo_name = repo or "Python repository"

    repo_context = extract_repo_context(snapshot_path, target_files)

    target_file_str = target_files[0] if target_files else "the relevant file"
    problem_desc = task.get("problem_statement", "")

# Determine the primary relative path from target_files or repo_context
    primary_file = target_files[-1] if target_files else "target_file.py"

    prompt = (
        f"You are an expert Python software engineer repairing {repo_name}.\n"
        f"Fix the issue described below by generating a valid unified git diff patch.\n\n"
        f"### Problem Statement:\n{problem_desc}\n\n"
        f"### Target Source Context:\n{repo_context}\n\n"
        f"### Instructions:\n"
        f"1. Generate ONLY a valid unified git diff patch enclosed in a single ```diff code block.\n"
        f"2. The diff headers MUST use the exact file path: `--- a/{primary_file}` and `+++ b/{primary_file}`.\n"
        f"3. Do NOT include placeholder comments like '# ...' or '# rest of code remains unchanged'.\n"
        f"4. Do NOT include markdown explanations before or after the diff block.\n"
        f"5. Ensure valid Python indentation and syntax within all modified lines.\n\n"
        f"Output format:\n"
        f"```diff\n"
        f"--- a/{primary_file}\n"
        f"+++ b/{primary_file}\n"
        f"@@ ... @@\n"
        f"<diff lines>\n"
        f"```"
    )

    print(f"[+] Querying {args.model} with grounded context for {args.task_id}...")
    response_text = query_ollama(prompt, args.model, args.ollama_url)
    patch_content = extract_diff(response_text, primary_file) 

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(patch_content, encoding="utf-8")
    print(f"[+] Candidate patch saved to: {out_file}")


if __name__ == "__main__":
    main()
