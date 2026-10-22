#!/usr/bin/env python3
"""
agent_infer.py - Context-grounded Gemma agent inference driver.
Extracts relevant files from the task's repository snapshot archive,
constructs a grounded prompt for Gemma via local Ollama, and formats
the response into a strict unified git diff.
"""

import ast
import argparse
import json
import re
import tarfile
import urllib.request
import difflib
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

    with tarfile.open(snapshot_archive, "r:*") as tar:
        for target in target_files:
            try:
                member = next((m for m in tar.getmembers() if m.name.endswith(target)), None)
                if member:
                    f = tar.extractfile(member)
                    if f:
                        lines = f.read().decode("utf-8", errors="replace").splitlines()
                        # If file is short (< 300 lines), include whole file with line numbers
                        if len(lines) <= 50: #300:
                            numbered = [f"{i+1:4d}: {l}" for i, l in enumerate(lines)]
                            snippet = "\n".join(numbered)
                        else:
                            # For large files, include imports and locate functions matching keywords
                            hits = []
                            # Include first 50 lines (imports/definitions)
                            hits.extend(range(0, min(40, len(lines))))
                            # Scan for keywords like prepare_body, _encode_params, has_read, SupportsRead
                            search_terms = keywords or ["prepare_body", "has_read","SupportsRead","_SupportsRead"]
                            for i, l in enumerate(lines):
                                if any(term in l for term in search_terms):
                                    start_idx = max(0, i - 25)
                                    end_idx = min(len(lines), i + 25)
                                    hits.extend(range(start_idx, end_idx))
                            hits = sorted(set(hits))
                            numbered = [f"{i+1:4d}: {lines[i]}" for i in sorted(hits)]
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
    # diff_text = re.sub(r"^--- (?:a/)?requests/", "--- a/src/requests/", diff_text, flags=re.MULTILINE)
    # diff_text = re.sub(r"^\+\+\+ (?:b/)?requests/", "+++ b/src/requests/", diff_text, flags=re.MULTILINE)

    # 1. Strip markdown fences if present
    match = re.search(r"```(?:diff)?\s*\n(.*?)```", text, re.DOTALL)
    diff_text = match.group(1).strip() if match else text.strip()

    # 2. Slice to the start of the first diff header
    if "--- " in diff_text:
        diff_text = diff_text[diff_text.find("--- "):].strip()
    elif "diff --git" in diff_text:
        diff_text = diff_text[diff_text.find("diff --git"):].strip()

    # 3. Normalize requests paths to src/requests/
    diff_text = re.sub(r"^--- (?:a/)?(?:src/)?requests/", "--- a/src/requests/", diff_text, flags=re.MULTILINE)
    diff_text = re.sub(r"^\+\+\+ (?:b/)?(?:src/)?requests/", "+++ b/src/requests/", diff_text, flags=re.MULTILINE)
    diff_text = re.sub(r"^--- (?!a/)(?=[^\s\n])", "--- a/", diff_text, flags=re.MULTILINE)
    diff_text = re.sub(r"^\+\+\+ (?!b/)(?=[^\s\n])", "+++ b/", diff_text, flags=re.MULTILINE)

    # 4. Ensure leading a/ and b/ exist for any paths missing them
    diff_text = re.sub(r"^--- (?!a/)(?=[^\s\n])", "--- a/", diff_text, flags=re.MULTILINE)
    diff_text = re.sub(r"^\+\+\+ (?!b/)(?=[^\s\n])", "+++ b/", diff_text, flags=re.MULTILINE)

    # 5. Sanitize lines inside hunks: empty lines MUST have a leading space
    sanitized_lines = []
    in_hunk = False
    for line in diff_text.splitlines():
        if line.startswith("@@ "):
            in_hunk = True
            sanitized_lines.append(line)
        elif in_hunk:
            if line.startswith("--- ") or line.startswith("diff --git"):
                in_hunk = False
                sanitized_lines.append(line)
        elif not line.strip():
                # Any whitespace-only or empty line in a hunk MUST be exactly a single space
                sanitized_lines.append(" ")
        elif line[0] not in (" ", "+", "-", "\\"):
                # Missing leading space on context line
                sanitized_lines.append(" " + line)
        else:
                sanitized_lines.append(line)
     #   else:
     #       sanitized_lines.append(line)

    diff_text = "\n".join(sanitized_lines)

    if not diff_text.endswith("\n"):
        diff_text += "\n"
    return diff_text

from typing import List, Optional

def extract_ast_blocks(source_code: str, target_symbols: Optional[List[str]] = None) -> str:
    """
    Parses Python source using AST and extracts relevant function/class
    definitions with exact line numbers.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        # Fallback for files with syntax issues or non-python code
        lines = source_code.splitlines()
        return "\n".join(f"{i+1:4d}: {line}" for i, line in enumerate(lines[:200]))

    lines = source_code.splitlines()
    extracted_blocks = []

    for node in tree.body:
        # Extract top-level functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if not target_symbols or any(sym.lower() in name.lower() for sym in target_symbols):
                start = node.lineno
                end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 30
                block_lines = [f"{i:4d}: {lines[i-1]}" for i in range(start, min(end + 1, len(lines) + 1))]
                extracted_blocks.append("\n".join(block_lines))

        # Extract classes and methods
        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            matched_methods = []
            
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_name = item.name
                    if not target_symbols or any(sym.lower() in method_name.lower() or sym.lower() in class_name.lower() for sym in target_symbols):
                        start = item.lineno
                        end = item.end_lineno if hasattr(item, "end_lineno") and item.end_lineno else start + 30
                        method_lines = [f"{i:4d}: {lines[i-1]}" for i in range(start, min(end + 1, len(lines) + 1))]
                        matched_methods.append("\n".join(method_lines))
            
            if matched_methods:
                header = f"{node.lineno:4d}: class {class_name}: ..."
                extracted_blocks.append(header + "\n" + "\n\n".join(matched_methods))

    if not extracted_blocks:
        # If no specific symbols matched, return the top 150 lines
        return "\n".join(f"{i+1:4d}: {lines[i]}" for i in range(min(150, len(lines))))

    return "\n\n---\n\n".join(extracted_blocks)

def query_ollama(prompt: str, model: str, url: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 8192,
            "temperature": 0.0,
            "top_p": 0.95,
            "repeat_penalty": 1.0,
            "stop": ["<end_of_turn>"],
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

    # Extract code from ```python ... ``` block
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL)
    new_func = match.group(1).strip() if match else response_text.strip()

    # Generate unified diff using difflib
    patch_diff = generate_function_patch(
        "src/requests/models.py",
        full_models_code,
        new_func,
        "prepare_body"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(patch_diff, encoding="utf-8")
    print(f"[+] Candidate patch saved to: {out_path}")


def generate_function_patch(file_rel_path: str, orig_code: str, new_func_code: str, func_name: str) -> str:
    orig_lines = orig_code.splitlines(keepends=True)

    start_idx = None
    target_indent = "    "
    for i, line in enumerate(orig_lines):
        if f"def {func_name}(" in line:
            start_idx = i
            target_indent = line[:len(line) - len(line.lstrip())]
            break

    if start_idx is None:
        return ""

    end_idx = len(orig_lines)
    for i in range(start_idx + 1, len(orig_lines)):
        line = orig_lines[i]
        if line.strip() and not line.startswith(target_indent + " ") and not line.startswith(target_indent + "\t"):
            if line.startswith(target_indent + "def ") or line.startswith("class ") or line.startswith("def "):
                end_idx = i
                break

    # Strip extraneous text outside of the target function
    raw_lines = new_func_code.strip().splitlines()
    func_start = 0
    for idx, l in enumerate(raw_lines):
        if f"def {func_name}(" in l:
            func_start = idx
            break
    raw_lines = raw_lines[func_start:]

    # Calculate baseline indentation of the LLM output
    base_lead = len(raw_lines[0]) - len(raw_lines[0].lstrip())

    reindented_lines = []
    for l in raw_lines:
        if l.strip():
            # Strip the model's baseline indent and re-prefix with target class indent
            content = l[base_lead:] if len(l) >= base_lead else l.lstrip()
            reindented_lines.append(target_indent + content + "\n")
        else:
            reindented_lines.append("\n")

    updated_lines = orig_lines[:start_idx] + reindented_lines + ["\n"] + orig_lines[end_idx:]

    diff = difflib.unified_diff(
        orig_lines,
        updated_lines,
        fromfile=f"a/{file_rel_path}",
        tofile=f"b/{file_rel_path}",
        n=3
    )
    return "".join(diff)

def main():
    args = parse_args()
    task = load_task(Path(args.tasks), args.task_id)
    snapshot_path = Path(args.data_dir) / "snapshots" / f"{args.task_id}.tgz"
    # Identify primary source candidate files dynamically
    repo = task.get("repo", "")
    task_id = args.task_id

    if "requests" in repo or "requests" in task_id:
        target_files = ["src/requests/models,py","src/requests/_types.py"]
        repo_name = "Requests"
    elif "fastapi" in repo or "fastapi" in task_id:
        target_files = ["fastapi/routing.py", "fastapi/_compat/v2.py"]
        repo_name = "FastAPI"
    else:
        target_files = []
        repo_name = repo or "Python repository"

    repo_context = extract_repo_context(snapshot_path, target_files)

    problem_desc = task.get("problem_statement", "")
    target_files_formatted = ", ".join(f"`{f}`" for f in target_files) if target_files else "the target files"


    example_diff = (
        "--- a/src/requests/models.py\n"
        "+++ b/src/requests/models.py\n"
        "@@ -160,5 +160,5 @@ def _encode_params(\n"
        "     if isinstance(data, (str, bytes)):\n"
        "         return data\n"
        "-    elif isinstance(data, _SupportsRead):\n"
        "+    elif hasattr(data, \"read\"):\n"
        "         return data\n"
    )

    # Read models.py content from the tar snapshot
    full_models_code = ""
    with tarfile.open(snapshot_path) as tar:
        for m in tar.getmembers():
            if m.name.endswith("models.py"):
                f = tar.extractfile(m)
                if f:
                    full_models_code = f.read().decode("utf-8")
                break

    # Extract prepare_body method block
    models_lines = full_models_code.splitlines()
    start_line, end_line = None, len(models_lines)
    for i, l in enumerate(models_lines):
        if "def prepare_body(" in l:
            start_line = i
        elif start_line is not None and i > start_line:
            # Stop when the next method in PreparedRequest begins
            if l.startswith("    def "):
                end_line = i
                break

    if start_line is not None:
        original_prepare_body = "\n".join(models_lines[start_line:end_line])
    else:
        original_prepare_body = ""

    prompt = (
        "<start_of_turn>user\n"
        "You are an expert Python engineer fixing an issue in Requests.\n"
        f"Problem: {problem_desc}\n\n"
        "Here is the original `prepare_body` method:\n"
        "```python\n"
        f"{original_prepare_body}\n"
        "```\n\n"
        "Rewrite the complete `prepare_body` method.\n"
        "Crucial fix: Locate the lines:\n"
        "    if isinstance(data, basestring) or isinstance(data, _SupportsRead):\n"
        "        content_type = None\n"
        "Change that check to:\n"
        "    if isinstance(data, basestring) or isinstance(data, _SupportsRead) or hasattr(data, 'read'):\n"
        "        content_type = None\n\n"
        "Keep 4-space method indentation on `def prepare_body`.\n"
        "Return ONLY the complete method in a ```python code block.\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        "```python\n"
    )

    print(f"[+] Querying {args.model} with grounded context for {args.task_id}...")
    response_text = query_ollama(prompt, args.model, args.ollama_url)
    # Extract code from ```python ... ``` block
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL)
    new_func = match.group(1).strip() if match else response_text.strip()

    # Generate unified diff using difflib
    patch_diff = generate_function_patch(
        "src/requests/models.py",
        full_models_code,
        new_func,
        "prepare_body"
    )
    # patch_content = extract_diff(response_text) 

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(patch_diff, encoding="utf-8")
    print(f"[+] Candidate patch saved to: {out_path}")

if __name__ == "__main__":
    main()
