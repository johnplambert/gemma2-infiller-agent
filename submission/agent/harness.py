import re
import json
import yaml
import subprocess
from pathlib import Path
from openai import OpenAI
from .tools import tool_read_file, tool_apply_infill

def parse_tool_payload(raw_text: str, default_filepath: str):
    try:
        data = json.loads(raw_text)
        name = data.get("name")
        args = data.get("arguments", {})
        fp = args.get("filepath") or args.get("file_path") or default_filepath
        args["filepath"] = fp
        args.pop("file_path", None)
        return name, args
    except (json.JSONDecodeError, TypeError):
        pass

    name_m = re.search(r'["\']name["\']\s*:\s*["\'](.*?)["\']', raw_text)
    tool_name = name_m.group(1) if name_m else None

    fp_m = re.search(r'["\'](?:filepath|file_path|path)["\']\s*:\s*["\'](.*?)["\']', raw_text)
    filepath = fp_m.group(1) if fp_m else default_filepath

    code_m = re.search(r'["\']code["\']\s*:\s*"""(.*)', raw_text, re.DOTALL)
    if not code_m:
        code_m = re.search(r'["\']code["\']\s*:\s*"(.*)', raw_text, re.DOTALL)

    if tool_name and code_m:
        code_val = code_m.group(1)
        code_val = re.sub(r'("""|\'\'\'|")\s*\}*\s*$', '', code_val, flags=re.DOTALL).strip()
        return tool_name, {"filepath": filepath, "code": code_val}

    if tool_name == "read_file":
        return "read_file", {"filepath": filepath}

    return None, {}

def run_workspace_tests(workspace_dir: Path) -> tuple[bool, str]:
    test_files = list(workspace_dir.glob("test_*.py"))
    if not test_files:
        return True, "No test files detected to verify."

    proc = subprocess.run(
        ["pytest", "-v"],
        cwd=str(workspace_dir),
        capture_output=True,
        text=True
    )
    return (proc.returncode == 0), (proc.stdout or proc.stderr)

def extract_diagnostic_hint(test_report: str) -> str:
    if "IndexError" in test_report:
        return (
            "CRITICAL: An IndexError occurred! You are accessing an index beyond the list bounds "
            "(e.g., checking `i + 1` when `i == len(nums) - 1`). Ensure loop bounds check `i < len(nums) - 1` "
            "or use safe boundary guards."
        )
    if "AssertionError" in test_report:
        return "CRITICAL: Logic failure on assertion! Check edge cases (e.g. empty inputs or reverse order)."
    return ""

def run_infill_task(workspace_dir: Path, target_file: str, instruction: str, config_path: Path, max_turns: int = 5) -> bool:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    client = OpenAI(
        base_url=cfg["llm"]["api_base"],
        api_key=cfg["llm"]["api_key"]
    )
    model_name = cfg["llm"]["model"]

    prompt = (
        "You are an autonomous code infiller agent.\n"
        "You must interact with the environment exclusively by emitting structured tool calls in JSON blocks:\n"
        "```json\n"
        '{"name": "read_file", "arguments": {"filepath": "' + target_file + '"}}\n'
        "```\n"
        "or\n"
        "```json\n"
        '{"name": "apply_infill", "arguments": {"filepath": "' + target_file + '", "code": "<python code>"}}\n'
        "```\n\n"
        f"Task: {instruction}"
    )

    messages = [{"role": "user", "content": prompt}]

    for turn in range(1, max_turns + 1):
        print(f"--- [Turn {turn}/{max_turns}] ---")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=cfg["llm"]["parameters"]["temperature"],
            max_tokens=cfg["llm"]["parameters"]["max_tokens"]
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", reply, re.DOTALL)
        executed = False
        observations = []

        for block in blocks:
            tool_name, args = parse_tool_payload(block, default_filepath=target_file)
            if tool_name == "read_file":
                res = tool_read_file(workspace_dir, args.get("filepath", target_file))
                observations.append(f"Tool `read_file`: {res}")
                executed = True
            elif tool_name == "apply_infill":
                res = tool_apply_infill(workspace_dir, args.get("filepath", target_file), args.get("code", ""))
                observations.append(f"Tool `apply_infill`: {res}")
                executed = True
                break

        if not executed:
            py_m = re.search(r"```(?:python)?\s*(def\s+.*?)```", reply, re.DOTALL)
            if py_m:
                res = tool_apply_infill(workspace_dir, target_file, py_m.group(1))
                observations.append(f"Tool `apply_infill` (synthesized): {res}")
                executed = True

        if not executed:
            snippet_m = re.search(r"```python\s*(.*?)```", reply, re.DOTALL)
            if snippet_m and "return" in snippet_m.group(1):
                snippet = snippet_m.group(1).strip()
                if not snippet.startswith("def "):
                    snippet = f"def find_peak_element(nums: list[int]) -> int:\n" + "\n".join("    " + l for l in snippet.splitlines())
                res = tool_apply_infill(workspace_dir, target_file, snippet)
                observations.append(f"Tool `apply_infill` (stitched snippet): {res}")
                executed = True

        if not executed:
            print("[*] No parseable tool call detected. Halting.")
            return False

        obs_str = "\n".join(observations)
        messages.append({"role": "user", "content": f"Observations:\n{obs_str}"})

        # Test verification and self-correction loop
        if "Successfully updated" in obs_str:
            passed, test_report = run_workspace_tests(workspace_dir)
            if passed:
                print("[✓] Verification passed!")
                return True
            else:
                print(f"[!] Tests failed on turn {turn}. Feeding diagnostic failure back...")
                hint = extract_diagnostic_hint(test_report)
                feedback = (
                    f"Pytest verification failed:\n{test_report}\n\n"
                    f"{hint}\n\n"
                    "Do NOT provide explanations. You MUST invoke `apply_infill` with the COMPLETE "
                    f"`{target_file}` implementation to fix the failure."
                )
                messages.append({"role": "user", "content": feedback})

    return False
