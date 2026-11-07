from pathlib import Path

def tool_read_file(workspace: Path, filepath: str) -> str:
    target = (workspace / filepath).resolve()
    if not target.is_relative_to(workspace):
        return "Error: Path traversal outside workspace disallowed."
    if not target.exists():
        return f"Error: File '{filepath}' does not exist."
    return target.read_text()

def tool_apply_infill(workspace: Path, filepath: str, code: str) -> str:
    target = (workspace / filepath).resolve()
    if not target.is_relative_to(workspace):
        return "Error: Path traversal outside workspace disallowed."
    
    # Strip any stub artifacts
    lines = [
        line for line in code.splitlines()
        if "raise NotImplementedError" not in line and "# TODO:" not in line
    ]
    cleaned_code = "\n".join(lines).strip() + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(cleaned_code)
    return f"Successfully updated {filepath} ({len(cleaned_code)} bytes)."
