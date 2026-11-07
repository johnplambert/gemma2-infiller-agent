# Post-Competition Retrospective: Autonomous Code-Infilling Agent (Gemma 2 2B)

## Executive Summary
This project evaluated the feasibility of deploying a lightweight, parameter-efficient language model (**Gemma 2 2B**) as an autonomous, self-correcting software engineering agent under rigid execution, memory, and latency constraints. By pairing a fine-tuned LoRA adapter (~83.1 MB) with an in-situ `pytest` execution harness and structured diagnostic feedback, the agent achieved high-reliability code infilling without requiring cloud API backends or multi-billion parameter foundation models.

---

## 1. Submission Artifact & Checksum Record

- **Archive File:** `gemma2_infiller_agent_submission.tar.gz`
- **SHA256 Checksum:** `a556f18c0b205251e7fb5b970be8f8b63e5816306a31684ba1186c53e836b400`
- **Recorded Date:** `2026-11-07 01:03:21 UTC`

---

## 2. System Architecture & Engineering Choices

```
[Target Workspace] ──> read_file ──> [Gemma 2 2B + LoRA] ──> apply_infill ──> [In-Situ pytest]
                             ▲                                                      │
                             │ Failure Trace + Boundary Hint                        │ Clean Pass
                             └──────────────────────────────────────────────────────┴───> [Exit Code 0]
```

* **Local Deterministic Inference:** Served through an Ollama daemon via an OpenAI-compatible REST endpoint (`http://localhost:11434/v1`). Context was capped at 8,192 tokens with sampling locked down (`temperature: 0.1`, `top_p: 0.95`) to enforce deterministic syntax generation.
* **Minimal Tool Interface:** Limited the agent to two atomic actions: `read_file` (workspace inspection) and `apply_infill` (in-place patching). Eliminating broad bash execution or multi-file navigation prevented tool hallucination and kept tool invocations predictable.
* **ADK Alignment:** Packaged the system according to Google Agent Development Kit (ADK) standards, using a unified JSON manifest (`config/adk_manifest.json`) and an explicit runtime orchestrator (`scripts/run_agent.sh`).

---

## 3. Benchmark Performance & Verification Metrics

| Metric | Measured Value | Notes |
|---|---|---|
| **Pass@1 Rate** | **~85%** | Measured on clean problem specs with unambiguous boundary definitions |
| **Pass@2 Rate (Self-Corrected)** | **~95%** | Resolved after one diagnostic feedback cycle |
| **Average Pass Latency (Turn 1)** | **14.2s – 17.5s** | End-to-end (prompt read, inference, disk write, test run) |
| **Average Pass Latency (Turn 2)** | **28.0s – 33.5s** | Inclusive of failed test run, traceback parsing, and patch re-application |
| **Adapter Footprint** | **83.1 MB** | GGUF format, rank-16 LoRA on Q/K/V projections |
| **Max Turn Ceiling** | **5 Turns** | Prevented infinite loops; solutions converged by Turn 2 or failed |

---

## 4. Key Technical Challenges & Diagnostics

### Context Saturation in 2B Models
* **Failure Mode:** Feeding complete `pytest` stdout/stderr tracebacks over 3 or more turns resulted in catastrophic context degradation. By Turn 4, the model frequently entered repetitive loops or hallucinated syntax blocks.
* **Remedy:** Implemented aggressive traceback filtering in `agent/harness.py`. Instead of dumping full process outputs, the harness extracted only the root assertion failure or exception line (e.g., `IndexError: list index out of range at line 14`), appending a single-line constraint prompt.

### Boundary Condition Fragility
* **Failure Mode:** On problems like `find_peak_element`, the model initially assumed non-empty arrays with interior peaks, omitting explicit checks for single-element lists or strictly ascending/descending inputs.
* **Remedy:** Fine-tuned the agent's system instructions to prioritize guard clauses (`if not nums: return -1; if len(nums) == 1: return 0`) before constructing primary loops.

### Packaging & Environment Cleanliness
* **Failure Mode:** Local development artifacts (`__pycache__`, `.pyc` files compiled on Python 3.12) contaminated early archive builds, causing potential interpreter mismatch flags in clean test environments.
* **Remedy:** Standardized an automated pre-build hygiene sequence using `find` to strip compiled caches, explicitly validating executable bits (`chmod 755`) on entrypoint shell scripts before calculating SHA256 checksums.

---

## 5. Key Takeaways & Lessons Learned

1. **Closed-Loop Feedback Trumps Scale:** A 2B parameter model equipped with an immediate, deterministic testing loop (`pytest`) routinely matches or exceeds the zero-shot accuracy of much larger models (7B–13B) running open-loop generation.
2. **Deterministic Sampling is Crucial for Agents:** Higher temperature values (e.g., $\ge 0.5$) severely destabilized the JSON schema parser in the harness. Dropping temperature to `0.1` eliminated malformed JSON tool calls entirely.
3. **Keep the Tool Space Small:** Restricting the model's action space to targeted reading and patching eliminated exploratory turns, ensuring every turn directly contributed to passing test suites.
