# Gemma 2 Autonomous Code-Infilling Agent (ADK Submission)

An autonomous software engineering agent powered by a fine-tuned Gemma 2 2B model. The system operates via a structured, JSON-based tool-use protocol (`read_file`, `apply_infill`) coupled with a closed-loop verification harness powered by `pytest` for multi-turn diagnostic self-correction.

---

## 1. System Specifications

| Component | Specification |
|---|---|
| **Base Architecture** | Gemma 2 (2 Billion Parameters) |
| **Adapter Type** | LoRA Adapter (`infiller-adapter.gguf`, ~83.1 MB) |
| **Inference Backend** | Ollama local daemon (OpenAI-compatible REST API) |
| **Context Window** | 8,192 tokens |
| **Sampling Defaults** | Temperature: `0.1`–`0.2`, Top-P: `0.95` |
| **Supported Tools** | `read_file`, `apply_infill` |
| **Verification Loop** | In-situ `pytest` execution with targeted diagnostic feedback |

---

## 2. Directory Layout

```text
submission/
├── agent/
│   ├── __init__.py
│   ├── harness.py          # Agentic loop, tool parser, and self-correction runner
│   └── tools.py            # Sandboxed filesystem inspection and patch tools
├── config/
│   ├── adk_manifest.json   # Google Agent Development Kit registration manifest
│   └── agent_config.yaml   # LLM parameters, timeout gates, and endpoint settings
├── models/
│   ├── Modelfile           # Ollama build definition specifying base and LoRA adapter
│   └── infiller-adapter.gguf # Serialized fine-tuned LoRA weights
├── scripts/
│   ├── start_ollama.sh     # Daemon check, bootstrap, and model registration
│   └── run_agent.sh        # Main submission entrypoint for evaluation runners
├── requirements.txt        # Runtime dependencies (openai, pyyaml, pytest)
└── README.md               # Architecture documentation and usage guide
