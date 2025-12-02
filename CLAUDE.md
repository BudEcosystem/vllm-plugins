# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a monorepo containing multiple vLLM plugins that extend vLLM's functionality without forking. Plugins use Python's `entry_points` mechanism for discovery and are loaded by vLLM at startup in all processes (main, workers, GPU/CPU workers).

## Commands

### Development Setup
```bash
./scripts/install_dev.sh          # Install all plugins in dev mode
pip install -e "plugins/vllm-entropy-decoder/.[dev]"  # Install single plugin
pre-commit install                # Install pre-commit hooks
```

### Testing
```bash
./scripts/test_all.sh             # Run all plugin tests
pytest plugins/vllm-entropy-decoder/tests/  # Run single plugin tests
pytest plugins/vllm-entropy-decoder/tests/test_entropy_processor.py::TestEntropyLogitsProcessor::test_apply_2d_input  # Single test
```

### Docker
```bash
docker build -t vllm-plugins-test .
docker-compose up -d
python scripts/test_integration.py --base-url http://localhost:8000
```

### Linting
```bash
pre-commit run --all-files        # Run all pre-commit hooks
./scripts/lint.sh                 # Run ruff + mypy
./scripts/lint.sh --fix           # Auto-fix issues
```

### Verify Plugin Installation
```bash
python -c "from importlib.metadata import entry_points; print(list(entry_points(group='vllm.logits_processors')))"
```

## Architecture

### Plugin Types and Entry Points

| Entry Point Group | Purpose | Registration Target |
|-------------------|---------|---------------------|
| `vllm.general_plugins` | Models, patches, extensions | Function that registers |
| `vllm.logits_processors` | Custom decoding strategies | LogitsProcessor subclass |
| `vllm.stat_logger_plugins` | Metrics/logging | Logger class directly |
| `vllm.platform_plugins` | Hardware backends | Platform class |

### Logits Processor Plugins (vLLM v1)

**Critical requirements** - Processors MUST:
1. **Inherit from base class**: `from vllm.v1.sample.logits_processor.interface import LogitsProcessor`
2. **Use exact constructor signature**: `__init__(self, vllm_config, device, is_pin_memory)`
3. **Implement required methods**:
   - `is_argmax_invariant() -> bool` - Return `False` for most processors
   - `update_state(batch_update)` - Handle batch changes
   - `apply(logits) -> logits` - The processing logic (NOT `__call__`)
   - `validate_params(sampling_params)` - Classmethod for validation

**Global-only behavior**: vLLM v1 does NOT support per-request logits processor selection. Plugins apply to ALL requests when installed. Deploy only ONE decoding strategy per server.

**Configuration via environment variables** (since vLLM instantiates processors):
- `ENTROPY_DECODER_TEMPERATURE`, `ENTROPY_DECODER_THRESHOLD_LOW`, etc.
- `COT_DECODER_CONFIDENCE_THRESHOLD`, `COT_DECODER_SHARPENING_FACTOR`, etc.

### General Plugins Pattern

Registration functions must be **re-entrant** (safe to call multiple times):

```python
_registered = False

def register() -> None:
    global _registered
    if _registered:
        return
    # ... registration logic ...
    _registered = True
```

### Key Components

- **`vllm-entropy-decoder`**: Adaptive entropy/varentropy-based decoding
- **`vllm-cot-decoder`**: Confidence-weighted Chain-of-Thought decoding
- **`vllm-patches`**: VLLMPatch[T] generic for surgical class modifications
- **`vllm-custom-models`**: Register custom model architectures
- **`shared/vllm_plugin_utils/`**: Common version checking utilities

### Project Structure Pattern

```
plugins/vllm-{name}/
├── pyproject.toml          # Entry point: [project.entry-points."vllm.logits_processors"]
├── src/vllm_{name}/
│   ├── processor.py        # LogitsProcessor subclass
│   └── metrics.py          # Helper modules
└── tests/
    └── test_{name}_processor.py
```

## Environment Variables

- `VLLM_PLUGINS` - Comma-separated list of plugins to load (empty = all)
- `VLLM_CUSTOM_PATCHES` - Comma-separated patch names or `*` for all

## Common Errors

| Error | Fix |
|-------|-----|
| `must be a subclass of LogitsProcessor` | Add `(LogitsProcessor)` to class definition |
| `has no attribute 'is_argmax_invariant'` | Add method returning `False` |
| `has no attribute 'update_state'` | Add method tracking `batch_update.batch_size` |
| Processor not called | Use `apply()` method, not `__call__` |
