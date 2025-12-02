# vLLM Plugins

A monorepo containing multiple vLLM plugins for extending vLLM's functionality without maintaining a fork.

## Repository Structure

```
vllm-plugins/
├── plugins/                        # Plugin packages
│   ├── vllm-plugin-example/       # Template plugin with best practices
│   ├── vllm-custom-models/        # Custom model architectures
│   ├── vllm-patches/              # Surgical patches for vLLM classes
│   ├── vllm-custom-loggers/       # Custom stat loggers for metrics
│   ├── vllm-entropy-decoder/      # Adaptive entropy-based decoding
│   └── vllm-cot-decoder/          # Confidence-weighted CoT decoding
├── shared/                         # Shared utilities
│   └── vllm_plugin_utils/         # Version checking, common helpers
├── scripts/                        # Development scripts
│   ├── build_all.sh               # Build all plugins
│   ├── test_all.sh                # Run all tests
│   ├── install_dev.sh             # Install all in dev mode
│   └── lint.sh                    # Lint all code
└── docs/                           # Documentation
    └── plugin-development.md      # Comprehensive development guide
```

## Quick Start

### Install All Plugins (Development)

```bash
./scripts/install_dev.sh
```

### Install a Specific Plugin

```bash
cd plugins/vllm-custom-models
pip install -e ".[dev]"
```

### Run Tests

```bash
./scripts/test_all.sh
```

## Available Plugins

| Plugin | Entry Point Group | Description |
|--------|-------------------|-------------|
| `vllm-plugin-example` | `vllm.general_plugins` | Template demonstrating best practices |
| `vllm-custom-models` | `vllm.general_plugins` | Register custom model architectures |
| `vllm-patches` | `vllm.general_plugins` | Apply surgical patches to vLLM classes |
| `vllm-custom-loggers` | `vllm.stat_logger_plugins` | Custom metrics and logging backends |
| `vllm-entropy-decoder` | `vllm.logits_processors` | Adaptive entropy-based decoding strategy |
| `vllm-cot-decoder` | `vllm.logits_processors` | Confidence-weighted Chain-of-Thought decoding |

## Plugin Types

vLLM supports several plugin entry point groups:

- **`vllm.general_plugins`**: General extensions, custom models, patches
- **`vllm.platform_plugins`**: Hardware backend integrations
- **`vllm.stat_logger_plugins`**: Custom metrics/logging
- **`vllm.logits_processors`**: Custom decoding strategies and logits manipulation (applies globally)
- **`vllm.io_processor_plugins`**: Input/output processing

> **Note on logits processors**: These plugins apply to ALL requests when installed.
> vLLM v1 does not support per-request selection. Deploy one strategy per server instance.

## Usage Examples

### Custom Models

After installing `vllm-custom-models`, your custom architectures are automatically available:

```python
from vllm import LLM

# Load a custom model (architecture registered by plugin)
llm = LLM(model="path/to/custom/model")
```

### Patches

Control which patches apply via environment variable:

```bash
# Enable specific patches
VLLM_CUSTOM_PATCHES=PrioritySchedulerPatch python app.py

# Enable all available patches
VLLM_CUSTOM_PATCHES=* python app.py
```

### Custom Decoding Strategies

> **Important**: Logits processor plugins apply **globally** to all requests when installed.
> vLLM v1 does not support per-request logits processor selection via the OpenAI API.
> Install only ONE decoding strategy plugin per deployment.

Use entropy-based or CoT-based decoding by installing the appropriate plugin:

```bash
# For entropy-based adaptive decoding
pip install -e plugins/vllm-entropy-decoder/

# OR for confidence-weighted CoT decoding (not both)
pip install -e plugins/vllm-cot-decoder/
```

Once installed, the decoder applies automatically to all requests:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="your-model")

# The installed decoder is applied automatically - no need to specify it
sampling_params = SamplingParams(temperature=0.8)

output = llm.generate("Your prompt here", sampling_params)
```

To switch strategies, use different Docker images or reinstall with the desired plugin.

### Selective Plugin Loading

Load only specific plugins:

```bash
VLLM_PLUGINS=custom_models,vllm_patches python app.py
```

## Development

### Creating a New Plugin

1. Copy the example plugin as a template:
   ```bash
   cp -r plugins/vllm-plugin-example plugins/vllm-my-plugin
   ```

2. Rename packages and update `pyproject.toml`

3. Implement your plugin logic in `register.py`

4. Add tests in `tests/`

See [docs/plugin-development.md](docs/plugin-development.md) for the comprehensive development guide.

### Key Guidelines

- **Re-entrancy**: Plugin registration functions must be safe to call multiple times
- **Version compatibility**: Always check/declare vLLM version requirements
- **Minimal changes**: Patches should be surgical, not wholesale replacements
- **Graceful degradation**: Handle missing dependencies gracefully

## Documentation

- [Plugin Development Guide](docs/plugin-development.md) - Comprehensive guide with best practices
- [vLLM Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system.html) - Official documentation
- [vLLM Plugin Blog Post](https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html) - Architecture deep-dive

## License

Apache-2.0
