# vLLM Plugin Example

A template plugin demonstrating best practices for building vLLM plugins.

## Installation

```bash
pip install -e .
```

## Usage

Once installed, the plugin is automatically loaded by vLLM. You can verify it's loaded by checking the logs during vLLM startup.

To selectively load this plugin:

```bash
VLLM_PLUGINS=example_plugin python -c "import vllm"
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Structure

```
vllm-plugin-example/
├── pyproject.toml          # Package configuration and entry points
├── src/
│   └── vllm_plugin_example/
│       ├── __init__.py     # Package init with version
│       └── register.py     # Entry point function
└── tests/
    └── test_example.py     # Plugin tests
```
