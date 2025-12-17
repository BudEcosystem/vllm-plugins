# vLLM Dynamic Plugin Loader

Dynamic plugin loader for vLLM - load, unload, and hot-swap plugins at runtime without restarting the server.

## Features

- **Dynamic Plugin Loading**: Install plugins from multiple sources at runtime
  - Mounted volumes: Watch a directory for new plugin files
  - PyPI/wheel URLs: Install from package repositories
  - Git repositories: Clone and install directly

- **Hot-Swap Logits Processors**: Seamlessly swap decoding strategies without restart
  - Graceful transition with in-flight request handling
  - Multi-process coordination across vLLM workers

- **REST API**: Full management interface at `/plugins/*`

## Installation

```bash
# Basic installation
pip install -e plugins/vllm-dynamic-loader/

# With REST API support
pip install -e "plugins/vllm-dynamic-loader/.[api]"

# With development dependencies
pip install -e "plugins/vllm-dynamic-loader/.[dev]"
```

## Quick Start

### 1. Start vLLM with Dynamic Loader

The plugin registers automatically via entry points:

```bash
# Set watch directory (optional)
export VLLM_PLUGIN_WATCH_DIR=/path/to/plugins

# Start vLLM normally
vllm serve your-model
```

### 2. Install Plugins at Runtime

**Via REST API:**

```bash
# Install from PyPI
curl -X POST http://localhost:8000/plugins/install/pypi \
  -H "Content-Type: application/json" \
  -d '{"package": "vllm-entropy-decoder", "auto_load": true}'

# Install from Git
curl -X POST http://localhost:8000/plugins/install/git \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/org/my-plugin", "branch": "main"}'
```

**Via Mounted Volume:**

Simply drop plugin files into the watch directory:

```bash
cp my_processor.py /path/to/plugins/
# Automatically detected and loaded!
```

### 3. Hot-Swap Logits Processors

```bash
# List available processors
curl http://localhost:8000/plugins/processors/available

# Swap to entropy decoder
curl -X POST http://localhost:8000/plugins/hot-swap \
  -H "Content-Type: application/json" \
  -d '{"processor": "entropy"}'

# Switch to passthrough (no processing)
curl -X POST http://localhost:8000/plugins/hot-swap \
  -H "Content-Type: application/json" \
  -d '{"processor": null}'
```

## API Integration

The plugin API is integrated with vLLM using one of these methods (in order of preference):

1. **ASGI Middleware Patching** (preferred): The plugin attempts to patch vLLM's FastAPI application to add `/plugins/*` routes directly.

2. **Standalone Server** (fallback): If patching fails, a standalone server starts on port 8001 (configurable via `VLLM_PLUGIN_API_PORT`).

3. **Manual Integration**: You can manually add the routes to your custom vLLM setup:

```python
from vllm_dynamic_loader.api import create_router

# Add to existing FastAPI app
router = create_router()
app.include_router(router)

# Or use the middleware directly
from vllm_dynamic_loader.api.middleware import PluginAPIMiddleware
app = PluginAPIMiddleware(your_app)
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_PLUGIN_WATCH_DIR` | `/plugins` | Directory to watch for plugins |
| `VLLM_PLUGIN_WATCH_ENABLED` | `true` | Enable directory watching |
| `VLLM_PLUGIN_REGISTRY` | `~/.local/share/vllm-plugins/registry.json` | Registry file path |
| `VLLM_PLUGIN_AUTO_ACTIVATE` | `true` | Auto-activate installed plugins |
| `VLLM_PLUGIN_API_ENABLED` | `true` | Enable REST API |
| `VLLM_PLUGIN_API_PORT` | `8001` | Port for standalone API (if patching fails) |
| `HOTSWAP_DEFAULT_PROCESSOR` | `passthrough` | Default logits processor |
| `HOTSWAP_GRACEFUL_TIMEOUT_MS` | `5000` | Timeout for graceful swap |

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/plugins/` | List all plugins |
| GET | `/plugins/status` | Get loader status |
| GET | `/plugins/{id}` | Get plugin details |
| POST | `/plugins/{id}/load` | Load a plugin |
| POST | `/plugins/{id}/unload` | Unload a plugin |
| DELETE | `/plugins/{id}` | Uninstall a plugin |
| POST | `/plugins/install/pypi` | Install from PyPI |
| POST | `/plugins/install/url` | Install from wheel URL |
| POST | `/plugins/install/git` | Install from git repo |
| POST | `/plugins/hot-swap` | Hot-swap logits processor |
| GET | `/plugins/processors/available` | List available processors |
| GET | `/plugins/processors/active` | Get active processor info |

## Docker Usage

```dockerfile
FROM budstudio/vllm:latest

# Install dynamic loader
COPY plugins/vllm-dynamic-loader /app/plugins/vllm-dynamic-loader
RUN pip install -e /app/plugins/vllm-dynamic-loader/.[api]

# Create plugin directory
RUN mkdir -p /plugins

# Set environment
ENV VLLM_PLUGIN_WATCH_DIR=/plugins
ENV VLLM_PLUGIN_WATCH_ENABLED=true
```

```yaml
# docker-compose.yml
services:
  vllm:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./my-plugins:/plugins
    environment:
      - VLLM_PLUGIN_WATCH_DIR=/plugins
      - HOTSWAP_DEFAULT_PROCESSOR=entropy
```

## Writing Dynamic Plugins

Create a single-file plugin:

```python
# my_processor.py
import torch
from vllm.v1.sample.logits_processor.interface import LogitsProcessor

class MyProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.device = device
        self.batch_size = 0

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update):
        if batch_update:
            self.batch_size = batch_update.batch_size

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # Your custom logic here
        return logits * 0.9

# Entry point for auto-detection
def register():
    pass  # Optional registration logic
```

## Known Limitations

1. **Python Module Unloading**: Python cannot truly unload modules. Deactivated plugins remain in memory until restart.

2. **Global Logits Processors**: vLLM v1 applies processors globally to ALL requests. Only ONE processor can be active at a time.

3. **Multi-Process Coordination**: Uses file-based polling (100ms default interval).

## Development

```bash
# Install dev dependencies
pip install -e "plugins/vllm-dynamic-loader/.[dev]"

# Run tests
pytest plugins/vllm-dynamic-loader/tests/

# Run linting
ruff check plugins/vllm-dynamic-loader/
mypy plugins/vllm-dynamic-loader/
```

## License

Apache-2.0
