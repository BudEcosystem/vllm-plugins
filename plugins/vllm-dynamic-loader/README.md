# vLLM Dynamic Plugin Loader

Dynamic plugin loader for vLLM - load, unload, and hot-swap plugins at runtime without restarting the server.

## Features

- **Configuration File**: Define plugins to install at startup via YAML config
  - PyPI packages with version constraints
  - Git repositories with branch/tag/commit support
  - Local filesystem paths for development

- **Dynamic Plugin Loading**: Install plugins from multiple sources at runtime
  - Mounted volumes: Watch a directory for new plugin files (optional)
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

### 1. Create a Plugin Configuration File (Recommended)

Create `~/.config/vllm/plugins.yaml` to define plugins that should be installed at startup:

```yaml
# Plugins to install at vLLM startup
plugins:
  # Install from PyPI
  - source: pypi
    package: vllm-entropy-decoder
    version: ">=0.1.0"

  # Install from a local folder (great for development)
  - source: local
    path: ~/projects/my-vllm-plugin
    editable: true  # default: true

  # Install from Git repository
  - source: git
    url: https://github.com/org/vllm-plugins
    ref: main
    subdirectory: plugins/vllm-cot-decoder  # for monorepos

# Optional: set default logits processor
default_processor: entropy
```

### 2. Start vLLM

The plugin registers automatically via entry points and installs plugins from the config file:

```bash
# Start vLLM normally - plugins are installed from config
vllm serve your-model

# Or specify a custom config file location
VLLM_PLUGIN_CONFIG=/path/to/plugins.yaml vllm serve your-model
```

### 3. Install Plugins at Runtime (Optional)

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

**Via Mounted Volume (requires enabling watcher):**

Enable the directory watcher and drop plugin files into the watch directory:

```bash
# Enable the watcher (disabled by default)
export VLLM_PLUGIN_WATCH_ENABLED=true
export VLLM_PLUGIN_WATCH_DIR=/path/to/plugins

# Start vLLM, then drop plugins into the directory
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

### Plugin Configuration File

The recommended way to manage plugins is via a YAML configuration file.

**File Location (in priority order):**
1. Path specified by `VLLM_PLUGIN_CONFIG` environment variable
2. `~/.config/vllm/plugins.yaml` (XDG config directory)

> See [examples/plugins.yaml](examples/plugins.yaml) for a complete example with all options.

**Full Configuration Reference:**

```yaml
plugins:
  # PyPI package
  - source: pypi
    package: vllm-entropy-decoder  # required
    version: ">=0.1.0"             # optional: version constraint
    enabled: true                  # optional: skip if false (default: true)
    vllm_min: "0.6.0"              # optional: minimum vLLM version (inclusive)
    vllm_max: "0.8.0"              # optional: maximum vLLM version (exclusive)

  # Local filesystem path
  - source: local
    path: ~/projects/my-plugin    # required: supports ~ expansion
    editable: true                # optional: pip install -e (default: true)
    vllm_min: "0.6.0"             # optional: minimum vLLM version

  # Git repository
  - source: git
    url: https://github.com/org/repo  # required
    ref: main                         # optional: branch, tag, or commit
    subdirectory: plugins/my-plugin   # optional: for monorepos
    editable: true                    # optional (default: true)

# Optional: default logits processor for hot-swap
default_processor: entropy
```

**vLLM Version Constraints:**

Use `vllm_min` and `vllm_max` to specify compatible vLLM versions:
- `vllm_min`: Minimum required version (inclusive). Plugin won't install if vLLM is older.
- `vllm_max`: Maximum supported version (exclusive). Plugin won't install if vLLM is newer.
- If not specified, the plugin installs regardless of vLLM version.

```yaml
plugins:
  # Only for vLLM 0.6.x
  - source: pypi
    package: vllm-legacy-plugin
    vllm_min: "0.6.0"
    vllm_max: "0.7.0"

  # For vLLM 0.7.0 and newer
  - source: pypi
    package: vllm-modern-plugin
    vllm_min: "0.7.0"

  # No version constraint - always installs
  - source: pypi
    package: vllm-universal-plugin
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_PLUGIN_CONFIG` | - | Path to plugin configuration file (YAML) |
| `VLLM_PLUGIN_WATCH_DIR` | `/plugins` | Directory to watch for plugins |
| `VLLM_PLUGIN_WATCH_ENABLED` | `false` | Enable directory watching (use config file instead) |
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

### Using Configuration File (Recommended)

```dockerfile
FROM budstudio/vllm:latest

# Install dynamic loader
COPY plugins/vllm-dynamic-loader /app/plugins/vllm-dynamic-loader
RUN pip install -e /app/plugins/vllm-dynamic-loader/.[api]

# Copy plugin configuration
COPY plugins.yaml /root/.config/vllm/plugins.yaml
```

```yaml
# docker-compose.yml
services:
  vllm:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./plugins.yaml:/root/.config/vllm/plugins.yaml:ro
      - ./local-plugins:/app/local-plugins:ro  # for local source plugins
    environment:
      - HOTSWAP_DEFAULT_PROCESSOR=entropy
```

### Using Directory Watcher

```dockerfile
FROM budstudio/vllm:latest

# Install dynamic loader
COPY plugins/vllm-dynamic-loader /app/plugins/vllm-dynamic-loader
RUN pip install -e /app/plugins/vllm-dynamic-loader/.[api]

# Create plugin directory
RUN mkdir -p /plugins

# Enable watcher
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
      - VLLM_PLUGIN_WATCH_ENABLED=true
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

### Plugin Manifest File

Plugin developers can include a `vllm-plugin.yaml` manifest file in their plugin root to declare metadata and version compatibility. This allows the plugin to specify its own vLLM version requirements.

> See [examples/vllm-plugin.yaml](examples/vllm-plugin.yaml) for a complete example with all options.

**Create `vllm-plugin.yaml` in your plugin root:**

```yaml
# Plugin metadata
name: my-vllm-plugin
description: A custom logits processor for adaptive decoding
version: 1.0.0
author: Your Name
license: Apache-2.0
homepage: https://github.com/yourname/my-vllm-plugin

# Categorization
tags:
  - logits-processor
  - decoding
  - experimental

# vLLM version compatibility
vllm_min: "0.6.0"    # Minimum vLLM version (inclusive)
vllm_max: "0.9.0"    # Maximum vLLM version (exclusive)

# Additional pip dependencies (optional)
dependencies:
  - numpy>=1.20
  - scipy
```

**How version constraints work:**

1. **User config takes precedence**: If `vllm_min`/`vllm_max` is specified in `plugins.yaml`, those values are used.
2. **Manifest as fallback**: If the user doesn't specify version constraints, the plugin's manifest constraints are checked.
3. **No constraints = always install**: If neither user config nor manifest specifies constraints, the plugin installs regardless of vLLM version.

**Supported manifest filenames** (checked in order):
- `vllm-plugin.yaml`
- `vllm-plugin.yml`
- `vllm_plugin.yaml`

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
