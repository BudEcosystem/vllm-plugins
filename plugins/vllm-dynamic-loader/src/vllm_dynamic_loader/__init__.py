"""vLLM Dynamic Plugin Loader.

This plugin enables runtime loading, unloading, and hot-swapping of vLLM plugins
without restarting the server. It supports multiple plugin sources:

- Mounted volumes: Watch a directory for new plugin files
- PyPI packages: Install from PyPI or wheel URLs
- Git repositories: Clone and install from git repos

Key components:
- DynamicPluginLoader: Main orchestrator for plugin lifecycle
- HotSwapProxyProcessor: Enables seamless logits processor swapping
- REST API: Management endpoints at /plugins/*

Environment Variables:
    VLLM_PLUGIN_WATCH_DIR: Directory to watch for plugins (default: /plugins)
    VLLM_PLUGIN_WATCH_ENABLED: Enable directory watching (default: true)
    VLLM_PLUGIN_REGISTRY: Registry file path (default: ~/.local/share/vllm-plugins/registry.json)
    VLLM_PLUGIN_AUTO_ACTIVATE: Auto-activate installed plugins (default: true)
    HOTSWAP_DEFAULT_PROCESSOR: Default logits processor (default: passthrough)
    HOTSWAP_GRACEFUL_TIMEOUT_MS: Timeout for graceful swap (default: 5000)
"""

__version__ = "0.1.0"

from vllm_dynamic_loader.core.loader import DynamicPluginLoader
from vllm_dynamic_loader.core.registry import PluginInfo, PluginRegistry, PluginState

__all__ = [
    "DynamicPluginLoader",
    "PluginRegistry",
    "PluginInfo",
    "PluginState",
]
