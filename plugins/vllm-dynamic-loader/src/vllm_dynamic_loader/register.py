"""Plugin registration and initialization.

This module handles registration of the dynamic plugin loader with vLLM's
plugin system. It is the entry point called by vLLM at startup.

Environment Variables:
    VLLM_PLUGIN_WATCH_DIR: Directory to watch for plugins (default: /plugins)
    VLLM_PLUGIN_WATCH_ENABLED: Enable directory watching (default: true)
    VLLM_PLUGIN_AUTO_ACTIVATE: Auto-activate installed plugins (default: true)
    VLLM_PLUGIN_API_ENABLED: Enable REST API (default: true)
    VLLM_PLUGIN_API_PORT: Port for standalone API server if patching fails (default: 8001)
"""

import logging
import os
import sys
import threading
from multiprocessing import current_process
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_dynamic_loader.core.loader import DynamicPluginLoader

# Configure logging for this plugin to match vLLM's log level
# This ensures our log messages are visible alongside vLLM's logs
_log_level_str = os.environ.get("VLLM_LOGGING_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_str, logging.INFO)

# Set up handler if not already configured
logger = logging.getLogger(__name__)
logger.setLevel(_log_level)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(_log_level)
    formatter = logging.Formatter(
        "%(levelname)s %(asctime)s [%(name)s] %(message)s", datefmt="%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Track registration state (re-entrant safety)
_registered = False
_loader_instance = None
_api_server_thread = None


def register() -> None:
    """Register the dynamic plugin loader with vLLM.

    This is the entry point called by vLLM's plugin system.
    Must be re-entrant (safe to call multiple times).
    """
    global _registered, _loader_instance, _api_server_thread

    if _registered:
        logger.debug("Dynamic loader already registered, skipping")
        return

    process_name = current_process().name
    logger.info(f"Registering dynamic plugin loader in {process_name}")

    try:
        from vllm_dynamic_loader.config import get_config
        from vllm_dynamic_loader.core.loader import DynamicPluginLoader

        config = get_config()

        # Initialize the loader
        _loader_instance = DynamicPluginLoader(config=config)

        # Start directory watcher only in main process
        if process_name == "MainProcess":
            if config.watch_enabled:
                _loader_instance.start_watching()
                logger.info(f"Started plugin directory watcher at {config.watch_directory}")
            else:
                logger.info("Plugin directory watching disabled")

            # Initialize REST API
            api_enabled = os.environ.get("VLLM_PLUGIN_API_ENABLED", "true").lower() == "true"
            if api_enabled:
                _setup_api()

            # Log configuration
            logger.info(
                f"Dynamic loader config: "
                f"watch_dir={config.watch_directory}, "
                f"registry={config.registry_path}, "
                f"auto_activate={config.auto_activate}"
            )

        _registered = True
        logger.info(f"Dynamic plugin loader registered successfully in {process_name}")

    except Exception as e:
        logger.error(f"Failed to register dynamic plugin loader: {e}")
        # Don't raise - allow vLLM to continue without dynamic loading
        _registered = True  # Prevent repeated failures


def _setup_api() -> None:
    """Set up the REST API for plugin management."""
    global _api_server_thread

    try:
        from vllm_dynamic_loader.api.middleware import patch_vllm_app

        # Try to patch vLLM's app first
        if patch_vllm_app():
            logger.info("Plugin API available at /plugins/*")
            return

        # Patching failed - start standalone server
        standalone_port = int(os.environ.get("VLLM_PLUGIN_API_PORT", "8001"))
        _start_standalone_api(standalone_port)

    except ImportError:
        logger.info(
            "FastAPI not installed. Plugin API disabled. "
            "Install with: pip install vllm-dynamic-loader[api]"
        )
    except Exception as e:
        logger.error(f"Failed to setup plugin API: {e}")


def _start_standalone_api(port: int) -> None:
    """Start the standalone API server in a background thread."""
    global _api_server_thread

    def run_server():
        try:
            from vllm_dynamic_loader.api.middleware import run_standalone_server

            logger.info(f"Starting standalone plugin API server on port {port}")
            run_standalone_server(host="0.0.0.0", port=port)
        except Exception as e:
            logger.error(f"Standalone API server failed: {e}")

    _api_server_thread = threading.Thread(target=run_server, daemon=True)
    _api_server_thread.start()
    logger.info(f"Plugin API available at http://localhost:{port}/plugins/")


def get_loader() -> "DynamicPluginLoader":
    """Get the dynamic plugin loader instance.

    Returns:
        The singleton DynamicPluginLoader instance.

    Raises:
        RuntimeError: If the loader hasn't been initialized.
    """
    global _loader_instance
    if _loader_instance is None:
        raise RuntimeError(
            "Dynamic plugin loader not initialized. " "Ensure register() has been called by vLLM."
        )
    return _loader_instance


def is_registered() -> bool:
    """Check if the dynamic loader has been registered.

    Returns:
        True if registered, False otherwise.
    """
    return _registered
