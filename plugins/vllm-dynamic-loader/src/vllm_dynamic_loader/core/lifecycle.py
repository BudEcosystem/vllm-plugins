"""Plugin lifecycle management for vLLM integration."""

import logging
import os
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from vllm_dynamic_loader.core.discovery import (
    EntryPointDiscovery,
    discover_package_entry_points,
)
from vllm_dynamic_loader.core.registry import (
    PluginInfo,
    PluginRegistry,
    PluginState,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Configure logging for this module to match vLLM's log level
_log_level_str = os.environ.get("VLLM_LOGGING_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_str, logging.INFO)

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


class PluginLifecycleManager:
    """Manages the lifecycle of dynamically loaded plugins.

    Handles activation (loading entry points, calling register functions)
    and deactivation of plugins with proper cleanup.
    """

    def __init__(self, vllm_config: Optional["VllmConfig"] = None):
        """Initialize the lifecycle manager.

        Args:
            vllm_config: Optional vLLM configuration for processor instantiation.
        """
        self.registry = PluginRegistry()
        self.discovery = EntryPointDiscovery()
        self.vllm_config = vllm_config

        # Callbacks for different plugin types
        self._load_callbacks: Dict[str, Callable[[str, Any], None]] = {}
        self._unload_callbacks: Dict[str, Callable[[str], None]] = {}

        # Track active components for unloading
        self._active_components: Dict[str, Dict[str, Any]] = {}

    def register_load_callback(self, group: str, callback: Callable[[str, Any], None]) -> None:
        """Register a callback for when entry points of a group are loaded.

        Args:
            group: Entry point group (e.g., "vllm.general_plugins").
            callback: Function to call with (name, loaded_object).
        """
        self._load_callbacks[group] = callback
        logger.debug(f"Registered load callback for {group}")

    def register_unload_callback(self, group: str, callback: Callable[[str], None]) -> None:
        """Register a callback for when entry points of a group are unloaded.

        Args:
            group: Entry point group.
            callback: Function to call with entry point name.
        """
        self._unload_callbacks[group] = callback
        logger.debug(f"Registered unload callback for {group}")

    def activate_plugin(self, plugin_info: PluginInfo) -> bool:
        """Activate a newly installed plugin.

        This discovers and loads all entry points from the plugin.

        Args:
            plugin_info: Information about the plugin to activate.

        Returns:
            True if activation succeeded, False otherwise.
        """
        try:
            self.registry.update_state(plugin_info.id, PluginState.LOADING)

            # Discover entry points from this package
            if plugin_info.name:
                eps = discover_package_entry_points(plugin_info.name)
                plugin_info.entry_points = eps

                # Load entry points for each vLLM group
                for group, ep_names in eps.items():
                    self._load_group_entry_points(plugin_info.id, group, ep_names)

            self.registry.update_state(plugin_info.id, PluginState.LOADED)
            logger.info(f"Plugin {plugin_info.id} ({plugin_info.name}) activated successfully")
            return True

        except Exception as e:
            error_msg = f"Failed to activate plugin: {e}"
            logger.error(error_msg)
            self.registry.update_state(plugin_info.id, PluginState.FAILED, error_msg)
            return False

    def _load_group_entry_points(self, plugin_id: str, group: str, ep_names: List[str]) -> None:
        """Load entry points for a specific group.

        Args:
            plugin_id: ID of the plugin being loaded.
            group: Entry point group.
            ep_names: Names of entry points to load.
        """

        self.discovery.invalidate_caches()

        # Reimport for fresh data
        import importlib.metadata as fresh_metadata

        all_eps = fresh_metadata.entry_points()
        # Python 3.10+ returns SelectableGroups with select() method
        # Python 3.9 returns a dict-like object
        if hasattr(all_eps, "select"):
            eps = list(all_eps.select(group=group))
        else:
            # Python 3.9 compatibility
            eps = list(all_eps.get(group, []))  # type: ignore[union-attr]

        for ep in eps:
            if ep.name in ep_names:
                try:
                    loaded = ep.load()

                    # Store for potential unloading
                    if plugin_id not in self._active_components:
                        self._active_components[plugin_id] = {}
                    self._active_components[plugin_id][f"{group}:{ep.name}"] = loaded

                    # Call registered callback
                    if group in self._load_callbacks:
                        self._load_callbacks[group](ep.name, loaded)

                    # Special handling for different plugin types
                    self._handle_loaded_entry_point(group, ep.name, loaded)

                    logger.info(f"Loaded entry point {group}:{ep.name}")

                except Exception as e:
                    logger.error(f"Failed to load {group}:{ep.name}: {e}")

    def _handle_loaded_entry_point(self, group: str, name: str, loaded: Any) -> None:
        """Handle special loading behavior for different plugin types.

        Args:
            group: Entry point group.
            name: Entry point name.
            loaded: The loaded object.
        """
        if group == "vllm.general_plugins":
            # General plugins have a register() function
            if callable(loaded):
                try:
                    loaded()
                    logger.info(f"Called register function for {name}")
                except Exception as e:
                    logger.error(f"Failed to call register for {name}: {e}")

        elif group == "vllm.logits_processors":
            # Logits processors are classes - vLLM will instantiate them
            # Automatically trigger hot-swap to make the processor active
            self._trigger_hotswap(name)

        elif group == "vllm.stat_logger_plugins":
            # Stat loggers are classes instantiated by vLLM
            logger.info(f"Loaded stat logger '{name}'")

    def _trigger_hotswap(self, processor_name: str) -> None:
        """Trigger a hot-swap to activate a newly loaded processor.

        Args:
            processor_name: Name of the processor to activate.
        """
        try:
            from vllm_dynamic_loader.hotswap.coordinator import SwapCoordinator
            from vllm_dynamic_loader.config import get_config

            config = get_config()

            # Only auto-hotswap if enabled
            if not config.auto_activate:
                logger.info(
                    f"Loaded logits processor '{processor_name}'. "
                    "Auto-activation disabled. Use hot-swap API to activate."
                )
                return

            coordinator = SwapCoordinator()
            result = coordinator.request_swap(processor_name, timeout_seconds=10.0)

            if result.get("results"):
                success_count = sum(
                    1 for r in result["results"].values()
                    if r.get("status") == "success"
                )
                logger.info(
                    f"Hot-swapped to processor '{processor_name}' "
                    f"({success_count} process(es) acknowledged)"
                )
            else:
                logger.warning(
                    f"Hot-swap requested for '{processor_name}' but no processes acknowledged. "
                    "The processor will be active on next request if HotSwapProxyProcessor is in use."
                )

        except Exception as e:
            logger.warning(
                f"Could not auto-hotswap to '{processor_name}': {e}. "
                "Use the hot-swap API manually: POST /plugins/hot-swap"
            )

    def deactivate_plugin(self, plugin_id: str) -> bool:
        """Deactivate and prepare a plugin for unloading.

        WARNING: Complete unloading of Python modules is inherently
        problematic. This marks the plugin as unloaded and calls
        cleanup callbacks, but the code may remain in memory.

        Args:
            plugin_id: ID of the plugin to deactivate.

        Returns:
            True if deactivation succeeded, False otherwise.
        """
        plugin = self.registry.get(plugin_id)
        if not plugin:
            logger.warning(f"Plugin {plugin_id} not found in registry")
            return False

        try:
            self.registry.update_state(plugin_id, PluginState.UNLOADING)

            # Call unload callbacks
            if plugin_id in self._active_components:
                for key in self._active_components[plugin_id]:
                    group, name = key.split(":", 1)
                    if group in self._unload_callbacks:
                        try:
                            self._unload_callbacks[group](name)
                        except Exception as e:
                            logger.error(f"Unload callback failed for {key}: {e}")

                del self._active_components[plugin_id]

            self.registry.update_state(plugin_id, PluginState.UNLOADED)
            logger.info(f"Plugin {plugin_id} deactivated")

            # Note: We can't truly unload Python modules, but we've cleaned up
            # what we can. A server restart may be needed for complete removal.

            return True

        except Exception as e:
            error_msg = f"Failed to deactivate plugin: {e}"
            logger.error(error_msg)
            self.registry.update_state(plugin_id, PluginState.FAILED, error_msg)
            return False

    def get_active_plugins(self) -> List[PluginInfo]:
        """Get all currently active plugins.

        Returns:
            List of loaded plugin info.
        """
        return [p for p in self.registry.get_all() if p.state == PluginState.LOADED]

    def get_plugin_status(self) -> Dict[str, Dict]:
        """Get status of all plugins.

        Returns:
            Dict mapping plugin IDs to status info.
        """
        return {
            p.id: {
                "name": p.name,
                "source": p.source.value,
                "state": p.state.value,
                "entry_points": p.entry_points,
                "error": p.error,
                "loaded_at": p.loaded_at,
            }
            for p in self.registry.get_all()
        }

    def get_loaded_components(self, plugin_id: str) -> Dict[str, Any]:
        """Get loaded components for a plugin.

        Args:
            plugin_id: Plugin ID.

        Returns:
            Dict mapping component keys to loaded objects.
        """
        return self._active_components.get(plugin_id, {}).copy()
