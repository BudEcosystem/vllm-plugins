"""Main plugin loader orchestrator."""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from vllm_dynamic_loader.config import PluginLoaderConfig, get_config
from vllm_dynamic_loader.core.lifecycle import PluginLifecycleManager
from vllm_dynamic_loader.core.registry import PluginInfo, PluginRegistry, PluginSource, PluginState
from vllm_dynamic_loader.sources.base import InstallResult
from vllm_dynamic_loader.sources.git_installer import GitInstaller
from vllm_dynamic_loader.sources.package_installer import PackageInstaller
from vllm_dynamic_loader.sources.volume_watcher import VolumeWatcher

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


class DynamicPluginLoader:
    """Orchestrates dynamic plugin loading from multiple sources.

    This is the main entry point for the dynamic plugin loading system.
    It coordinates:
    - Plugin discovery from mounted volumes
    - Installation from PyPI and wheel URLs
    - Installation from git repositories
    - Plugin lifecycle management (activation/deactivation)
    """

    _instance: Optional["DynamicPluginLoader"] = None

    def __init__(
        self,
        config: Optional[PluginLoaderConfig] = None,
        vllm_config: Optional["VllmConfig"] = None,
    ):
        """Initialize the dynamic plugin loader.

        Args:
            config: Plugin loader configuration.
            vllm_config: Optional vLLM configuration for lifecycle integration.
        """
        self.config = config or get_config()
        self.vllm_config = vllm_config

        self.registry = PluginRegistry(self.config.registry_path)
        self.lifecycle = PluginLifecycleManager(vllm_config)

        # Initialize source handlers
        self.volume_watcher: Optional[VolumeWatcher] = None
        self.package_installer = PackageInstaller(
            on_install=self._on_plugin_installed,
            on_uninstall=self._on_plugin_uninstalled,
            on_update=self._on_plugin_updated,
            download_dir=self.config.package_download_dir,
        )
        self.git_installer = GitInstaller(
            on_install=self._on_plugin_installed,
            on_uninstall=self._on_plugin_uninstalled,
            on_update=self._on_plugin_updated,
            clone_dir=self.config.git_clone_dir,
            editable=self.config.git_editable,
        )

        # Store singleton reference
        DynamicPluginLoader._instance = self

        logger.info(
            f"DynamicPluginLoader initialized: "
            f"watch_dir={self.config.watch_directory}, "
            f"registry={self.config.registry_path}"
        )

    @classmethod
    def get_instance(cls) -> Optional["DynamicPluginLoader"]:
        """Get the singleton instance."""
        return cls._instance

    def start_watching(self) -> None:
        """Start the directory watcher for mounted volume plugins."""
        if self.volume_watcher is None:
            self.volume_watcher = VolumeWatcher(
                watch_directory=self.config.watch_directory,
                on_install=self._on_plugin_installed,
                on_uninstall=self._on_plugin_uninstalled,
                on_update=self._on_plugin_updated,
                auto_start=True,
                poll_interval=self.config.watch_poll_interval,
            )
            logger.info(f"Started watching {self.config.watch_directory}")

    def stop_watching(self) -> None:
        """Stop the directory watcher."""
        if self.volume_watcher:
            self.volume_watcher.stop()
            self.volume_watcher = None
            logger.info("Stopped volume watcher")

    def _on_plugin_installed(self, plugin_info: PluginInfo) -> None:
        """Handle plugin installation callback."""
        logger.info(f"Plugin installed: {plugin_info.id} ({plugin_info.name})")
        self.registry.register(plugin_info)

        if self.config.auto_activate:
            self.lifecycle.activate_plugin(plugin_info)

    def _on_plugin_uninstalled(self, plugin_id: str) -> None:
        """Handle plugin uninstallation callback."""
        logger.info(f"Plugin uninstalled: {plugin_id}")
        self.lifecycle.deactivate_plugin(plugin_id)
        self.registry.remove(plugin_id)

    def _on_plugin_updated(self, plugin_info: PluginInfo) -> None:
        """Handle plugin update callback."""
        logger.info(f"Plugin updated: {plugin_info.id}")
        # Deactivate old, activate new
        self.lifecycle.deactivate_plugin(plugin_info.id)
        self.lifecycle.activate_plugin(plugin_info)

    # ==================== Public API ====================

    def install_from_pypi(self, package_spec: str, **kwargs) -> InstallResult:
        """Install a plugin from PyPI.

        Args:
            package_spec: Package specification (e.g., "vllm-my-plugin>=0.1.0").
            **kwargs: Additional options (upgrade, force_reinstall, extra_index_url).

        Returns:
            InstallResult with installation outcome.
        """
        return self.package_installer.install(package_spec, **kwargs)

    def install_from_url(self, url: str, **kwargs) -> InstallResult:
        """Install a plugin from a wheel URL.

        Args:
            url: URL to the wheel file.
            **kwargs: Additional options (upgrade, force_reinstall).

        Returns:
            InstallResult with installation outcome.
        """
        return self.package_installer.install(url, **kwargs)

    def install_from_wheel(self, path: str, **kwargs) -> InstallResult:
        """Install a plugin from a local wheel.

        Args:
            path: Path to the wheel file.
            **kwargs: Additional options (upgrade, force_reinstall).

        Returns:
            InstallResult with installation outcome.
        """
        return self.package_installer.install(path, **kwargs)

    def install_from_git(self, url: str, **kwargs) -> InstallResult:
        """Install a plugin from a git repository.

        Args:
            url: Git URL with optional ref (e.g., "https://github.com/user/repo@branch").
            **kwargs: Additional options (branch, tag, commit, depth, editable).

        Returns:
            InstallResult with installation outcome.
        """
        return self.git_installer.install(url, **kwargs)

    def uninstall(self, plugin_id: str) -> bool:
        """Uninstall a plugin.

        Args:
            plugin_id: ID of the plugin to uninstall.

        Returns:
            True if successful.
        """
        plugin = self.registry.get(plugin_id)
        if not plugin:
            logger.warning(f"Plugin {plugin_id} not found")
            return False

        if plugin.source == PluginSource.VOLUME:
            return self.volume_watcher.uninstall(plugin_id) if self.volume_watcher else False
        elif plugin.source == PluginSource.PACKAGE:
            return self.package_installer.uninstall(plugin_id)
        elif plugin.source == PluginSource.GIT:
            return self.git_installer.uninstall(plugin_id)

        return False

    def update(self, plugin_id: str) -> InstallResult:
        """Update a plugin.

        Args:
            plugin_id: ID of the plugin to update.

        Returns:
            InstallResult with update outcome.
        """
        plugin = self.registry.get(plugin_id)
        if not plugin:
            return InstallResult(success=False, error=f"Plugin {plugin_id} not found")

        if plugin.source == PluginSource.VOLUME:
            if self.volume_watcher:
                return self.volume_watcher.update(plugin_id)
            return InstallResult(success=False, error="Volume watcher not enabled")
        elif plugin.source == PluginSource.PACKAGE:
            return self.package_installer.update(plugin_id)
        elif plugin.source == PluginSource.GIT:
            return self.git_installer.update(plugin_id)

        return InstallResult(success=False, error="Unknown source type")

    def load_plugin(self, plugin_id: str) -> bool:
        """Manually load/activate a plugin.

        Args:
            plugin_id: ID of the plugin to load.

        Returns:
            True if successful.
        """
        plugin = self.registry.get(plugin_id)
        if not plugin:
            logger.warning(f"Plugin {plugin_id} not found")
            return False

        return self.lifecycle.activate_plugin(plugin)

    def unload_plugin(self, plugin_id: str) -> bool:
        """Manually unload/deactivate a plugin.

        Args:
            plugin_id: ID of the plugin to unload.

        Returns:
            True if successful.
        """
        return self.lifecycle.deactivate_plugin(plugin_id)

    def list_plugins(self) -> List[Dict]:
        """List all registered plugins.

        Returns:
            List of plugin info dictionaries.
        """
        return [
            {
                "id": p.id,
                "name": p.name,
                "source": p.source.value,
                "state": p.state.value,
                "version": p.version,
                "entry_points": p.entry_points,
                "error": p.error,
            }
            for p in self.registry.get_all()
        ]

    def get_plugin(self, plugin_id: str) -> Optional[Dict]:
        """Get details of a specific plugin.

        Args:
            plugin_id: Plugin ID.

        Returns:
            Plugin info dictionary or None.
        """
        plugin = self.registry.get(plugin_id)
        if plugin:
            return plugin.to_dict()
        return None

    def get_status(self) -> Dict:
        """Get overall loader status.

        Returns:
            Status information dictionary.
        """
        plugins = self.registry.get_all()
        return {
            "total_plugins": len(plugins),
            "loaded": sum(1 for p in plugins if p.state == PluginState.LOADED),
            "failed": sum(1 for p in plugins if p.state == PluginState.FAILED),
            "volume_watcher_active": self.volume_watcher is not None,
            "watch_directory": str(self.config.watch_directory),
            "registry_path": str(self.config.registry_path),
            "plugins": self.lifecycle.get_plugin_status(),
        }

    def shutdown(self) -> None:
        """Shutdown the loader and cleanup resources."""
        self.stop_watching()
        logger.info("DynamicPluginLoader shutdown complete")
