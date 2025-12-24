"""Main plugin loader orchestrator."""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from vllm_dynamic_loader.config import PluginLoaderConfig, get_config
from vllm_dynamic_loader.core.lifecycle import PluginLifecycleManager
from vllm_dynamic_loader.core.registry import PluginInfo, PluginRegistry, PluginSource, PluginState
from vllm_dynamic_loader.plugin_config import (
    PluginConfigFile,
    PluginSourceType,
    PluginSpec,
    load_plugin_config,
)
from vllm_dynamic_loader.plugin_manifest import PluginManifest, load_manifest
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

    def install_initial_plugins(self) -> Dict[str, bool]:
        """Install plugins from the configuration file.

        This method loads the plugin configuration file and installs
        any plugins that are not already in the registry.

        Returns:
            Dict mapping plugin identifiers to installation success status.
        """
        config = load_plugin_config()
        if config is None:
            return {}

        results: Dict[str, bool] = {}

        for spec in config.plugins:
            if not spec.enabled:
                logger.debug(f"Skipping disabled plugin: {spec}")
                continue

            plugin_id = self._get_plugin_id_from_spec(spec)

            # Load manifest for local sources (we have the path before install)
            manifest: Optional[PluginManifest] = None
            if spec.source == PluginSourceType.LOCAL and spec.path:
                from pathlib import Path

                local_path = Path(spec.path).expanduser().resolve()
                if local_path.exists():
                    manifest = load_manifest(local_path)
                    if manifest and manifest.name:
                        logger.debug(f"Loaded manifest for {plugin_id}: {manifest.name}")

            # Check vLLM version compatibility (using spec and/or manifest)
            if not self._check_vllm_compatibility(spec, manifest):
                constraint_str = self._get_version_constraint_source(spec, manifest)
                logger.info(
                    f"Skipping plugin {plugin_id}: "
                    f"vLLM version not compatible (requires {constraint_str})"
                )
                continue

            # Check if already installed
            existing = self.registry.get(plugin_id)
            if existing is not None:
                logger.debug(f"Plugin {plugin_id} already installed, skipping")
                results[plugin_id] = True
                continue

            logger.info(f"Installing plugin from config: {plugin_id}")

            try:
                result = self._install_from_spec(spec)
                results[plugin_id] = result.success
                if result.success:
                    logger.info(f"Successfully installed plugin: {plugin_id}")
                else:
                    logger.warning(f"Failed to install plugin {plugin_id}: {result.error}")
            except Exception as e:
                logger.error(f"Error installing plugin {plugin_id}: {e}")
                results[plugin_id] = False

        # Set default processor if specified
        if config.default_processor:
            logger.info(f"Setting default processor to: {config.default_processor}")
            # This will be picked up by the hot-swap system
            import os

            os.environ.setdefault("HOTSWAP_DEFAULT_PROCESSOR", config.default_processor)

        installed_count = sum(1 for success in results.values() if success)
        logger.info(
            f"Initial plugin installation complete: "
            f"{installed_count}/{len(results)} plugins installed"
        )

        return results

    def _get_plugin_id_from_spec(self, spec: PluginSpec) -> str:
        """Get a plugin identifier from a plugin specification.

        Args:
            spec: Plugin specification.

        Returns:
            A unique identifier string for the plugin.
        """
        if spec.source == PluginSourceType.PYPI:
            # Use package name without version constraint
            package_name = spec.package
            # Strip version specifiers if present in package name
            for op in [">=", "<=", "==", "!=", ">", "<", "~="]:
                if op in package_name:
                    package_name = package_name.split(op)[0]
            return package_name
        elif spec.source == PluginSourceType.GIT:
            # Use repo name from URL, plus subdirectory if specified
            url = spec.url.rstrip("/")
            if url.endswith(".git"):
                url = url[:-4]
            repo_name = url.split("/")[-1]
            if spec.subdirectory:
                # Use the last component of the subdirectory path
                subdir_name = spec.subdirectory.rstrip("/").split("/")[-1]
                return f"{repo_name}/{subdir_name}"
            return repo_name
        elif spec.source == PluginSourceType.LOCAL:
            # Use the folder name from the path
            from pathlib import Path

            path = Path(spec.path).expanduser().resolve()
            return f"local/{path.name}"
        return str(spec)

    def _check_vllm_compatibility(
        self,
        spec: PluginSpec,
        manifest: Optional[PluginManifest] = None,
    ) -> bool:
        """Check if the plugin is compatible with the current vLLM version.

        Version constraints are checked in order of precedence:
        1. User-specified constraints in plugins.yaml (spec.vllm_min/max)
        2. Plugin manifest constraints (manifest.vllm_min/max)
        3. If neither specified, plugin is considered compatible

        Args:
            spec: Plugin specification with optional vllm_min/vllm_max constraints.
            manifest: Optional plugin manifest with version constraints.

        Returns:
            True if compatible (or no constraints specified), False otherwise.
        """
        # Determine version constraints (user config takes precedence over manifest)
        vllm_min = spec.vllm_min
        vllm_max = spec.vllm_max

        # Fall back to manifest if user didn't specify
        if manifest is not None:
            if vllm_min is None:
                vllm_min = manifest.vllm_min
            if vllm_max is None:
                vllm_max = manifest.vllm_max

        # No version constraints - always compatible
        if vllm_min is None and vllm_max is None:
            return True

        try:
            from vllm_plugin_utils.version import check_vllm_version, get_vllm_version

            # Log current version for debugging
            current_version = get_vllm_version()
            logger.debug(f"Current vLLM version: {current_version}")

            return check_vllm_version(
                min_version=vllm_min or "0.0.0",
                max_version=vllm_max,
            )
        except ImportError:
            # vllm_plugin_utils not available, try direct check
            try:
                import vllm
                from packaging.version import Version

                current = Version(vllm.__version__)

                if vllm_min:
                    if current < Version(vllm_min):
                        return False

                if vllm_max:
                    if current >= Version(vllm_max):
                        return False

                return True
            except ImportError:
                # Can't check version, assume compatible
                logger.warning("Cannot determine vLLM version, skipping compatibility check")
                return True

    def _get_version_constraint_source(
        self,
        spec: PluginSpec,
        manifest: Optional[PluginManifest] = None,
    ) -> str:
        """Get a human-readable string describing the version constraints.

        Args:
            spec: Plugin specification.
            manifest: Optional plugin manifest.

        Returns:
            String describing version constraints for logging.
        """
        vllm_min = spec.vllm_min or (manifest.vllm_min if manifest else None)
        vllm_max = spec.vllm_max or (manifest.vllm_max if manifest else None)

        if vllm_min and vllm_max:
            return f"{vllm_min} <= vLLM < {vllm_max}"
        elif vllm_min:
            return f"vLLM >= {vllm_min}"
        elif vllm_max:
            return f"vLLM < {vllm_max}"
        return "any version"

    def _install_from_spec(self, spec: PluginSpec) -> InstallResult:
        """Install a plugin from a specification.

        Args:
            spec: Plugin specification.

        Returns:
            InstallResult with the installation outcome.
        """
        if spec.source == PluginSourceType.PYPI:
            return self.package_installer.install(spec.package_spec)
        elif spec.source == PluginSourceType.GIT:
            # Build the git URL with optional subdirectory
            git_url = spec.url
            if spec.subdirectory:
                # Append subdirectory in pip-compatible format
                git_url = f"{git_url}#subdirectory={spec.subdirectory}"

            kwargs = {}
            if spec.ref:
                kwargs["ref"] = spec.ref
            if spec.editable:
                kwargs["editable"] = spec.editable

            # Add version check callback if user didn't specify version constraints
            # This allows the plugin manifest to enforce its own constraints
            if spec.vllm_min is None and spec.vllm_max is None:

                def version_check(manifest: PluginManifest) -> bool:
                    return self._check_vllm_compatibility(spec, manifest)

                kwargs["version_check"] = version_check

            return self.git_installer.install(git_url, **kwargs)
        elif spec.source == PluginSourceType.LOCAL:
            # Install from local filesystem path
            from pathlib import Path

            local_path = Path(spec.path).expanduser().resolve()
            if not local_path.exists():
                return InstallResult(success=False, error=f"Path not found: {spec.path}")
            if not local_path.is_dir():
                return InstallResult(success=False, error=f"Path is not a directory: {spec.path}")

            # Use package installer for local path (it handles editable installs)
            return self.package_installer.install(
                str(local_path), editable=spec.editable
            )
        else:
            return InstallResult(success=False, error=f"Unknown source type: {spec.source}")

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
