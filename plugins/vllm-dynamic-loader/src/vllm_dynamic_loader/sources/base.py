"""Base class for plugin source handlers."""

import abc
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from vllm_dynamic_loader.core.registry import PluginInfo, PluginSource

if TYPE_CHECKING:
    from vllm_dynamic_loader.plugin_manifest import PluginManifest

logger = logging.getLogger(__name__)


@dataclass
class InstallResult:
    """Result of a plugin installation attempt."""

    success: bool
    plugin_info: Optional[PluginInfo] = None
    package_name: Optional[str] = None
    install_path: Optional[str] = None
    version: Optional[str] = None
    error: Optional[str] = None
    manifest: Optional[Any] = None  # PluginManifest, using Any to avoid circular import


class SourceHandler(abc.ABC):
    """Abstract base class for plugin source handlers.

    Each source handler manages a specific way of obtaining plugins:
    - VolumeWatcher: Monitors a directory for plugin files
    - PackageInstaller: Installs from PyPI or wheel URLs
    - GitInstaller: Clones and installs from git repositories
    """

    def __init__(
        self,
        on_install: Optional[Callable[[PluginInfo], None]] = None,
        on_uninstall: Optional[Callable[[str], None]] = None,
        on_update: Optional[Callable[[PluginInfo], None]] = None,
    ):
        """Initialize the source handler.

        Args:
            on_install: Callback when a plugin is installed.
            on_uninstall: Callback when a plugin is uninstalled.
            on_update: Callback when a plugin is updated.
        """
        self.on_install = on_install
        self.on_uninstall = on_uninstall
        self.on_update = on_update

    @property
    @abc.abstractmethod
    def source_type(self) -> PluginSource:
        """Return the source type for this handler."""
        pass

    @abc.abstractmethod
    def install(self, source: str, **kwargs) -> InstallResult:
        """Install a plugin from the source.

        Args:
            source: Source-specific location (path, URL, etc.).
            **kwargs: Additional source-specific options.

        Returns:
            InstallResult with installation outcome.
        """
        pass

    @abc.abstractmethod
    def uninstall(self, plugin_id: str) -> bool:
        """Uninstall a plugin.

        Args:
            plugin_id: ID of the plugin to uninstall.

        Returns:
            True if uninstallation succeeded.
        """
        pass

    @abc.abstractmethod
    def update(self, plugin_id: str) -> InstallResult:
        """Update a plugin to the latest version.

        Args:
            plugin_id: ID of the plugin to update.

        Returns:
            InstallResult with update outcome.
        """
        pass

    def _notify_install(self, plugin_info: PluginInfo) -> None:
        """Notify listeners of installation."""
        if self.on_install:
            try:
                self.on_install(plugin_info)
            except Exception as e:
                logger.error(f"Install callback failed: {e}")

    def _notify_uninstall(self, plugin_id: str) -> None:
        """Notify listeners of uninstallation."""
        if self.on_uninstall:
            try:
                self.on_uninstall(plugin_id)
            except Exception as e:
                logger.error(f"Uninstall callback failed: {e}")

    def _notify_update(self, plugin_info: PluginInfo) -> None:
        """Notify listeners of update."""
        if self.on_update:
            try:
                self.on_update(plugin_info)
            except Exception as e:
                logger.error(f"Update callback failed: {e}")
