"""Package installer for PyPI, wheel URLs, and local wheels."""

import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional
from urllib.parse import urlparse

import requests

from vllm_dynamic_loader.core.registry import PluginInfo, PluginSource, PluginState
from vllm_dynamic_loader.sources.base import InstallResult, SourceHandler
from vllm_dynamic_loader.utils.pip_wrapper import PipWrapper

logger = logging.getLogger(__name__)


class PackageInstaller(SourceHandler):
    """Installs plugins from PyPI, wheel URLs, or local wheel files."""

    # Pattern for PyPI package specifications
    PYPI_PATTERN = re.compile(
        r"^(?P<name>[a-zA-Z0-9][-a-zA-Z0-9._]*)"
        r"(?P<extras>\[[-a-zA-Z0-9._,]+\])?"
        r"(?P<constraint>==|>=|<=|>|<|!=|~=)?"
        r"(?P<version>[-a-zA-Z0-9._]+)?$"
    )

    def __init__(
        self,
        on_install: Optional[Callable[[PluginInfo], None]] = None,
        on_uninstall: Optional[Callable[[str], None]] = None,
        on_update: Optional[Callable[[PluginInfo], None]] = None,
        download_dir: Optional[str] = None,
    ):
        """Initialize the package installer.

        Args:
            on_install: Callback when a plugin is installed.
            on_uninstall: Callback when a plugin is uninstalled.
            on_update: Callback when a plugin is updated.
            download_dir: Directory for downloading wheels (temp if not specified).
        """
        super().__init__(on_install, on_uninstall, on_update)

        self._pip = PipWrapper()
        self._download_dir = Path(download_dir) if download_dir else None
        self._installed: Dict[str, PluginInfo] = {}

    @property
    def source_type(self) -> PluginSource:
        return PluginSource.PACKAGE

    def install(self, source: str, **kwargs) -> InstallResult:
        """Install a plugin from a package source.

        Args:
            source: One of:
                - PyPI package name: "vllm-my-plugin"
                - PyPI with version: "vllm-my-plugin>=0.1.0"
                - Wheel URL: "https://example.com/plugin.whl"
                - Local wheel path: "/path/to/plugin.whl"
            **kwargs:
                upgrade: bool - Upgrade if already installed
                force_reinstall: bool - Force reinstall
                extra_index_url: str - Additional PyPI index

        Returns:
            InstallResult with installation outcome.
        """
        upgrade = kwargs.get("upgrade", False)
        force_reinstall = kwargs.get("force_reinstall", False)
        extra_index_url = kwargs.get("extra_index_url")

        try:
            # Determine source type
            if self._is_url(source):
                return self._install_from_url(source, upgrade, force_reinstall)
            elif self._is_local_wheel(source):
                return self._install_from_local_wheel(source, upgrade, force_reinstall)
            else:
                return self._install_from_pypi(source, upgrade, force_reinstall, extra_index_url)

        except Exception as e:
            error_msg = f"Failed to install {source}: {e}"
            logger.error(error_msg)
            return InstallResult(success=False, error=error_msg)

    def _is_url(self, source: str) -> bool:
        """Check if source is a URL."""
        parsed = urlparse(source)
        return parsed.scheme in ("http", "https")

    def _is_local_wheel(self, source: str) -> bool:
        """Check if source is a local wheel file."""
        return source.endswith(".whl") and os.path.isfile(source)

    def _install_from_pypi(
        self,
        package_spec: str,
        upgrade: bool = False,
        force_reinstall: bool = False,
        extra_index_url: Optional[str] = None,
    ) -> InstallResult:
        """Install a package from PyPI."""
        logger.info(f"Installing from PyPI: {package_spec}")

        match = self.PYPI_PATTERN.match(package_spec)
        if not match:
            return InstallResult(success=False, error=f"Invalid package spec: {package_spec}")

        package_name = match.group("name")

        result = self._pip.install_package(
            package_spec,
            upgrade=upgrade,
            force_reinstall=force_reinstall,
            extra_index_url=extra_index_url,
        )

        if result.success:
            plugin_id = f"pypi_{hashlib.md5(package_spec.encode()).hexdigest()[:8]}"

            plugin_info = PluginInfo(
                id=plugin_id,
                name=package_name,
                source=PluginSource.PACKAGE,
                source_path=package_spec,
                version=result.version,
                state=PluginState.INSTALLED,
            )

            self._installed[plugin_id] = plugin_info
            self._notify_install(plugin_info)

            return InstallResult(
                success=True,
                plugin_info=plugin_info,
                package_name=package_name,
                version=result.version,
            )
        else:
            return InstallResult(success=False, error=result.error)

    def _install_from_url(
        self,
        url: str,
        upgrade: bool = False,
        force_reinstall: bool = False,
    ) -> InstallResult:
        """Install a wheel from a URL."""
        logger.info(f"Installing from URL: {url}")

        # Download the wheel
        try:
            download_path = self._download_wheel(url)
        except Exception as e:
            return InstallResult(success=False, error=f"Failed to download: {e}")

        # Install the downloaded wheel
        result = self._pip.install_wheel(
            str(download_path), upgrade=upgrade, force_reinstall=force_reinstall
        )

        if result.success:
            plugin_id = f"url_{hashlib.md5(url.encode()).hexdigest()[:8]}"

            plugin_info = PluginInfo(
                id=plugin_id,
                name=result.package_name or download_path.stem,
                source=PluginSource.PACKAGE,
                source_path=url,
                install_path=str(download_path),
                version=result.version,
                state=PluginState.INSTALLED,
            )

            self._installed[plugin_id] = plugin_info
            self._notify_install(plugin_info)

            return InstallResult(
                success=True,
                plugin_info=plugin_info,
                package_name=result.package_name,
                version=result.version,
            )
        else:
            return InstallResult(success=False, error=result.error)

    def _install_from_local_wheel(
        self,
        path: str,
        upgrade: bool = False,
        force_reinstall: bool = False,
    ) -> InstallResult:
        """Install a local wheel file."""
        logger.info(f"Installing from local wheel: {path}")

        result = self._pip.install_wheel(path, upgrade=upgrade, force_reinstall=force_reinstall)

        if result.success:
            plugin_id = f"wheel_{hashlib.md5(path.encode()).hexdigest()[:8]}"

            plugin_info = PluginInfo(
                id=plugin_id,
                name=result.package_name or Path(path).stem,
                source=PluginSource.PACKAGE,
                source_path=path,
                version=result.version,
                state=PluginState.INSTALLED,
            )

            self._installed[plugin_id] = plugin_info
            self._notify_install(plugin_info)

            return InstallResult(
                success=True,
                plugin_info=plugin_info,
                package_name=result.package_name,
                version=result.version,
            )
        else:
            return InstallResult(success=False, error=result.error)

    def _download_wheel(self, url: str) -> Path:
        """Download a wheel from URL."""
        # Determine download directory
        if self._download_dir:
            download_dir = self._download_dir
            download_dir.mkdir(parents=True, exist_ok=True)
        else:
            download_dir = Path(tempfile.mkdtemp(prefix="vllm_plugin_wheel_"))

        # Extract filename from URL
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename.endswith(".whl"):
            filename = f"plugin_{hashlib.md5(url.encode()).hexdigest()[:8]}.whl"

        download_path = download_dir / filename

        # Download with streaming
        logger.info(f"Downloading wheel to {download_path}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded {download_path.stat().st_size} bytes")
        return download_path

    def uninstall(self, plugin_id: str) -> bool:
        """Uninstall a plugin."""
        if plugin_id not in self._installed:
            logger.warning(f"Plugin {plugin_id} not found")
            return False

        plugin_info = self._installed[plugin_id]

        result = self._pip.uninstall(plugin_info.name)
        if result.success:
            del self._installed[plugin_id]
            self._notify_uninstall(plugin_id)
            return True
        else:
            logger.error(f"Failed to uninstall {plugin_info.name}: {result.error}")
            return False

    def update(self, plugin_id: str) -> InstallResult:
        """Update a plugin to the latest version."""
        if plugin_id not in self._installed:
            return InstallResult(success=False, error=f"Plugin {plugin_id} not found")

        plugin_info = self._installed[plugin_id]

        result = self.install(plugin_info.source_path, upgrade=True)
        if result.success and result.plugin_info:
            self._notify_update(result.plugin_info)
        return result

    def get_installed(self) -> Dict[str, PluginInfo]:
        """Get all installed plugins."""
        return self._installed.copy()
