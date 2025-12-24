"""Plugin configuration file parsing and validation.

This module handles loading and parsing YAML configuration files that specify
plugins to be installed at startup.

Configuration file locations (in priority order):
1. Path specified by VLLM_PLUGIN_CONFIG environment variable
2. ~/.config/vllm/plugins.yaml (XDG config directory)
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class PluginSourceType(str, Enum):
    """Source type for plugin installation."""

    PYPI = "pypi"
    GIT = "git"
    LOCAL = "local"


@dataclass
class PluginSpec:
    """Specification for a single plugin to install.

    Attributes:
        source: The source type (pypi, git, or local).
        package: Package name for PyPI sources.
        version: Version constraint for PyPI sources (e.g., ">=0.1.0").
        url: Git repository URL for git sources.
        ref: Git reference (branch, tag, commit) for git sources.
        subdirectory: Subdirectory within the git repo containing the plugin (git only).
        path: Local filesystem path for local sources.
        editable: Whether to install in editable mode (git/local).
        enabled: Whether this plugin should be installed (default: True).
        vllm_min: Minimum vLLM version required (inclusive), e.g., "0.6.0".
        vllm_max: Maximum vLLM version supported (exclusive), e.g., "0.8.0".
    """

    source: PluginSourceType
    package: Optional[str] = None
    version: Optional[str] = None
    url: Optional[str] = None
    ref: Optional[str] = None
    subdirectory: Optional[str] = None
    path: Optional[str] = None
    editable: bool = True  # Default to editable for local development
    enabled: bool = True
    vllm_min: Optional[str] = None  # Minimum vLLM version (inclusive)
    vllm_max: Optional[str] = None  # Maximum vLLM version (exclusive)

    def __post_init__(self):
        """Validate the plugin specification."""
        if self.source == PluginSourceType.PYPI:
            if not self.package:
                raise ValueError("PyPI plugins require 'package' field")
        elif self.source == PluginSourceType.GIT:
            if not self.url:
                raise ValueError("Git plugins require 'url' field")
        elif self.source == PluginSourceType.LOCAL:
            if not self.path:
                raise ValueError("Local plugins require 'path' field")

    @property
    def package_spec(self) -> str:
        """Get the full package specification for PyPI installs."""
        if self.source != PluginSourceType.PYPI:
            raise ValueError("package_spec only valid for PyPI sources")
        if self.version:
            return f"{self.package}{self.version}"
        return self.package

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginSpec":
        """Create a PluginSpec from a dictionary.

        Args:
            data: Dictionary with plugin configuration.

        Returns:
            PluginSpec instance.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        source_str = data.get("source")
        if not source_str:
            raise ValueError("Plugin spec missing required 'source' field")

        try:
            source = PluginSourceType(source_str.lower())
        except ValueError:
            raise ValueError(
                f"Invalid source type '{source_str}'. "
                f"Must be one of: {[s.value for s in PluginSourceType]}"
            )

        return cls(
            source=source,
            package=data.get("package"),
            version=data.get("version"),
            url=data.get("url"),
            ref=data.get("ref"),
            subdirectory=data.get("subdirectory"),
            path=data.get("path"),
            editable=data.get("editable", True),
            enabled=data.get("enabled", True),
            vllm_min=data.get("vllm_min"),
            vllm_max=data.get("vllm_max"),
        )


@dataclass
class PluginConfigFile:
    """Configuration file contents.

    Attributes:
        plugins: List of plugin specifications to install.
        default_processor: Optional default logits processor for hot-swap.
    """

    plugins: List[PluginSpec] = field(default_factory=list)
    default_processor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginConfigFile":
        """Create a PluginConfigFile from a dictionary.

        Args:
            data: Dictionary with configuration file contents.

        Returns:
            PluginConfigFile instance.
        """
        plugins = []
        for idx, plugin_data in enumerate(data.get("plugins", [])):
            try:
                spec = PluginSpec.from_dict(plugin_data)
                plugins.append(spec)
            except ValueError as e:
                logger.warning(f"Invalid plugin spec at index {idx}: {e}")
                continue

        return cls(
            plugins=plugins,
            default_processor=data.get("default_processor"),
        )


def get_config_file_path() -> Optional[Path]:
    """Get the path to the plugin configuration file.

    Returns:
        Path to the configuration file, or None if not found.

    Priority:
        1. VLLM_PLUGIN_CONFIG environment variable
        2. ~/.config/vllm/plugins.yaml
    """
    # Check environment variable first
    env_path = os.environ.get("VLLM_PLUGIN_CONFIG")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path
        logger.warning(f"VLLM_PLUGIN_CONFIG path does not exist: {env_path}")
        return None

    # Check XDG config directory
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    xdg_path = Path(xdg_config_home) / "vllm" / "plugins.yaml"
    if xdg_path.exists():
        return xdg_path

    # No config file found
    return None


def load_plugin_config() -> Optional[PluginConfigFile]:
    """Load the plugin configuration file.

    Returns:
        PluginConfigFile if found and valid, None otherwise.
    """
    config_path = get_config_file_path()
    if not config_path:
        logger.debug("No plugin configuration file found")
        return None

    logger.info(f"Loading plugin configuration from {config_path}")

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            logger.debug("Plugin configuration file is empty")
            return PluginConfigFile()

        if not isinstance(data, dict):
            logger.warning(f"Invalid plugin configuration: expected dict, got {type(data)}")
            return None

        config = PluginConfigFile.from_dict(data)
        logger.info(f"Loaded {len(config.plugins)} plugin(s) from configuration file")
        return config

    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse plugin configuration file: {e}")
        return None
    except OSError as e:
        logger.warning(f"Failed to read plugin configuration file: {e}")
        return None
