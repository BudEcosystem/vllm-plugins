"""Plugin manifest file parsing.

Plugin developers can include a `vllm-plugin.yaml` manifest file in their
plugin root directory to declare metadata and version compatibility.

Example vllm-plugin.yaml:
```yaml
name: my-vllm-plugin
description: A custom logits processor for vLLM
version: 1.0.0
author: Your Name
license: Apache-2.0
tags:
  - logits-processor
  - decoding
vllm_min: "0.6.0"
vllm_max: "0.8.0"
```
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Manifest filename (checked in order)
MANIFEST_FILENAMES = ["vllm-plugin.yaml", "vllm-plugin.yml", "vllm_plugin.yaml"]


@dataclass
class PluginManifest:
    """Plugin manifest containing metadata and compatibility info.

    Attributes:
        name: Human-readable plugin name.
        description: Short description of what the plugin does.
        version: Plugin version string.
        author: Plugin author name or organization.
        license: License identifier (e.g., "Apache-2.0", "MIT").
        homepage: URL to plugin homepage or repository.
        tags: List of tags for categorization.
        vllm_min: Minimum vLLM version required (inclusive).
        vllm_max: Maximum vLLM version supported (exclusive).
        entry_points: Dict of entry point groups to names (optional override).
        dependencies: List of additional pip dependencies.
    """

    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    homepage: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    vllm_min: Optional[str] = None
    vllm_max: Optional[str] = None
    entry_points: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginManifest":
        """Create a PluginManifest from a dictionary.

        Args:
            data: Dictionary with manifest contents.

        Returns:
            PluginManifest instance.
        """
        return cls(
            name=data.get("name"),
            description=data.get("description"),
            version=data.get("version"),
            author=data.get("author"),
            license=data.get("license"),
            homepage=data.get("homepage"),
            tags=data.get("tags", []),
            vllm_min=data.get("vllm_min"),
            vllm_max=data.get("vllm_max"),
            entry_points=data.get("entry_points", {}),
            dependencies=data.get("dependencies", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary.

        Returns:
            Dictionary representation of the manifest.
        """
        result = {}
        if self.name:
            result["name"] = self.name
        if self.description:
            result["description"] = self.description
        if self.version:
            result["version"] = self.version
        if self.author:
            result["author"] = self.author
        if self.license:
            result["license"] = self.license
        if self.homepage:
            result["homepage"] = self.homepage
        if self.tags:
            result["tags"] = self.tags
        if self.vllm_min:
            result["vllm_min"] = self.vllm_min
        if self.vllm_max:
            result["vllm_max"] = self.vllm_max
        if self.entry_points:
            result["entry_points"] = self.entry_points
        if self.dependencies:
            result["dependencies"] = self.dependencies
        return result

    def has_version_constraints(self) -> bool:
        """Check if the manifest specifies vLLM version constraints.

        Returns:
            True if vllm_min or vllm_max is specified.
        """
        return self.vllm_min is not None or self.vllm_max is not None


def find_manifest_path(plugin_path: Path) -> Optional[Path]:
    """Find the manifest file in a plugin directory.

    Args:
        plugin_path: Path to the plugin directory.

    Returns:
        Path to the manifest file, or None if not found.
    """
    if not plugin_path.is_dir():
        return None

    for filename in MANIFEST_FILENAMES:
        manifest_path = plugin_path / filename
        if manifest_path.exists():
            return manifest_path

    return None


def load_manifest(plugin_path: Path) -> Optional[PluginManifest]:
    """Load the plugin manifest from a directory.

    Args:
        plugin_path: Path to the plugin directory.

    Returns:
        PluginManifest if found and valid, None otherwise.
    """
    manifest_path = find_manifest_path(plugin_path)
    if manifest_path is None:
        logger.debug(f"No manifest found in {plugin_path}")
        return None

    try:
        with open(manifest_path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            logger.debug(f"Empty manifest file: {manifest_path}")
            return PluginManifest()

        if not isinstance(data, dict):
            logger.warning(f"Invalid manifest format in {manifest_path}: expected dict")
            return None

        manifest = PluginManifest.from_dict(data)
        logger.debug(f"Loaded manifest from {manifest_path}: {manifest.name or 'unnamed'}")
        return manifest

    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse manifest {manifest_path}: {e}")
        return None
    except OSError as e:
        logger.warning(f"Failed to read manifest {manifest_path}: {e}")
        return None


def load_manifest_from_string(content: str) -> Optional[PluginManifest]:
    """Load a plugin manifest from a YAML string.

    Args:
        content: YAML content string.

    Returns:
        PluginManifest if valid, None otherwise.
    """
    try:
        data = yaml.safe_load(content)
        if data is None or not isinstance(data, dict):
            return None
        return PluginManifest.from_dict(data)
    except yaml.YAMLError:
        return None
