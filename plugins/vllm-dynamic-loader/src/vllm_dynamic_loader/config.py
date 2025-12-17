"""Configuration via environment variables."""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PluginLoaderConfig:
    """Configuration for the dynamic plugin loader."""

    # Volume watcher settings
    watch_directory: str = field(
        default_factory=lambda: os.environ.get("VLLM_PLUGIN_WATCH_DIR", "/plugins")
    )
    watch_enabled: bool = field(
        default_factory=lambda: os.environ.get("VLLM_PLUGIN_WATCH_ENABLED", "true").lower()
        == "true"
    )
    watch_poll_interval: float = field(
        default_factory=lambda: float(os.environ.get("VLLM_PLUGIN_WATCH_INTERVAL", "5.0"))
    )

    # Registry settings
    registry_path: str = field(
        default_factory=lambda: os.environ.get(
            "VLLM_PLUGIN_REGISTRY",
            os.path.join(
                os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
                "vllm-plugins",
                "registry.json",
            ),
        )
    )

    # Git installer settings
    git_clone_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("VLLM_PLUGIN_GIT_DIR")
    )
    git_editable: bool = field(
        default_factory=lambda: os.environ.get("VLLM_PLUGIN_GIT_EDITABLE", "true").lower() == "true"
    )

    # Package installer settings
    package_download_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("VLLM_PLUGIN_DOWNLOAD_DIR")
    )
    extra_index_url: Optional[str] = field(
        default_factory=lambda: os.environ.get("VLLM_PLUGIN_EXTRA_INDEX")
    )

    # Lifecycle settings
    auto_activate: bool = field(
        default_factory=lambda: os.environ.get("VLLM_PLUGIN_AUTO_ACTIVATE", "true").lower()
        == "true"
    )

    # Hot-swap settings
    hotswap_default_processor: str = field(
        default_factory=lambda: os.environ.get("HOTSWAP_DEFAULT_PROCESSOR", "passthrough")
    )
    hotswap_graceful_timeout_ms: int = field(
        default_factory=lambda: int(os.environ.get("HOTSWAP_GRACEFUL_TIMEOUT_MS", "5000"))
    )
    hotswap_poll_interval_ms: int = field(
        default_factory=lambda: int(os.environ.get("HOTSWAP_POLL_INTERVAL_MS", "100"))
    )

    # Security settings
    trusted_sources: List[str] = field(
        default_factory=lambda: [
            s.strip()
            for s in os.environ.get("VLLM_PLUGIN_TRUSTED_SOURCES", "").split(",")
            if s.strip()
        ]
    )


# Global config instance
_config: Optional[PluginLoaderConfig] = None


def get_config() -> PluginLoaderConfig:
    """Get the plugin loader configuration (singleton)."""
    global _config
    if _config is None:
        _config = PluginLoaderConfig()
    return _config


def reset_config() -> None:
    """Reset the configuration (for testing)."""
    global _config
    _config = None
