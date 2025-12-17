"""Plugin registry for tracking dynamically loaded plugins."""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import FileLock

logger = logging.getLogger(__name__)


class PluginState(Enum):
    """Plugin lifecycle states."""

    PENDING = "pending"
    INSTALLING = "installing"
    INSTALLED = "installed"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"


class PluginSource(Enum):
    """Plugin source types."""

    VOLUME = "volume"
    PACKAGE = "package"
    GIT = "git"
    BUILTIN = "builtin"


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""

    id: str  # Unique identifier
    name: str  # Package/plugin name
    source: PluginSource  # How it was installed
    source_path: str  # Original source location
    install_path: Optional[str] = None  # Where it was installed
    version: Optional[str] = None  # Package version
    entry_points: Dict[str, List[str]] = field(default_factory=dict)  # group -> [names]
    state: PluginState = PluginState.PENDING
    error: Optional[str] = None  # Error message if failed
    installed_at: Optional[float] = None
    loaded_at: Optional[float] = None
    content_hash: Optional[str] = None  # For change detection

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source.value,
            "source_path": self.source_path,
            "install_path": self.install_path,
            "version": self.version,
            "entry_points": self.entry_points,
            "state": self.state.value,
            "error": self.error,
            "installed_at": self.installed_at,
            "loaded_at": self.loaded_at,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginInfo":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            source=PluginSource(data["source"]),
            source_path=data["source_path"],
            install_path=data.get("install_path"),
            version=data.get("version"),
            entry_points=data.get("entry_points", {}),
            state=PluginState(data.get("state", "pending")),
            error=data.get("error"),
            installed_at=data.get("installed_at"),
            loaded_at=data.get("loaded_at"),
            content_hash=data.get("content_hash"),
        )


class PluginRegistry:
    """Thread-safe plugin registry with file-based persistence.

    This registry maintains the state of all dynamically loaded plugins
    and persists them to a JSON file for multi-process coordination.
    """

    _instance: Optional["PluginRegistry"] = None
    _lock = threading.Lock()
    _initialized: bool

    def __new__(cls, registry_path: Optional[str] = None) -> "PluginRegistry":
        """Singleton pattern for process-local instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, registry_path: Optional[str] = None):
        """Initialize the registry.

        Args:
            registry_path: Path to the registry JSON file.
                          If not specified, uses config default.
        """
        if self._initialized:
            return

        from vllm_dynamic_loader.config import get_config

        config = get_config()
        self.registry_path = Path(registry_path or config.registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.registry_path.with_suffix(".lock")

        self._local_cache: Dict[str, PluginInfo] = {}
        self._registry_lock = threading.RLock()
        self._last_read_mtime: float = 0

        # Load existing registry
        self._load_from_file()

        self._initialized = True
        logger.info(f"PluginRegistry initialized: {self.registry_path}")

    def _load_from_file(self) -> None:
        """Load registry from file into local cache."""
        with FileLock(self.lock_path):
            if self.registry_path.exists():
                try:
                    data = json.loads(self.registry_path.read_text())
                    self._local_cache = {
                        pid: PluginInfo.from_dict(pdata) for pid, pdata in data.items()
                    }
                    self._last_read_mtime = self.registry_path.stat().st_mtime
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to load registry: {e}")
                    self._local_cache = {}

    def _save_to_file(self) -> None:
        """Save local cache to file."""
        with FileLock(self.lock_path):
            data = {pid: pinfo.to_dict() for pid, pinfo in self._local_cache.items()}
            self.registry_path.write_text(json.dumps(data, indent=2))
            self._last_read_mtime = self.registry_path.stat().st_mtime

    def _sync_if_stale(self) -> None:
        """Reload from file if it has been modified externally."""
        try:
            if self.registry_path.exists():
                current_mtime = self.registry_path.stat().st_mtime
                if current_mtime > self._last_read_mtime:
                    self._load_from_file()
        except OSError:
            pass

    def register(self, plugin: PluginInfo) -> None:
        """Register a plugin in the registry.

        Args:
            plugin: Plugin information to register.
        """
        with self._registry_lock:
            self._sync_if_stale()
            self._local_cache[plugin.id] = plugin
            self._save_to_file()
            logger.info(f"Registered plugin: {plugin.id} ({plugin.name})")

    def update_state(self, plugin_id: str, state: PluginState, error: Optional[str] = None) -> None:
        """Update plugin state.

        Args:
            plugin_id: ID of the plugin to update.
            state: New state.
            error: Optional error message.
        """
        with self._registry_lock:
            self._sync_if_stale()
            if plugin_id in self._local_cache:
                plugin = self._local_cache[plugin_id]
                plugin.state = state
                plugin.error = error
                if state == PluginState.INSTALLED:
                    plugin.installed_at = time.time()
                elif state == PluginState.LOADED:
                    plugin.loaded_at = time.time()
                self._save_to_file()
                logger.debug(f"Plugin {plugin_id} state -> {state.value}")

    def get(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin by ID.

        Args:
            plugin_id: Plugin ID to look up.

        Returns:
            PluginInfo if found, None otherwise.
        """
        with self._registry_lock:
            self._sync_if_stale()
            return self._local_cache.get(plugin_id)

    def get_by_name(self, name: str) -> Optional[PluginInfo]:
        """Get plugin by name.

        Args:
            name: Plugin name to look up.

        Returns:
            PluginInfo if found, None otherwise.
        """
        with self._registry_lock:
            self._sync_if_stale()
            for plugin in self._local_cache.values():
                if plugin.name == name:
                    return plugin
            return None

    def get_by_source(self, source_path: str) -> List[PluginInfo]:
        """Get all plugins from a specific source.

        Args:
            source_path: Source path to match.

        Returns:
            List of matching plugins.
        """
        with self._registry_lock:
            self._sync_if_stale()
            return [p for p in self._local_cache.values() if source_path in p.source_path]

    def list_by_type(self, source: PluginSource) -> List[PluginInfo]:
        """List plugins by source type.

        Args:
            source: Plugin source type.

        Returns:
            List of plugins from that source.
        """
        with self._registry_lock:
            self._sync_if_stale()
            return [p for p in self._local_cache.values() if p.source == source]

    def list_by_state(self, state: PluginState) -> List[PluginInfo]:
        """List plugins by state.

        Args:
            state: Plugin state to filter by.

        Returns:
            List of plugins in that state.
        """
        with self._registry_lock:
            self._sync_if_stale()
            return [p for p in self._local_cache.values() if p.state == state]

    def list_active(self) -> List[PluginInfo]:
        """List all active (loaded) plugins.

        Returns:
            List of loaded plugins.
        """
        return self.list_by_state(PluginState.LOADED)

    def get_all(self) -> List[PluginInfo]:
        """Get all registered plugins.

        Returns:
            List of all plugins.
        """
        with self._registry_lock:
            self._sync_if_stale()
            return list(self._local_cache.values())

    def remove(self, plugin_id: str) -> Optional[PluginInfo]:
        """Remove a plugin from the registry.

        Args:
            plugin_id: ID of the plugin to remove.

        Returns:
            Removed plugin info if found, None otherwise.
        """
        with self._registry_lock:
            self._sync_if_stale()
            plugin = self._local_cache.pop(plugin_id, None)
            if plugin:
                self._save_to_file()
                logger.info(f"Removed plugin: {plugin_id}")
            return plugin

    def clear(self) -> None:
        """Clear all plugins from the registry."""
        with self._registry_lock:
            self._local_cache.clear()
            self._save_to_file()
            logger.info("Registry cleared")

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None
