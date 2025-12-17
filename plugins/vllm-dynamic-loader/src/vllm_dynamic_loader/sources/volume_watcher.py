"""File system watcher for mounted plugin volumes."""

import hashlib
import importlib
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Set

from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from vllm_dynamic_loader.core.registry import PluginInfo, PluginSource, PluginState
from vllm_dynamic_loader.sources.base import InstallResult, SourceHandler
from vllm_dynamic_loader.utils.hash_utils import compute_directory_hash, compute_file_hash
from vllm_dynamic_loader.utils.pip_wrapper import PipWrapper

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


@dataclass
class WatchedPlugin:
    """Tracks a plugin detected in the watched directory."""

    path: Path
    is_package: bool  # True for directory packages, False for single .py files
    content_hash: str
    plugin_id: Optional[str] = None
    package_name: Optional[str] = None  # pip package name for uninstall


class VolumePluginEventHandler(FileSystemEventHandler):
    """Handles file system events for plugin detection."""

    def __init__(self, watcher: "VolumeWatcher"):
        super().__init__()
        self.watcher = watcher
        # Debounce: track recent events to avoid duplicates
        self._recent_events: Dict[str, float] = {}
        self._debounce_seconds = 1.0

    def _should_process(self, path: str) -> bool:
        """Check if event should be processed (debouncing)."""
        now = time.time()
        if path in self._recent_events:
            if now - self._recent_events[path] < self._debounce_seconds:
                return False
        self._recent_events[path] = now

        # Cleanup old entries
        cutoff = now - self._debounce_seconds * 2
        self._recent_events = {k: v for k, v in self._recent_events.items() if v > cutoff}

        return True

    def on_created(self, event):
        """Handle file/directory creation."""
        if not self._should_process(event.src_path):
            return

        path = Path(event.src_path)

        # Only process items at the top level of the watch directory
        if path.parent != self.watcher.watch_directory:
            return

        # Skip __init__.py files
        if path.name == "__init__.py":
            return

        if isinstance(event, DirCreatedEvent):
            # Check if it's a Python package
            self.watcher._schedule_check(path)
        elif isinstance(event, FileCreatedEvent):
            if path.suffix == ".py":
                self.watcher._handle_new_plugin(path)
            elif path.suffix == ".whl":
                self.watcher._handle_wheel_file(path)

    def on_modified(self, event):
        """Handle file modification."""
        if not self._should_process(event.src_path):
            return

        if isinstance(event, FileModifiedEvent):
            path = Path(event.src_path)
            if path.suffix == ".py" or path.name in ("pyproject.toml", "setup.py"):
                self.watcher._handle_plugin_modified(path)

    def on_deleted(self, event):
        """Handle file/directory deletion."""
        if not self._should_process(event.src_path):
            return

        path = Path(event.src_path)
        if isinstance(event, (FileDeletedEvent, DirDeletedEvent)):
            self.watcher._handle_plugin_deleted(path)


class VolumeWatcher(SourceHandler):
    """Watches a directory for plugin additions, modifications, and deletions."""

    def __init__(
        self,
        watch_directory: str,
        on_install: Optional[Callable[[PluginInfo], None]] = None,
        on_uninstall: Optional[Callable[[str], None]] = None,
        on_update: Optional[Callable[[PluginInfo], None]] = None,
        auto_start: bool = False,
        poll_interval: float = 5.0,
    ):
        """Initialize the volume watcher.

        Args:
            watch_directory: Directory to watch for plugins.
            on_install: Callback when a plugin is installed.
            on_uninstall: Callback when a plugin is uninstalled.
            on_update: Callback when a plugin is updated.
            auto_start: Whether to start watching immediately.
            poll_interval: Interval for periodic scans (fallback).
        """
        super().__init__(on_install, on_uninstall, on_update)

        self.watch_directory = Path(watch_directory)
        self.poll_interval = poll_interval

        self._pip = PipWrapper()
        self._observer: Optional[Observer] = None
        self._watched_plugins: Dict[str, WatchedPlugin] = {}
        self._lock = threading.RLock()
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None

        # Ensure watch directory exists
        self.watch_directory.mkdir(parents=True, exist_ok=True)

        if auto_start:
            self.start()

    @property
    def source_type(self) -> PluginSource:
        return PluginSource.VOLUME

    def _cleanup_stale_plugins(self) -> None:
        """Clean up plugins whose source files were deleted while server was stopped."""
        from vllm_dynamic_loader.core.registry import PluginRegistry

        try:
            registry = PluginRegistry()
            volume_plugins = registry.list_by_type(PluginSource.VOLUME)

            for plugin in volume_plugins:
                source_path = Path(plugin.source_path)
                # Only clean up plugins from this watch directory
                if not str(source_path).startswith(str(self.watch_directory)):
                    continue

                if not source_path.exists():
                    logger.info(f"Source deleted while stopped: {source_path}, cleaning up {plugin.name}")

                    # Uninstall the pip package
                    if plugin.name:
                        result = self._pip.uninstall(plugin.name)
                        if result.success:
                            logger.info(f"Uninstalled stale package: {plugin.name}")
                        else:
                            logger.warning(f"Failed to uninstall {plugin.name}: {result.error}")

                    # Remove from registry
                    registry.remove(plugin.id)

        except Exception as e:
            logger.warning(f"Error during stale plugin cleanup: {e}")

    def start(self) -> None:
        """Start watching the directory."""
        if self._running:
            return

        self._running = True

        # Clean up plugins that were deleted while server was stopped
        self._cleanup_stale_plugins()

        # Initial scan
        self._scan_directory()

        # Try to start watchdog observer (may fail on systems with low inotify limits)
        try:
            self._observer = Observer()
            handler = VolumePluginEventHandler(self)
            self._observer.schedule(handler, str(self.watch_directory), recursive=True)
            self._observer.start()
            logger.info(f"Started inotify watcher for {self.watch_directory}")
        except OSError as e:
            # Handle inotify limit errors gracefully (errno 24=EMFILE, 28=ENOSPC)
            if e.errno in (24, 28):
                logger.warning(
                    f"Could not start inotify watcher (limit reached): {e}. "
                    f"Falling back to polling-only mode (interval: {self.poll_interval}s)"
                )
                self._observer = None
            else:
                raise

        # Start poll thread as fallback (always runs)
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        logger.info(f"Started watching {self.watch_directory}")

    def stop(self) -> None:
        """Stop watching the directory."""
        self._running = False

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        if self._poll_thread:
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None

        logger.info("Stopped volume watcher")

    def _poll_loop(self) -> None:
        """Periodic fallback scan loop."""
        while self._running:
            time.sleep(self.poll_interval)
            if self._running:
                self._scan_directory()

    def _scan_directory(self) -> None:
        """Scan the watch directory for plugins."""
        with self._lock:
            current_paths: Set[str] = set()

            try:
                for item in self.watch_directory.iterdir():
                    if item.name.startswith(".") or item.name.startswith("__"):
                        continue

                    path_key = str(item)
                    current_paths.add(path_key)

                    if item.is_file() and item.suffix == ".py":
                        self._check_single_file_plugin(item)
                    elif item.is_file() and item.suffix == ".whl":
                        self._check_wheel_file(item)
                    elif item.is_dir():
                        self._check_package_plugin(item)

                # Check for deletions
                removed = set(self._watched_plugins.keys()) - current_paths
                for path_key in removed:
                    watched = self._watched_plugins.pop(path_key, None)
                    if watched and watched.plugin_id:
                        self._handle_plugin_deleted(Path(path_key))

            except Exception as e:
                logger.error(f"Error scanning directory: {e}")

    def _check_single_file_plugin(self, path: Path) -> None:
        """Check if a .py file is a new or modified plugin."""
        # Skip __init__.py - these are package markers, not standalone plugins
        if path.name == "__init__.py":
            return

        # Skip files that are inside a package directory (not at top level)
        if path.parent != self.watch_directory:
            return

        path_key = str(path)
        try:
            content_hash = compute_file_hash(path)
        except Exception:
            return

        if path_key in self._watched_plugins:
            watched = self._watched_plugins[path_key]
            if watched.content_hash != content_hash:
                # Modified
                watched.content_hash = content_hash
                self._handle_plugin_modified(path)
        else:
            # New plugin
            self._watched_plugins[path_key] = WatchedPlugin(
                path=path, is_package=False, content_hash=content_hash
            )
            self._handle_new_plugin(path)

    def _check_wheel_file(self, path: Path) -> None:
        """Check if a wheel file is new."""
        path_key = str(path)
        if path_key not in self._watched_plugins:
            try:
                content_hash = compute_file_hash(path)
                self._watched_plugins[path_key] = WatchedPlugin(
                    path=path, is_package=False, content_hash=content_hash
                )
                self._handle_wheel_file(path)
            except Exception as e:
                logger.error(f"Error checking wheel file {path}: {e}")

    def _check_package_plugin(self, path: Path) -> None:
        """Check if a directory is a Python package plugin."""
        # Check for package indicators
        has_init = (path / "__init__.py").exists()
        has_pyproject = (path / "pyproject.toml").exists()
        has_setup = (path / "setup.py").exists()

        if not (has_init or has_pyproject or has_setup):
            return

        path_key = str(path)
        try:
            content_hash = compute_directory_hash(path)
        except Exception:
            return

        if path_key in self._watched_plugins:
            watched = self._watched_plugins[path_key]
            if watched.content_hash != content_hash:
                # Modified
                watched.content_hash = content_hash
                self._handle_plugin_modified(path)
        else:
            # New plugin
            self._watched_plugins[path_key] = WatchedPlugin(
                path=path, is_package=True, content_hash=content_hash
            )
            self._handle_new_plugin(path)

    def _schedule_check(self, path: Path) -> None:
        """Schedule a path for checking (with delay for copy completion)."""

        def delayed_check():
            time.sleep(2.0)  # Wait for copy to complete
            if path.is_dir():
                self._check_package_plugin(path)
            elif path.is_file() and path.suffix == ".py":
                self._check_single_file_plugin(path)

        threading.Thread(target=delayed_check, daemon=True).start()

    def _handle_new_plugin(self, path: Path) -> None:
        """Handle detection of a new plugin."""
        logger.info(f"New plugin detected: {path}")
        result = self.install(str(path))
        if result.success and result.plugin_info:
            path_key = str(path)
            if path_key in self._watched_plugins:
                self._watched_plugins[path_key].plugin_id = result.plugin_info.id
                self._watched_plugins[path_key].package_name = result.package_name

    def _handle_wheel_file(self, path: Path) -> None:
        """Handle detection of a new wheel file."""
        logger.info(f"New wheel file detected: {path}")
        result = self._install_wheel(path)
        if result.success and result.plugin_info:
            path_key = str(path)
            if path_key in self._watched_plugins:
                self._watched_plugins[path_key].plugin_id = result.plugin_info.id
                self._watched_plugins[path_key].package_name = result.package_name

    def _handle_plugin_modified(self, path: Path) -> None:
        """Handle modification of a plugin."""
        logger.info(f"Plugin modified: {path}")

        # Find the plugin root (may be a file within a package)
        plugin_path = self._find_plugin_root(path)
        if plugin_path:
            path_key = str(plugin_path)
            if path_key in self._watched_plugins:
                watched = self._watched_plugins[path_key]
                if watched.plugin_id:
                    result = self.update(watched.plugin_id)
                    if result.success:
                        logger.info(f"Plugin {watched.plugin_id} hot-reloaded")

    def _handle_plugin_deleted(self, path: Path) -> None:
        """Handle deletion of a plugin."""
        logger.info(f"Plugin deleted: {path}")

        path_key = str(path)
        watched = self._watched_plugins.pop(path_key, None)
        if watched and watched.plugin_id:
            self.uninstall(watched.plugin_id)

    def _find_plugin_root(self, path: Path) -> Optional[Path]:
        """Find the root of a plugin given a file path."""
        current = path.parent if path.is_file() else path

        while current != self.watch_directory and current != current.parent:
            if str(current) in self._watched_plugins:
                return current
            current = current.parent

        return path if str(path) in self._watched_plugins else None

    def _sanitize_package_name(self, name: str) -> str:
        """Create a valid PEP508 package name from a file stem."""
        import re

        # Replace invalid characters with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        # Collapse multiple underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")
        # Ensure it doesn't start with a digit
        if not sanitized or sanitized[0].isdigit():
            sanitized = "plugin_" + sanitized
        return f"vllm_dynplugin_{sanitized}"

    def _detect_entry_point_type(self, path: Path) -> tuple:
        """Detect the entry point type and check for register function.

        Returns:
            Tuple of (entry_point_group, has_register_function)
        """
        entry_point_group = "vllm.general_plugins"
        has_register = False

        try:
            content = path.read_text()

            # Check for register function
            has_register = "def register(" in content or "def register():" in content

            # Detect entry point type
            if "LogitsProcessor" in content:
                entry_point_group = "vllm.logits_processors"
            elif "StatLoggerBase" in content:
                entry_point_group = "vllm.stat_logger_plugins"
            elif "Platform" in content:
                entry_point_group = "vllm.platform_plugins"
        except Exception:
            pass

        return entry_point_group, has_register

    def _install_single_file(self, path: Path, plugin_id: str) -> InstallResult:
        """Install a single .py file as a plugin.

        Creates a minimal package structure in a temp directory.
        """
        # Detect entry point type and check for register function
        entry_point_group, has_register = self._detect_entry_point_type(path)

        # Skip files without a register function for general plugins
        if entry_point_group == "vllm.general_plugins" and not has_register:
            logger.debug(f"Skipping {path.name} - no register() function found")
            return InstallResult(
                success=False, error=f"File {path.name} has no register() function"
            )

        # Create temp package structure with sanitized name
        temp_dir = tempfile.mkdtemp(prefix=f"vllm_plugin_{plugin_id}_")
        package_name = self._sanitize_package_name(path.stem)
        package_dir = Path(temp_dir) / package_name
        package_dir.mkdir()

        logger.info(f"Creating package {package_name} from {path.name} in {temp_dir}")

        # Copy the file
        shutil.copy(path, package_dir / "__init__.py")

        # Create entry point name (sanitized)
        entry_point_name = path.stem.replace("-", "_").strip("_")

        # Create minimal pyproject.toml
        # Only add entry point if there's a register function (for general plugins)
        # or if it's a different plugin type (logits processors, etc.)
        if has_register or entry_point_group != "vllm.general_plugins":
            entry_point_section = f"""
[project.entry-points."{entry_point_group}"]
{entry_point_name} = "{package_name}:register"
"""
        else:
            entry_point_section = ""

        pyproject_content = f"""[project]
name = "{package_name}"
version = "0.1.0"
description = "Dynamically loaded vLLM plugin from {path.name}"
{entry_point_section}
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
"""
        (Path(temp_dir) / "pyproject.toml").write_text(pyproject_content)

        # Install the package (pip_wrapper will add to sys.path)
        result = self._pip.install_local(temp_dir, editable=True)
        result.install_path = temp_dir
        result.package_name = package_name

        if result.success:
            # Double-check sys.path includes temp_dir
            if temp_dir not in sys.path:
                sys.path.insert(0, temp_dir)
                logger.debug(f"Explicitly added {temp_dir} to sys.path")

            # Invalidate import caches to ensure module is findable
            importlib.invalidate_caches()
            logger.info(f"Successfully installed {package_name} from {path.name}")
        else:
            logger.error(f"Failed to install {package_name}: {result.error}")

        return InstallResult(
            success=result.success,
            package_name=result.package_name,
            install_path=result.install_path,
            version=result.version,
            error=result.error,
        )

    def _install_wheel(self, path: Path) -> InstallResult:
        """Install a wheel file."""
        plugin_id = f"wheel_{hashlib.md5(str(path).encode()).hexdigest()[:8]}"

        result = self._pip.install_wheel(str(path))

        if result.success:
            plugin_info = PluginInfo(
                id=plugin_id,
                name=result.package_name or path.stem,
                source=PluginSource.VOLUME,
                source_path=str(path),
                version=result.version,
                state=PluginState.INSTALLED,
            )

            self._notify_install(plugin_info)

            return InstallResult(
                success=True,
                plugin_info=plugin_info,
                package_name=result.package_name,
                version=result.version,
            )

        return InstallResult(success=False, error=result.error)

    def install(self, source: str, **kwargs) -> InstallResult:
        """Install a plugin from a path within the watched directory."""
        path = Path(source)

        if not path.exists():
            return InstallResult(success=False, error=f"Path does not exist: {source}")

        try:
            plugin_id = f"volume_{hashlib.md5(source.encode()).hexdigest()[:8]}"

            # Determine installation method
            if path.is_file() and path.suffix == ".py":
                # Single file plugin - create temp package structure
                pip_result = self._install_single_file(path, plugin_id)
            elif path.is_file() and path.suffix == ".whl":
                # Wheel file - install directly
                return self._install_wheel(path)
            else:
                # Package plugin - install with pip
                pip_result = InstallResult(success=False)
                result = self._pip.install_local(str(path), editable=True)
                pip_result.success = result.success
                pip_result.package_name = result.package_name
                pip_result.install_path = result.install_path
                pip_result.version = result.version
                pip_result.error = result.error

            if pip_result.success:
                content_hash = (
                    compute_file_hash(path) if path.is_file() else compute_directory_hash(path)
                )

                plugin_info = PluginInfo(
                    id=plugin_id,
                    name=pip_result.package_name or path.stem,
                    source=PluginSource.VOLUME,
                    source_path=source,
                    install_path=pip_result.install_path,
                    version=pip_result.version,
                    content_hash=content_hash,
                    state=PluginState.INSTALLED,
                )

                self._notify_install(plugin_info)

                return InstallResult(
                    success=True,
                    plugin_info=plugin_info,
                    package_name=pip_result.package_name,
                    install_path=pip_result.install_path,
                    version=pip_result.version,
                )
            else:
                return InstallResult(success=False, error=pip_result.error)

        except Exception as e:
            error_msg = f"Failed to install {source}: {e}"
            logger.error(error_msg)
            return InstallResult(success=False, error=error_msg)

    def uninstall(self, plugin_id: str, package_name: Optional[str] = None) -> bool:
        """Uninstall a plugin.

        Args:
            plugin_id: The plugin ID to uninstall.
            package_name: Optional package name to uninstall via pip.
                         If not provided, will try to find it from watched plugins.
        """
        # Find the plugin in watched list
        for path_key, watched in list(self._watched_plugins.items()):
            if watched.plugin_id == plugin_id:
                # Try to uninstall the pip package to remove entry points
                # Priority: explicit arg > stored name > derived name
                pkg_name = package_name or watched.package_name
                if not pkg_name:
                    # Fallback: derive package name from path
                    pkg_name = self._sanitize_package_name(watched.path.stem)

                if pkg_name:
                    result = self._pip.uninstall(pkg_name)
                    if result.success:
                        logger.info(f"Uninstalled package {pkg_name}")
                    else:
                        logger.warning(f"Failed to uninstall package {pkg_name}: {result.error}")

                self._notify_uninstall(plugin_id)
                del self._watched_plugins[path_key]
                return True
        return False

    def update(self, plugin_id: str) -> InstallResult:
        """Update a plugin (reinstall from source)."""
        for path_key, watched in self._watched_plugins.items():
            if watched.plugin_id == plugin_id:
                # Uninstall first
                self.uninstall(plugin_id)

                # Reinstall
                result = self.install(str(watched.path))
                if result.success and result.plugin_info:
                    watched.plugin_id = result.plugin_info.id
                    self._notify_update(result.plugin_info)
                return result

        return InstallResult(success=False, error=f"Plugin {plugin_id} not found")

    def get_watched_plugins(self) -> Dict[str, WatchedPlugin]:
        """Get all watched plugins."""
        with self._lock:
            return self._watched_plugins.copy()
