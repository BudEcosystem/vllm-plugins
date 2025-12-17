"""Entry point discovery with cache invalidation support."""

import importlib
import importlib.metadata
import logging
import sys
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def _get_entry_points_for_group(group: str) -> List[importlib.metadata.EntryPoint]:
    """Get entry points for a group with Python 3.9/3.10+ compatibility.

    Args:
        group: The entry point group name.

    Returns:
        List of entry points in the group.
    """
    eps = importlib.metadata.entry_points()
    # Python 3.10+ returns SelectableGroups with select() method
    # Python 3.9 returns a dict-like object
    if hasattr(eps, "select"):
        return list(eps.select(group=group))
    else:
        # Python 3.9 compatibility - eps is dict-like
        return list(eps.get(group, []))  # type: ignore[union-attr]


# Known vLLM plugin entry point groups
VLLM_ENTRY_POINT_GROUPS = [
    "vllm.general_plugins",
    "vllm.logits_processors",
    "vllm.stat_logger_plugins",
    "vllm.platform_plugins",
]


class EntryPointDiscovery:
    """Discovers and loads entry points with cache invalidation.

    This class handles the critical challenge of discovering newly installed
    packages after pip install without restarting the Python process.
    """

    def __init__(self):
        """Initialize the discovery system."""
        # Cache of known entry points before dynamic loading
        self._baseline_eps: Dict[str, Set[str]] = {}
        self._capture_baseline()

    def _capture_baseline(self) -> None:
        """Capture baseline entry points at initialization."""
        for group in VLLM_ENTRY_POINT_GROUPS:
            try:
                eps = _get_entry_points_for_group(group)
                self._baseline_eps[group] = {ep.name for ep in eps}
                logger.debug(f"Baseline for {group}: {self._baseline_eps[group]}")
            except Exception as e:
                logger.warning(f"Failed to capture baseline for {group}: {e}")
                self._baseline_eps[group] = set()

    def invalidate_caches(self) -> None:
        """Invalidate all import-related caches.

        This is critical for discovering newly installed packages.
        importlib.metadata caches entry_points() results.
        """
        # Clear importlib caches
        importlib.invalidate_caches()

        # Force reimport of importlib.metadata to refresh its caches
        # This is a known workaround for entry_points caching
        modules_to_clear = [
            "importlib.metadata",
            "importlib._bootstrap",
            "importlib._bootstrap_external",
        ]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

        # Clear meta_path finder caches
        for finder in sys.meta_path:
            if hasattr(finder, "invalidate_caches"):
                try:
                    finder.invalidate_caches()
                except Exception:
                    pass

        logger.debug("Import caches invalidated")

    def discover_all_entry_points(self, group: str) -> List[importlib.metadata.EntryPoint]:
        """Discover all entry points in a group after cache invalidation.

        Args:
            group: The entry point group to check.

        Returns:
            List of all entry points in the group.
        """
        self.invalidate_caches()

        # Reimport to get fresh data
        import importlib.metadata as fresh_metadata

        try:
            eps = fresh_metadata.entry_points()
            # Python 3.10+ returns SelectableGroups with select() method
            # Python 3.9 returns a dict-like object
            if hasattr(eps, "select"):
                return list(eps.select(group=group))
            else:
                # Python 3.9 compatibility
                return list(eps.get(group, []))  # type: ignore[union-attr]
        except Exception as e:
            logger.error(f"Error discovering entry points for {group}: {e}")
            return []

    def discover_new_entry_points(self, group: str) -> List[importlib.metadata.EntryPoint]:
        """Discover entry points added since baseline or last update.

        Args:
            group: The entry point group to check.

        Returns:
            List of newly discovered entry points.
        """
        current_eps = self.discover_all_entry_points(group)
        current_names = {ep.name for ep in current_eps}

        baseline = self._baseline_eps.get(group, set())
        new_names = current_names - baseline

        if new_names:
            logger.info(f"Discovered {len(new_names)} new entry points in {group}: {new_names}")

        return [ep for ep in current_eps if ep.name in new_names]

    def load_entry_point(self, entry_point: importlib.metadata.EntryPoint) -> Optional[Any]:
        """Load an entry point and return the loaded object.

        Args:
            entry_point: The entry point to load.

        Returns:
            The loaded object (class, function, etc.) or None on failure.
        """
        try:
            loaded = entry_point.load()
            logger.info(f"Successfully loaded entry point: {entry_point.name}")
            return loaded
        except Exception as e:
            logger.error(f"Failed to load entry point {entry_point.name}: {e}")
            return None

    def discover_and_load_all(
        self, group: str, callback: Optional[Callable[[str, Any], None]] = None
    ) -> Dict[str, Any]:
        """Discover and load all new entry points in a group.

        Args:
            group: Entry point group.
            callback: Optional callback(name, loaded_obj) for each loaded EP.

        Returns:
            Dict mapping entry point names to loaded objects.
        """
        results = {}
        new_eps = self.discover_new_entry_points(group)

        for ep in new_eps:
            loaded = self.load_entry_point(ep)
            if loaded is not None:
                results[ep.name] = loaded
                if callback:
                    try:
                        callback(ep.name, loaded)
                    except Exception as e:
                        logger.error(f"Callback failed for {ep.name}: {e}")

        # Update baseline with newly discovered
        if group in self._baseline_eps:
            self._baseline_eps[group].update(results.keys())

        return results

    def update_baseline(self, group: str, names: Set[str]) -> None:
        """Update the baseline for a group.

        Args:
            group: Entry point group.
            names: Entry point names to add to baseline.
        """
        if group not in self._baseline_eps:
            self._baseline_eps[group] = set()
        self._baseline_eps[group].update(names)

    def get_baseline(self, group: str) -> Set[str]:
        """Get the current baseline for a group.

        Args:
            group: Entry point group.

        Returns:
            Set of entry point names in the baseline.
        """
        return self._baseline_eps.get(group, set()).copy()


def discover_package_entry_points(package_name: str) -> Dict[str, List[str]]:
    """Discover all entry points defined by a specific package.

    Args:
        package_name: The installed package name.

    Returns:
        Dict mapping group names to list of entry point names.
    """
    importlib.invalidate_caches()

    # Reimport for fresh data
    if "importlib.metadata" in sys.modules:
        del sys.modules["importlib.metadata"]
    import importlib.metadata as fresh_metadata

    results: Dict[str, List[str]] = {}

    try:
        dist = fresh_metadata.distribution(package_name)
        eps = dist.entry_points

        for ep in eps:
            if ep.group not in results:
                results[ep.group] = []
            results[ep.group].append(ep.name)

        logger.info(f"Discovered entry points for {package_name}: {results}")

    except fresh_metadata.PackageNotFoundError:
        logger.warning(f"Package {package_name} not found")
    except Exception as e:
        logger.error(f"Error discovering entry points for {package_name}: {e}")

    return results


def load_module_class(module_path: str, class_name: str) -> Optional[type]:
    """Dynamically load a class from a module.

    Args:
        module_path: Dotted module path (e.g., "vllm_entropy_decoder.processor").
        class_name: Name of the class to load.

    Returns:
        The loaded class or None on failure.
    """
    try:
        # Invalidate caches first
        importlib.invalidate_caches()

        # Import the module
        module = importlib.import_module(module_path)

        # Get the class
        cls: type = getattr(module, class_name)
        logger.info(f"Loaded {module_path}:{class_name}")
        return cls

    except ImportError as e:
        logger.error(f"Failed to import module {module_path}: {e}")
        return None
    except AttributeError as e:
        logger.error(f"Class {class_name} not found in {module_path}: {e}")
        return None
