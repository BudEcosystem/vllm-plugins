"""Base classes for vLLM patching framework.

This module provides the core infrastructure for applying surgical patches
to vLLM classes without modifying the upstream codebase.
"""

import logging
from typing import Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PatchManager:
    """Central registry for managing applied patches.

    The PatchManager tracks which patches have been applied to prevent
    duplicate application and provides introspection capabilities.
    """

    _applied_patches: dict[str, "VLLMPatch"] = {}
    _target_classes: dict[type, set[str]] = {}

    @classmethod
    def register(cls, patch: "VLLMPatch") -> None:
        """Register a patch as applied."""
        patch_name = patch.__class__.__name__
        cls._applied_patches[patch_name] = patch

        target = patch.get_target_class()
        if target:
            if target not in cls._target_classes:
                cls._target_classes[target] = set()
            cls._target_classes[target].add(patch_name)

        logger.debug(f"Registered patch: {patch_name}")

    @classmethod
    def is_applied(cls, patch_name: str) -> bool:
        """Check if a patch has been applied."""
        return patch_name in cls._applied_patches

    @classmethod
    def get_applied_patches(cls) -> dict[str, "VLLMPatch"]:
        """Get all applied patches."""
        return cls._applied_patches.copy()

    @classmethod
    def get_patches_for_class(cls, target_class: type) -> set[str]:
        """Get names of patches applied to a specific class."""
        return cls._target_classes.get(target_class, set()).copy()

    @classmethod
    def reset(cls) -> None:
        """Reset the patch registry. Mainly for testing."""
        cls._applied_patches.clear()
        cls._target_classes.clear()


class VLLMPatch(Generic[T]):
    """Base class for vLLM patches.

    VLLMPatch provides a structured way to modify vLLM classes without
    monkey-patching or maintaining a fork. Subclasses define methods
    that will be added to or replace methods on the target class.

    Type parameter T specifies the target class to patch.

    Example:
        from vllm.core.scheduler import Scheduler

        class PrioritySchedulerPatch(VLLMPatch[Scheduler]):
            '''Add priority-based scheduling.'''

            def get_priority(self, request) -> int:
                '''New method added to Scheduler.'''
                return request.metadata.get("priority", 0)

            def _schedule_with_priority(self, waiting_queue):
                '''Override existing method.'''
                sorted_queue = sorted(
                    waiting_queue,
                    key=lambda r: self.get_priority(r),
                    reverse=True
                )
                return self._original__schedule(sorted_queue)

        # Apply the patch
        PrioritySchedulerPatch.apply()

    Naming conventions:
        - New methods: regular method names (e.g., `get_priority`)
        - Override methods: prefix with `_` and suffix with target method
          (e.g., `_schedule_with_priority` to override `_schedule`)
        - Access original: `self._original_<method_name>`
    """

    _target_class: Optional[type[T]] = None
    _applied: bool = False

    @classmethod
    def get_target_class(cls) -> Optional[type[T]]:
        """Get the target class from the generic type parameter.

        This extracts the class from VLLMPatch[TargetClass].
        """
        if cls._target_class is not None:
            return cls._target_class

        # Extract from __orig_bases__ if available (Python 3.9+)
        for base in getattr(cls, "__orig_bases__", []):
            if hasattr(base, "__origin__") and base.__origin__ is VLLMPatch:
                args = getattr(base, "__args__", ())
                if args:
                    cls._target_class = args[0]
                    return cls._target_class

        return None

    @classmethod
    def apply(cls) -> bool:
        """Apply the patch to the target class.

        Returns:
            True if patch was applied, False if already applied or failed.
        """
        patch_name = cls.__name__

        if PatchManager.is_applied(patch_name):
            logger.debug(f"Patch {patch_name} already applied, skipping")
            return False

        target = cls.get_target_class()
        if target is None:
            logger.error(f"Cannot determine target class for {patch_name}")
            return False

        logger.info(f"Applying patch {patch_name} to {target.__name__}")

        # Create an instance to get the methods
        instance = cls()

        # Get all public methods defined in the patch class (not inherited from VLLMPatch)
        patch_methods = {
            name: getattr(instance, name)
            for name in dir(instance)
            if not name.startswith("__")
            and callable(getattr(instance, name))
            and name not in dir(VLLMPatch)
        }

        for method_name, method in patch_methods.items():
            # Skip class methods and static methods from base
            if method_name in ("apply", "get_target_class"):
                continue

            # Store original method if it exists
            if hasattr(target, method_name):
                original = getattr(target, method_name)
                setattr(target, f"_original_{method_name}", original)
                logger.debug(f"  Stored original {method_name}")

            # Apply the patch method
            setattr(target, method_name, method)
            logger.debug(f"  Patched {method_name}")

        # Register the patch
        PatchManager.register(instance)
        cls._applied = True

        logger.info(f"Patch {patch_name} applied successfully")
        return True
