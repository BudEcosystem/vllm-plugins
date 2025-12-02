"""Version compatibility utilities for patches."""

import warnings
from typing import Callable, TypeVar

from packaging.version import Version

F = TypeVar("F", bound=Callable)


def get_vllm_version() -> Version:
    """Get the currently installed vLLM version."""
    try:
        import vllm

        return Version(vllm.__version__)
    except ImportError as err:
        raise ImportError("vLLM is not installed") from err


def min_vllm_version(version: str) -> Callable[[type], type]:
    """Decorator to specify minimum vLLM version for a patch class.

    Use this to ensure patches only apply to compatible vLLM versions.
    If the installed version is too old, a warning is issued and the
    patch's apply() method becomes a no-op.

    Args:
        version: Minimum required vLLM version (e.g., "0.9.1")

    Example:
        @min_vllm_version("0.9.1")
        class MySchedulerPatch(VLLMPatch[Scheduler]):
            def custom_method(self):
                return "patched"
    """

    def decorator(cls: type) -> type:
        cls._min_vllm_version = version  # type: ignore[attr-defined]

        original_apply = getattr(cls, "apply", None)

        def guarded_apply(kls: type) -> bool:
            try:
                current = get_vllm_version()
                required = Version(version)
                if current < required:
                    warnings.warn(
                        f"{kls.__name__} requires vLLM >= {version}, "
                        f"but {current} is installed. Patch not applied.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return False
            except ImportError:
                warnings.warn(
                    f"Cannot verify vLLM version for {kls.__name__}. Patch not applied.",
                    UserWarning,
                    stacklevel=2,
                )
                return False

            if original_apply:
                result = original_apply.__func__(kls)  # type: ignore[union-attr]
                return bool(result)
            return True

        cls.apply = classmethod(guarded_apply)  # type: ignore[attr-defined,assignment]
        return cls

    return decorator
