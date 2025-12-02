"""Version checking utilities for vLLM plugins.

These utilities help ensure plugins are compatible with the installed vLLM version.
"""

import functools
import warnings
from typing import Callable, Optional, TypeVar

from packaging.version import Version

F = TypeVar("F", bound=Callable)


def get_vllm_version() -> Version:
    """Get the currently installed vLLM version.

    Returns:
        Version object representing the installed vLLM version.

    Raises:
        ImportError: If vLLM is not installed.
    """
    try:
        import vllm

        return Version(vllm.__version__)
    except ImportError as err:
        raise ImportError(
            "vLLM is not installed. Please install vLLM first: pip install vllm"
        ) from err


def check_vllm_version(min_version: str, max_version: Optional[str] = None) -> bool:
    """Check if the installed vLLM version is within the specified range.

    Args:
        min_version: Minimum required vLLM version (inclusive).
        max_version: Maximum supported vLLM version (exclusive). Optional.

    Returns:
        True if the installed version is compatible, False otherwise.
    """
    current = get_vllm_version()
    min_ver = Version(min_version)

    if current < min_ver:
        return False

    if max_version is not None:
        max_ver = Version(max_version)
        if current >= max_ver:
            return False

    return True


def min_vllm_version(version: str) -> Callable[[F], F]:
    """Decorator to specify the minimum vLLM version required for a class or function.

    This decorator checks the vLLM version at class/function definition time and
    issues a warning if the installed version is too old.

    Args:
        version: Minimum required vLLM version string (e.g., "0.9.1").

    Returns:
        Decorated class or function.

    Example:
        @min_vllm_version("0.9.1")
        class MyPatch(VLLMPatch[Scheduler]):
            pass
    """

    def decorator(cls_or_func: F) -> F:
        @functools.wraps(cls_or_func)
        def wrapper(*args, **kwargs):
            if not check_vllm_version(version):
                current = get_vllm_version()
                warnings.warn(
                    f"{cls_or_func.__name__} requires vLLM >= {version}, "
                    f"but {current} is installed. This may cause compatibility issues.",
                    UserWarning,
                    stacklevel=2,
                )
            return cls_or_func(*args, **kwargs)

        # Store version requirement as attribute for introspection
        wrapper._min_vllm_version = version  # type: ignore
        return wrapper  # type: ignore

    return decorator
