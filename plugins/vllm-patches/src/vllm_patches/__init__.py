"""vLLM Patches - Surgical modifications to vLLM without forking.

This plugin provides a framework for applying targeted patches to vLLM classes
at runtime, enabling clean modifications without maintaining a fork.

Key features:
- Type-safe patching with VLLMPatch base class
- Version compatibility checking with @min_vllm_version decorator
- Runtime activation via VLLM_CUSTOM_PATCHES environment variable
- Patch registry for tracking applied modifications
"""

__version__ = "0.1.0"

from vllm_patches.base import PatchManager, VLLMPatch
from vllm_patches.version import min_vllm_version

__all__ = [
    "VLLMPatch",
    "PatchManager",
    "min_vllm_version",
]
