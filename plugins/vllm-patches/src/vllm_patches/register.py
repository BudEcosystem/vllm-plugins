"""Plugin registration and patch activation.

This module handles:
1. Registering the plugin with vLLM's plugin system
2. Activating patches based on VLLM_CUSTOM_PATCHES environment variable

Environment Variables:
    VLLM_CUSTOM_PATCHES: Comma-separated list of patch names to activate.
                         If not set, no patches are applied automatically.
                         Use "*" to apply all available patches.

Example:
    VLLM_CUSTOM_PATCHES=PrioritySchedulerPatch,CustomSamplerPatch python app.py
    VLLM_CUSTOM_PATCHES=* python app.py  # Apply all patches
"""

import logging
import os

from vllm_patches.base import VLLMPatch

logger = logging.getLogger(__name__)

# Track registration state
_registered = False

# Registry of available patches
# Map patch name to patch class
AVAILABLE_PATCHES: dict[str, type[VLLMPatch]] = {
    # Add your patches here:
    # "PrioritySchedulerPatch": PrioritySchedulerPatch,
    # "CustomSamplerPatch": CustomSamplerPatch,
}


def get_enabled_patches() -> list[str]:
    """Get list of patches to enable from environment variable."""
    env_value = os.environ.get("VLLM_CUSTOM_PATCHES", "")

    if not env_value:
        return []

    if env_value.strip() == "*":
        return list(AVAILABLE_PATCHES.keys())

    return [p.strip() for p in env_value.split(",") if p.strip()]


def register() -> None:
    """Register the patches plugin with vLLM.

    This is the entry point called by vLLM's plugin system.
    It reads VLLM_CUSTOM_PATCHES and applies the specified patches.
    """
    global _registered

    if _registered:
        logger.debug("Patches plugin already registered, skipping")
        return

    logger.info("Registering vLLM patches plugin")

    enabled = get_enabled_patches()

    if not enabled:
        logger.info("No patches enabled (set VLLM_CUSTOM_PATCHES to enable)")
        _registered = True
        return

    logger.info(f"Enabling patches: {enabled}")

    applied_count = 0
    for patch_name in enabled:
        if patch_name not in AVAILABLE_PATCHES:
            logger.warning(f"Unknown patch: {patch_name}")
            continue

        patch_class = AVAILABLE_PATCHES[patch_name]
        if patch_class.apply():
            applied_count += 1

    _registered = True
    logger.info(f"Patches plugin registered ({applied_count}/{len(enabled)} patches applied)")
