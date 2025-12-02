"""Plugin registration for vLLM.

This module contains the entry point function that vLLM calls to register the plugin.
The function must be re-entrant (safe to call multiple times) because vLLM may call
it in multiple processes (main process, worker processes, etc.).
"""

import logging

logger = logging.getLogger(__name__)

# Track registration state to ensure re-entrancy
_registered = False


def register() -> None:
    """Register the example plugin with vLLM.

    This function is called by vLLM's plugin system via the entry point defined
    in pyproject.toml. It must be:

    1. Re-entrant: Safe to call multiple times without side effects
    2. Fast: Called before vLLM initialization in every process
    3. Side-effect free: Should only register capabilities, not modify state

    The function is invoked by `load_general_plugins()` which runs in:
    - Main process
    - All worker processes
    - GPU/CPU workers
    - Auxiliary processes

    Example registration patterns:
    - Register custom models via ModelRegistry.register_model()
    - Register custom samplers
    - Apply patches to vLLM classes
    """
    global _registered

    # Ensure re-entrancy - only register once per process
    if _registered:
        logger.debug("Example plugin already registered, skipping")
        return

    logger.info("Registering example vLLM plugin")

    # Example: Register a custom model (uncomment and modify as needed)
    # from vllm import ModelRegistry
    # if "MyCustomModel" not in ModelRegistry.get_supported_archs():
    #     ModelRegistry.register_model(
    #         "MyCustomModel",
    #         "vllm_plugin_example.models.my_model:MyCustomModel"
    #     )

    _registered = True
    logger.info("Example vLLM plugin registered successfully")
