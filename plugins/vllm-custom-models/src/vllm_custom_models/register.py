"""Plugin registration for custom models.

This module registers custom model architectures with vLLM's ModelRegistry.
Models are registered at plugin load time, making them available for inference.
"""

import logging

logger = logging.getLogger(__name__)

# Track registration state for re-entrancy
_registered = False

# Model registry mapping: architecture name -> module path
# Format: "ArchitectureName": "vllm_custom_models.models.module_name:ClassName"
MODEL_REGISTRY: dict[str, str] = {
    # Example entries (uncomment and modify as needed):
    # "MyCustomLlama": "vllm_custom_models.models.custom_llama:MyCustomLlamaForCausalLM",
    # "MyCustomMistral": "vllm_custom_models.models.custom_mistral:MyCustomMistralForCausalLM",
}


def register() -> None:
    """Register custom models with vLLM's ModelRegistry.

    This function is called by vLLM's plugin system. It registers all custom
    model architectures defined in MODEL_REGISTRY, making them available for
    inference via vLLM.

    The registration is idempotent - calling multiple times has no effect
    after the first successful registration.

    Models are registered using their architecture name, which should match
    the `architectures` field in the model's config.json file.

    Example config.json:
        {
            "architectures": ["MyCustomLlama"],
            ...
        }
    """
    global _registered

    if _registered:
        logger.debug("Custom models already registered, skipping")
        return

    logger.info("Registering custom models with vLLM")

    try:
        from vllm import ModelRegistry
    except ImportError:
        logger.warning("vLLM not available, skipping model registration")
        return

    registered_count = 0
    for arch_name, model_path in MODEL_REGISTRY.items():
        if arch_name not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch_name, model_path)
            logger.info(f"Registered model architecture: {arch_name}")
            registered_count += 1
        else:
            logger.debug(f"Model architecture {arch_name} already registered")

    _registered = True
    logger.info(f"Custom models registration complete ({registered_count} models)")
