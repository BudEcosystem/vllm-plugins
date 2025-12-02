"""Tests for model registration."""


def test_register_is_reentrant():
    """Test that register can be called multiple times safely."""
    from vllm_custom_models.register import register

    # Should not raise on multiple calls
    register()
    register()


def test_model_registry_format():
    """Test that MODEL_REGISTRY has correct format."""
    from vllm_custom_models.register import MODEL_REGISTRY

    for arch_name, model_path in MODEL_REGISTRY.items():
        assert isinstance(arch_name, str)
        assert isinstance(model_path, str)
        assert ":" in model_path, f"Model path should be 'module:class' format: {model_path}"
