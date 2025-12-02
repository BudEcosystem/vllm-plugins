"""Tests for the example plugin."""


def test_register_is_reentrant():
    """Test that the register function can be called multiple times safely."""
    from vllm_plugin_example.register import register

    # Should not raise on multiple calls
    register()
    register()
    register()


def test_plugin_version():
    """Test that plugin version is defined."""
    from vllm_plugin_example import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)
