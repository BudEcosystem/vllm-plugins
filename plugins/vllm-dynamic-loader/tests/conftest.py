"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env(monkeypatch, temp_dir):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("VLLM_PLUGIN_WATCH_DIR", str(temp_dir / "plugins"))
    monkeypatch.setenv("VLLM_PLUGIN_REGISTRY", str(temp_dir / "registry.json"))
    monkeypatch.setenv("VLLM_PLUGIN_WATCH_ENABLED", "false")
    monkeypatch.setenv("VLLM_PLUGIN_AUTO_ACTIVATE", "false")

    # Reset config singleton
    from vllm_dynamic_loader.config import reset_config

    reset_config()

    yield

    reset_config()


@pytest.fixture
def registry(mock_env, temp_dir):
    """Create a test registry instance."""
    from vllm_dynamic_loader.core.registry import PluginRegistry

    # Reset singleton
    PluginRegistry.reset_instance()

    registry = PluginRegistry(str(temp_dir / "registry.json"))
    yield registry

    PluginRegistry.reset_instance()


@pytest.fixture
def watch_dir(temp_dir):
    """Create a watch directory for volume watcher tests."""
    watch_path = temp_dir / "plugins"
    watch_path.mkdir(parents=True, exist_ok=True)
    return watch_path
