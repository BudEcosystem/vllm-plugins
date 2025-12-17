"""Tests for the plugin registry."""


from vllm_dynamic_loader.core.registry import (
    PluginInfo,
    PluginSource,
    PluginState,
)


class TestPluginInfo:
    """Tests for PluginInfo dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        info = PluginInfo(
            id="test_123",
            name="test-plugin",
            source=PluginSource.PACKAGE,
            source_path="test-plugin==1.0.0",
            version="1.0.0",
            state=PluginState.INSTALLED,
        )

        data = info.to_dict()

        assert data["id"] == "test_123"
        assert data["name"] == "test-plugin"
        assert data["source"] == "package"
        assert data["state"] == "installed"
        assert data["version"] == "1.0.0"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "test_123",
            "name": "test-plugin",
            "source": "package",
            "source_path": "test-plugin==1.0.0",
            "version": "1.0.0",
            "state": "loaded",
        }

        info = PluginInfo.from_dict(data)

        assert info.id == "test_123"
        assert info.name == "test-plugin"
        assert info.source == PluginSource.PACKAGE
        assert info.state == PluginState.LOADED


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_register_and_get(self, registry):
        """Test registering and retrieving a plugin."""
        info = PluginInfo(
            id="test_001",
            name="test-plugin",
            source=PluginSource.PACKAGE,
            source_path="test-plugin",
        )

        registry.register(info)
        retrieved = registry.get("test_001")

        assert retrieved is not None
        assert retrieved.id == "test_001"
        assert retrieved.name == "test-plugin"

    def test_get_nonexistent(self, registry):
        """Test getting a nonexistent plugin returns None."""
        assert registry.get("nonexistent") is None

    def test_update_state(self, registry):
        """Test updating plugin state."""
        info = PluginInfo(
            id="test_002",
            name="test-plugin",
            source=PluginSource.PACKAGE,
            source_path="test-plugin",
            state=PluginState.PENDING,
        )

        registry.register(info)
        registry.update_state("test_002", PluginState.LOADED)

        retrieved = registry.get("test_002")
        assert retrieved.state == PluginState.LOADED
        assert retrieved.loaded_at is not None

    def test_list_by_state(self, registry):
        """Test listing plugins by state."""
        info1 = PluginInfo(
            id="test_003",
            name="plugin1",
            source=PluginSource.PACKAGE,
            source_path="plugin1",
            state=PluginState.LOADED,
        )
        info2 = PluginInfo(
            id="test_004",
            name="plugin2",
            source=PluginSource.PACKAGE,
            source_path="plugin2",
            state=PluginState.PENDING,
        )

        registry.register(info1)
        registry.register(info2)

        loaded = registry.list_by_state(PluginState.LOADED)
        assert len(loaded) == 1
        assert loaded[0].id == "test_003"

    def test_remove(self, registry):
        """Test removing a plugin."""
        info = PluginInfo(
            id="test_005",
            name="test-plugin",
            source=PluginSource.PACKAGE,
            source_path="test-plugin",
        )

        registry.register(info)
        removed = registry.remove("test_005")

        assert removed is not None
        assert removed.id == "test_005"
        assert registry.get("test_005") is None

    def test_persistence(self, temp_dir):
        """Test that registry persists to file."""
        from vllm_dynamic_loader.core.registry import PluginRegistry

        registry_path = str(temp_dir / "test_registry.json")

        # Create registry and add plugin
        PluginRegistry.reset_instance()
        registry1 = PluginRegistry(registry_path)
        info = PluginInfo(
            id="persist_001",
            name="persistent-plugin",
            source=PluginSource.GIT,
            source_path="https://github.com/test/repo",
        )
        registry1.register(info)

        # Reset and create new instance - should load from file
        PluginRegistry.reset_instance()
        registry2 = PluginRegistry(registry_path)

        retrieved = registry2.get("persist_001")
        assert retrieved is not None
        assert retrieved.name == "persistent-plugin"

        PluginRegistry.reset_instance()
