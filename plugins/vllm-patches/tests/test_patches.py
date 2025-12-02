"""Tests for the patching framework."""

from vllm_patches.base import PatchManager, VLLMPatch


class DummyTarget:
    """Dummy class for testing patches."""

    def existing_method(self) -> str:
        return "original"

    def another_method(self) -> int:
        return 42


class DummyPatch(VLLMPatch[DummyTarget]):
    """Test patch for DummyTarget."""

    def existing_method(self) -> str:
        return "patched"

    def new_method(self) -> str:
        return "new"


class TestVLLMPatch:
    def setup_method(self):
        """Reset state before each test."""
        PatchManager.reset()
        # Reset DummyTarget
        DummyTarget.existing_method = lambda self: "original"
        if hasattr(DummyTarget, "new_method"):
            delattr(DummyTarget, "new_method")
        if hasattr(DummyTarget, "_original_existing_method"):
            delattr(DummyTarget, "_original_existing_method")
        DummyPatch._applied = False

    def test_get_target_class(self):
        """Test that target class is correctly extracted from generic."""
        assert DummyPatch.get_target_class() is DummyTarget

    def test_apply_adds_new_method(self):
        """Test that apply adds new methods to target."""
        assert not hasattr(DummyTarget, "new_method")

        DummyPatch.apply()

        target = DummyTarget()
        assert hasattr(target, "new_method")

    def test_apply_overrides_existing_method(self):
        """Test that apply can override existing methods."""
        target = DummyTarget()
        assert target.existing_method() == "original"

        DummyPatch.apply()

        assert target.existing_method() == "patched"

    def test_apply_stores_original(self):
        """Test that original method is preserved."""
        DummyPatch.apply()

        assert hasattr(DummyTarget, "_original_existing_method")

    def test_apply_is_idempotent(self):
        """Test that applying twice doesn't cause issues."""
        result1 = DummyPatch.apply()
        result2 = DummyPatch.apply()

        assert result1 is True
        assert result2 is False  # Already applied

    def test_patch_manager_tracks_patches(self):
        """Test that PatchManager tracks applied patches."""
        assert not PatchManager.is_applied("DummyPatch")

        DummyPatch.apply()

        assert PatchManager.is_applied("DummyPatch")
        assert "DummyPatch" in PatchManager.get_applied_patches()


class TestPatchManager:
    def setup_method(self):
        PatchManager.reset()

    def test_reset_clears_registry(self):
        """Test that reset clears all tracked patches."""
        DummyPatch.apply()
        assert PatchManager.is_applied("DummyPatch")

        PatchManager.reset()

        assert not PatchManager.is_applied("DummyPatch")


def test_register_is_reentrant():
    """Test that register can be called multiple times."""
    from vllm_patches.register import register

    register()
    register()  # Should not raise
