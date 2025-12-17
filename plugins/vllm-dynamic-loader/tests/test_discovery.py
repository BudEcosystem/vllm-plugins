"""Tests for entry point discovery."""


from vllm_dynamic_loader.core.discovery import VLLM_ENTRY_POINT_GROUPS, EntryPointDiscovery


class TestEntryPointDiscovery:
    """Tests for EntryPointDiscovery."""

    def test_initialization(self):
        """Test discovery initializes with baseline."""
        discovery = EntryPointDiscovery()

        # Should have captured baselines for all groups
        for group in VLLM_ENTRY_POINT_GROUPS:
            assert group in discovery._baseline_eps

    def test_invalidate_caches(self):
        """Test cache invalidation doesn't raise errors."""
        discovery = EntryPointDiscovery()

        # Should not raise
        discovery.invalidate_caches()

    def test_discover_all_entry_points(self):
        """Test discovering all entry points in a group."""
        discovery = EntryPointDiscovery()

        # This should return at least built-in vLLM processors
        eps = discovery.discover_all_entry_points("vllm.logits_processors")

        # Result should be a list (may be empty if no processors installed)
        assert isinstance(eps, list)

    def test_discover_new_entry_points(self):
        """Test discovering only new entry points."""
        discovery = EntryPointDiscovery()

        # Initially, no new entry points (everything in baseline)
        new_eps = discovery.discover_new_entry_points("vllm.general_plugins")

        # May have new ones or not, but should be a list
        assert isinstance(new_eps, list)

    def test_update_baseline(self):
        """Test updating the baseline."""
        discovery = EntryPointDiscovery()

        # Add to baseline
        discovery.update_baseline("vllm.logits_processors", {"test_processor"})

        # Check it's in baseline
        baseline = discovery.get_baseline("vllm.logits_processors")
        assert "test_processor" in baseline

    def test_get_baseline(self):
        """Test getting baseline for a group."""
        discovery = EntryPointDiscovery()

        # Should return a set (copy)
        baseline = discovery.get_baseline("vllm.general_plugins")
        assert isinstance(baseline, set)

        # Should be a copy, not the original
        baseline.add("test")
        assert "test" not in discovery._baseline_eps["vllm.general_plugins"]
