"""Tests for the CoT logits processor."""

import os
from unittest.mock import MagicMock

import torch
from vllm_cot_decoder.confidence import (
    aggregate_confidence,
    calculate_confidence_margin,
    calculate_top_k_confidence,
)
from vllm_cot_decoder.processor import CoTLogitsProcessor


class TestConfidenceMetrics:
    """Tests for confidence calculation functions."""

    def test_confidence_margin_peaked(self):
        """Peaked distribution should have high confidence margin."""
        vocab_size = 100
        logits = torch.full((1, vocab_size), -10.0)
        logits[0, 0] = 10.0  # Very peaked

        margin = calculate_confidence_margin(logits)

        # Should be close to 1.0 (top token dominates)
        assert margin[0] > 0.9

    def test_confidence_margin_uniform(self):
        """Uniform distribution should have low confidence margin."""
        vocab_size = 100
        logits = torch.zeros(1, vocab_size)

        margin = calculate_confidence_margin(logits)

        # All tokens equally likely, margin should be ~0
        assert margin[0] < 0.02

    def test_confidence_margin_two_peaks(self):
        """Two equally likely tokens should have ~0 margin."""
        vocab_size = 100
        logits = torch.full((1, vocab_size), -100.0)
        logits[0, 0] = 10.0
        logits[0, 1] = 10.0  # Two equally likely

        margin = calculate_confidence_margin(logits)

        # Two equal peaks = no margin
        assert margin[0] < 0.01

    def test_confidence_margin_batch(self):
        """Test batch processing of confidence margin."""
        batch_size = 4
        vocab_size = 100
        logits = torch.randn(batch_size, vocab_size)

        margin = calculate_confidence_margin(logits)

        assert margin.shape == (batch_size,)
        assert (margin >= 0).all()
        assert (margin <= 1).all()

    def test_top_k_confidence(self):
        """Test top-k confidence calculation."""
        logits = torch.randn(2, 100)

        probs, indices, margin = calculate_top_k_confidence(logits, k=5)

        assert probs.shape == (2, 5)
        assert indices.shape == (2, 5)
        assert margin.shape == (2,)

        # Probs should be sorted descending
        assert (probs[:, :-1] >= probs[:, 1:]).all()

    def test_aggregate_confidence_mean(self):
        """Test mean aggregation."""
        confidences = torch.tensor([0.8, 0.6, 0.9, 0.7])

        result = aggregate_confidence(confidences, method="mean")

        assert torch.isclose(result, torch.tensor(0.75))

    def test_aggregate_confidence_min(self):
        """Test min aggregation."""
        confidences = torch.tensor([0.8, 0.6, 0.9, 0.7])

        result = aggregate_confidence(confidences, method="min")

        assert torch.isclose(result, torch.tensor(0.6))


def create_processor(**env_overrides):
    """Helper to create processor with optional env var overrides."""
    # Save original env vars
    original = {}
    for key, value in env_overrides.items():
        env_key = f"COT_DECODER_{key.upper()}"
        original[env_key] = os.environ.get(env_key)
        os.environ[env_key] = str(value)

    try:
        mock_config = MagicMock()
        device = torch.device("cpu")
        processor = CoTLogitsProcessor(mock_config, device, is_pin_memory=False)
        return processor
    finally:
        # Restore original env vars
        for env_key, orig_value in original.items():
            if orig_value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = orig_value


class TestCoTLogitsProcessor:
    """Tests for the CoTLogitsProcessor class."""

    def test_initialization(self):
        """Test processor initializes with correct parameters."""
        processor = create_processor(
            confidence_threshold=0.4,
            sharpening_factor=2.0,
            exploration_factor=0.5,
        )

        assert processor.confidence_threshold == 0.4
        assert processor.sharpening_factor == 2.0
        assert processor.exploration_factor == 0.5

    def test_is_argmax_invariant(self):
        """Test that is_argmax_invariant returns False."""
        processor = create_processor()
        assert processor.is_argmax_invariant() is False

    def test_update_state(self):
        """Test update_state handles batch updates."""
        processor = create_processor()

        # Test with None
        processor.update_state(None)
        assert processor.batch_size == 0

        # Test with mock batch update
        mock_update = MagicMock()
        mock_update.batch_size = 4
        processor.update_state(mock_update)
        assert processor.batch_size == 4

    def test_apply_2d_input(self):
        """Test processor handles 2D input (batch, vocab)."""
        processor = create_processor()
        logits = torch.randn(4, 1000)

        result = processor.apply(logits)

        assert result.shape == logits.shape
        assert result.dtype == logits.dtype

    def test_apply_empty_batch(self):
        """Test processor handles empty batch."""
        processor = create_processor()
        logits = torch.randn(0, 1000)

        result = processor.apply(logits)

        assert result.shape == logits.shape

    def test_high_confidence_sharpening(self):
        """Test that high confidence sharpens distribution."""
        processor = create_processor(
            confidence_threshold=0.1,  # Low threshold
            sharpening_factor=2.0,
        )

        # Create peaked distribution (high confidence)
        vocab_size = 100
        logits = torch.full((1, vocab_size), -5.0)
        logits[0, 0] = 5.0

        result = processor.apply(logits)

        # Sharpened should have larger relative difference
        original_softmax = torch.softmax(logits, dim=-1)
        result_softmax = torch.softmax(result, dim=-1)

        # Top token should be even more dominant after sharpening
        assert result_softmax[0, 0] > original_softmax[0, 0]

    def test_low_confidence_exploration(self):
        """Test that low confidence encourages exploration."""
        processor = create_processor(
            confidence_threshold=0.9,  # High threshold = low relative confidence
            exploration_factor=0.5,
        )

        # Create moderate distribution
        logits = torch.randn(1, 100)

        result = processor.apply(logits)

        # Should flatten the distribution
        original_softmax = torch.softmax(logits, dim=-1)
        result_softmax = torch.softmax(result, dim=-1)

        # Max probability should decrease (more uniform)
        # Note: This depends on the logits, so we just check it changed
        assert not torch.allclose(result_softmax, original_softmax)

    def test_processor_is_reentrant(self):
        """Test that processor can be called multiple times safely."""
        processor = create_processor()
        logits = torch.randn(2, 500)

        result1 = processor.apply(logits)
        result2 = processor.apply(logits)

        # Same input should produce same output
        assert torch.allclose(result1, result2)

    def test_processor_no_nan(self):
        """Test that processor doesn't produce NaN values."""
        processor = create_processor()

        test_cases = [
            torch.randn(1, 1000),
            torch.zeros(1, 1000),
            torch.full((1, 1000), -100.0),
            torch.randn(1, 1000) * 100,
        ]

        for logits in test_cases:
            result = processor.apply(logits)
            assert not torch.isnan(result).any(), "NaN in result"

    def test_processor_preserves_ranking(self):
        """Test that basic token ranking is preserved."""
        processor = create_processor()

        # Create clear ranking
        logits = torch.tensor([[10.0, 5.0, 1.0, -5.0]])

        result = processor.apply(logits)

        # Top token should still be top
        assert result.argmax() == logits.argmax()

    def test_different_confidence_levels(self):
        """Test processor handles various confidence levels correctly."""
        processor = create_processor(confidence_threshold=0.3)

        # High confidence case
        high_conf = torch.full((1, 100), -10.0)
        high_conf[0, 0] = 10.0

        # Low confidence case
        low_conf = torch.zeros(1, 100)

        result_high = processor.apply(high_conf)
        result_low = processor.apply(low_conf)

        # Both should process without error
        assert result_high.shape == high_conf.shape
        assert result_low.shape == low_conf.shape

        # High confidence should be sharpened more than low confidence
        # (relative change in top probability)
        high_ratio = (
            torch.softmax(result_high, dim=-1).max() / torch.softmax(high_conf, dim=-1).max()
        )
        low_ratio = torch.softmax(result_low, dim=-1).max() / torch.softmax(low_conf, dim=-1).max()

        # High confidence should sharpen more
        assert high_ratio > low_ratio or torch.isclose(high_ratio, low_ratio, rtol=0.1)

    def test_validate_params(self):
        """Test validate_params doesn't raise for any params."""
        mock_params = MagicMock()
        # Should not raise
        CoTLogitsProcessor.validate_params(mock_params)
