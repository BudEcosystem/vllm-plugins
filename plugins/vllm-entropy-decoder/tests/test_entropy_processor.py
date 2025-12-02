"""Tests for the entropy logits processor."""

import os
from unittest.mock import MagicMock

import torch
from vllm_entropy_decoder.metrics import (
    calculate_entropy,
    calculate_varentropy,
    calculate_varentropy_logsoftmax,
)
from vllm_entropy_decoder.processor import EntropyLogitsProcessor


class TestEntropyMetrics:
    """Tests for entropy calculation functions."""

    def test_calculate_entropy_uniform(self):
        """Uniform distribution should have maximum entropy."""
        vocab_size = 1000
        logits = torch.zeros(1, vocab_size)  # Uniform distribution
        entropy = calculate_entropy(logits)

        # Max entropy for uniform dist is log2(vocab_size)
        expected_max = torch.log2(torch.tensor(vocab_size, dtype=torch.float))
        assert torch.isclose(entropy[0], expected_max, rtol=0.01)

    def test_calculate_entropy_peaked(self):
        """Peaked distribution should have low entropy."""
        vocab_size = 1000
        logits = torch.full((1, vocab_size), -100.0)
        logits[0, 0] = 100.0  # Very peaked at first token

        entropy = calculate_entropy(logits)
        assert entropy[0] < 0.1  # Should be near zero

    def test_calculate_varentropy_logsoftmax_shapes(self):
        """Test output shapes match input batch dimension."""
        batch_size = 4
        vocab_size = 1000
        logits = torch.randn(batch_size, vocab_size)

        entropy, varentropy = calculate_varentropy_logsoftmax(logits)

        assert entropy.shape == (batch_size,)
        assert varentropy.shape == (batch_size,)

    def test_entropy_varentropy_consistency(self):
        """Test that separate and combined calculations match."""
        logits = torch.randn(2, 500)

        # Combined calculation
        entropy_combined, varentropy_combined = calculate_varentropy_logsoftmax(logits)

        # Separate calculations
        entropy_separate = calculate_entropy(logits)
        varentropy_separate = calculate_varentropy(logits, entropy_separate)

        assert torch.allclose(entropy_combined, entropy_separate, rtol=1e-5)
        assert torch.allclose(varentropy_combined, varentropy_separate, rtol=1e-5)


def create_processor(**env_overrides):
    """Helper to create processor with optional env var overrides."""
    # Save original env vars
    original = {}
    for key, value in env_overrides.items():
        env_key = f"ENTROPY_DECODER_{key.upper()}"
        original[env_key] = os.environ.get(env_key)
        os.environ[env_key] = str(value)

    try:
        mock_config = MagicMock()
        device = torch.device("cpu")
        processor = EntropyLogitsProcessor(mock_config, device, is_pin_memory=False)
        return processor
    finally:
        # Restore original env vars
        for env_key, orig_value in original.items():
            if orig_value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = orig_value


class TestEntropyLogitsProcessor:
    """Tests for the EntropyLogitsProcessor class."""

    def test_initialization(self):
        """Test processor initializes with correct parameters."""
        processor = create_processor(
            temperature=0.7,
            threshold_low=0.2,
            threshold_high=2.5,
        )

        assert processor.temperature == 0.7
        assert processor.entropy_threshold_low == 0.2
        assert processor.entropy_threshold_high == 2.5

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
        """Test that low entropy/varentropy sharpens distribution."""
        processor = create_processor(threshold_low=0.1)

        # Create very peaked distribution (low entropy)
        vocab_size = 100
        logits = torch.full((1, vocab_size), -50.0)
        logits[0, 0] = 50.0

        result = processor.apply(logits)

        # Sharpened logits should have larger magnitude differences
        original_diff = logits[0, 0] - logits[0, 1]
        result_diff = result[0, 0] - result[0, 1]
        assert result_diff > original_diff

    def test_high_uncertainty_flattening(self):
        """Test that high entropy/varentropy flattens distribution."""
        processor = create_processor(
            temperature=1.0,
            varentropy_threshold=0.01,  # Low threshold to trigger
        )

        # Create moderate distribution
        logits = torch.randn(1, 100) * 2

        result = processor.apply(logits)

        # Result should be modified (not identical to input)
        assert not torch.allclose(result, logits)

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

        # Test with various inputs
        test_cases = [
            torch.randn(1, 1000),
            torch.zeros(1, 1000),
            torch.full((1, 1000), -100.0),
            torch.randn(1, 1000) * 100,
        ]

        for logits in test_cases:
            result = processor.apply(logits)
            assert not torch.isnan(result).any(), f"NaN in result for input shape {logits.shape}"

    def test_min_p_filtering(self):
        """Test that min_p filtering removes low probability tokens.

        min_p filtering only applies in the 'default' branch of _adapt_logits.
        To reach that branch, entropy must be:
        - >= entropy_threshold_low (not Case 1)
        - <= entropy_threshold_high (not Case 2)
        - And varentropy conditions must not match Cases 3 or 4
        """
        processor = create_processor(
            min_p=0.1,
            temperature=1.0,
            threshold_low=0.0,  # entropy >= 0 won't trigger Case 1
            threshold_high=100,  # entropy <= 100 won't trigger Case 2
            varentropy_threshold=100,  # varentropy <= 100 won't trigger Cases 3/4
        )

        # Create distribution where some tokens should be filtered
        # This has moderate entropy (not too peaked, not uniform)
        logits = torch.tensor([[10.0, 5.0, 0.0, -5.0, -10.0]])

        result = processor.apply(logits)

        # Low probability tokens should be -inf after min_p filtering
        assert result[0, -1] == float("-inf") or result[0, -1] < -1000

    def test_validate_params(self):
        """Test validate_params doesn't raise for any params."""
        mock_params = MagicMock()
        # Should not raise
        EntropyLogitsProcessor.validate_params(mock_params)
