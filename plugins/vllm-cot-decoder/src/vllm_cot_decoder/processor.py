"""Confidence-weighted logits processor for vLLM v1.

This processor implements a confidence-aware decoding strategy inspired by
Chain-of-Thought decoding, adjusting logits based on the model's confidence
in its predictions.
"""

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch
from vllm.v1.sample.logits_processor.interface import LogitsProcessor

from vllm_cot_decoder.confidence import calculate_confidence_margin

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor.interface import BatchUpdate

logger = logging.getLogger(__name__)


class CoTLogitsProcessor(LogitsProcessor):
    """Confidence-weighted logits processor for vLLM v1.

    This processor analyzes the confidence margin (difference between top-2
    token probabilities) and adjusts the logits distribution accordingly:

    - High confidence margin: Sharpen the distribution (boost the top token)
    - Low confidence margin: Flatten the distribution (encourage exploration)

    This captures the core insight from Chain-of-Thought decoding: when the
    model is confident, trust it; when uncertain, explore alternatives.

    Configuration via environment variables:
        COT_DECODER_CONFIDENCE_THRESHOLD: Margin above this = confident (default: 0.3)
        COT_DECODER_SHARPENING_FACTOR: Boost factor when confident (default: 1.5)
        COT_DECODER_EXPLORATION_FACTOR: Flatten factor when uncertain (default: 0.8)
        COT_DECODER_MIN_CONFIDENCE: Below this = max exploration (default: 0.05)
    """

    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool):
        """Initialize the CoT logits processor.

        Args:
            vllm_config: vLLM configuration object
            device: Target device for tensors
            is_pin_memory: Whether to use pinned memory
        """
        self.device = device
        self.is_pin_memory = is_pin_memory

        # Configuration from environment variables with defaults
        self.confidence_threshold = float(os.environ.get("COT_DECODER_CONFIDENCE_THRESHOLD", "0.3"))
        self.sharpening_factor = float(os.environ.get("COT_DECODER_SHARPENING_FACTOR", "1.5"))
        self.exploration_factor = float(os.environ.get("COT_DECODER_EXPLORATION_FACTOR", "0.8"))
        self.min_confidence = float(os.environ.get("COT_DECODER_MIN_CONFIDENCE", "0.05"))

        # Batch tracking
        self.batch_size = 0

        logger.info(
            f"CoTLogitsProcessor initialized: threshold={self.confidence_threshold}, "
            f"sharpen={self.sharpening_factor}, explore={self.exploration_factor}, device={device}"
        )

    def is_argmax_invariant(self) -> bool:
        """Confidence-based modification can change the argmax, so not invariant."""
        return False

    def update_state(self, batch_update: Optional["BatchUpdate"]) -> None:
        """Update internal state based on batch changes.

        Args:
            batch_update: Information about added, removed, and moved requests.
                         None if no changes to the batch.
        """
        if batch_update:
            self.batch_size = batch_update.batch_size

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply confidence-based modification to logits.

        Args:
            logits: Raw logits tensor of shape (batch_size, vocab_size)

        Returns:
            Modified logits tensor with same shape as input
        """
        if logits.size(0) == 0:
            return logits

        # Calculate confidence margin for each sample
        confidence = calculate_confidence_margin(logits)

        # Process each sample
        modified_logits = self._apply_confidence_adjustment(logits, confidence)

        return modified_logits

    def _apply_confidence_adjustment(
        self, logits: torch.Tensor, confidence: torch.Tensor
    ) -> torch.Tensor:
        """Apply confidence-based adjustment to logits.

        Args:
            logits: Logits tensor of shape (batch, vocab_size)
            confidence: Confidence margin tensor of shape (batch,)

        Returns:
            Adjusted logits tensor
        """
        batch_size = logits.size(0)
        modified_logits = logits.clone()

        for i in range(batch_size):
            conf = confidence[i].item()

            if conf > self.confidence_threshold:
                # High confidence: Sharpen the distribution
                # Divide by factor > 1 to increase differences
                temperature = 1.0 / self.sharpening_factor
                modified_logits[i] = logits[i] / temperature
                logger.debug(f"High confidence ({conf:.3f}): sharpening")

            elif conf < self.min_confidence:
                # Very low confidence: Maximum exploration
                # Multiply by factor to flatten (divide by small number)
                temperature = 1.0 / (self.exploration_factor * 0.5)
                modified_logits[i] = logits[i] / temperature
                logger.debug(f"Very low confidence ({conf:.3f}): max exploration")

            else:
                # Moderate confidence: Gradual adjustment
                # Interpolate between sharpening and exploration
                # based on where confidence falls relative to threshold
                ratio = conf / self.confidence_threshold

                if ratio > 1.0:
                    # Above threshold but not extremely confident
                    # Slight sharpening
                    blend = min((ratio - 1.0) / 2.0, 1.0)
                    temperature = 1.0 - blend * (1.0 - 1.0 / self.sharpening_factor)
                else:
                    # Below threshold: encourage exploration
                    blend = 1.0 - ratio
                    temperature = 1.0 + blend * (1.0 / self.exploration_factor - 1.0)

                modified_logits[i] = logits[i] / max(temperature, 0.1)
                logger.debug(f"Moderate confidence ({conf:.3f}): temp={temperature:.3f}")

        return modified_logits

    def _boost_top_tokens(
        self, logits: torch.Tensor, k: int = 5, boost_factor: float = 1.2
    ) -> torch.Tensor:
        """Boost the top-k tokens relative to others.

        Alternative adjustment strategy that directly boosts promising tokens.

        Args:
            logits: Single sample logits (vocab_size,)
            k: Number of top tokens to boost
            boost_factor: Multiplicative boost for top tokens

        Returns:
            Adjusted logits
        """
        top_k_vals, top_k_idx = torch.topk(logits, k=min(k, logits.size(-1)))

        boosted = logits.clone()
        boosted[top_k_idx] = boosted[top_k_idx] * boost_factor

        return boosted

    @classmethod
    def validate_params(cls, sampling_params: "SamplingParams") -> None:
        """Validate sampling parameters.

        Args:
            sampling_params: The sampling parameters to validate.

        Raises:
            ValueError: If parameters are invalid.
        """
        # No custom per-request parameters to validate
        # This processor applies globally with env var configuration
        pass
