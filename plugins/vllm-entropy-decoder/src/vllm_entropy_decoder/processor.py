"""Entropy-based logits processor for vLLM v1.

This processor implements adaptive sampling based on entropy and varentropy
metrics, dynamically adjusting the logits to influence sampling behavior.
"""

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as functional
from vllm.v1.sample.logits_processor.interface import LogitsProcessor

from vllm_entropy_decoder.metrics import calculate_varentropy_logsoftmax

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor.interface import BatchUpdate

logger = logging.getLogger(__name__)


class EntropyLogitsProcessor(LogitsProcessor):
    """Adaptive entropy-based logits processor for vLLM v1.

    This processor analyzes the entropy and varentropy of the logits
    distribution and modifies logits to adapt sampling behavior:

    - Low entropy + low varentropy: Model is confident, sharpen distribution
    - High entropy + low varentropy: Uncertain but consistent, increase temperature
    - Low entropy + high varentropy: Mixed confidence, moderate adjustment
    - High entropy + high varentropy: Very uncertain, flatten for exploration

    Configuration via environment variables:
        ENTROPY_DECODER_TEMPERATURE: Base temperature (default: 0.666)
        ENTROPY_DECODER_MIN_P: Minimum probability threshold (default: 0.03)
        ENTROPY_DECODER_THRESHOLD_LOW: Low entropy threshold (default: 0.1)
        ENTROPY_DECODER_THRESHOLD_HIGH: High entropy threshold (default: 3.0)
        ENTROPY_DECODER_VARENTROPY_THRESHOLD: Varentropy threshold (default: 5.0)
    """

    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool):
        """Initialize the entropy logits processor.

        Args:
            vllm_config: vLLM configuration object
            device: Target device for tensors
            is_pin_memory: Whether to use pinned memory
        """
        self.device = device
        self.is_pin_memory = is_pin_memory

        # Configuration from environment variables with defaults
        self.temperature = float(os.environ.get("ENTROPY_DECODER_TEMPERATURE", "0.666"))
        self.min_p = float(os.environ.get("ENTROPY_DECODER_MIN_P", "0.03"))
        self.entropy_threshold_low = float(os.environ.get("ENTROPY_DECODER_THRESHOLD_LOW", "0.1"))
        self.entropy_threshold_high = float(os.environ.get("ENTROPY_DECODER_THRESHOLD_HIGH", "3.0"))
        self.varentropy_threshold = float(
            os.environ.get("ENTROPY_DECODER_VARENTROPY_THRESHOLD", "5.0")
        )

        # Batch tracking
        self.batch_size = 0

        logger.info(
            f"EntropyLogitsProcessor initialized: temp={self.temperature}, "
            f"entropy_low={self.entropy_threshold_low}, entropy_high={self.entropy_threshold_high}, "
            f"varentropy_thresh={self.varentropy_threshold}, device={device}"
        )

    def is_argmax_invariant(self) -> bool:
        """Entropy-based modification can change the argmax, so not invariant."""
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
        """Apply entropy-based modification to logits.

        Args:
            logits: Raw logits tensor of shape (batch_size, vocab_size)

        Returns:
            Modified logits tensor with same shape as input
        """
        if logits.size(0) == 0:
            return logits

        # Calculate entropy metrics
        entropy, varentropy = calculate_varentropy_logsoftmax(logits)

        # Process each sample in the batch
        modified_logits = logits.clone()

        for i in range(logits.size(0)):
            ent = entropy[i].item()
            var_ent = varentropy[i].item()

            modified_logits[i] = self._adapt_logits(logits[i], ent, var_ent)

        return modified_logits

    def _adapt_logits(
        self, logits: torch.Tensor, entropy: float, varentropy: float
    ) -> torch.Tensor:
        """Adapt logits based on entropy and varentropy values.

        Args:
            logits: Single sample logits (vocab_size,)
            entropy: Entropy value for this sample
            varentropy: Varentropy value for this sample

        Returns:
            Modified logits tensor
        """
        # Case 1: Low entropy, low varentropy - model is confident
        # Sharpen the distribution to reinforce confidence
        if entropy < self.entropy_threshold_low and varentropy < self.entropy_threshold_low:
            logger.debug(
                f"High confidence mode: entropy={entropy:.3f}, varentropy={varentropy:.3f}"
            )
            # Sharpen by dividing by a factor < 1 (equivalent to lower temperature)
            return logits / 0.5

        # Case 2: High entropy, low varentropy - consistently uncertain
        # Increase temperature to allow more exploration
        elif entropy > self.entropy_threshold_high and varentropy < self.entropy_threshold_low:
            logger.debug(
                f"Consistent uncertainty mode: entropy={entropy:.3f}, varentropy={varentropy:.3f}"
            )
            temp_adjustment = 1.3 + 0.1 * min(entropy - self.entropy_threshold_high, 2.0)
            return logits / (self.temperature * temp_adjustment)

        # Case 3: Low entropy, high varentropy - mixed signals
        # Moderate exploration by slightly increasing temperature
        elif entropy < self.varentropy_threshold and varentropy > self.varentropy_threshold:
            logger.debug(
                f"Mixed confidence mode: entropy={entropy:.3f}, varentropy={varentropy:.3f}"
            )
            temp_adjustment = 1.2 + 0.15 * min(varentropy - self.varentropy_threshold, 3.0)
            return logits / (self.temperature * temp_adjustment)

        # Case 4: High entropy, high varentropy - very uncertain
        # Flatten distribution for maximum exploration
        elif entropy > self.varentropy_threshold and varentropy > self.varentropy_threshold:
            logger.debug(
                f"High uncertainty mode: entropy={entropy:.3f}, varentropy={varentropy:.3f}"
            )
            temp_adjustment = 2.0 + 0.2 * min(varentropy - self.varentropy_threshold, 5.0)
            return logits / (self.temperature * temp_adjustment)

        # Default: Apply standard temperature scaling with min_p filtering
        else:
            logger.debug(f"Standard mode: entropy={entropy:.3f}, varentropy={varentropy:.3f}")
            scaled_logits = logits / self.temperature

            # Apply min_p filtering
            if self.min_p > 0:
                probs = functional.softmax(scaled_logits, dim=-1)
                max_prob = probs.max()
                mask = probs < (self.min_p * max_prob)
                scaled_logits = torch.where(
                    mask, torch.full_like(scaled_logits, float("-inf")), scaled_logits
                )

            return scaled_logits

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
