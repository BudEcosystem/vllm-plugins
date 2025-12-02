"""vLLM Entropy Decoder - Adaptive entropy-based decoding strategy.

This plugin provides an entropy-aware logits processor that dynamically
adjusts sampling behavior based on model uncertainty metrics:
- Entropy: Measures overall uncertainty in the probability distribution
- Varentropy: Variance of entropy, indicating consistency of uncertainty

The processor adapts its behavior:
- Low entropy/varentropy: Allow greedy-like sampling (model is confident)
- High entropy, low varentropy: Increase temperature for exploration
- High varentropy: Adjust top-k to explore diverse options
"""

__version__ = "0.1.0"

from vllm_entropy_decoder.metrics import (
    calculate_entropy,
    calculate_varentropy,
    calculate_varentropy_logsoftmax,
)
from vllm_entropy_decoder.processor import EntropyLogitsProcessor

__all__ = [
    "EntropyLogitsProcessor",
    "calculate_entropy",
    "calculate_varentropy",
    "calculate_varentropy_logsoftmax",
]
