"""vLLM CoT Decoder - Confidence-weighted Chain-of-Thought decoding.

This plugin provides a confidence-aware logits processor inspired by
Chain-of-Thought decoding strategies. It adjusts the token distribution
based on the model's confidence margin (difference between top-2 probabilities).

Key concepts:
- Confidence margin: p_top1 - p_top2 (how much more likely the best token is)
- High confidence: Sharpen distribution to reinforce the model's choice
- Low confidence: Flatten distribution to encourage exploration

This captures the essence of CoT scoring without requiring full path exploration.
"""

__version__ = "0.1.0"

from vllm_cot_decoder.confidence import (
    calculate_confidence_margin,
    calculate_top_k_confidence,
)
from vllm_cot_decoder.processor import CoTLogitsProcessor

__all__ = [
    "CoTLogitsProcessor",
    "calculate_confidence_margin",
    "calculate_top_k_confidence",
]
