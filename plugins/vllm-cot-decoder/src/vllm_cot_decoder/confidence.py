"""Confidence calculation utilities for CoT decoding.

These functions compute confidence metrics used to assess how certain
the model is about its predictions.
"""


import torch
import torch.nn.functional as functional


def calculate_confidence_margin(logits: torch.Tensor) -> torch.Tensor:
    """Calculate confidence margin as difference between top-2 probabilities.

    The confidence margin (Δ) measures how much more probable the top token
    is compared to the second-best option. This is the core metric from
    CoT-decoding papers for scoring reasoning paths.

    Args:
        logits: Raw logits tensor of shape (..., vocab_size)

    Returns:
        Confidence margin tensor of shape (...), values in [0, 1]
    """
    probs = functional.softmax(logits, dim=-1)

    # Get top-2 probabilities
    top_2_probs, _ = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1)

    if top_2_probs.size(-1) < 2:
        # Only one token available, maximum confidence
        return torch.ones(logits.shape[:-1], device=logits.device)

    # Confidence = p_top1 - p_top2
    confidence = top_2_probs[..., 0] - top_2_probs[..., 1]

    return confidence


def calculate_top_k_confidence(
    logits: torch.Tensor, k: int = 5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate confidence metrics for top-k tokens.

    Provides detailed confidence information about the top-k most likely
    tokens, useful for understanding the model's probability distribution.

    Args:
        logits: Raw logits tensor of shape (..., vocab_size)
        k: Number of top tokens to analyze

    Returns:
        Tuple of:
        - top_k_probs: Probabilities of top-k tokens (..., k)
        - top_k_indices: Indices of top-k tokens (..., k)
        - margin: Confidence margin between top-1 and top-2 (...)
    """
    probs = functional.softmax(logits, dim=-1)

    k = min(k, probs.size(-1))
    top_k_probs, top_k_indices = torch.topk(probs, k=k, dim=-1)

    # Calculate margin
    if k >= 2:
        margin = top_k_probs[..., 0] - top_k_probs[..., 1]
    else:
        margin = torch.ones(logits.shape[:-1], device=logits.device)

    return top_k_probs, top_k_indices, margin


def aggregate_confidence(confidences: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Aggregate confidence scores across a sequence.

    Args:
        confidences: Confidence scores of shape (seq_len,) or (batch, seq_len)
        method: Aggregation method - "mean", "min", or "product"

    Returns:
        Aggregated confidence score
    """
    if method == "mean":
        return confidences.mean(dim=-1)
    elif method == "min":
        return confidences.min(dim=-1).values
    elif method == "product":
        # Geometric mean for numerical stability
        return torch.exp(torch.log(confidences + 1e-10).mean(dim=-1))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
