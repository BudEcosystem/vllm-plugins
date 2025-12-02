"""Entropy and varentropy calculation utilities.

These metrics measure model uncertainty and are used to adapt sampling behavior.
"""


import torch
import torch.nn.functional as functional

# ln(2) for converting between natural log and log base 2
LN_2 = 0.69314718056


def calculate_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Calculate Shannon entropy from logits.

    Args:
        logits: Raw logits tensor of shape (..., vocab_size)
        dim: Dimension along which to compute entropy

    Returns:
        Entropy tensor in bits (base-2)
    """
    log_probs = functional.log_softmax(logits, dim=dim)
    probs = torch.exp(log_probs)
    # Entropy: -sum(p * log(p)), converted to base-2
    entropy = -torch.sum(probs * log_probs, dim=dim) / LN_2
    return entropy


def calculate_varentropy(
    logits: torch.Tensor, entropy: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Calculate variance of entropy (varentropy) from logits.

    Varentropy measures the spread of information content across tokens.
    High varentropy indicates the model is uncertain about which tokens
    are uncertain.

    Args:
        logits: Raw logits tensor of shape (..., vocab_size)
        entropy: Pre-computed entropy tensor
        dim: Dimension along which to compute

    Returns:
        Varentropy tensor
    """
    log_probs = functional.log_softmax(logits, dim=dim)
    probs = torch.exp(log_probs)

    # Varentropy: E[(log(p) + H)^2] where H is entropy
    # This measures how much individual token entropies deviate from mean
    entropy_expanded = entropy.unsqueeze(dim)
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy_expanded) ** 2, dim=dim)
    return varentropy


def calculate_varentropy_logsoftmax(
    logits: torch.Tensor, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate both entropy and varentropy efficiently in one pass.

    This is the primary function for computing uncertainty metrics,
    optimized to avoid redundant softmax computations.

    Args:
        logits: Raw logits tensor of shape (..., vocab_size)
        dim: Dimension along which to compute

    Returns:
        Tuple of (entropy, varentropy) tensors in base-2
    """
    log_probs = functional.log_softmax(logits, dim=dim)
    probs = torch.exp(log_probs)

    # Entropy in base-2
    entropy = -torch.sum(probs * log_probs, dim=dim) / LN_2

    # Varentropy
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(dim)) ** 2, dim=dim)

    return entropy, varentropy
