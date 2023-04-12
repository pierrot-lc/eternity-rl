"""This module present different sampling methods for action selection.

Those methods are useful to control the exploration and exploitation tradeoff.
"""

import torch
from torch.distributions import Categorical


@torch.no_grad()
def nucleus_sampling(distributions: torch.Tensor, top_p: float) -> torch.Tensor:
    """Sample actions from the given logits using
    the nucleus sampling method.

    From the paper : https://arxiv.org/abs/1904.09751.
    This implementation is my own.

    ---
    Args:
        distributions: The distributions of the actions.
            Shape of [batch_size, n_actions].
        top_p: The probability mass to keep.

    ---
    Returns:
        The sampled actions.
            Shape [batch_size,].
    """
    ordered_distributions, action_ids = distributions.sort(dim=-1, descending=True)
    cumulated_probs = ordered_distributions.cumsum(dim=-1)
    mask = cumulated_probs <= top_p
    mask[..., 0] = True  # Ensure that at least one action is sampled.

    # Build the nucleus distributions.
    nucleus_distributions = torch.zeros_like(
        distributions, dtype=torch.float, device=distributions.device
    )
    nucleus_distributions[mask] = ordered_distributions[mask]
    nucleus_distributions = nucleus_distributions / nucleus_distributions.sum(
        dim=-1, keepdim=True
    )  # Normalize the distributions.

    # Sample from the distributions.
    categorical = Categorical(probs=nucleus_distributions)
    sampled_nucleus_actions = categorical.sample().unsqueeze(1)
    sampled_actions = torch.gather(action_ids, dim=-1, index=sampled_nucleus_actions)
    sampled_actions = sampled_actions.squeeze(1)

    return sampled_actions
