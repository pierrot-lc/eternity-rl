"""This module present different sampling methods for action selection.

Those methods are useful to control the exploration and exploitation tradeoff.
"""

import torch
from torch.distributions import Categorical, Dirichlet


def nucleus_distributions(distributions: torch.Tensor, top_p: float) -> torch.Tensor:
    """Compute the nucleus distributions from the given distributions."""
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

    # Reorder the distributions to match the original order.
    _, inverse_action_ids = action_ids.sort(dim=-1)
    nucleus_distributions = torch.gather(
        nucleus_distributions, dim=-1, index=inverse_action_ids
    )

    return nucleus_distributions


@torch.no_grad()
def nucleus_sampling(distributions: torch.Tensor, top_p: float) -> torch.Tensor:
    """Sample actions from the given distributions using
    the nucleus sampling method.

    From the paper: https://arxiv.org/abs/1904.09751.
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
    distributions = nucleus_distributions(distributions, top_p)
    categorical = Categorical(probs=distributions)
    sampled_actions = categorical.sample()

    return sampled_actions


def epsilon_greedy_distributions(
    distributions: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Compute the epsilon-greedy distributions from the given distributions."""
    batch_size, n_actions = distributions.shape

    # Create the random distributions.
    random_probs = torch.zeros_like(
        distributions, dtype=torch.float, device=distributions.device
    )
    random_probs = random_probs.fill_(1 / n_actions)

    # Create the greedy distributions.
    greedy_probs = torch.zeros(
        batch_size * n_actions, dtype=torch.float, device=distributions.device
    )
    greedy_actions = distributions.argmax(dim=-1).flatten()
    offsets = torch.arange(
        start=0, end=batch_size * n_actions, step=n_actions, device=distributions.device
    )
    greedy_probs[greedy_actions + offsets] = 1
    greedy_probs = greedy_probs.view(batch_size, n_actions)

    # Combine the two distributions.
    epsilon_greedy_distributions = (1 - epsilon) * greedy_probs + epsilon * random_probs
    return epsilon_greedy_distributions


@torch.no_grad()
def epsilon_greedy_sampling(
    distributions: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Sample actions from the given distributions using
    the epsilon-greedy method. It samples the greedy action
    with probability 1 - epsilon and samples a random action otherwise.

    ---
    Args:
        distributions: The distributions of the actions.
            Shape of [batch_size, n_actions].
        epsilon: The probability of sampling a random action.

    ---
    Returns:
        The sampled actions.
            Shape [batch_size,].
    """
    distributions = epsilon_greedy_distributions(distributions, epsilon)
    categorical = Categorical(probs=distributions)
    sampled_actions = categorical.sample()

    return sampled_actions


def epsilon_distributions(distributions: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Compute the epsilon distributions from the given distributions."""
    _, n_actions = distributions.shape

    # Create the random distributions.
    random_probs = torch.zeros_like(
        distributions, dtype=torch.float, device=distributions.device
    )
    random_probs = random_probs.fill_(1 / n_actions)

    # Combine the two distributions.
    epsilon_distributions = (1 - epsilon) * distributions + epsilon * random_probs
    return epsilon_distributions


def epsilon_sampling(distributions: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Sample actions from the given distributions using
    the epsilon method. It samples using the given distribution
    with probability 1 - epsilon and samples a random action otherwise.

    ---
    Args:
        distributions: The distributions of the actions.
            Shape of [batch_size, n_actions].
        epsilon: The probability of sampling a random action.

    ---
    Returns:
        The sampled actions.
            Shape [batch_size,].
    """
    distributions = epsilon_distributions(distributions, epsilon)
    categorical = Categorical(probs=distributions)
    sampled_actions = categorical.sample()

    return sampled_actions


def dirichlet_distributions(
    distributions: torch.Tensor, concentration: float, exploration: float
) -> torch.Tensor:
    """Compute the dirichlet distributions from the given distributions."""
    batch_size, n_actions = distributions.shape

    concentration_tensor = torch.ones_like(distributions)
    concentration_tensor.fill_(concentration)
    exploration_distribution = Dirichlet(concentration_tensor).sample()

    return (1 - exploration) * distributions + exploration * exploration_distribution


def dirichlet_sampling(
    distributions: torch.Tensor, concentration: float, exploration: float
) -> torch.Tensor:
    """Sample actions from the given distributions using
    the dirichlet method. It samples using the given distribution
    with probability 1 - exploration and samples a random action otherwise.

    ---
    Args:
        distributions: The distributions of the actions.
            Shape of [batch_size, n_actions].
        concentration: The concentration of the dirichlet distribution.
        exploration: The probability of sampling a random action.

    ---
    Returns:
        The sampled actions.
            Shape [batch_size,].
    """
    distributions = dirichlet_distributions(distributions, concentration, exploration)
    categorical = Categorical(probs=distributions)
    sampled_actions = categorical.sample()

    return sampled_actions
