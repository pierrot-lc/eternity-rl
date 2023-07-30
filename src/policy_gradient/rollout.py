"""Play some steps."""
from tqdm import tqdm
from collections import defaultdict

import torch
import einops
from tensordict import TensorDictBase, TensorDict

from ..environment import EternityEnv
from ..model import Policy


def rollout(
    env: EternityEnv,
    model: Policy,
    sampling_mode: str,
    steps: int,
    disable_logs: bool,
) -> TensorDictBase:
    """Play some steps.

    ---
    Args:
        env: The environments to play in.
        model: The model to use to play.
        sampling_mode: The sampling mode to use.
        steps: The number of steps to play.
        disable_logs: Whether to disable the logs.
    """
    traces = defaultdict(list)

    for _ in tqdm(range(steps), desc="Rollout", leave=False, disable=disable_logs):
        sample = dict()

        sample["states"] = env.render()
        sample["actions"], sample["log-probs"], _, sample["values"] = model(
            sample["states"], sampling_mode
        )
        _, sample["rewards"], terminated, _, infos = env.step(sample["actions"])
        sample["masks"] = ~terminated | infos["just-won"]
        sample["dones"] = terminated
        sample["values"] *= ~terminated

        for name, tensor in sample.items():
            traces[name].append(tensor)

    # To [batch_size, steps, ...].
    for name, tensors in traces.items():
        traces[name] = torch.stack(tensors, dim=1)

    final_states = env.render()
    *_, final_values = model(final_states, sampling_mode)
    final_values *= ~env.terminated
    traces["next-values"] = torch.concat(
        (traces["values"][:, 1:], final_values.unsqueeze(1)), dim=1
    )

    return TensorDict(traces, batch_size=traces["states"].shape[0], device=env.device)


def cumulative_decay_return(
    rewards: torch.Tensor, masks: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Compute the cumulative decayed return of a batch of games.
    It is efficiently implemented using tensor operations.

    Thanks to the kind stranger here: https://discuss.pytorch.org/t/cumulative-sum-with-decay-factor/69788/2.
    For `gamma != 1`, this function may not be numerically stable.

    ---
    Args:
        rewards: The rewards of the games.
            Shape of [batch_size, max_steps].
        masks: The mask indicating which steps are actual plays.
            Shape of [batch_size, max_steps].
        gamma: The discount factor.

    ---
    Returns:
        The cumulative decayed return of the games.
            Shape of [batch_size, max_steps].
    """
    if gamma == 1:
        rewards = torch.flip(rewards, dims=(1,))
        masks = torch.flip(masks, dims=(1,))
        returns = torch.cumsum(masks * rewards, dim=1)
        returns = torch.flip(returns, dims=(1,))
        return returns

    # Compute the gamma powers.
    powers = (rewards.shape[1] - 1) - torch.arange(
        rewards.shape[1], device=rewards.device
    )
    powers = gamma**powers
    powers = einops.repeat(powers, "t -> b t", b=rewards.shape[0])

    # Compute the cumulative decayed return.
    rewards = torch.flip(rewards, dims=(1,))
    masks = torch.flip(masks, dims=(1,))
    returns = torch.cumsum(masks * rewards * powers, dim=1) / powers
    returns = torch.flip(returns, dims=(1,))

    return returns
