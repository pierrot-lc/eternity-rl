"""Play some steps."""
from collections import defaultdict

import einops
import torch
from tensordict import TensorDict, TensorDictBase
from tqdm import tqdm

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
        _, sample["rewards"], terminated, truncated, infos = env.step(sample["actions"])
        sample["truncated"] = truncated
        sample["dones"] = terminated
        sample["values"] *= ~terminated

        for name, tensor in sample.items():
            traces[name].append(tensor)

        if (terminated | truncated).sum() > 0:
            reset_ids = torch.arange(0, env.batch_size, device=env.device)
            reset_ids = reset_ids[terminated | truncated]
            env.reset(reset_ids)

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


def split_reset_rollouts(traces: TensorDictBase) -> TensorDictBase:
    """Split the samples that have been reset during the rollouts.
    For each element of the batch, if at some point the trace is either
    done or truncated, the rollout is splitted.

    ---
    Args:
        traces: A dict containing at least the following entries:
            dones: The rollout dones of the given states.
                Shape of [batch_size, steps].
            truncated: The rollout truncated of the given states.
                Shape of [batch_size, steps].

            The other entries must be of shape [batch_size, steps, ...].

    ---
    Returns:
        The splitted traces.
    """
    resets = traces["dones"] | traces["truncated"]
    resets[:, -1] = True

    batch_size, steps = resets.shape
    split_batch_size = resets.sum()
    device = resets.device

    # NOTE: The construction of the mask is done as follows:
    # 1. Find the length of each splitted rollouts.
    # 2. Fill the mask with "True" at the end of the each rollouts.
    # 3. Use `cummax` to propagate the "True" until the start of the rollouts.
    masks = torch.zeros(
        (split_batch_size, steps),
        dtype=torch.bool,
        device=device,
    )

    reset_indices = torch.arange(0, steps, device=device)
    reset_indices = einops.repeat(reset_indices, "s -> b s", b=batch_size)
    reset_indices = reset_indices[resets]  # Shape of [split_batch_size].
    shifted_reset_indices = torch.roll(reset_indices, shifts=1, dims=(0,))
    episode_lenghts = (reset_indices - shifted_reset_indices) % steps

    masks.scatter_(dim=1, index=episode_lenghts.unsqueeze(1), src=1)

    masks = torch.roll(masks, shifts=1, dims=(1,))
    masks[:, 0] = False
    masks = torch.cummax(masks, dim=1).values
    masks = ~masks

    split_traces = {
        "masks": masks,
    }

    for name, tensor in traces.items():
        split_tensor = torch.zeros(
            (split_batch_size, steps, *tensor.shape[2:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        split_tensor[masks] = tensor

        split_traces[name] = split_tensor

    return TensorDict(split_traces, batch_size=split_batch_size, device=device)


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
