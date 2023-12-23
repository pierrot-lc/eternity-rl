"""Play some steps."""
from collections import defaultdict

import einops
import torch
from tensordict import TensorDict, TensorDictBase
from tqdm import tqdm

from ..environment import EternityEnv
from ..model import Policy, Critic


def rollout(
    env: EternityEnv,
    policy: Policy,
    critic: Critic,
    sampling_mode: str,
    steps: int,
    disable_logs: bool,
) -> TensorDictBase:
    """Play some steps.

    ---
    Args:
        env: The environments to play in.
        policy: The policy to use.
        critic: The critic to use.
        sampling_mode: The sampling mode to use.
        steps: The number of steps to play.
        disable_logs: Whether to disable the logs.

    ---
    Returns:
        The traces of the played steps.
    """
    traces = defaultdict(list)

    for step_id in tqdm(
        range(steps), desc="Rollout", leave=False, disable=disable_logs
    ):
        sample = dict()
        sample["states"] = env.render()
        sample["conditionals"] = env.n_steps
        sample["best-boards"] = env.best_boards

        sample["actions"], sample["log-probs"], _ = policy(
            sample["states"],
            sample["best-boards"],
            sample["conditionals"],
            sampling_mode,
        )
        sample["values"] = critic(
            sample["states"],
            sample["best-boards"],
            sample["conditionals"],
        )

        _, sample["rewards"], sample["dones"], sample["truncated"], _ = env.step(
            sample["actions"]
        )

        sample["next-values"] = critic(
            env.render(),
            env.best_boards,
            env.n_steps,
        )

        sample["next-values"] *= (~sample["dones"]).float()

        if (sample["dones"] | sample["truncated"]).sum() > 0:
            reset_ids = torch.arange(0, env.batch_size, device=env.device)
            reset_ids = reset_ids[sample["dones"] | sample["truncated"]]
            env.reset(reset_ids)

        # BUG: There's an issue with cuda that hallucinates values when stacking
        # more than 128 tensors. To avoid this issue, we have to offload the tensors
        # so that we can stack on CPU.
        for name, tensor in sample.items():
            traces[name].append(tensor.cpu())

    for name, tensors in traces.items():
        traces[name] = torch.stack(tensors, dim=1)  # Stack on CPU.
        traces[name] = traces[name].to(env.device)  # Back to GPU.

    return TensorDict(traces, batch_size=traces["states"].shape[0], device=env.device)


def split_reset_rollouts(traces: TensorDictBase) -> TensorDictBase:
    """Split the samples that have been reset during the rollouts.
    For each element of the batch, if at some point the trace is either
    done or truncated, the rollout is split.

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
        The split traces.
            The traces are augmented with a "masks" entry.
            A sample is masked if it is some padding.
    """
    resets = traces["dones"] | traces["truncated"]
    resets[:, -1] = True

    batch_size, steps = resets.shape
    split_batch_size = resets.sum()
    device = resets.device

    # NOTE: The construction of the mask is done as follows:
    # 1. Find the length of each split rollouts.
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
    episodes_length = (reset_indices - shifted_reset_indices) % steps

    episodes_length -= 1  # Get indices.
    episodes_length[episodes_length == -1] = steps - 1  # `-1` is not valid for scatter.
    masks.scatter_(1, episodes_length.unsqueeze(1), 1)

    masks = torch.roll(masks, shifts=1, dims=(1,))
    masks[:, 0] = False
    masks = torch.cummax(masks, dim=1).values
    masks = ~masks

    split_traces = dict()
    for name, tensor in traces.items():
        split_tensor = torch.zeros(
            (split_batch_size, steps, *tensor.shape[2:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        split_tensor[masks] = einops.rearrange(tensor, "b s ... -> (b s) ...")
        split_traces[name] = split_tensor

    split_traces["masks"] = masks

    return TensorDict(
        split_traces, batch_size=split_traces["dones"].shape[0], device=device
    )


def mask_reset_rollouts(traces: TensorDictBase) -> TensorDictBase:
    """Mask the samples that have been reset during the rollouts.
    This make sure the samples are only about the original envs when rollout has been
    collected.

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
        The masked traces. A new entry "masks" is added to the traces.
        A mask is True if the sample is not a padding.
    """
    masks = traces["dones"] | traces["truncated"]
    masks = torch.cummax(masks, dim=1).values
    masks = torch.roll(masks, shifts=1, dims=(1,))
    masks = ~masks
    masks[:, 0] = True

    traces["masks"] = masks


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
