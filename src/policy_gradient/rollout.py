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
    state_shape = env.render().shape
    traces = {
        "states": torch.zeros(
            (env.batch_size, steps, *state_shape[1:]),
            dtype=torch.long,
            device=env.device,
        ),
        "conditionals": torch.zeros(
            (env.batch_size, steps), dtype=torch.long, device=env.device
        ),
        "actions": torch.zeros(
            (env.batch_size, steps, 4), dtype=torch.long, device=env.device
        ),
        "log-probs": torch.zeros(
            (env.batch_size, steps, 4), dtype=torch.float, device=env.device
        ),
        "values": torch.zeros(
            (env.batch_size, steps), dtype=torch.float, device=env.device
        ),
        "next-values": torch.zeros(
            (env.batch_size, steps), dtype=torch.float, device=env.device
        ),
        "rewards": torch.zeros(
            (env.batch_size, steps), dtype=torch.float, device=env.device
        ),
        "dones": torch.zeros(
            (env.batch_size, steps), dtype=torch.bool, device=env.device
        ),
        "truncated": torch.zeros(
            (env.batch_size, steps), dtype=torch.bool, device=env.device
        ),
    }

    for step_id in tqdm(
        range(steps), desc="Rollout", leave=False, disable=disable_logs
    ):
        traces["states"][:, step_id] = env.render()
        traces["conditionals"][:, step_id] = env.n_steps

        (
            traces["actions"][:, step_id],
            traces["log-probs"][:, step_id],
            _,
            traces["values"][:, step_id],
        ) = model(
            traces["states"][:, step_id],
            traces["conditionals"][:, step_id],
            sampling_mode,
        )

        (
            _,
            traces["rewards"][:, step_id],
            traces["dones"][:, step_id],
            traces["truncated"][:, step_id],
            _,
        ) = env.step(traces["actions"][:, step_id])

        *_, traces["next-values"][:, step_id] = model(
            env.render(),
            env.n_steps,
            sampling_mode,
        )

        if (traces["dones"][:, step_id] | traces["truncated"][:, step_id]).sum() > 0:
            reset_ids = torch.arange(0, env.batch_size, device=env.device)
            reset_ids = reset_ids[
                traces["dones"][:, step_id] | traces["truncated"][:, step_id]
            ]
            env.reset(reset_ids)

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
