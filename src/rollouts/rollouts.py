"""Play some steps."""
from collections import defaultdict

import einops
import torch
from tensordict import TensorDict, TensorDictBase
from tqdm import tqdm

from ..environment import EternityEnv
from ..mcts import MCTSTree
from ..model import Critic, Policy


def policy_rollouts(
    env: EternityEnv,
    policy: Policy,
    critic: Critic,
    steps: int,
    disable_logs: bool,
    sampling_mode: str = "softmax",
) -> TensorDictBase:
    """Play some steps.

    ---
    Args:
        env: The environments to play in.
        policy: The policy to use.
        critic: The critic to use.
        steps: The number of steps to play.
        disable_logs: Whether to disable the logs.
        sampling_mode: The sampling mode to use for the policy.

    ---
    Returns:
        traces: The traces of the played steps.
    """
    traces = defaultdict(list)

    for step_id in tqdm(
        range(steps), desc="Rollout", leave=False, disable=disable_logs
    ):
        sample = dict()
        sample["states"] = env.render()

        sample["actions"], sample["log-probs"], _ = policy(
            sample["states"], sampling_mode=sampling_mode
        )
        sample["values"] = critic(sample["states"])

        _, sample["rewards"], sample["dones"], sample["truncated"], _ = env.step(
            sample["actions"]
        )

        sample["next-values"] = critic(env.render())

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


@torch.inference_mode()
def exploit_rollouts(
    env: EternityEnv,
    policy: Policy,
    steps: int,
    disable_logs: bool,
    sampling_mode: str = "greedy",
):
    """Play some steps, do not collect anything nor use the critic.

    ---
    Args:
        env: The environments to play in.
        policy: The policy to use.
        steps: The number of steps to play.
        disable_logs: Whether to disable the logs.
        sampling_mode: The sampling mode to use for the policy.
    """
    for step_id in tqdm(
        range(steps), desc="Exploit rollout", leave=False, disable=disable_logs
    ):
        actions, *_ = policy(env.render(), sampling_mode=sampling_mode)

        _, _, dones, truncated, _ = env.step(actions)

        if (dones | truncated).sum() > 0:
            reset_ids = torch.arange(0, env.batch_size, device=env.device)
            reset_ids = reset_ids[dones | truncated]
            env.reset(reset_ids)


@torch.inference_mode()
def mcts_rollouts(
    env: EternityEnv,
    policy: Policy,
    critic: Critic,
    mcts: MCTSTree,
    steps: int,
    sampling_mode: str,
    disable_logs: bool,
) -> TensorDictBase:
    """Play some steps using a MCTS tree to look for the best
    move at each step.

    ---
    Args:
        env: The environments to play in.
        policy: The policy to use.
        critic: The critic to use.
        mcts: The MCTS tree to use.
        steps: The number of steps to play.
        sampling_mode: The sampling mode to use for the action selection.
        disable_logs: Whether to disable the logs.

    ---
    Returns:
        The traces of the played steps.
    """
    traces = defaultdict(list)

    for step_id in tqdm(
        range(steps), desc="MCTS Rollout", leave=False, disable=disable_logs
    ):
        sample = dict()
        mcts.reset(env, policy, critic)

        sample["states"] = env.render()
        sample["probs"], sample["values"], sample["actions"] = mcts.evaluate(
            disable_logs
        )

        # Make sure there's no "nan" values.
        assert torch.isfinite(sample["probs"]).all()
        assert torch.isfinite(sample["values"]).all()

        print(sample["probs"][0])
        print(sample["values"][0])

        match sampling_mode:
            case "softmax":
                action_ids = Policy.sample_actions(sample["probs"], mode=sampling_mode)
                sample["values"] = (sample["values"] * sample["probs"]).sum(dim=1)
            case "greedy":
                action_ids = sample["values"].argmax(dim=1)
                sample["values"] = sample["values"].max(dim=1).values
            case "uniform":
                action_ids = Policy.sample_actions(sample["probs"], mode=sampling_mode)
                sample["values"] = sample["values"].mean(dim=1)
            case _:
                raise ValueError(f"Invalid sampling mode: {sampling_mode}")

        sampled_actions = sample["actions"][mcts.batch_range, action_ids]
        _, rewards, dones, truncated, _ = env.step(
            sampled_actions
        )

        if (dones | truncated).sum() > 0:
            reset_ids = torch.arange(0, env.batch_size, device=env.device)
            reset_ids = reset_ids[dones | truncated]
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
