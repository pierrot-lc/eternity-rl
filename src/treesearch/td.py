"""Soft Monte Carlo Tree Search implementation.
Differences with pure MCTS:
    - Only simple rollouts to get a crude estimate of the next best actions.
    - No exploration/exploitation intermediate states.
    - No backpropagation, pure simulation.

This simplications are there to enable better exploitation of the batch parallelism.
"""
from collections import defaultdict
from math import prod

import torch
from einops import rearrange, repeat
from tensordict import TensorDict
from torchrl.data import ReplayBuffer
from tqdm import tqdm

from ..environment import EternityEnv
from ..model import N_SIDES, Policy
from ..reinforce.rollout_buffer import RolloutBuffer
from ..sampling import epsilon_greedy_sampling


class TDTreeSearch:
    def __init__(
        self,
        n_copies: int,
        gamma: float,
    ):
        self.n_copies = n_copies
        self.gamma = gamma

        self.action_returns = torch.empty((0,))
        self.action_visits = torch.empty((0,))

    @torch.inference_mode()
    def run(
        self,
        original_env: EternityEnv,
        model: Policy,
        replay_buffer: ReplayBuffer,
        sampling_mode: str,
        rollout_depth: int,
        disable_logs: bool,
    ) -> torch.Tensor:
        """Do the simulations and return the best action found for each instance."""
        traces = defaultdict(list)
        duplicated_env = self.init_run(original_env)

        for _ in tqdm(
            range(rollout_depth), desc="Rollout", leave=False, disable=disable_logs
        ):
            self.simulation(duplicated_env, model, traces, sampling_mode)

        for name, tensors in traces.items():
            # Build proper traces of shape [batch_size, rollout_depth, ...].
            traces[name] = torch.stack(tensors, dim=1)

        # Compute the estimated returns. Do not forget to add
        # the last states' estimated returns using the value network.
        final_states = duplicated_env.render()
        *_, final_values = model(final_states, sampling_mode)
        final_values *= duplicated_env.terminated
        traces["returns"] = traces["rewards"].clone()
        traces["returns"][:, -1] += self.gamma * final_values
        traces["returns"] = RolloutBuffer.cumulative_decay_return(
            traces["returns"], traces["masks"], self.gamma
        )

        # Find the best environment among all duplicated instances.
        # Replace the original env state by the new env states.
        rollout_lengths = traces["masks"].sum(dim=1) - 1
        final_returns = torch.gather(
            traces["returns"],
            dim=1,
            index=rollout_lengths.unsqueeze(1),
        ).squeeze(1)
        final_returns = rearrange(final_returns, "(b d) -> b d", d=self.n_copies)
        best_env_ids = torch.argmax(final_returns, dim=1).unsqueeze(1)

        for member_name in ["instances", "terminated", "max_matches"]:
            member_value = getattr(duplicated_env, member_name)
            member_value = rearrange(
                member_value, "(b d) ... -> b d ...", d=self.n_copies
            )
            gather_ids = best_env_ids
            for dimension_size in member_value.shape[2:]:
                gather_ids = repeat(gather_ids, "... -> ... r", r=dimension_size)
            member_value = torch.gather(member_value, 1, gather_ids).squeeze(1)
            setattr(original_env, member_name, member_value)

        if duplicated_env.best_matches_found > original_env.best_matches_found:
            original_env.best_board = duplicated_env.best_board
            original_env.best_matches_found = duplicated_env.best_matches_found

        original_env.total_won += (
            (original_env.max_matches == original_env.best_matches_possible)
            .sum()
            .cpu()
            .item()
        )

        # Add the unmasked samples to the rollout buffer.
        samples = dict()
        masks = traces["masks"].flatten()
        for name, values in traces.items():
            values = rearrange(values, "b d ... -> (b d) ...")
            samples[name] = values[masks]

        samples = TensorDict(
            samples, batch_size=masks.sum().cpu().item(), device=masks.device
        )
        replay_buffer.extend(samples)

    @torch.inference_mode()
    def simulation(
        self,
        env: EternityEnv,
        model: Policy,
        traces: dict[str, list[torch.Tensor]],
        sampling_mode: str,
    ):
        """Do a look-ahead simulation and store the results in the replay buffer."""
        sample = dict()
        sample["states"] = env.render()
        sample["actions"], *_ = model(sample["states"], sampling_mode)
        _, sample["rewards"], terminated, _, infos = env.step(sample["actions"])

        sample["masks"] = ~terminated | infos["just_won"]
        for name, tensor in sample.items():
            traces[name].append(tensor)

    def init_run(self, env: EternityEnv) -> EternityEnv:
        """Duplicate the instances and initialize the environment."""
        instances = env.instances.clone()
        instances = torch.repeat_interleave(instances, self.n_copies, dim=0)
        env = EternityEnv(instances, env.device, env.rng.seed())

        self.action_returns = torch.zeros(
            (
                env.batch_size,
                env.n_pieces,
                env.n_pieces,
                N_SIDES,
                N_SIDES,
            ),
            dtype=torch.float32,
            device=env.device,
        )
        self.action_visits = torch.zeros_like(
            self.action_returns,
            dtype=torch.long,
            device=env.device,
        )

        return env

    @staticmethod
    def batched_add(
        input_tensor: torch.Tensor, actions: torch.Tensor, to_add: torch.Tensor | float
    ) -> torch.Tensor:
        """Add to the input tensor the elements at the given actions indices.
        This function is actually not limited to 4 actions, but can take any number
        of actions, as long as the `input_tensor` and `actions` are properly setup.

        ---
        Args:
            input_tensor: Tensor to add the elements to.
                Shape of [batch_size, n_actions_1, n_actions_2, n_actions_3, n_actions_4].
            actions: Indices to which we add the elements in the input.
                Shape of [batch_size, 4].
            to_add: Elements to add to the input.
                Shape of [batch_size,].

        ---
        Returns:
            The input tensor with the elements added.
                Shape of [batch_size, n_actions_1, n_actions_2, n_actions_3, n_actions_4].
        """
        batch_size, *actions_shape = input_tensor.shape
        n_elements = prod(actions_shape)

        # assert (
        #     len(actions_shape) == actions.shape[1]
        # ), "The number of actions does not match the number of dimensions in the input tensor."
        #
        # for action_id in range(len(actions_shape)):
        #     assert actions_shape[action_id] >= actions[:, action_id].max()

        indices = torch.zeros(batch_size, dtype=torch.long, device=input_tensor.device)
        for action_id, n_actions in enumerate(actions_shape):
            n_elements //= n_actions
            indices += actions[:, action_id] * n_elements

        n_elements = prod(actions_shape)
        offsets = torch.arange(
            start=0,
            end=n_elements * batch_size,
            step=n_elements,
            device=input_tensor.device,
        )
        indices += offsets

        input_tensor = input_tensor.flatten()
        input_tensor[indices] += to_add
        input_tensor = input_tensor.reshape(batch_size, *actions_shape)
        return input_tensor

    @staticmethod
    def best_actions(scores: torch.Tensor, sampling_mode: str) -> torch.Tensor:
        """Return the coordinates maximizing the score for each instance.

        ---
        Args:
            scores: The scores of each pairs of (instance, actions).
                Tensor of shape [batch_size, n_actions_1, n_actions_2, n_actions_3, n_actions_4].

        ---
        Returns:
            The coordinates of the best actions for each instance.
                Tensor of shape [batch_size, 4].
        """
        actions_shape = scores.shape[1:]
        n_elements = prod(actions_shape)
        scores = scores.flatten(start_dim=1)

        match sampling_mode:
            case "greedy":
                best_scores = scores.argmax(dim=1)
            case "epsilon-greedy":
                distributions = torch.softmax(scores, dim=-1)
                best_scores = epsilon_greedy_sampling(distributions, epsilon=0.05)
            case _:
                raise ValueError(f"Unknown sampling mode: {sampling_mode}")

        best_actions = []
        for n_actions in actions_shape:
            n_elements //= n_actions
            coord, best_scores = best_scores // n_elements, best_scores % n_elements
            best_actions.append(coord)

        return torch.stack(best_actions, dim=1)
