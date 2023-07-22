"""Soft Monte Carlo Tree Search implementation.
Differences with pure MCTS:
    - Only simple rollouts to get a crude estimate of the next best actions.
    - No exploration/exploitation intermediate states.
    - No backpropagation, pure simulation.

This simplications are there to enable better exploitation of the batch parallelism.
"""
from math import prod

import torch
from einops import rearrange
from tensordict import TensorDict
from torchrl.data import ReplayBuffer

from ..environment import EternityEnv
from ..model import N_SIDES, Policy
from ..sampling import epsilon_greedy_sampling


class TDTreeSearch:
    def __init__(
        self,
        n_copies: int,
        gamma: float,
    ):
        self.n_copies = n_copies
        self.gamma = gamma

    @torch.inference_mode()
    def run(
        self,
        env: EternityEnv,
        model: Policy,
        replay_buffer: ReplayBuffer,
        sampling_mode: str,
    ) -> torch.Tensor:
        """Do the simulations and return the best action found for each instance."""
        env, action_returns, action_visits = self.init_run(env)
        action_returns, action_visits = self.simulation(
            env, model, replay_buffer, action_returns, action_visits, sampling_mode
        )

        # Merge the simulations from duplicated instances.
        action_returns = rearrange(
            action_returns, "(b d) ... -> b d ...", d=self.n_copies
        )
        action_returns = action_returns.sum(dim=1)
        action_visits = rearrange(
            action_visits, "(b d) ... -> b d ...", d=self.n_copies
        )
        action_visits = action_visits.sum(dim=1)

        action_returns[action_visits == 0] = -1  # Do not select empty visits.
        action_visits[action_visits == 0] = 1  # Make sure we do not divide by 0.

        scores = action_returns / action_visits
        return TDTreeSearch.best_actions(scores, sampling_mode="greedy")

    @torch.inference_mode()
    def simulation(
        self,
        env: EternityEnv,
        model: Policy,
        replay_buffer: ReplayBuffer,
        action_returns: torch.Tensor,
        action_visits: torch.Tensor,
        sampling_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Do a look-ahead simulation and store the results in the replay buffer."""
        sample = dict()
        sample["states"] = env.render()
        sample["actions"], *_ = model(sample["states"], sampling_mode)
        sample["next-states"], sample["rewards"], terminated, _, infos = env.step(
            sample["actions"]
        )
        *_, next_values = model(sample["states"], sampling_mode)

        returns = sample["rewards"] + self.gamma * next_values
        action_returns = TDTreeSearch.batched_add(
            action_returns, sample["actions"], returns
        )
        action_visits = TDTreeSearch.batched_add(action_visits, sample["actions"], 1)

        to_keep = ~terminated | infos["just_won"]
        for name, tensor in sample.items():
            sample[name] = tensor[to_keep]

        sample = TensorDict(sample, batch_size=env.batch_size, device=env.device)
        replay_buffer.extend(sample)

        return action_returns, action_visits

    def init_run(
        self, env: EternityEnv
    ) -> tuple[EternityEnv, torch.Tensor, torch.Tensor]:
        """Duplicate the instances and initialize the environment."""
        instances = env.instances.clone()
        instances = torch.repeat_interleave(instances, self.n_copies, dim=0)
        env = EternityEnv(instances, env.device, env.rng.seed())
        action_returns = torch.zeros(
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
        action_visits = torch.zeros_like(
            action_returns,
            dtype=torch.long,
            device=env.device,
        )

        return env, action_returns, action_visits

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
