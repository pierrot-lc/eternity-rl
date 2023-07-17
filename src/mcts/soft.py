"""Soft Monte Carlo Tree Search implementation.
Differences with pure MCTS:
    - Only simple rollouts to get a crude estimate of the next best actions.
    - No exploration/exploitation intermediate states.
    - No backpropagation, pure simulation.

This simplications are there to enable better exploitation of the batch parallelism.
"""
from math import prod

import torch
from torchrl.data import ReplayBuffer
from tensordict import TensorDict
from einops import rearrange

from ..environment import EternityEnv
from ..model import N_SIDES, Policy


class SoftMCTS:
    def __init__(
        self,
        env: EternityEnv,
        model: Policy,
        n_env_copies: int,
    ):
        self.model = model
        self.n_env_copies = n_env_copies
        self.device = env.device

        # Duplicate the instances and initialize the environment.
        instances = env.instances.clone()
        instances = torch.repeat_interleave(instances, n_env_copies, dim=0)
        self.env = EternityEnv(instances, self.device, env.rng.seed())

        self.action_returns = torch.zeros(
            (
                self.env.batch_size,
                self.env.n_pieces,
                self.env.n_pieces,
                N_SIDES,
                N_SIDES,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.action_visits = torch.zeros_like(
            self.action_returns,
            dtype=torch.long,
            device=self.device,
        )

    @torch.inference_mode()
    def simulation(self, gamma: float, sampling_mode: str, replay_buffer: ReplayBuffer):
        """Do a look-ahead simulation and store the results in the replay buffer."""
        sample = dict()
        sample["states"] = self.env.render()
        sample["actions"], *_ = self.model(sample["states"], sampling_mode)
        sample["next-states"], sample["rewards"], terminated, _, infos = self.env.step(
            sample["actions"]
        )
        *_, next_values = self.model(sample["states"], sampling_mode)

        returns = sample["rewards"] + gamma * next_values
        self.action_returns = SoftMCTS.batched_add(
            self.action_returns, sample["actions"], returns
        )
        self.action_visits = SoftMCTS.batched_add(
            self.action_visits, sample["actions"], 1
        )

        to_keep = ~terminated | infos["just_won"]
        for name, tensor in sample.items():
            sample[name] = tensor[to_keep]

        sample = TensorDict(sample, batch_size=self.env.batch_size, device=self.device)
        replay_buffer.extend(sample)

    @torch.inference_mode()
    def run(
        self, gamma: float, sampling_mode: str, replay_buffer: ReplayBuffer
    ) -> torch.Tensor:
        """Do the simulations and return the best action found for each instance."""
        self.simulation(gamma, sampling_mode, replay_buffer)

        # Merge the simulations from duplicated instances.
        action_returns = rearrange(
            self.action_returns, "(b d) ... -> b d ...", d=self.n_env_copies
        )
        action_returns = action_returns.sum(dim=1)
        action_visits = rearrange(
            self.action_visits, "(b d) ... -> b d ...", d=self.n_env_copies
        )
        action_visits = action_visits.sum(dim=1)

        action_returns[action_visits == 0] = -1  # Do not select empty visits.
        action_visits[action_visits == 0] = 1  # Make sure we do not divide by 0.

        scores = action_returns / action_visits
        return SoftMCTS.best_actions(scores)

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
    def best_actions(scores: torch.Tensor) -> torch.Tensor:
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
        best_scores = scores.argmax(dim=1)

        best_actions = []
        for n_actions in actions_shape:
            n_elements //= n_actions
            coord, best_scores = best_scores // n_elements, best_scores % n_elements
            best_actions.append(coord)

        return torch.stack(best_actions, dim=1)
