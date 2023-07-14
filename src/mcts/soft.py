"""Soft Monte Carlo Tree Search implementation.
Differences with pure MCTS:
    - Only simple rollouts to get a crude estimate of the next best actions.
    - No exploration/exploitation intermediate states.
    - No backpropagation, pure simulation.

This simplications are there to enable better exploitation of the batch parallelism.
"""
import torch
from einops import rearrange
from math import prod

from ..model import Policy, N_SIDES
from ..environment import EternityEnv


class SoftMCTS:
    def __init__(self, env: EternityEnv, model: Policy, device: str | torch.device):
        self.env = env
        self.model = model
        self.device = device
        self.n_simulations = 10
        self.sampling_mode = "sample"

        self.root_instances = env.render().detach()
        self.root_step_id = env.step_id

        self.action_returns = torch.zeros(
            (env.batch_size, env.board_size, env.board_size, N_SIDES, N_SIDES),
            dtype=torch.float32,
            device=device,
        )
        self.action_visits = torch.zeros(
            (env.batch_size, env.board_size, env.board_size, N_SIDES, N_SIDES),
            dtype=torch.long,
            device=device,
        )

    @torch.inference_mode()
    def simulation(self):
        """Do a Monte Carlo simulation from the root instances until
        the episodes are finished.
        """
        batch_size, board_height, board_width, _, _ = self.action_returns.shape

        states, _ = self.env.reset(self.root_instances)
        root_actions, _, _ = self.model(states, self.sampling_mode)
        states, rewards, _, _, infos = self.env.step(root_actions)

        while not self.env.truncated and not torch.all(self.env.terminated):
            actions, _, _ = self.model(states, self.sampling_mode)
            states, rewards, _, _, infos = self.env.step(actions)

        returns = self.env.max_matches / self.env.best_matches
        self.action_returns = SoftMCTS.batched_add(
            self.action_returns, root_actions, returns
        )
        self.action_visits = SoftMCTS.batched_add(self.action_visits, root_actions, 1)

    @torch.inference_mode()
    def run(self) -> torch.Tensor:
        """Do the simulations and return the best action found for each instance."""
        for _ in range(self.n_simulations):
            self.simulation()

        scores = self.action_returns / self.action_visits
        return SoftMCTS.best_actions(scores)

    @staticmethod
    def batched_add(
        input_tensor: torch.Tensor, actions: torch.Tensor, to_add: torch.Tensor | float
    ) -> torch.Tensor:
        """Add to the input tensor the elements at the given actions indices.

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
        (
            batch_size,
            n_actions_1,
            n_actions_2,
            n_actions_3,
            n_actions_4,
        ) = input_tensor.shape
        input_tensor = input_tensor.flatten()
        indices = (
            actions[:, 0] * n_actions_1
            + actions[:, 1] * n_actions_2
            + actions[:, 2] * n_actions_3
            + actions[:, 3] * n_actions_4
        )
        offsets = torch.arange(
            start=0,
            end=input_tensor.shape[0],
            step=input_tensor.shape[0] // batch_size,
        )
        indices = indices + offsets

        input_tensor[indices] += to_add
        input_tensor = rearrange(
            input_tensor,
            "(b a1 a2 a3 a4) -> b a1 a2 a3 a4",
            a1=n_actions_1,
            a2=n_actions_2,
            a3=n_actions_3,
            a4=n_actions_4,
        )
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
