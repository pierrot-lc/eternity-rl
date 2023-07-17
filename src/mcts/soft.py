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
from tqdm import tqdm

from ..environment import EternityEnv
from ..model import N_SIDES, Policy


class SoftMCTS:
    def __init__(
        self,
        env: EternityEnv,
        model: Policy,
        sampling_mode: str,
        max_depth: int,
        n_simulations: int,
        n_env_copies: int,
        device: str | torch.device,
    ):
        self.model = model
        self.sampling_mode = sampling_mode
        self.n_simulations = n_simulations
        self.n_env_copies = n_env_copies
        self.device = device

        # Duplicate the instances and initialize the environment.
        instances = env.instances.clone()
        instances = torch.repeat_interleave(instances, n_env_copies, dim=0)
        self.env = EternityEnv(instances, max_depth, device, env.rng.seed())

        self.root_instances = self.env.render().clone()

        self.action_returns = torch.zeros(
            (
                self.env.batch_size,
                self.env.n_pieces,
                self.env.n_pieces,
                N_SIDES,
                N_SIDES,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.action_visits = torch.zeros_like(
            self.action_returns,
            dtype=torch.long,
            device=device,
        )

    @torch.inference_mode()
    def simulation(self):
        """Do a Monte Carlo simulation from the root instances until
        the episodes are finished.
        """
        states, _ = self.env.reset(self.root_instances)
        root_actions, *_ = self.model(states, self.sampling_mode)
        states, *_ = self.env.step(root_actions)

        while not self.env.truncated and not torch.all(self.env.terminated):
            actions, *_ = self.model(states, self.sampling_mode)
            states, *_ = self.env.step(actions)

        returns = self.env.max_matches / self.env.best_matches
        self.action_returns = SoftMCTS.batched_add(
            self.action_returns, root_actions, returns
        )
        self.action_visits = SoftMCTS.batched_add(self.action_visits, root_actions, 1)

    @torch.inference_mode()
    def run(self, disable_tqdm: bool) -> torch.Tensor:
        """Do the simulations and return the best action found for each instance."""
        for _ in tqdm(
            range(self.n_simulations),
            desc="Soft MCTS",
            leave=False,
            disable=disable_tqdm,
        ):
            self.simulation()

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
