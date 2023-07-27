"""Implements a simple rollout buffer."""
import einops
import torch

from ..model import N_ACTIONS, N_SIDES


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        max_steps: int,
        board_width: int,
        board_height: int,
        n_classes: int,
        device: str,
    ):
        self.buffer_size = buffer_size
        self.max_steps = max_steps
        self.board_width = board_width
        self.board_height = board_height
        self.n_classes = n_classes
        self.device = device
        self.pointer = 0

        self.create_buffers()
        self.reset()

    def create_buffers(self):
        """Create buffers of shape `[buffer_size, max_steps]`.
        Buffers are stored on the specified device.
        """
        self.state_buffer = torch.zeros(
            (
                self.buffer_size,
                self.max_steps,
                N_SIDES,
                self.board_height,
                self.board_width,
            ),
            dtype=torch.long,
            device=self.device,
        )
        self.action_buffer = torch.zeros(
            (self.buffer_size, self.max_steps, N_ACTIONS),
            dtype=torch.long,
            device=self.device,
        )
        self.reward_buffer = torch.zeros(
            (self.buffer_size, self.max_steps),
            dtype=torch.float32,
            device=self.device,
        )
        self.mask_buffer = torch.zeros(
            (self.buffer_size, self.max_steps),
            dtype=torch.bool,
            device=self.device,
        )
        self.return_buffer = torch.zeros(
            (self.buffer_size, self.max_steps),
            dtype=torch.float32,
            device=self.device,
        )
        self.advantage_buffer = torch.zeros(
            (self.buffer_size, self.max_steps),
            dtype=torch.float32,
            device=self.device,
        )

        self.all_steps = torch.arange(
            start=0,
            end=self.buffer_size * self.max_steps,
            device=self.device,
        )
        self.valid_steps = torch.zeros(
            (self.buffer_size * self.max_steps),
            dtype=torch.int64,
            device=self.device,
        )

    def reset(self):
        """Reset the buffer."""
        self.pointer = 0
        self.mask_buffer.fill_(False)

    def store(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ):
        """Add the given sample to the buffer.

        ---
        Args:
            states: Batch of states.
                Shape of [batch_size, N_SIDES, board_height, board_width].
            actions: Batch of actions.
                Shape of [batch_size, N_ACTIONS].
            rewards: Batch of rewards.
                Shape of [batch_size,].
            masks: Batch of masks.
                Shape of [batch_size,].
        """
        self.state_buffer[:, self.pointer] = states.to(self.device)
        self.action_buffer[:, self.pointer] = actions.to(self.device)
        self.reward_buffer[:, self.pointer] = rewards.to(self.device)
        self.mask_buffer[:, self.pointer] = masks.to(self.device)

        self.pointer += 1
        self.pointer %= self.max_steps

    def finalize(self, advantage_type: str):
        """Compute the returns and advantages of the trajectories."""
        self.return_buffer, self.mask_buffer = RolloutBuffer.cumulative_max_cut(
            self.reward_buffer, self.mask_buffer
        )

        match advantage_type:
            case "estimated":
                returns = self.return_buffer[:, 0]
                self.advantage_buffer = self.return_buffer - returns.mean()
                self.advantage_buffer = self.advantage_buffer / (returns.std() + 1e-5)
            case "no-advantage":
                self.advantage_buffer = self.return_buffer
            case "max":
                self.advantage_buffer = self.return_buffer / self.return_buffer.max()
            case _:
                raise ValueError(f"Unknown advantage type: {advantage_type}")

        # Save the unmasked steps in the rollout.
        self.valid_steps = self.all_steps[self.flatten_masks]

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch from the buffer.
        Ignores the masked steps.

        ---
        Args:
            batch_size: The batch size.

        ---
        Returns:
            A batch of samples as a dictionary:
                - advantages: Batch of advantages.
                    Shape of [batch_size,].
                - logprobs: Log-probabilities of the actions taken.
                    Shape of [batch_size, n_actions].
                - entropies: Batch of entropies.
                    Shape of [batch_size, n_actions].
        """
        # Without replacement.
        indices = torch.randperm(len(self.valid_steps), device=self.device)[:batch_size]
        steps = self.valid_steps[indices]

        return {
            "states": self.flatten_states[steps],
            "actions": self.flatten_actions[steps],
            "advantages": self.flatten_advantages[steps],
        }

    @staticmethod
    @torch.no_grad()
    def cumulative_max_cut(
        rewards: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use a cumulative max over the rewards and cut the episodes that end with
        a lower reward than their cumulative max.

        ---
        Args:
            rewards: The rewards of the games.
                Shape of [batch_size, max_steps].
            masks: The mask indicating which steps are actual plays.
                Shape of [batch_size, max_steps].

        ---
        Returns:
            The returns and masks of the games.
        """
        # Compute the reversed cumulative max.
        rewards = torch.flip(rewards, dims=(1,))
        masks = torch.flip(masks, dims=(1,))
        cummax = torch.cummax(masks * rewards, dim=1)
        returns, indices = cummax.values, cummax.indices

        # Cut the episodes that end with a lower reward than their cumulative max.
        best_indices = indices.max(dim=1).values
        cutted_masks = indices == best_indices.unsqueeze(1)

        # Flip back the tensors.
        returns = torch.flip(returns, dims=(1,))
        cutted_masks = torch.flip(cutted_masks, dims=(1,))
        return returns, cutted_masks

    @staticmethod
    @torch.no_grad()
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

    @property
    def flatten_advantages(self) -> torch.Tensor:
        """Flatten the advantages buffer.

        ---
        Returns:
            The flattened advantages buffer.
                Shape of [buffer_size * max_steps].
        """
        return einops.rearrange(self.advantage_buffer, "b t -> (b t)")

    @property
    def flatten_states(self) -> torch.Tensor:
        """Flatten the states buffer.

        ---
        Returns:
            The flattened states buffer.
                Shape of [buffer_size * max_steps, N_SIDES, board_height, board_width].
        """
        return einops.rearrange(self.state_buffer, "b t s h w -> (b t) s h w")

    @property
    def flatten_actions(self) -> torch.Tensor:
        """Flatten the actions buffer.

        ---
        Returns:
            The flattened actions buffer.
                Shape of [buffer_size * max_steps, N_ACTIONS].
        """
        return einops.rearrange(self.action_buffer, "b t a -> (b t) a")

    @property
    def flatten_masks(self) -> torch.Tensor:
        """Flatten the masks buffer.

        ---
        Returns:
            The flattened masks buffer.
                Shape of [buffer_size * max_steps].
        """
        return einops.rearrange(self.mask_buffer, "b t -> (b t)")
