"""Implements a simple rollout buffer."""
import einops
import torch


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        max_steps: int,
        board_size: int,
        n_classes: int,
        device: str,
    ):
        self.buffer_size = buffer_size
        self.max_steps = max_steps
        self.board_size = board_size
        self.n_classes = n_classes
        self.device = device
        self.pointer = 0

        self.create_buffers()
        self.reset()

    def create_buffers(self):
        """Create buffers of shape `[buffer_size, max_steps`].
        Buffers are stored on cpu.
        """
        self.observation_buffer = torch.zeros(
            (self.buffer_size, self.max_steps, 4, self.board_size, self.board_size),
            dtype=torch.int64,
            device=self.device,
        )
        self.timestep_buffer = torch.zeros(
            (self.buffer_size, self.max_steps),
            dtype=torch.int64,
            device=self.device,
        )
        self.action_buffer = torch.zeros(
            (self.buffer_size, self.max_steps, 4),
            dtype=torch.int64,
            device=self.device,
        )
        self.reward_buffer = torch.zeros(
            (self.buffer_size, self.max_steps),
            dtype=torch.float32,
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
        self.mask_buffer = torch.zeros(
            (self.buffer_size, self.max_steps),
            dtype=torch.bool,
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

    def store(
        self,
        observations: torch.Tensor,
        timesteps: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ):
        """Add the given sample to the buffer.

        ---
        Args:
            observations: Batch of observations.
                Shape of [batch_size, 4, board_size, board_size].
            actions: Batch of actions.
                Shape of [batch_sizen, 4].
            rewards: Batch of rewards.
                Shape of [batch_size,].
            masks: Batch of masks.
                Shape of [batch_size,].
        """
        self.observation_buffer[:, self.pointer] = observations.to(self.device)
        self.timestep_buffer[:, self.pointer] = timesteps.to(self.device)
        self.action_buffer[:, self.pointer] = actions.to(self.device)
        self.reward_buffer[:, self.pointer] = rewards.to(self.device)
        self.mask_buffer[:, self.pointer] = masks.to(self.device)

        self.pointer += 1
        self.pointer %= self.max_steps

    def finalize(self, advantage_type: str, gamma: float):
        """Compute the returns and advantages of the trajectories."""
        self.return_buffer = RolloutBuffer.cumulative_decay_return(
            self.reward_buffer, self.mask_buffer, gamma
        )

        match advantage_type:
            case "estimated":
                returns = self.return_buffer[:, 0]
                self.advantage_buffer = self.return_buffer - returns.mean()
                self.advantage_buffer = self.advantage_buffer / (returns.std() + 1e-5)
            case "no-advantage":
                self.advantage_buffer = self.return_buffer
            case _:
                raise ValueError(f"Unknown advantage type: {advantage_type}")

        # Compute the valid steps we can sample from (those that are not masked).
        steps = torch.arange(
            start=0, end=self.max_steps * self.buffer_size, device=self.device
        )
        steps = einops.rearrange(steps, "(b t) -> b t", b=self.buffer_size)
        self.valid_steps = steps[self.mask_buffer]  # Valid steps we can sample from.

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch from the buffer.
        Ignores the masked steps.

        ---
        Args:
            batch_size: The batch size.

        ---
        Returns:
            A batch of samples as a dictionary:
                - observations: Batch of observations.
                    Shape of [batch_size, 4, board_size, board_size].
                - timesteps: Batch of timesteps.
                    Shape of [batch_size,].
                - actions: Batch of actions.
                    Shape of [batch_sizen, 4].
                - advantages: Batch of advantages.
                    Shape of [batch_size,].
        """
        # Without replacement.
        indices = torch.randperm(len(self.valid_steps), device=self.device)[:batch_size]
        steps = self.valid_steps[indices]

        return {
            "observations": self.flatten_observations[steps],
            "timesteps": self.flatten_timesteps[steps],
            "actions": self.flatten_actions[steps],
            "advantages": self.flatten_advantages[steps],
            "returns": self.flatten_returns[steps],
        }

    @staticmethod
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
            returns = torch.cummax(masks * rewards, dim=1).values
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
    def flatten_observations(self) -> torch.Tensor:
        """Flatten the observations buffer.

        ---
        Returns:
            The flattened observations buffer.
                Shape of [buffer_size * max_steps, 4, board_size, board_size].
        """
        return einops.rearrange(self.observation_buffer, "b t c h w -> (b t) c h w")

    @property
    def flatten_actions(self) -> torch.Tensor:
        """Flatten the actions buffer.

        ---
        Returns:
            The flattened actions buffer.
                Shape of [buffer_size * max_steps, 4].
        """
        return einops.rearrange(self.action_buffer, "b t c -> (b t) c")

    @property
    def flatten_returns(self) -> torch.Tensor:
        """Flatten the returns buffer.

        ---
        Returns:
            The flattened returns buffer.
                Shape of [buffer_size * max_steps].
        """
        return einops.rearrange(self.return_buffer, "b t -> (b t)")

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
    def flatten_timesteps(self) -> torch.Tensor:
        """Flatten the timesteps buffer.

        ---
        Returns:
            The flattened timesteps buffer.
                Shape of [buffer_size * max_steps].
        """
        return einops.rearrange(self.timestep_buffer, "b t -> (b t)")
