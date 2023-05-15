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

        self.create_buffers()

    def create_buffers(self):
        """Create buffers of shape `[buffer_size, max_steps`].
        Buffers are stored on cpu.
        """
        self.observation_buffer = torch.zeros(
            (self.buffer_size, self.max_steps, 4, self.board_size, self.board_size),
            dtype=torch.int64,
        )
        self.action_buffer = torch.zeros(
            (self.buffer_size, self.max_steps, 4),
            dtype=torch.int64,
        )
        self.reward_buffer = torch.zeros(
            (self.buffer_size, self.max_steps),
            dtype=torch.float32,
        )
        self.return_buffer = torch.zeros(
            (self.buffer_size, self.max_steps),
            dtype=torch.float32,
        )
        self.advantage_buffer = torch.zeros(
            (self.buffer_size, self.max_steps),
            dtype=torch.float32,
        )
        self.mask_buffer = torch.zeros(
            (self.buffer_size, self.max_steps),
            dtype=torch.bool,
        )

        # Flattened buffers for sampling.
        self.flatten_observation = torch.zeros(
            (self.buffer_size * self.max_steps, 4, self.board_size, self.board_size),
            dtype=torch.int64,
        )
        self.flatten_action = torch.zeros(
            (self.buffer_size * self.max_steps, 4),
            dtype=torch.int64,
        )
        self.flatten_advantage = torch.zeros(
            (self.buffer_size * self.max_steps),
            dtype=torch.float32,
        )
        self.valid_steps = torch.zeros(
            (self.buffer_size * self.max_steps),
            dtype=torch.int64,
        )

        self.pointer = 0

    def store(
        self,
        observations: torch.Tensor,
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
        self.observation_buffer[:, self.pointer] = observations.cpu()
        self.action_buffer[:, self.pointer] = actions.cpu()
        self.reward_buffer[:, self.pointer] = rewards.cpu()
        self.mask_buffer[:, self.pointer] = masks.cpu()

        self.pointer += 1
        self.pointer %= self.max_steps

    def finalize(self, advantage_type: str, gamma: float):
        """Compute the returns and advantages for the trajectories."""
        self.return_buffer = RolloutBuffer.cumulative_decay_return(
            self.reward_buffer, self.mask_buffer, gamma
        )

        match advantage_type:
            case "estimated":
                returns = self.return_buffer[:, 0]
                self.advantage_buffer = (self.return_buffer - returns.mean()) / (
                    returns.std() + 1e-8
                )
            case "no-advantage":
                self.advantage_buffer = self.return_buffer
            case _:
                raise ValueError(f"Unknown advantage type: {advantage_type}")

        # To sample efficiently from the buffers.
        self.flatten_observation = einops.rearrange(
            self.observation_buffer, "b t c h w -> (b t) c h w"
        )
        self.flatten_action = einops.rearrange(self.action_buffer, "b t c -> (b t) c")
        self.flatten_advantage = einops.rearrange(self.advantage_buffer, "b t -> (b t)")

        # Compute the valid steps we can sample from (those that are not masked).
        steps = torch.arange(start=0, end=self.max_steps * self.buffer_size)
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
                - actions: Batch of actions.
                    Shape of [batch_sizen, 4].
                - advantages: Batch of advantages.
                    Shape of [batch_size,].
        """
        # Without replacement.
        indices = torch.randperm(len(self.valid_steps))[:batch_size]
        steps = self.valid_steps[indices]

        return {
            "observations": self.flatten_observation[steps].to(self.device),
            "actions": self.flatten_action[steps].to(self.device),
            "advantages": self.flatten_advantage[steps].to(self.device),
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
