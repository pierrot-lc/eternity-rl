"""A batched version of the environment.
All actions are made on the batch.
"""
import einops
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch

from .gym import EAST, NORTH, SOUTH, WEST

# Defines convs that will compute vertical and horizontal matches.
# Shapes are [out_channels, in_channels, kernel_height, kernel_width].
# See `BatchedEternityEnv.matches` for more information.
# Don't forget that the y-axis is reversed!!
HORIZONTAL_CONV = torch.zeros((1, 4, 2, 1))
HORIZONTAL_CONV[0, SOUTH, 1, 0] = 1
HORIZONTAL_CONV[0, NORTH, 0, 0] = -1

VERTICAL_CONV = torch.zeros((1, 4, 1, 2))
VERTICAL_CONV[0, EAST, 0, 0] = 1
VERTICAL_CONV[0, WEST, 0, 1] = -1

# Convs to detect the 0-0 matches.
HORIZONTAL_ZERO_CONV = torch.zeros((1, 4, 2, 1))
HORIZONTAL_ZERO_CONV[0, SOUTH, 1, 0] = 1
HORIZONTAL_ZERO_CONV[0, NORTH, 0, 0] = 1

VERTICAL_ZERO_CONV = torch.zeros((1, 4, 1, 2))
VERTICAL_ZERO_CONV[0, EAST, 0, 0] = 1
VERTICAL_ZERO_CONV[0, WEST, 0, 1] = 1


class BatchedEternityEnv(gym.Env):
    """A batched version of the environment.
    All computations are done on GPU using torch tensors
    instead of numpy arrays.
    """

    metadata = {"render.modes": ["computer"]}

    def __init__(self, batch_instances: torch.LongTensor, seed: int = 0):
        """Initialize the environment.

        ---
        Args:
            batch_instances: The instances of this environment.
                Shape of [batch_size, 4, size, size].
            seed: The seed for the random number generator.
        """
        assert len(batch_instances.shape) == 4, "Tensor must have 4 dimensions."
        assert batch_instances.shape[1] == 4, "The pieces must have 4 sides."
        assert (
            batch_instances.shape[2] == batch_instances.shape[3]
        ), "Instances are not squares."
        assert torch.all(batch_instances >= 0), "Classes must be positives."

        super().__init__()
        self.instances = batch_instances
        self.rng = torch.Generator().manual_seed(seed)

        # Instances infos.
        self.size = self.instances.shape[-1]
        self.n_pieces = self.size * self.size
        self.n_class = self.instances.max().cpu().item()
        self.max_steps = self.n_pieces * 4
        self.best_matches = 2 * self.size * (self.size - 1)
        self.batch_size = self.instances.shape[0]

        # Dynamic infos.
        self.tot_steps = 0

        # Spaces
        # Those spaces do not take into account that
        # this env is a batch of multiple instances.
        self.action_space = spaces.MultiDiscrete(
            [
                self.n_pieces,  # Tile id to swap.
                self.n_pieces,  # Tile id to swap.
                4,  # How much rolls for the first tile.
                4,  # How much rolls for the first tile.
            ]
        )
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.instances.shape[1:], dtype=np.uint8
        )

    def reset(self):
        """Reset the environment.

        Scrambles the instances and reset their infos.
        """
        self.scramble_instances()
        self.tot_steps = 0

        return self.render()

    @property
    def matches(self) -> torch.Tensor:
        """The number of matches for each instance.
        Uses convolutions to vectorize the computations.

        The main idea is to compute the sum $class_id_1 - class_id_2$,
        where "1" and "2" represents two neighbour tiles.
        If this sum equals to 0, then the class_id are the same.

        We still have to make sure that we're not computing 0-0 matchings,
        so we also compute $class_id_1 + class_id_2$ and check if this is equal
        to 0 (which would mean that both class_id are equal to 0).
        """
        n_matches = torch.zeros(self.batch_size)

        for conv in [HORIZONTAL_CONV, VERTICAL_CONV]:
            res = torch.conv2d(self.instances.float(), conv)
            n_matches += (res == 0).float().flatten(start_dim=1).sum(dim=1)

        # Remove the 0-0 matches from the count.
        for conv in [HORIZONTAL_ZERO_CONV, VERTICAL_ZERO_CONV]:
            res = torch.conv2d(self.instances.float(), conv)
            n_matches -= (res == 0).float().flatten(start_dim=1).sum(dim=1)

        return n_matches.long()

    def scramble_instances(self):
        """Scrambles the instances to start from a new valid configuration."""
        # Scrambles the tiles.
        self.instances = einops.rearrange(
            self.instances, "b c h w -> b (h w) c", w=self.size
        )  # Shape of [batch_size, n_pieces, 4].
        permutations = torch.stack(
            [
                torch.randperm(self.n_pieces, generator=self.rng)
                for _ in range(self.batch_size)
            ]
        )  # Shape of [batch_size, n_pieces].
        # Permute tiles according to `permutations`.
        for instance_id in range(self.batch_size):
            self.instances[instance_id] = self.instances[instance_id][
                permutations[instance_id]
            ]

        # Randomly rolls the tiles.
        # Note: we can't really make it in a vectorized way.
        # A way to do it would be to use `torch.vmap`, which is still in beta.
        rolls = torch.randint(
            low=0, high=4, size=(self.batch_size, self.n_pieces), generator=self.rng
        )
        for instance_id in range(self.batch_size):
            for piece_id in range(self.n_pieces):
                piece = self.instances[instance_id, piece_id]
                piece = torch.roll(piece, shifts=rolls[instance_id, piece_id].item())
                self.instances[instance_id, piece_id] = piece
        self.instances = einops.rearrange(
            self.instances, "b (h w) c -> b c h w", w=self.size
        )

    def render(self, mode: str = "computer") -> torch.Tensor:
        """Render the environment.

        ---
        Args:
            mode: The rendering type.

        ---
        Returns:
            The observation of shape [batch_size, 4, size, size].
        """
        match mode:
            case "computer":
                return self.instances
            case _:
                raise RuntimeError(f"Unknown rendering type: {mode}.")
