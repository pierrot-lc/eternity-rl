"""A batched version of the environment.
All actions are made on the batch.
"""
from pathlib import Path
from typing import Any

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch
from einops import rearrange, repeat

from .draw import draw_instance

ENV_DIR = Path("./instances")
ENV_ORDERED = [
    "eternity_trivial_A.txt",
    "eternity_trivial_B.txt",
    "eternity_A.txt",
    "eternity_B.txt",
    "eternity_C.txt",
    "eternity_D.txt",
    "eternity_E.txt",
    "eternity_complet.txt",
]

# Ids of the sides of the tiles.
# The y-axis has its origin at the bottom.
# The x-axis has its origin at the left.
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

ORIGIN_NORTH = 0
ORIGIN_SOUTH = 1
ORIGIN_WEST = 2
ORIGIN_EAST = 3

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


class EternityEnv(gym.Env):
    """A batched version of the environment.
    All computations are done on GPU using torch tensors
    instead of numpy arrays.
    """

    metadata = {"render.modes": ["computer"]}

    def __init__(
        self,
        instances: torch.Tensor,
        reward_type: str,
        max_steps: int,
        device: str,
        seed: int = 0,
    ):
        """Initialize the environment.

        ---
        Args:
            instances: The instances of this environment.
                Long tensor of shape of [batch_size, 4, size, size].
            reward_type: The type of reward to use.
                Can be either "delta" or "win".
            max_steps: The maximum number of steps.
            device: The device to use.
            seed: The seed for the random number generator.
        """
        assert len(instances.shape) == 4, "Tensor must have 4 dimensions."
        assert instances.shape[1] == 4, "The pieces must have 4 sides."
        assert instances.shape[2] == instances.shape[3], "Instances are not squares."
        assert torch.all(instances >= 0), "Classes must be positives."
        assert reward_type in ["delta", "win"], "Unknown reward type."

        super().__init__()
        self.instances = instances.to(device)
        self.reward_type = reward_type
        self.device = device
        self.rng = torch.Generator(device).manual_seed(seed)

        # Instances infos.
        self.board_size = self.instances.shape[-1]
        self.n_pieces = self.board_size * self.board_size
        self.n_classes = int(self.instances.max().cpu().item() + 1)
        self.max_steps = max_steps
        self.best_matches = 2 * self.board_size * (self.board_size - 1)
        self.batch_size = self.instances.shape[0]

        # Dynamic infos.
        self.step_id = 0
        self.truncated = False
        self.terminated = torch.zeros(self.batch_size, dtype=torch.bool, device=device)

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

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        Scrambles the instances and reset their infos.
        """
        self.scramble_instances()
        self.step_id = 0
        self.truncated = False
        self.terminated = torch.zeros(
            self.batch_size, dtype=torch.bool, device=self.device
        )

        return self.render(), dict()

    @torch.no_grad()
    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, dict[str, Any]]:
        """Do a batched step through all instances.

        ---
        Args:
            actions: Batch of actions to apply.
                Long tensor of shape of [batch_size, actions]
                where actions are a tuple (tile_id_1, shift_1, tile_id_2, shift_2).

        ---
        Returns:
            observations: The observation of the environments.
                Shape of [batch_size, 4, size, size].
            rewards: The reward of the environments.
                Shape of [batch_size,].
            terminated: Whether the environments are terminated (won).
                Shape of [batch_size,].
            truncated: Whether the environments are truncated (max steps reached).
            infos: Additional infos.
                - `just_won`: Whether the environments has just been won.
        """
        infos = dict()
        self.step_id += 1
        tiles_id_1, tiles_id_2 = actions[:, 0], actions[:, 2]
        shifts_1, shifts_2 = actions[:, 1], actions[:, 3]

        # Save the previous matches (useful for the `delta` reward).
        previous_matches = self.matches
        previous_terminated = self.terminated.clone()

        self.roll_tiles(tiles_id_1, shifts_1)
        self.roll_tiles(tiles_id_2, shifts_2)
        self.swap_tiles(tiles_id_1, tiles_id_2)

        # New matches counts.
        matches = self.matches

        # Maintain the previous terminated states.
        self.terminated |= matches == self.best_matches
        self.truncated = self.step_id >= self.max_steps
        infos["just_won"] = self.terminated & ~previous_terminated

        match self.reward_type:
            case "win":
                # Only give a reward at the end of the episode.
                # Either when the environment is done or if the episode is truncated.
                if not self.truncated:
                    rewards = matches * infos["just_won"] / self.best_matches
                else:
                    rewards = matches * ~self.terminated / self.best_matches
            case "delta":
                # Give a reward at each step.
                rewards = (matches - previous_matches) / self.best_matches
                rewards = rewards * ~previous_terminated
            case _:
                raise ValueError(f"Unknown reward type {self.reward_type}.")

        return self.render(), rewards, self.terminated, self.truncated, infos

    def roll_tiles(self, tile_ids: torch.Tensor, shifts: torch.Tensor):
        """Rolls tiles at the given ids for the given shifts.
        It actually shifts all tiles, but with 0-shifts
        except for the pointed tile ids.

        ---
        Args:
            tile_ids: The id of the tiles to roll.
                Shape of [batch_size,].
            shifts: The number of shifts for each tile.
                Shape of [batch_size,].
        """
        total_shifts = torch.zeros(
            self.batch_size * self.n_pieces, dtype=torch.long, device=self.device
        )
        offsets = torch.arange(
            0, self.batch_size * self.n_pieces, self.n_pieces, device=self.device
        )
        total_shifts[tile_ids + offsets] = shifts

        self.instances = rearrange(self.instances, "b c h w -> (b h w) c")
        self.instances = self.batched_roll(self.instances, total_shifts)
        self.instances = rearrange(
            self.instances,
            "(b h w) c -> b c h w",
            b=self.batch_size,
            h=self.board_size,
            w=self.board_size,
        )

    def swap_tiles(self, tile_ids_1: torch.Tensor, tile_ids_2: torch.Tensor):
        """Swap two tiles in each element of the batch.

        ---
        Args:
            tile_ids_1: The id of the first tiles to swap.
                Shape of [batch_size,].
            tile_ids_2: The id of the second tiles to swap.
                Shape of [batch_size,].
        """
        offsets = torch.arange(
            0, self.batch_size * self.n_pieces, self.n_pieces, device=self.device
        )
        tile_ids_1 = tile_ids_1 + offsets
        tile_ids_2 = tile_ids_2 + offsets
        self.instances = rearrange(self.instances, "b c h w -> (b h w) c")
        self.instances[tile_ids_1], self.instances[tile_ids_2] = (
            self.instances[tile_ids_2],
            self.instances[tile_ids_1],
        )
        self.instances = rearrange(
            self.instances,
            "(b h w) c -> b c h w",
            b=self.batch_size,
            h=self.board_size,
            w=self.board_size,
        )

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
        n_matches = torch.zeros(self.batch_size, device=self.device)

        for conv in [HORIZONTAL_CONV, VERTICAL_CONV]:
            res = torch.conv2d(self.instances.float(), conv.to(self.device))
            n_matches += (res == 0).float().flatten(start_dim=1).sum(dim=1)

        # Remove the 0-0 matches from the count.
        for conv in [HORIZONTAL_ZERO_CONV, VERTICAL_ZERO_CONV]:
            res = torch.conv2d(self.instances.float(), conv.to(self.device))
            n_matches -= (res == 0).float().flatten(start_dim=1).sum(dim=1)

        return n_matches.long()

    def scramble_instances(self):
        """Scrambles the instances to start from a new valid configuration."""
        # Scrambles the tiles.
        self.instances = rearrange(
            self.instances, "b c h w -> b (h w) c", w=self.board_size
        )  # Shape of [batch_size, n_pieces, 4].
        permutations = torch.stack(
            [
                torch.randperm(self.n_pieces, generator=self.rng, device=self.device)
                for _ in range(self.batch_size)
            ]
        )  # Shape of [batch_size, n_pieces].
        # Permute tiles according to `permutations`.
        for instance_id in range(self.batch_size):
            self.instances[instance_id] = self.instances[instance_id][
                permutations[instance_id]
            ]

        # Randomly rolls the tiles.
        shifts = torch.randint(
            low=0,
            high=4,
            size=(self.batch_size * self.n_pieces,),
            generator=self.rng,
            device=self.device,
        )
        self.instances = rearrange(self.instances, "b p c -> (b p) c")
        self.instances = self.batched_roll(self.instances, shifts)
        self.instances = rearrange(
            self.instances,
            "(b h w) c -> b c h w",
            b=self.batch_size,
            h=self.board_size,
            w=self.board_size,
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

    def save_best_env(self, filepath: Path | str):
        """Render the best environment and save it on disk."""
        best_env = self.matches.argmax()
        best_score = self.matches[best_env].cpu().item()
        board = self.instances[best_env].cpu().numpy()
        draw_instance(board, best_score, filepath)

    @staticmethod
    def batched_roll(input_tensor: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        """Batched version of `torch.roll`.
        It applies a circular shifts to the last dimension of the tensor.

        ---
        Args:
            input_tensor: The tensor to roll.
                Shape of [batch_size, hidden_size].
            shifts: The number of shifts for each element of the batch.
                Shape of [batch_size,].

        ---
        Returns:
            The rolled tensor.
        """
        batch_size, hidden_size = input_tensor.shape
        # In the right range of values (circular shift).
        shifts = shifts % hidden_size
        # To match the same direction as `torch.roll`.
        shifts = hidden_size - shifts

        # Compute the indices that will select the right part of the extended tensor.
        offsets = torch.arange(hidden_size, device=input_tensor.device)
        offsets = repeat(offsets, "h -> b h", b=batch_size)
        select_indices = shifts.unsqueeze(1) + offsets

        # Extend the input tensor with circular padding.
        input_tensor = torch.concat((input_tensor, input_tensor), dim=1)

        rolled_tensor = torch.gather(input_tensor, dim=1, index=select_indices)
        return rolled_tensor

    @classmethod
    def from_file(
        cls,
        instance_path: Path,
        batch_size: int,
        reward_type: str,
        max_steps: int,
        device: str,
        seed: int = 0,
    ):
        instance = read_instance_file(instance_path)
        instances = repeat(instance, "c h w -> b c h w", b=batch_size)
        return cls(instances, reward_type, max_steps, device, seed)


def read_instance_file(instance_path: Path | str) -> torch.Tensor:
    """Read the instance file and return a matrix containing the ordered elements.

    ---
    Returns:
        data: Matrix containing each tile.
            Shape of [4, y-axis, x-axis].
            Origin is in the bottom left corner.
    """
    with open(instance_path, "r") as instance_file:
        instance_file = iter(instance_file)
        n = int(next(instance_file))
        data = torch.zeros((4, n * n), dtype=torch.long)

        # Read all tiles.
        for element_id, element in enumerate(instance_file):
            class_ids = element.split(" ")
            class_ids = [int(c) for c in class_ids]

            # Put the classes in the right order according to our convention.
            class_ids[NORTH], class_ids[EAST], class_ids[SOUTH], class_ids[WEST] = (
                class_ids[ORIGIN_NORTH],
                class_ids[ORIGIN_EAST],
                class_ids[ORIGIN_SOUTH],
                class_ids[ORIGIN_WEST],
            )

            data[:, element_id] = torch.tensor(class_ids)

    data = rearrange(data, "c (h w) -> c h w", h=n, w=n)
    return data


def next_instance(instance_path: Path) -> Path:
    instance_id = [
        i + 1 for i, path in enumerate(ENV_ORDERED) if ENV_DIR / path == instance_path
    ][0]
    instance_id = min(instance_id, len(ENV_ORDERED) - 1)
    return ENV_DIR / ENV_ORDERED[instance_id]
