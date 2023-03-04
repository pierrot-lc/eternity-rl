import os
from itertools import product
from pathlib import Path
from typing import Optional, Union

import einops
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

from .draw import draw_instance

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

WALL_ID = 0

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


class EternityEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array", "computer"]}

    def __init__(
        self,
        instance: Optional[np.ndarray] = None,
        instance_path: Optional[Union[Path, str]] = None,
        reward_type: str = "win_ratio",
        reward_penalty: float = 0.0,
        seed: int = 0,
    ):
        super().__init__()

        assert (instance is None) ^ (instance_path is None)

        self.rng = np.random.default_rng(seed)
        self.instance = instance if instance is not None else np.zeros((4, 0, 0))
        self.instance_path = instance_path
        if instance_path is not None:
            self.instance = read_instance_file(
                instance_path
            )  # Shape is [4, size, size].

        self.size = self.instance.shape[-1]
        self.n_class = self.instance.max() + 1
        self.n_pieces = self.size * self.size
        self.max_steps = self.n_pieces * 4
        self.best_matches = (
            2
            * self.size
            * (self.size - 1)  # Inner matches.
            # + 4 * self.size  # Walls matches at the borders.
        )
        self.matches = 0
        self.tot_steps = 0

        self.action_space = spaces.MultiDiscrete(
            [
                self.n_pieces,  # Tile id to swap
                self.n_pieces,  # Tile id to swap
                4,  # How much rolls for the first tile
                4,  # How much rolls for the second tile
            ]
        )

        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.render().shape, dtype=np.uint8
        )

        self.reward_type = reward_type
        self.reward_penalty = reward_penalty

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Swap the two chosen tiles and orient them in the best possible way.

        ---
        Args:
            action: Id of the tiles to swap and their rolling shift values.
                In the form of [tile_1_id, roll_1, tile_2_id, roll_2].

        ---
        Returns:
            observation: Array of observation, output of self.render.
            reward: Number of matching sides.
            done: Is there still sides that are not matched?
            info: -
        """
        previous_matches = self.matches

        rolls = [a for a in action[1::2]]
        coords = [(a // self.size, a % self.size) for a in action[::2]]

        # Roll tiles.
        self.roll_tile(coords[0], rolls[0])
        self.roll_tile(coords[1], rolls[1])

        # Swap tiles.
        self.swap_tiles(coords[0], coords[1])

        self.matches = self.count_matches()
        self.tot_steps += 1

        observation = self.render()
        win = self.matches == self.best_matches
        timeout = self.tot_steps == self.max_steps
        done = win or timeout

        match self.reward_type:
            case "win_ratio":
                reward = self.matches / self.best_matches if done else 0
            case "win":
                reward = int(win)
            case "delta":
                reward = (self.matches - previous_matches) / self.best_matches
            case "penalty":
                reward = 0
            case _:
                raise RuntimeError(f"Unknown reward type: {self.reward_type}.")

        reward -= self.reward_penalty  # Small penalty at each step.

        info = {
            "matches": self.matches,
            "ratio": self.matches / self.best_matches,
            "win": win,
        }

        if win and self.instance_path == os.path.join(ENV_DIR, "eternity_complet.txt"):
            self.render(mode="rgb_array", output_file=Path("solution.png"))

        return observation, reward, done, info

    def reset(
        self,
        instance: Optional[np.ndarray] = None,
        tot_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Scramble the tiles and randomly orient them."""
        if instance is None:
            # Randomly mix the tiles to create a new instance.
            instance = self.instance
            height = instance.shape[1]
            instance = einops.rearrange(instance, "c h w -> (h w) c")

            # Scramble the tiles.
            instance = self.rng.permutation(instance, axis=0)

            # Randomly orient the tiles.
            for tile_id, tile in enumerate(instance):
                shift_value = self.rng.integers(low=0, high=4)
                instance[tile_id] = np.roll(tile, shift_value)

            instance = einops.rearrange(instance, "(h w) c -> c h w", h=height)

        self.instance = instance
        self.matches = self.count_matches()
        self.tot_steps = tot_steps if tot_steps is not None else 0

        return self.render()

    def render(
        self, mode: str = "computer", output_file: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """Transform the instance into an observation.

        The observation is a map of shape [4, size, size].
        """
        match mode:
            case "computer":
                return self.instance
            case "rgb_array":
                return draw_instance(self.instance, self.count_matches(), output_file)
            case _:
                raise RuntimeError(f"Unknown rendering type: {mode}.")

    def count_tile_matches(self, coords: tuple[int, int]) -> int:
        """Count the matches a tile has with its neighbours.
        Walls are not counted as matches.
        """
        matches = 0
        tile = self.instance[:, coords[0], coords[1]]

        tile_sides = [NORTH, EAST, SOUTH, WEST]
        other_sides = [SOUTH, WEST, NORTH, EAST]
        other_coords = [
            (coords[0] + 1, coords[1]),  # (y, x)
            (coords[0], coords[1] + 1),
            (coords[0] - 1, coords[1]),
            (coords[0], coords[1] - 1),
        ]

        for side_t, side_o, coords_o in zip(tile_sides, other_sides, other_coords):
            if not (0 <= coords_o[0] < self.size) or not (0 <= coords_o[1] < self.size):
                continue  # Those coords are outside the square.

            tile_class = tile[side_t]
            other_class = self.instance[side_o, coords_o[0], coords_o[1]]

            # Do not count walls as matches.
            matches += int(tile_class == other_class != WALL_ID)

        return matches

    def count_walls_matches(self, coords: tuple[int, int]) -> int:
        """Count all walls matches for the given tile.
        A wall is considered a match if it is at the border of the map.
        """
        matches = 0
        tile = self.instance[:, coords[0], coords[1]]

        if coords[0] == 0 and tile[SOUTH] == WALL_ID:
            matches += 1
        if coords[0] == self.size - 1 and tile[NORTH] == WALL_ID:
            matches += 1
        if coords[1] == 0 and tile[WEST] == WALL_ID:
            matches += 1
        if coords[1] == self.size - 1 and tile[EAST] == WALL_ID:
            matches += 1

        return matches

    def count_matches(self) -> int:
        """Count all matches for the current state."""
        # Count all matches between each tile.
        matches = sum(
            self.count_tile_matches((y, x))
            for x, y in product(range(self.size), range(self.size))
        )
        matches = matches // 2  # Sides have all been checked twice.

        # Count all matches between walls and borders.
        # matches += sum(
        #     self.count_walls_matches((y, x))
        #     for x, y in product(range(self.size), range(self.size))
        # )

        return matches

    def swap_tiles(
        self, tile_1_coords: tuple[int, int], tile_2_coords: tuple[int, int]
    ):
        """Swap the two given tiles."""
        tile_1 = self.instance[:, tile_1_coords[0], tile_1_coords[1]].copy()
        tile_2 = self.instance[:, tile_2_coords[0], tile_2_coords[1]]

        self.instance[:, tile_1_coords[0], tile_1_coords[1]] = tile_2
        self.instance[:, tile_2_coords[0], tile_2_coords[1]] = tile_1

    def roll_tile(self, coords: tuple[int, int], shift_value: int):
        """Reorient the tile by doing a circular shift."""
        self.instance[:, coords[0], coords[1]] = np.roll(
            self.instance[:, coords[0], coords[1]], shift_value
        )

    def seed(self, seed: int):
        """Modify the seed."""
        self.rng = np.random.default_rng(seed)


def read_instance_file(instance_path: Union[Path, str]) -> np.ndarray:
    """Read the instance file and return a matrix containing the ordered elements.

    Output
    ------
        data: Matrix containing each tile.
            Shape of [4, y-axis, x-axis].
            Origin is in the bottom left corner.
    """
    with open(instance_path, "r") as instance_file:
        instance_file = iter(instance_file)

        n = int(next(instance_file))

        data = np.zeros((4, n * n), dtype=np.uint8)
        for element_id, element in enumerate(instance_file):
            class_ids = element.split(" ")
            class_ids = [int(c) for c in class_ids]
            data[:, element_id] = class_ids

    for tile_id in range(data.shape[-1]):
        tile = data[:, tile_id]
        tile[NORTH], tile[EAST], tile[SOUTH], tile[WEST] = (
            tile[ORIGIN_NORTH],
            tile[ORIGIN_EAST],
            tile[ORIGIN_SOUTH],
            tile[ORIGIN_WEST],
        )

    data = data.reshape((4, n, n))
    return data


def next_instance(instance_path: Path) -> Path:
    instance_id = [
        i + 1 for i, path in enumerate(ENV_ORDERED) if ENV_DIR / path == instance_path
    ][0]
    instance_id = min(instance_id, len(ENV_ORDERED) - 1)
    return ENV_DIR / ENV_ORDERED[instance_id]
