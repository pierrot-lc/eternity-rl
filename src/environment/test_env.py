from itertools import product
from pathlib import Path

import einops
import pytest
import torch

from .gym import (
    EAST,
    ENV_DIR,
    ENV_ORDERED,
    NORTH,
    SOUTH,
    WEST,
    EternityEnv,
    next_instance,
    read_instance_file,
)


def test_read_instance():
    instance = read_instance_file(Path("instances/eternity_A.txt"))
    real = torch.LongTensor(
        [
            [
                [0, 0, 0, 0],
                [2, 6, 4, 4],
                [12, 8, 8, 5],
                [6, 11, 4, 1],
            ],
            [
                [1, 8, 3, 0],
                [9, 4, 10, 0],
                [11, 7, 8, 0],
                [0, 3, 7, 0],
            ],
            [
                [2, 6, 4, 4],
                [12, 8, 8, 5],
                [3, 11, 4, 1],
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 8, 3],
                [0, 9, 4, 10],
                [0, 11, 7, 8],
                [3, 6, 3, 7],
            ],
        ]
    )
    assert torch.all(instance[:, 0, 0] == torch.LongTensor([0, 1, 2, 0]))
    assert torch.all(instance[:, 3, 3] == torch.LongTensor([1, 0, 0, 7]))
    assert torch.all(instance[:, 3, 2] == torch.LongTensor([4, 7, 0, 3]))
    assert torch.all(instance == real)


def test_env_sizes():
    previous_size = 0
    for path in ENV_ORDERED:
        size = read_instance_file(ENV_DIR / path).shape[-1]

        assert previous_size < size
        previous_size = size


def test_matches_tiles():
    def count_tile_matches(instance: torch.Tensor, x: int, y: int) -> int:
        matches = 0
        tile = instance[:, y, x]
        size = instance.shape[-1]

        tile_sides = [NORTH, EAST, SOUTH, WEST]
        other_sides = [SOUTH, WEST, NORTH, EAST]
        other_coords = [
            (y + 1, x),
            (y, x + 1),
            (y - 1, x),
            (y, x - 1),
        ]

        for side_t, side_o, coords_o in zip(tile_sides, other_sides, other_coords):
            if not (0 <= coords_o[0] < size) or not (0 <= coords_o[1] < size):
                continue  # Those coords are outside the square.

            tile_class = tile[side_t]
            other_class = instance[side_o, coords_o[0], coords_o[1]]

            # Do not count walls as matches.
            matches += int(tile_class == other_class != 0)  # Ignore walls.

        return matches

    def count_matches(instance: torch.Tensor) -> int:
        # Count all matches between each tile.
        size = instance.shape[-1]
        matches = sum(
            count_tile_matches(instance, x, y)
            for x, y in product(range(size), range(size))
        )
        matches = matches // 2  # Sides have all been checked twice.
        return matches

    env = EternityEnv.from_file(Path("./instances/eternity_A.txt"), 10, "cpu")
    assert torch.all(env.matches == 12)
    for _ in range(10):
        env.reset()
        matches = env.matches
        for instance_id, instance in enumerate(env.instances):
            assert count_matches(instance) == matches[instance_id]


@pytest.mark.parametrize(
    "instance_path",
    [
        "eternity_trivial_A.txt",
        "eternity_trivial_B.txt",
        "eternity_A.txt",
    ],
)
def test_batch_scramble(instance_path: str):
    env = EternityEnv.from_file(ENV_DIR / instance_path, 10, "cpu")
    reference = env.instances[0].clone()
    instance_ids = torch.randperm(env.batch_size)[: env.batch_size // 2]
    env.scramble_instances(instance_ids)

    def compare_pieces(piece_1: torch.Tensor, piece_2: torch.Tensor) -> bool:
        """True if pieces are equal rollwise."""
        for shifts in range(4):
            rolled_piece = torch.roll(piece_2, shifts)
            if torch.all(piece_1 == rolled_piece):
                return True
        return False

    def compare_instances(instance_1: torch.Tensor, instance_2: torch.Tensor) -> bool:
        """True if all pieces of both instances are found."""
        pieces_1 = list(einops.rearrange(instance_1, "c h w -> (h w) c"))
        pieces_2 = list(einops.rearrange(instance_2, "c h w -> (h w) c"))
        pieces_found = set()

        for piece_1 in pieces_1:
            for piece_id, piece_2 in enumerate(pieces_2):
                if compare_pieces(piece_1, piece_2) and piece_id not in pieces_found:
                    pieces_found.add(piece_id)
                    break

        return len(pieces_found) == len(pieces_2)

    for instance in env.instances:
        assert compare_instances(instance, reference)

    changed_instances = set(i.item() for i in instance_ids)
    for instance_id in range(env.batch_size):
        if instance_id in changed_instances:
            continue

        torch.all(env.instances[instance_id] == reference)


def test_batch_roll():
    input_tensor = torch.randn((10, 5))
    shifts = torch.randint(low=-15, high=15, size=(len(input_tensor),))
    rolled_tensor = EternityEnv.batched_roll(input_tensor, shifts)

    for inpt, shift, rolled in zip(input_tensor, shifts, rolled_tensor):
        assert torch.all(torch.roll(inpt, shift.item()) == rolled)


@pytest.mark.parametrize(
    "instance_path",
    [
        "eternity_trivial_A.txt",
        "eternity_trivial_B.txt",
        "eternity_A.txt",
    ],
)
def test_batch_roll_action(instance_path):
    env = EternityEnv.from_file(ENV_DIR / instance_path, 10, "cpu")
    instance_reference = env.instances[0].clone()
    tile_ids = torch.randint(low=0, high=env.n_pieces, size=(env.batch_size,))
    shifts = torch.randint(low=0, high=4, size=(env.batch_size,))
    env.roll_tiles(tile_ids, shifts)

    for instance_rolled, tile_id, shift in zip(env.instances, tile_ids, shifts):
        coords = (tile_id.item() // env.board_size, tile_id.item() % env.board_size)
        instance_copy = instance_reference.clone()
        instance_copy[:, coords[0], coords[1]] = torch.roll(
            instance_copy[:, coords[0], coords[1]], shift.item()
        )

        assert torch.all(instance_copy == instance_rolled)


@pytest.mark.parametrize(
    "instance_path",
    [
        "eternity_trivial_A.txt",
        "eternity_trivial_B.txt",
        "eternity_A.txt",
    ],
)
def test_batch_swap_action(instance_path):
    env = EternityEnv.from_file(ENV_DIR / instance_path, 10, "cpu")
    instance_reference = env.instances[0].clone()
    tile_ids_1 = torch.randint(low=0, high=env.n_pieces, size=(env.batch_size,))
    tile_ids_2 = torch.randint(low=0, high=env.n_pieces, size=(env.batch_size,))
    env.swap_tiles(tile_ids_1, tile_ids_2)

    for instance_swapped, tile_id_1, tile_id_2 in zip(
        env.instances, tile_ids_1, tile_ids_2
    ):
        coords = [
            (c.item() // env.board_size, c.item() % env.board_size)
            for c in [tile_id_1, tile_id_2]
        ]
        instance_copy = instance_reference.clone()
        tile = instance_copy[:, coords[0][0], coords[0][1]].clone()
        instance_copy[:, coords[0][0], coords[0][1]] = instance_copy[
            :, coords[1][0], coords[1][1]
        ]
        instance_copy[:, coords[1][0], coords[1][1]] = tile

        assert torch.all(instance_copy == instance_swapped)


@pytest.mark.parametrize(
    "instance_1, instance_2",
    [
        (
            "eternity_trivial_A.txt",
            "eternity_trivial_B.txt",
        ),
        (
            "eternity_A.txt",
            "eternity_B.txt",
        ),
        (
            "eternity_C.txt",
            "eternity_D.txt",
        ),
        (
            "eternity_E.txt",
            "eternity_complet.txt",
        ),
    ],
)
def test_instance_upgrade(instance_1: str, instance_2: str):
    assert next_instance(ENV_DIR / instance_1) == ENV_DIR / instance_2
