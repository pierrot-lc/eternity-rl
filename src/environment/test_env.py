from pathlib import Path

import numpy as np
import pytest

from .data_generation import generate_perfect_instance, generate_sample
from .gym import ENV_DIR, ENV_ORDERED, EternityEnv, next_instance, read_instance_file


def test_read_instance():
    instance = read_instance_file(Path("instances/eternity_A.txt"))
    real = np.array(
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
    assert np.all(instance[:, 0, 0] == np.array([0, 1, 2, 0]))
    assert np.all(instance[:, 3, 3] == np.array([1, 0, 0, 7]))
    assert np.all(instance[:, 3, 2] == np.array([4, 7, 0, 3]))
    assert np.all(instance == real)


def test_env_sizes():
    previous_size = 0
    for path in ENV_ORDERED:
        size = read_instance_file(ENV_DIR / path).shape[-1]

        assert previous_size < size
        previous_size = size


def test_matches_tiles():
    env = EternityEnv(instance_path=Path("instances/eternity_A.txt"))
    assert env.count_matches() == 19
    assert env.count_tile_matches((0, 0)) == 1
    assert env.count_tile_matches((1, 2)) == 3

    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(10):
        action = np.array(
            [
                rng.integers(0, env.n_pieces - 1),
                rng.integers(0, 4),
                rng.integers(0, env.n_pieces - 1),
                rng.integers(0, 4),
            ]
        )
        _, _, done, _ = env.step(action)
        if done:
            return

        assert env.count_matches() == env.matches


@pytest.mark.parametrize(
    "coords_1, coords_2",
    [
        ((0, 0), (3, 2)),
        ((0, 0), (2, 1)),
        ((0, 0), (0, 0)),
        ((1, 2), (2, 1)),
    ],
)
def test_swap_tiles(coords_1: tuple[int, int], coords_2: tuple[int, int]):
    env = EternityEnv(instance_path=Path("instances/eternity_A.txt"))

    tile_1 = env.instance[:, coords_1[0], coords_1[1]].copy()
    tile_2 = env.instance[:, coords_2[0], coords_2[1]].copy()
    env.swap_tiles(coords_1, coords_2)

    assert np.all(tile_1 == env.instance[:, coords_2[0], coords_2[1]])
    assert np.all(tile_2 == env.instance[:, coords_1[0], coords_1[1]])


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


@pytest.mark.parametrize(
    "coords, roll_value", [((0, 0), 1), ((3, 2), -1), ((1, 1), 10)]
)
def test_roll_tiles(coords: tuple[int, int], roll_value: int):
    env = EternityEnv(instance_path=Path("instances/eternity_A.txt"))
    tile = env.instance[:, coords[0], coords[1]].copy()

    env.roll_tile(coords, roll_value)
    assert np.all(env.instance[:, coords[0], coords[1]] == np.roll(tile, roll_value))


@pytest.mark.parametrize(
    "size, n_classes, seed",
    [
        (4, 4, 0),
        (4, 4, 1),
        (8, 12, 0),
    ],
)
def test_generate_perfect_instance(size: int, n_classes: int, seed: int):
    instance = generate_perfect_instance(size, n_classes, seed)
    assert instance.shape == (4, size, size)

    env = EternityEnv(instance=instance)
    assert env.count_matches() == env.best_matches


@pytest.mark.parametrize(
    "size, n_classes, seed",
    [
        (4, 4, 0),
        (4, 4, 1),
        (8, 12, 0),
    ],
)
def test_generate_sample(size: int, n_classes: int, seed: int):
    instance, actions = generate_sample(size, n_classes, seed)
    env = EternityEnv(instance=instance)
    for action in actions:
        env.step(action)

    assert env.count_matches() == env.best_matches
