import numpy as np
import pytest

from ..environment.gym import EternityEnv
from .data_generation import generate_perfect_instance, generate_sample


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
    "size, n_classes, n_steps, seed",
    [
        (4, 4, 6, 0),
        (4, 4, 6, 1),
        (8, 12, 10, 0),
    ],
)
def test_generate_sample(size: int, n_classes: int, n_steps: int, seed: int):
    """Test that the generated sample is valid.
    All actions should reproduce the next instance.
    All actions should lead to the best possible score.
    """
    instances, actions = generate_sample(size, n_classes, n_steps, seed)
    env = EternityEnv(instance=instances[0])

    for instance, action in zip(instances[1:], actions[:-1]):
        env.step(action)
        assert np.all(env.render() == instance)

    env.step(actions[-1])
    assert env.count_matches() == env.best_matches
