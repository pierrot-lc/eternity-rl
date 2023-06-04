import pytest
import torch

from .reinforce import Reinforce
from .rollout_buffer import RolloutBuffer


@pytest.mark.parametrize(
    "rewards, masks, gamma, returns",
    [
        (
            torch.FloatTensor([[2, 1, 1]]),
            torch.BoolTensor([[True, True, True]]),
            1,
            torch.FloatTensor([[4, 2, 1]]),
        ),
        (
            torch.FloatTensor([[2, 1, 1]]),
            torch.BoolTensor([[True, True, False]]),
            1,
            torch.FloatTensor([[3, 1, 0]]),
        ),
        (
            torch.FloatTensor([[2, 1, 1]]),
            torch.BoolTensor([[True, True, True]]),
            0.9,
            torch.FloatTensor([[2 + 0.9 * (1 + 0.9), 1 + 0.9, 1]]),
        ),
        (
            torch.FloatTensor([[2, 1, 1]]),
            torch.BoolTensor([[True, True, False]]),
            0.9,
            torch.FloatTensor([[2 + 0.9 * 1, 1, 0]]),
        ),
        (
            torch.FloatTensor([[2, 1, 1], [3, 2, 3]]),
            torch.BoolTensor([[True, True, False], [True, False, False]]),
            0.9,
            torch.FloatTensor([[2 + 0.9 * 1, 1, 0], [3, 0, 0]]),
        ),
        (
            torch.FloatTensor([[0.15, 0.003, 0.32, 0.45]]),
            torch.BoolTensor([[True, True, True, True]]),
            0.9,
            torch.FloatTensor([[0.73995, 0.6555, 0.7250, 0.45]]),
        ),
    ],
)
def test_cumulative_decay_return(
    rewards: torch.Tensor,
    masks: torch.Tensor,
    gamma: float,
    returns: torch.Tensor,
):
    assert torch.allclose(
        RolloutBuffer.cumulative_decay_return(rewards, masks, gamma), returns
    )
