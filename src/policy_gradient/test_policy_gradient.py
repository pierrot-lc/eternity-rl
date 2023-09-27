import pytest
import torch
from tensordict import TensorDict

from .rollout import cumulative_decay_return, split_reset_rollouts


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
    assert torch.allclose(cumulative_decay_return(rewards, masks, gamma), returns)


@pytest.mark.parametrize(
    "traces,split_traces",
    [
        (
            {
                "dones": torch.BoolTensor([[False, False, True]]),
                "truncated": torch.BoolTensor([[False, False, False]]),
            },
            {
                "dones": torch.BoolTensor([[False, False, True]]),
                "truncated": torch.BoolTensor([[False, False, False]]),
                "masks": torch.BoolTensor([[True, True, True]]),
            },
        ),
        (
            {
                "dones": torch.BoolTensor([[False, False, True]]),
                "truncated": torch.BoolTensor([[False, False, True]]),
            },
            {
                "dones": torch.BoolTensor([[False, False, True]]),
                "truncated": torch.BoolTensor([[False, False, True]]),
                "masks": torch.BoolTensor([[True, True, True]]),
            },
        ),
        (
            {
                "dones": torch.BoolTensor([[False, False, False]]),
                "truncated": torch.BoolTensor([[True, False, False]]),
            },
            {
                "dones": torch.BoolTensor(
                    [
                        [False, False, False],
                        [False, False, False],
                    ]
                ),
                "truncated": torch.BoolTensor(
                    [
                        [True, False, False],
                        [False, False, False],
                    ]
                ),
                "masks": torch.BoolTensor(
                    [
                        [True, False, False],
                        [True, True, False],
                    ]
                ),
            },
        ),
        (
            {
                "dones": torch.BoolTensor([[False, False, False]]),
                "truncated": torch.BoolTensor([[True, False, False]]),
                "rewards": torch.FloatTensor([[1.0, 3.0, 2.0]]),
                "actions": torch.LongTensor([[[0, 1], [3, 4], [1, 4]]]),
            },
            {
                "dones": torch.BoolTensor(
                    [
                        [False, False, False],
                        [False, False, False],
                    ]
                ),
                "truncated": torch.BoolTensor(
                    [
                        [True, False, False],
                        [False, False, False],
                    ]
                ),
                "rewards": torch.FloatTensor(
                    [
                        [1.0, 0.0, 0.0],
                        [3.0, 2.0, 0.0],
                    ]
                ),
                "actions": torch.LongTensor(
                    [
                        [[0, 1], [0, 0], [0, 0]],
                        [[3, 4], [1, 4], [0, 0]],
                    ]
                ),
                "masks": torch.BoolTensor(
                    [
                        [True, False, False],
                        [True, True, False],
                    ]
                ),
            },
        ),
    ],
)
def test_split_reset_rollouts(traces: dict, split_traces: dict):
    traces = TensorDict(traces, batch_size=traces["dones"].shape[0], device="cpu")
    traces = split_reset_rollouts(traces)

    for name in split_traces.keys():
        assert torch.all(split_traces[name] == traces[name])

    split_traces_keys = set(split_traces.keys())
    traces_keys = set(traces.keys())
    assert split_traces_keys == traces_keys
