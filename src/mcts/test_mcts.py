from itertools import product
import pytest
import torch

from .soft import SoftMCTS


@pytest.mark.parametrize(
    "input_tensor, actions, to_add",
    [
        (
            torch.randn(4, 4, 4, 2, 2),
            torch.LongTensor(
                [
                    [3, 2, 0, 1],
                    [3, 3, 0, 1],
                    [3, 3, 0, 1],
                    [3, 3, 0, 1],
                ]
            ),
            torch.randn(4),
        ),
        (
            torch.randn(4, 4, 4, 2, 2),
            torch.LongTensor(
                [
                    [3, 2, 0, 1],
                    [3, 3, 0, 1],
                    [3, 3, 0, 1],
                    [3, 3, 0, 1],
                ]
            ),
            1,
        ),
        (
            torch.randn(4, 8, 2, 5, 8),
            torch.LongTensor(
                [
                    [6, 0, 1, 7],
                    [3, 1, 2, 1],
                    [4, 1, 4, 0],
                    [2, 0, 0, 1],
                ]
            ),
            torch.randn(4),
        ),
    ],
)
def test_batched_add(
    input_tensor: torch.Tensor, actions: torch.Tensor, to_add: torch.Tensor | float
):
    true_output = input_tensor.detach()
    for sample_id in range(input_tensor.shape[0]):
        action = actions[sample_id]
        el = to_add[sample_id] if type(to_add) is torch.Tensor else to_add
        true_output[sample_id, action[0], action[1], action[2], action[3]] += el

    output = SoftMCTS.batched_add(input_tensor, actions, to_add)
    assert torch.all(output == true_output)


@pytest.mark.parametrize(
    "scores",
    [
        torch.randn(12, 4, 4, 2, 2),
        torch.randn(12, 8, 2, 5, 8),
        torch.randn(2, 4, 4, 2, 2),
    ],
)
def test_best_actions(scores: torch.Tensor):
    output = SoftMCTS.best_actions(scores)
    true_output = []
    for batch_id in range(scores.shape[0]):
        best_actions = None
        best_score = float("-inf")
        for action_1, action_2, action_3, action_4 in product(
            *[range(scores.shape[i]) for i in range(1, 5)]
        ):
            score = scores[batch_id, action_1, action_2, action_3, action_4]
            if score > best_score:
                best_score = score
                best_actions = (action_1, action_2, action_3, action_4)

        true_output.append(torch.LongTensor(best_actions))

    true_output = torch.stack(true_output, dim=0)
    assert torch.all(true_output == output)
