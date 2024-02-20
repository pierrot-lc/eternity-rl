"""Make sure the GNN is SO4-equivariant."""
import einops
import pytest
import torch
import torch.nn as nn

from environment import EternityEnv
from environment.constants import EAST, NORTH, SOUTH, WEST

from .gnn import GNNBackbone


def instanciate_boards(instance_path: str) -> torch.Tensor:
    env = EternityEnv.from_file(instance_path, episode_length=10, batch_size=5)
    boards = env.instances
    return einops.rearrange(boards, "b s h w -> b h w s")


def rotate_boards(boards: torch.Tensor) -> torch.Tensor:
    """Rotate the board to 90 degres.

    ---
    Args:
        boards: The boards to rotate.
            Shape of [batch_size, board_height, board_width, N_SIDES].

    ---
    Returns:
        The rotated boards.
            Shape of [batch_size, board_width, board_height, N_SIDES].
    """
    rotated = torch.rot90(boards, 1, (1, 2))

    rotated_ = rotated.clone()
    rotated[:, :, :, NORTH] = rotated_[:, :, :, EAST]
    rotated[:, :, :, EAST] = rotated_[:, :, :, SOUTH]
    rotated[:, :, :, SOUTH] = rotated_[:, :, :, WEST]
    rotated[:, :, :, WEST] = rotated_[:, :, :, NORTH]

    return rotated


@pytest.mark.parametrize(
    "boards, rotated",
    [
        (
            torch.LongTensor(
                [
                    [
                        [[0, 1, 2, 3], [5, 6, 7, 8]],
                        [[10, 11, 12, 13], [-1, -2, -3, -4]],
                    ],
                ]
            ),
            torch.LongTensor(
                [
                    [
                        [[6, 7, 8, 5], [-2, -3, -4, -1]],
                        [[1, 2, 3, 0], [11, 12, 13, 10]],
                    ],
                ]
            ),
        ),
        (
            torch.LongTensor(
                [
                    [
                        [[0, 1, 2, 3], [5, 6, 7, 8]],
                        [[10, 11, 12, 13], [-1, -2, -3, -4]],
                    ],
                    [
                        [[0, 1, 2, 3], [3, 4, 3, 9]],
                        [[10, 11, 12, 13], [-1, -2, -3, -4]],
                    ],
                ]
            ),
            torch.LongTensor(
                [
                    [
                        [[6, 7, 8, 5], [-2, -3, -4, -1]],
                        [[1, 2, 3, 0], [11, 12, 13, 10]],
                    ],
                    [
                        [[4, 3, 9, 3], [-2, -3, -4, -1]],
                        [[1, 2, 3, 0], [11, 12, 13, 10]],
                    ],
                ]
            ),
        ),
    ],
)
def test_rotate(boards: torch.Tensor, rotated: torch.Tensor):
    assert torch.all(rotate_boards(boards) == rotated)


@torch.inference_mode()
@pytest.mark.parametrize(
    "boards, model",
    [
        (
            instanciate_boards("./instances/eternity_A.txt"),
            nn.Sequential(
                einops.layers.torch.Rearrange("b s h w -> (h w) b s"),
                nn.Embedding(20, 16),
                einops.layers.torch.Reduce(
                    "(h w) b s e -> (h w) b e", reduction="mean", h=4
                ),
            ),
        ),  # One trivial example.
        (
            instanciate_boards("./instances/eternity_A.txt"),
            GNNBackbone(32, 2),
        ),
    ],
)
def test_rotation_equivariant(boards: torch.Tensor, model: nn.Module):
    _, height, width, _ = boards.shape
    references = model(einops.rearrange(boards, "b h w s -> b s h w"))
    references = einops.rearrange(references, "(h w) b e -> b h w e", h=height, w=width)

    rotated_references = references.clone()
    rotated_boards = boards.clone()
    for _ in range(4):
        rotated_references = torch.rot90(rotated_references, k=1, dims=(1, 2))
        rotated_boards = rotate_boards(rotated_boards)

        model_rotated = model(einops.rearrange(rotated_boards, "b h w s -> b s h w"))
        model_rotated = einops.rearrange(
            model_rotated, "(h w) b e -> b h w e", h=height, w=width
        )
        assert torch.allclose(model_rotated, rotated_references)

    assert torch.all(rotated_references == references)
    assert torch.all(rotated_boards == boards)
