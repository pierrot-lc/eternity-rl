"""Make sure the GNN is SO4-equivariant."""
import einops
import pytest
import torch
import torch.nn as nn

from ..environment import EternityEnv
from ..environment.constants import EAST, NORTH, SOUTH, WEST

from .backbones.gnn import GNNBackbone
from .policy import Policy
from .critic import Critic


def instanciate_boards(instance_path: str) -> torch.Tensor:
    env = EternityEnv.from_file(instance_path, episode_length=10, batch_size=5)
    env.reset()
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
            instanciate_boards("./instances/eternity_trivial_A.txt"),
            GNNBackbone(32, 2),
        ),
        (
            instanciate_boards("./instances/eternity_A.txt"),
            GNNBackbone(32, 4),
        ),
    ],
)
def test_rotation_equivariant_backbone(boards: torch.Tensor, model: nn.Module):
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
        assert torch.allclose(
            model_rotated, rotated_references, rtol=1e-3, atol=1e-6
        ), f"Absolute mean error is {( model_rotated - rotated_references ).abs().mean()}"

    assert torch.all(rotated_references == references)
    assert torch.all(rotated_boards == boards)


@torch.inference_mode()
@pytest.mark.parametrize(
    "boards, policy, critic",
    [
        (
            instanciate_boards("./instances/eternity_trivial_A.txt"),
            Policy(32, 2, 3, 2, 0.0),
            Critic(32, 2, 3, 2, 0.0),
        ),
        (
            instanciate_boards("./instances/eternity_A.txt"),
            Policy(32, 2, 3, 2, 0.0),
            Critic(32, 2, 3, 2, 0.0),
        ),
    ],
)
def test_rotation_equivariant_model(
    boards: torch.Tensor, policy: Policy, critic: Critic
):
    batch_size, height, width, _ = boards.shape
    input_boards = einops.rearrange(boards, "b h w s -> b s h w")
    actions, logprobs, entropies = policy(input_boards, sampling_mode="greedy")
    values = critic(input_boards)

    tile_ids = torch.arange(start=0, end=height * width)
    tile_ids = einops.rearrange(tile_ids, "(h w) -> h w", h=height, w=width)
    tile_ids = einops.repeat(tile_ids, "h w -> b h w", b=batch_size)
    tile_ids = einops.rearrange(tile_ids, "b h w -> b (h w)")

    tile_ids_1 = tile_ids[torch.arange(batch_size), actions[:, 0]]
    tile_ids_2 = tile_ids[torch.arange(batch_size), actions[:, 1]]

    rotated_boards = boards.clone()
    for _ in range(4):
        rotated_boards = rotate_boards(rotated_boards)
        tile_ids = einops.rearrange(tile_ids, "b (h w) -> b h w", h=height)
        tile_ids = torch.rot90(tile_ids, k=1, dims=(1, 2))
        tile_ids = einops.rearrange(tile_ids, "b h w -> b (h w)")
        input_boards = einops.rearrange(rotated_boards, "b h w s -> b s h w")

        rotated_actions, rotated_logprobs, rotated_entropies = policy(
            input_boards, sampling_mode="greedy"
        )
        rotated_values = critic(input_boards)

        rotated_tile_ids_1 = tile_ids[torch.arange(batch_size), rotated_actions[:, 0]]
        rotated_tile_ids_2 = tile_ids[torch.arange(batch_size), rotated_actions[:, 1]]
        assert torch.all(
            rotated_tile_ids_1 == tile_ids_1
        ), "Tile ids 1 are differents"
        assert torch.all(
            rotated_tile_ids_2 == tile_ids_2
        ), "Tile ids 2 are differents"
        assert torch.all(
            rotated_actions[:, 2:] == actions[:, 2:]
        ), "Shifts actions are differents"
        assert torch.allclose(
            logprobs, rotated_logprobs, atol=1e-6
        ), f"Absolute mean error is {( logprobs - rotated_logprobs ).abs().mean()}"
        assert torch.allclose(
            entropies, rotated_entropies, atol=1e-6
        ), f"Absolute mean error is {( entropies - rotated_entropies ).abs().mean()}"
        assert torch.allclose(
            values, rotated_values, atol=1e-6
        ), f"Absolute mean error is {( values - rotated_values ).abs().mean()}"
