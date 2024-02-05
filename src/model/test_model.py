import einops
import pytest
import torch

from ..environment import N_SIDES, EternityEnv, random_perfect_instances
from .backbone import Backbone
from .class_encoding import ClassEncoding


def init_env(seed: int) -> EternityEnv:
    generator = torch.Generator().manual_seed(seed)
    instances = random_perfect_instances(
        size=4, n_classes=6, n_instances=10, generator=generator
    )
    return EternityEnv(instances, episode_length=float("+inf"), device="cpu")


def rotate_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate the given tensor by 90 degres, assumes the tensor is 2D.

    ---
    Args:
        tensor: The 2D tensor to rotate.
            Shape of [batch_size, y_size, x_size, ...].

    ---
    Return:
        The rotated tensor.
            Shape of [batch_size, x_size, y_size, ...].
    """
    tensor = einops.rearrange(tensor, "b y x ... -> b x y ...")
    tensor = tensor.flip(dims=(1,))
    return tensor


@pytest.mark.parametrize(
    "tensor, rotated",
    [
        (
            torch.LongTensor(
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                    ],
                    [
                        [6, 5, 4],
                        [3, 2, 1],
                    ],
                ]
            ),
            torch.LongTensor(
                [
                    [
                        [3, 6],
                        [2, 5],
                        [1, 4],
                    ],
                    [
                        [4, 1],
                        [5, 2],
                        [6, 3],
                    ],
                ]
            ),
        ),
        (
            torch.LongTensor(
                [
                    [
                        [[1, 2], [3, 4]],
                        [[5, 6], [7, 8]],
                    ],
                ]
            ),
            torch.LongTensor(
                [
                    [
                        [[3, 4], [7, 8]],
                        [[1, 2], [5, 6]],
                    ],
                ]
            ),
        ),
    ],
)
def test_rotate_tensor(tensor: torch.Tensor, rotated: torch.Tensor):
    tensor = rotate_tensor(tensor)
    assert torch.all(tensor == rotated)


@torch.inference_mode()
@pytest.mark.parametrize(
    "boards, backbone",
    [
        (
            init_env(seed=0).reset()[0],
            Backbone(embedding_dim=12, n_heads=1, n_layers=1, dropout=0.0),
        ),
        (
            init_env(seed=1).reset()[0],
            Backbone(embedding_dim=12, n_heads=2, n_layers=1, dropout=0.0),
        ),
        (
            init_env(seed=2).reset()[0],
            Backbone(embedding_dim=12, n_heads=2, n_layers=3, dropout=0.0),
        ),
    ],
)
def test_equivariant_backbone(boards: torch.Tensor, backbone: Backbone):
    _, _, height, width = boards.shape
    encoding = backbone(boards)
    encoding = einops.rearrange(encoding, "(h w) b e -> b h w e", h=height, w=width)

    for _ in range(4):
        boards = einops.rearrange(boards, "b s h w -> b h w s")
        boards = rotate_tensor(boards)
        boards = einops.rearrange(boards, "b h w s -> b s h w")
        encoding = rotate_tensor(encoding)
        rotate_encoding = backbone(boards)
        rotate_encoding = einops.rearrange(
            rotate_encoding, "(h w) b e -> b h w e", h=height, w=width
        )
        assert torch.allclose(encoding, rotate_encoding, atol=1e-5)


@pytest.mark.parametrize(
    "embedding_dim, batch_size, board_size",
    [
        (10, 1, 4),
        (128, 64, 4),
        (256, 1024, 7),
    ],
)
def test_class_encoding(embedding_dim: int, batch_size: int, board_size: int):
    encoding_module = ClassEncoding(embedding_dim)
    random_input = torch.randint(
        low=0,
        high=embedding_dim - 1,
        size=(batch_size, board_size, board_size, N_SIDES),
        dtype=torch.long,
    )

    encoded_classes = encoding_module(random_input)

    assert encoded_classes.shape == torch.Size(
        (batch_size, board_size, board_size, N_SIDES, embedding_dim)
    ), "The shape of the encoded classes is not correct!"

    encoded_classes = encoded_classes.flatten(end_dim=-2)
    random_input = random_input.flatten()

    for class_id, encoded_vector in zip(random_input, encoded_classes):
        assert torch.all(encoding_module.class_enc.weight[class_id] == encoded_vector)

    ortho = encoding_module.class_enc.weight
    assert torch.allclose(
        ortho @ ortho.T, torch.eye(embedding_dim), atol=1e-4
    ), "Encodings are not orthogonal!"
