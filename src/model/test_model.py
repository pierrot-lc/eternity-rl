import pytest
import torch

from ..environment import N_SIDES
from .class_encoding import ClassEncoding


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
