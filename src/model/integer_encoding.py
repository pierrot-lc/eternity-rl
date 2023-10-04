"""A simple integer encoding module, using the classical positional encodings.
Can be used to encode a game's matches for example.
"""
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D


class IntegerEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_integer: int = 10000):
        super().__init__()

        pos_enc = PositionalEncoding1D(embedding_dim)
        x = torch.zeros((1, max_integer, embedding_dim))
        encodings = pos_enc(x).squeeze(0)

        self.integer_encoding = nn.Embedding(
            num_embeddings=max_integer,
            embedding_dim=embedding_dim,
            _weight=encodings,
            _freeze=True,
        )

    def forward(self, integers: torch.Tensor) -> torch.Tensor:
        """Embed the timesteps.

        ---
        Args:
            integers: The integers to encode.
                Long tensor of shape of [...].
        ---
        Returns:
            The positional encodings for the given integers.
                Shape of [..., embedding_dim].
        """
        return self.integer_encoding(integers)
