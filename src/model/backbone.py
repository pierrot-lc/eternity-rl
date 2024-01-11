from positional_encodings.torch_encodings import PositionalEncoding2D, Summer

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from ..environment import N_SIDES
from .class_encoding import ClassEncoding
from .transformer import TransformerEncoderLayer


class Backbone(nn.Module):
    """Encode the board and produce a final embedding of the
    wanted size.

    The board is encoded as follows:
        - Embed the classes of each side of the tiles.
        - Merge the classes of each tile into a single embedding.
        - Same thing for the best boards.
        - Add the encoding of the steps using cross-attention.
        - Merge the two boards.
        - Use a simple ResNet to compute latent representations.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.embed_board = nn.Sequential(
            # Encode the classes.
            ClassEncoding(embedding_dim),
            # Merge the classes of each tile into a single embedding.
            Rearrange("b t h w e -> b h w (t e)"),
            nn.Linear(N_SIDES * embedding_dim, embedding_dim),
            # Add the 2D positional encodings.
            Summer(PositionalEncoding2D(embedding_dim)),
            # To transformer layout.
            Rearrange("b h w e -> (h w) b e"),
        )

        self.encoder = nn.TransformerEncoder(
            TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=n_heads,
                dim_feedforward=4 * embedding_dim * 2 // 3,  # See SwiGLU paper.
                dropout=dropout,
                norm_first=True,  # Pre-norm.
                batch_first=False,
            ),
            num_layers=n_layers,
            enable_nested_tensor=False,  # Pre-norm can't profit from this.
        )

    def forward(
        self,
        boards: torch.Tensor,
    ) -> torch.Tensor:
        """Embed the game state.

        ---
        Args:
            tiles: The game state.
                Tensor of shape [batch_size, N_SIDES, board_height, board_width].

        ---
        Returns:
            The embedded game state as sequence of tiles.
                Shape of [board_height x board_width, batch_size, embedding_dim].
        """
        boards = self.embed_board(boards)
        tokens = self.encoder(boards)
        return tokens
