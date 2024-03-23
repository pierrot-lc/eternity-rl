"""Equivariant transformer backbone."""

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

from ...environment import N_SIDES
from ..class_encoding import ClassEncoding
from ..transformer import TransformerEncoderLayer
from .gnn import MessagePassingExter, MessagePassingInter


class EquivariantTransformerBackbone(nn.Module):
    """Encode the board and produce a final embedding of the
    wanted size.

    The board is encoded as follows:
        - Embed the classes of each side of the tiles.
        - Merge the classes of each tile into a single embedding.
        - Add 2D positional encoding.
        - Treat each tile as a token and use a transformer encoder
        to process the token sequence.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        assert embedding_dim % N_SIDES == 0
        super().__init__()

        self.embed_tokens = nn.Sequential(
            ClassEncoding(embedding_dim),
            Rearrange("b t h w e -> b h w t e"),
        )

        self.gnn_exter = MessagePassingExter(embedding_dim)
        self.gnn_inter = MessagePassingInter(embedding_dim)
        self.reduce = Reduce("b h w t e -> (h w) b e", reduction="mean")

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
            boards: The game state.
                Tensor of shape [batch_size, N_SIDES, board_height, board_width].

        ---
        Returns:
            tokens: The embedded game state as sequence of tiles.
                Shape of [board_height x board_width, batch_size, embedding_dim].
        """
        board_width = boards.shape[-1]

        # Shape of [batch_size, board_height, board_width, N_SIDES, embedding_dim].
        init_boards = self.embed_tokens(boards)
        boards = init_boards

        # Apply GNN layers to encode positional information.
        for _ in range(board_width):
            boards = self.gnn_exter(boards) + boards
            boards = self.gnn_inter(boards) + boards

        # Merge the nodes representing the same tile.
        tokens = self.reduce(boards + init_boards)
        tokens = self.encoder(tokens)
        return tokens
