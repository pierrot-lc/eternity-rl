from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from ..environment import N_SIDES
from .class_encoding import ClassEncoding
from .integer_encoding import IntegerEncoding
from .transformer import TransformerDecoderLayer


class EncodeSteps(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.steps_encoder = IntegerEncoding(embedding_dim)

        # The transformer is the same as the one used in LLaMa: https://arxiv.org/abs/2302.13971.
        self.tiles_encoder = TransformerDecoderLayer(
            embedding_dim,
            nhead=1,
            dim_feedforward=4 * embedding_dim * 2 // 3,
            dropout=0,
            batch_first=False,
            norm_first=True,
        )

    def forward(self, boards: torch.Tensor, n_steps: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention to the tiles of each board so that
        their latent representation is enhanced by the number of steps
        of the current game.

        ---
        Args:
            boards: The game state.
                Tensor of shape [batch_size, embedding_dim, board_height, board_width].
            n_steps: Number of steps of each game.
                Long tensor of shape [batch_size,].

        ---
        Returns:
            The enhanced game state.
                Tensor of shape [batch_size, embedding_dim, board_height, board_width].
        """
        board_height, board_width = boards.shape[2], boards.shape[3]

        tiles = rearrange(boards, "b e h w -> (h w) b e")
        n_steps = self.steps_encoder(n_steps)  # Shape of [batch_size, embedding_dim].
        n_steps = n_steps.unsqueeze(0)

        tiles = self.tiles_encoder(tiles, n_steps)

        boards = rearrange(tiles, "(h w) b e -> b e h w", h=board_height, w=board_width)
        return boards


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
        Conv2d = partial(nn.Conv2d, stride=1, padding="same")

        self.embed_board = nn.Sequential(
            # Encode the classes.
            ClassEncoding(embedding_dim),
            # Apply a CNN encoder to account for local same-class information.
            Rearrange("b t h w e -> b (t e) h w"),
            Conv2d(embedding_dim * N_SIDES, embedding_dim, kernel_size=1),
            Conv2d(embedding_dim, embedding_dim, kernel_size=3),
            nn.GroupNorm(num_groups=1, num_channels=embedding_dim),
        )
        self.embed_steps = EncodeSteps(embedding_dim)

        self.res_layers = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2d(embedding_dim, embedding_dim, kernel_size=3),
                    nn.GELU(),
                    nn.GroupNorm(num_groups=1, num_channels=embedding_dim),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        boards: torch.Tensor,
        best_boards: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Embed the game state.

        ---
        Args:
            tiles: The game state.
                Tensor of shape [batch_size, N_SIDES, board_height, board_width].
            best_tiles: The game best state.
                Tensor of shape [batch_size, N_SIDES, board_height, board_width].
            n_steps: Number of steps of each game.
                Long tensor of shape [batch_size,].

        ---
        Returns:
            The embedded game state as sequence of tiles.
                Shape of [board_height x board_width, batch_size, embedding_dim].
        """
        boards = self.embed_board(boards)
        for layer in self.res_layers:
            boards = layer(boards) + boards

        tiles = rearrange(boards, "b e h w -> (h w) b e")
        return tiles
