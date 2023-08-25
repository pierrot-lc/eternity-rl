import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from positional_encodings.torch_encodings import PositionalEncoding2D

from ..environment import N_SIDES


class Backbone(nn.Module):
    """Encode the board and produce a final embedding of the
    wanted size.

    The board is encoded as follows:
        - Embed the classes of each size of the tiles.
        - Merge the classes of each tile into a single embedding.
        - CNN layers to encode local features.
        - Flatten into 1D sequence of tokens.
        - Transformer layers to encode global features.

    The CNN layers are there to provide the tiles information about their localisation.
    The transformer layers are there to provide global overview of the states.
    """

    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        embedding_dim: int,
        n_heads: int,
        cnn_layers: int,
        transformer_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.embed_board = nn.Sequential(
            # Embed the classes of each size of the tiles.
            Rearrange("b t h w -> b h w t"),
            nn.Embedding(n_classes, n_channels),
            # Merge the classes of each tile into a single embedding.
            Rearrange("b h w t e -> b (t e) h w"),
            nn.Conv2d(N_SIDES * n_channels, n_channels, kernel_size=1, padding="same"),
            nn.GELU(),
            nn.GroupNorm(1, n_channels),
        )
        self.linear = nn.Sequential(
            nn.Linear(n_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.positional_enc = PositionalEncoding2D(embedding_dim)

        self.cnn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        n_channels,
                        n_channels,
                        kernel_size=3,
                        padding="same",
                    ),
                    nn.GELU(),
                    nn.GroupNorm(1, n_channels),
                )
                for _ in range(cnn_layers)
            ]
        )
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=n_heads,
                dim_feedforward=4 * embedding_dim,
                dropout=dropout,
                batch_first=False,
            ),
            num_layers=transformer_layers,
        )

    def forward(
        self,
        tiles: torch.Tensor,
    ) -> torch.Tensor:
        """Embed the game state.

        ---
        Args:
            tiles: The game state.
                Tensor of shape [batch_size, N_SIDES, board_height, board_width].

        ---
        Returns:
            The embedded game state.
                Shape of [board_height x board_width, batch_size, embedding_dim].
        """
        batch_size, _, board_height, board_width = tiles.shape

        tiles = self.embed_board(tiles)
        for layer in self.cnn_layers:
            tiles = layer(tiles) + tiles

        tokens = rearrange(tiles, "b e h w -> b h w e")
        tokens = self.linear(tokens)
        tokens = self.positional_enc(tokens) + tokens
        tokens = rearrange(tokens, "b h w e -> (h w) b e")
        tokens = self.transformer_layers(tokens)

        return tokens
