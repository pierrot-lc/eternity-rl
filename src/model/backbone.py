import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from .g_cnn import GCNN
from .time_encoding import TimeEncoding


class Backbone(nn.Module):
    """Encode the board and produce a final embedding of the
    wanted size.

    The board is encoded as follows:
        - Embed the classes of each size of the tiles.
        - Merge the classes of each tile into a single embedding.
        - Flatten into 1D sequence of tokens.
        - Add a token that encode the timestep.
        - Interleave layers of CNN and transformer.

    The CNN layers are there to provide the tiles information about their localisation.
    The transformer layers are there to provide global overview of the states.
    """

    def __init__(
        self,
        n_classes: int,
        embedding_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=n_heads,
                    dim_feedforward=2 * embedding_dim,
                    dropout=dropout,
                    batch_first=False,
                )
                for _ in range(n_layers)
            ]
        )
        self.cnn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm2d(embedding_dim),
                    nn.Conv2d(
                        embedding_dim,
                        embedding_dim,
                        kernel_size=3,
                        padding="same",
                    ),
                    nn.GELU(),
                )
                for _ in range(n_layers)
            ]
        )
        self.embed_board = nn.Sequential(
            # Embed the classes of each size of the tiles.
            Rearrange("b t w h -> b h w t"),
            nn.Embedding(n_classes, embedding_dim),
            nn.LayerNorm(embedding_dim),
            # Merge the classes of each tile into a single embedding.
            Rearrange("b h w t e -> b (t e) h w"),
            nn.Conv2d(4 * embedding_dim, embedding_dim, kernel_size=1, padding="same"),
            nn.GELU(),
            Rearrange("b c h w -> (h w) b c"),
        )

    def forward(
        self,
        tiles: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """Embed the game state.

        ---
        Args:
            tiles: The game state.
                Tensor of shape [batch_size, 4, board_height, board_width].
            hidden_state: The previous hidden state of the model.
                Tensor of shape [batch_size, embedding_dim].

        ---
        Returns:
            The embedded game state.
                Shape of [board_height x board_width, batch_size, embedding_dim].
        """
        batch_size, _, board_height, board_width = tiles.shape

        tiles = self.embed_board(tiles)
        tokens = torch.concat((hidden_state.unsqueeze(0), tiles), dim=0)

        for cnn_layer, transformer_layer in zip(
            self.cnn_layers, self.transformer_layers
        ):
            hidden_state, tiles = tokens[:1], tokens[1:]
            tiles = rearrange(tiles, "(h w) b e -> b e h w", h=board_height)
            tiles = cnn_layer(tiles) + tiles
            tiles = rearrange(tiles, "b e h w -> (h w) b e")
            tokens = torch.concat((hidden_state, tiles), dim=0)
            tokens = transformer_layer(tokens) + tokens

        return tokens[1:], tokens[0]
