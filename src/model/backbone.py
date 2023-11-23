import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer

from ..environment import N_SIDES
from .class_encoding import ClassEncoding
from .integer_encoding import IntegerEncoding
from .transformer import TransformerEncoderLayer, TransformerDecoderLayer


class Backbone(nn.Module):
    """Encode the board and produce a final embedding of the
    wanted size.

    The board is encoded as follows:
        - Embed the classes of each side of the tiles.
        - Merge the classes of each tile into a single embedding.
        - Add 2D positional encoding.
        - Flatten into 1D sequence of tokens.
        - Transformer encoder to compute features.
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

        self.steps_encoder = IntegerEncoding(embedding_dim)

        # The transformer is the same as the one used in LLaMa: https://arxiv.org/abs/2302.13971.
        # TODO: Add rotary embeddings?
        self.conditionals_encoder = nn.TransformerEncoder(
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
        self.tiles_encoder = nn.TransformerDecoder(
            TransformerDecoderLayer(
                embedding_dim,
                nhead=n_heads,
                dim_feedforward=4 * embedding_dim * 2 // 3,
                dropout=dropout,
                batch_first=False,
                norm_first=True,
            ),
            num_layers=n_layers,
        )

    def forward(
        self,
        tiles: torch.Tensor,
        best_tiles: torch.Tensor,
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
            The embedded game state.
                Shape of [board_height x board_width, batch_size, embedding_dim].
        """
        n_steps = self.steps_encoder(n_steps)
        tiles = self.embed_board(tiles)
        best_tiles = self.embed_board(best_tiles)

        conditional_tokens = torch.cat(
            (n_steps.unsqueeze(0), best_tiles),
            dim=0,
        )
        conditional_tokens = self.conditionals_encoder(conditional_tokens)
        tiles = self.tiles_encoder(tiles, conditional_tokens)
        return tiles
