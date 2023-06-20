import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange
from positional_encodings.torch_encodings import PositionalEncoding2D

from .time_encoding import TimeEncoding


class Summer(nn.Module):
    """This module replaces the `Summer` module from the `positional_encodings`
    package. The original `Summer` does not pass the output of the layer
    to the same device as the input.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the input with the output of the layer.
        Makes sure the output is on the same device.
        """
        return x + self.layer(x).to(x.device)


class Backbone(nn.Module):
    """Encode the board and produce a final embedding of the
    wanted size.

    The board is encoded as follows:
        - Embed the classes of each size of the tiles.
        - Merge the classes of each tile into a single embedding.
        - Add 2D positional encodings.
        - Flatten into 1D sequence of tokens.
        - Add a token that encode the timestep.
        - Use a transformer to encode the sequence.
        - Finally, use a dedicated query token to extract the final embedding.

    Note that the last query produce the embedding of the wanted size.
    The tokens in the transformers are of size `tile_embedding_dim`.
    """

    def __init__(
        self,
        n_classes: int,
        tile_embedding_dim: int,
        embedding_dim: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()

        # Input embeddings.
        self.embed_board = nn.Sequential(
            # Embed the classes of each size of the tiles.
            Rearrange("b t w h -> b h w t"),
            nn.Embedding(n_classes, tile_embedding_dim),
            nn.LayerNorm(tile_embedding_dim),
            # Merge the classes of each tile into a single embedding.
            Rearrange("b h w t e -> b (t e) h w"),
            nn.Conv2d(4 * tile_embedding_dim, tile_embedding_dim, 1, padding="same"),
            nn.GELU(),
            # Add 2D positional encodings.
            Rearrange("b c h w -> b h w c"),
            Summer(PositionalEncoding2D(tile_embedding_dim)),
            # Finally flatten into tokens for the transformer.
            Rearrange("b h w c -> b (h w) c"),
            nn.LayerNorm(tile_embedding_dim),
        )
        self.embed_timesteps = nn.Sequential(
            TimeEncoding(tile_embedding_dim),
            nn.LayerNorm(tile_embedding_dim),
        )

        # Main backbone.
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=tile_embedding_dim,
                nhead=1,  # One for each side of the tiles.
                dim_feedforward=2 * tile_embedding_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
        )

        # Extract the final embedding.
        self.output_query = nn.Parameter(torch.randn(1, embedding_dim))
        self.extract_embedding = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            kdim=tile_embedding_dim,
            vdim=tile_embedding_dim,
            num_heads=1,  # One for each action of the game.
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        tiles: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Embed the game state.

        ---
        Args:
            tiles: The game state.
                Tensor of shape [batch_size, 4, board_height, board_width].
            timesteps: The timestep of the game states.
                Tensor of shape [batch_size,].

        ---
        Returns:
            The embedding of the game state.
                Shape of [batch_size, embedding_dim].
        """
        # Project to shape of [batch_size, board_height x board_width, tile_embedding_dim].
        tiles = self.embed_board(tiles)
        timesteps = self.embed_timesteps(timesteps)
        tokens = torch.concat((timesteps.unsqueeze(1), tiles), dim=1)
        tokens = self.encoder(tokens)

        # Extract the final embeddings.
        query = self.output_query.to(tokens.device)
        query = repeat(self.output_query, "t e -> b t e", b=tokens.shape[0])
        embeddings, _ = self.extract_embedding(query, tokens, tokens)
        embeddings = self.layer_norm(embeddings)
        return embeddings.squeeze(1)
