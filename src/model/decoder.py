"""Simple cross-attention decoder.
"""
import torch
import torch.nn as nn
from einops import repeat


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_layers: int,
        dropout: float,
        n_queries: int = 4,
    ):
        super().__init__()

        self.queries = nn.Parameter(torch.randn(n_queries, embedding_dim))
        self.cross_attentions = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                embedding_dim,
                nhead=1,
                dim_feedforward=2 * embedding_dim,
                dropout=dropout,
                batch_first=False,
            ),
            num_layers=n_layers,
        )

    def forward(
        self,
        tiles: torch.Tensor,
    ) -> torch.Tensor:
        """Update the queries.

        ---
        Args:
            tiles: The game state.
                Tensor of shape [board_height x board_width, batch_size, embedding_dim].

        ---
        Returns:
            The updated queries.
                Shape of [n_queries, batch_size, embedding_dim].
        """
        batch_size = tiles.shape[1]
        queries = repeat(self.queries, "n d -> n b d", b=batch_size)
        queries = self.cross_attentions(queries, tiles)
        return queries
