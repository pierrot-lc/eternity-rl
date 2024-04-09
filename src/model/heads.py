import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from ..environment import N_SIDES
from .transformer import TransformerDecoderLayer, TransformerEncoderLayer


class SelectTile(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()

        self.decoder = nn.TransformerEncoder(
            TransformerEncoderLayer(
                embedding_dim,
                nhead=n_heads,
                dim_feedforward=4 * embedding_dim * 2 // 3,
                dropout=dropout,
                batch_first=False,
                norm_first=True,
            ),
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.select_tile = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            Rearrange("t b e -> b (t e)"),
            nn.Softmax(dim=1),
        )

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        """Select a tile by using a cross-attention operation between
        the tiles and some query.

        ---
        Args:
            tiles: The tiles, already embedded.
                Tensor of shape [n_tiles, batch_size, embedding_dim].
            query: The query.
                Tensor of shape [batch_size, embedding_dim].

        ---
        Returns:
            A probability distribution over the tiles.
                Tensor of shape [batch_size, n_tiles].
        """
        tiles = self.decoder(tiles)
        distributions = self.select_tile(tiles)
        return tiles, distributions


class SelectSide(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()

        self.decoder = nn.TransformerDecoder(
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
        self.predict_side = nn.Sequential(
            nn.Linear(embedding_dim, N_SIDES),
            nn.Softmax(dim=-1),
        )

    def forward(self, tiles: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Select a side by using a cross-attention operation between
        the tiles and some query.

        ---
        Args:
            tiles: The tiles, already embedded.
                Tensor of shape [n_tiles, batch_size, embedding_dim].
            query: The query.
                Tensor of shape [batch_size, embedding_dim].

        ---
        Returns:
            A probability distribution over the sides.
                Tensor of shape [batch_size, N_SIDES].
        """
        query = query.unsqueeze(0)
        query = self.decoder(query, tiles)
        return self.predict_side(query.squeeze(0))


class EstimateValue(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.decoder = nn.TransformerEncoder(
            TransformerEncoderLayer(
                embedding_dim,
                nhead=n_heads,
                dim_feedforward=4 * embedding_dim * 2 // 3,
                dropout=dropout,
                batch_first=False,
                norm_first=True,
            ),
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.predict_value = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Tanh(),
        )

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        """Predict the value of the given state using a cross-attention operation
        between the tiles and some query.

        ---
        Args:
            tiles: The tiles, already embedded.
                Tensor of shape [n_tiles, batch_size, embedding_dim].

        ---
        Returns:
            The predicted value.
                Tensor of shape [batch_size,].
        """
        tiles = self.decoder(tiles)
        value = self.predict_value(tiles.mean(dim=0))
        return value.squeeze(1)


class SymExp(nn.Module):
    """See https://arxiv.org/abs/2301.04104."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sign_x = torch.sign(x)
        abs_x = torch.abs(x)
        return sign_x * (torch.exp(abs_x) - 1)
