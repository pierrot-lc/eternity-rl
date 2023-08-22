import torch
import torch.nn as nn

from ..environment import N_SIDES


class SelectTile(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                embedding_dim,
                nhead=n_heads,
                dim_feedforward=4 * embedding_dim,
                dropout=dropout,
                batch_first=False,
            ),
            num_layers=n_layers,
        )
        self.attention_layer = nn.MultiheadAttention(
            embedding_dim,
            num_heads=1,
            dropout=0.0,
            bias=False,
            batch_first=False,
        )

        # Deactivate these parameters, as they're not used (DDP).
        self.attention_layer.out_proj.requires_grad_(False)

    def forward(self, tiles: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
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
        query = query.unsqueeze(0)
        query = self.decoder(query, tiles)
        _, node_distributions = self.attention_layer(
            query,
            tiles,
            tiles,
            need_weights=True,
            average_attn_weights=True,
        )
        return node_distributions.squeeze(1)


class SelectSide(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                embedding_dim,
                nhead=n_heads,
                dim_feedforward=4 * embedding_dim,
                dropout=dropout,
                batch_first=False,
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

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                embedding_dim,
                nhead=n_heads,
                dim_feedforward=2 * embedding_dim,
                dropout=dropout,
                batch_first=False,
            ),
            num_layers=n_layers,
        )
        self.predict_value = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            SymExp(),
        )

    def forward(self, tiles: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Predict the value of the given state using a cross-attention operation
        between the tiles and some query.

        ---
        Args:
            tiles: The tiles, already embedded.
                Tensor of shape [n_tiles, batch_size, embedding_dim].
            query: The query.
                Tensor of shape [batch_size, embedding_dim].

        ---
        Returns:
            The predicted value.
                Tensor of shape [batch_size,].
        """
        query = query.unsqueeze(0)
        query = self.decoder(query, tiles)
        value = self.predict_value(query.squeeze(0))
        return value.squeeze(1)


class SymExp(nn.Module):
    """See https://arxiv.org/abs/2301.04104."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sign_x = torch.sign(x)
        abs_x = torch.abs(x)
        return sign_x * (torch.exp(abs_x) - 1)
