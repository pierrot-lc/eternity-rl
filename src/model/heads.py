import torch
import torch.nn as nn


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
        node_distributions = node_distributions.squeeze(1)
        return node_distributions


class SelectSide(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        n_sides: int,
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
        self.attention_layer = nn.MultiheadAttention(
            embedding_dim,
            num_heads=1,
            dropout=0.0,
            bias=False,
            batch_first=False,
        )
        self.predict_side = nn.Sequential(
            nn.Linear(embedding_dim, n_sides),
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
                Tensor of shape [batch_size, n_sides].
        """
        query = query.unsqueeze(0)
        query = self.decoder(query, tiles)
        side_embeddings, _ = self.attention_layer(
            query,
            tiles,
            tiles,
            need_weights=True,
        )
        side_embeddings = side_embeddings.squeeze(0)
        side_distributions = self.predict_side(side_embeddings)
        return side_distributions
