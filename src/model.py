import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class CNNPolicy(nn.Module):
    def __init__(
        self,
        n_classes: int,
        embedding_dim: int,
        board_width: int,
        board_height: int,
    ):
        super().__init__()

        self.embed_tiles = nn.Sequential(
            Rearrange("b t h w -> b h w t"),
            nn.Embedding(n_classes, embedding_dim),
            nn.LayerNorm(embedding_dim),
            Rearrange("b h w t e -> b (t e) h w"),
        )

        self.cnn = nn.Sequential(
            nn.
                )

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        """

        Args:
            tiles: The game state.
                Tensor of shape [batch_size, 4, board_height, board_width].

        Returns:
            swap_height:
        """
        embed = self.embed_tiles(tiles)
