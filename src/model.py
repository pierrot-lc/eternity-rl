import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class CNNPolicy(nn.Module):
    def __init__(
        self,
        n_classes: int,
        embedding_dim: int,
        n_layers: int,
        board_width: int,
        board_height: int,
    ):
        super().__init__()

        self.embed_tiles = nn.Sequential(
            Rearrange("b t h w -> b h w t"),
            nn.Embedding(n_classes, embedding_dim),
            nn.LayerNorm(embedding_dim),
            Rearrange("b h w t e -> b (t e) h w"),
            nn.Conv2d(4 * embedding_dim, embedding_dim, 3, padding="same"),
            nn.LayerNorm([embedding_dim, board_height, board_width]),
            nn.GELU(),
        )

        self.residuals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embedding_dim, embedding_dim, 3, padding="same"),
                    nn.LayerNorm([embedding_dim, board_height, board_width]),
                    nn.GELU(),
                )
                for _ in range(n_layers)
            ]
        )

        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim * board_width * board_height, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        self.select_1 = nn.ModuleDict(
            {
                "tile": nn.Linear(embedding_dim, board_width * board_height),
                "roll": nn.Linear(embedding_dim, 4),
            }
        )
        self.select_2 = nn.ModuleDict(
            {
                "tile": nn.Linear(embedding_dim, board_width * board_height),
                "roll": nn.Linear(embedding_dim, 4),
            }
        )

    def forward(
        self, tiles: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """

        Args:
            tiles: The game state.
                Tensor of shape [batch_size, 4, board_height, board_width].

        Returns:
            swap_height:
        """
        embed = self.embed_tiles(tiles)
        for layer in self.residuals:
            embed = layer(embed) + embed
        embed = self.flatten(embed)
        tile_1 = {key: layer(embed) for key, layer in self.select_1.items()}
        tile_2 = {key: layer(embed) for key, layer in self.select_2.items()}
        return tile_1, tile_2
