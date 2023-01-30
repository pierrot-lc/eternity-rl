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
            nn.Conv2d(4 * embedding_dim, embedding_dim, 3, padding="same"),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),
            nn.Conv2d(embedding_dim, 15, 3, padding="same"),
            nn.BatchNorm2d(15),
            nn.GELU(),
            nn.Conv2d(15, 15, 3, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(15),
            # Flatten.
            nn.Flatten(),
            nn.LazyLinear(embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        self.select_1 = nn.ModuleDict(
            {
                "width": nn.Linear(embedding_dim, board_width),
                "height": nn.Linear(embedding_dim, board_height),
                "roll": nn.Linear(embedding_dim, 4),
            }
        )
        self.select_2 = nn.ModuleDict(
            {
                "width": nn.Linear(embedding_dim, board_width),
                "height": nn.Linear(embedding_dim, board_height),
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
        embed = self.cnn(embed)
        tile_1 = {key: layer(embed) for key, layer in self.select_1.items()}
        tile_2 = {key: layer(embed) for key, layer in self.select_2.items()}
        return tile_1, tile_2
