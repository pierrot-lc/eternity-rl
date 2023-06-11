import torch
import torch.nn as nn
from einops import rearrange

from .time_encoding import TimeEncoding


class Backbone(nn.Module):
    def __init__(
        self,
        n_classes: int,
        board_width: int,
        board_height: int,
        embedding_dim: int,
        res_layers: int,
        maxpool_kernel: int,
        zero_init_residuals: bool,
    ):
        super().__init__()

        self.embed_classes = nn.Sequential(
            nn.Embedding(n_classes, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.embed_board = nn.Sequential(
            nn.Conv2d(4 * embedding_dim, embedding_dim, 1, padding="same"),
            nn.GELU(),
            nn.LayerNorm([embedding_dim, board_height, board_width]),
        )
        self.residuals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embedding_dim, embedding_dim, 3, padding="same"),
                    nn.GELU(),
                    nn.LayerNorm([embedding_dim, board_height, board_width]),
                )
                for _ in range(res_layers)
            ]
        )
        self.project_board = nn.Sequential(
            nn.MaxPool2d(kernel_size=maxpool_kernel),
            nn.Flatten(),
            nn.LazyLinear(embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )

        self.embed_timesteps = TimeEncoding(embedding_dim)
        self.project_timesteps = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )

        if zero_init_residuals:
            self.init_residuals()

    def init_residuals(self):
        """Zero out the weights of the residual convolutional layers."""
        for module in self.residuals.modules():
            if isinstance(module, nn.Conv2d):
                for param in module.parameters():
                    param.data.zero_()

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
            timestep: The timestep of the game states.
                Tensor of shape [batch_size,].

        ---
        Returns:
            The embedding of the game state.
                Shape of [batch_size, embedding_dim].
        """
        tiles = rearrange(tiles, "b t w h -> b h w t")
        embed = self.embed_classes(tiles)

        embed = rearrange(embed, "b h w t e -> b (t e) h w")
        embed = self.embed_board(embed)

        # Shape is of [batch_size, embedding_dim, height, width].
        for layer in self.residuals:
            embed = layer(embed) + embed

        embed = self.project_board(embed)
        # Shape is of [batch_size, embedding_dim].

        # Add the positional encodings for the given timesteps.
        encodings = self.embed_timesteps(timesteps)
        embed = torch.concat((embed, encodings), dim=1)
        embed = self.project_timesteps(embed)

        return embed
