"""Convolutional backbone."""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from ...environment import N_SIDES
from ..class_encoding import ClassEncoding


class ConvBackbone(nn.Module):
    """Use 3x3 convolutions to process the board. The board is initially encoded
    using the same process as the transformer backbone without the 2D positional
    encodings.
    """

    def __init__(self, embedding_dim: int, n_layers: int):
        assert embedding_dim % N_SIDES == 0
        super().__init__()

        self.embed_board = nn.Sequential(
            # Encode the classes.
            ClassEncoding(embedding_dim),
            # Merge the classes of each tile into a single embedding.
            # Applies the same projection to all sides.
            nn.Linear(embedding_dim, embedding_dim // N_SIDES),
            # To CNN shape.
            Rearrange("b t h w e -> b (t e) h w"),
        )

        self.space_mixin = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=3,
                        padding="same",
                        groups=embedding_dim // N_SIDES,
                    ),
                    nn.GELU(),
                    nn.GroupNorm(
                        num_groups=embedding_dim // N_SIDES, num_channels=embedding_dim
                    ),
                )
                for _ in range(n_layers)
            ]
        )
        self.channel_mixin = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=1,
                        padding="same",
                    ),
                    nn.GELU(),
                    nn.GroupNorm(num_groups=1, num_channels=embedding_dim),
                )
                for _ in range(n_layers)
            ]
        )

        self.to_sequence = Rearrange("b e h w -> (h w) b e")

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        """Embed the game state.

        ---
        Args:
            boards: The game state.
                Tensor of shape [batch_size, N_SIDES, board_height, board_width].

        ---
        Returns:
            tokens: The embedded game state as sequence of tiles.
                Shape of [board_height x board_width, batch_size, embedding_dim].
        """
        boards = self.embed_board(boards)

        for space, channel in zip(self.space_mixin, self.channel_mixin):
            boards = space(boards) + channel(boards) + boards

        return self.to_sequence(boards)
