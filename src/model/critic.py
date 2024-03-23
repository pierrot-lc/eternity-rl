import torch
import torch.nn as nn
from einops import repeat
from torchinfo import summary

from ..environment import N_SIDES
from .backbones import GNNBackbone, TransformerBackbone, EquivariantTransformerBackbone
from .heads import EstimateValue


class Critic(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        backbone_layers: int,
        decoder_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.backbone = TransformerBackbone(
            embedding_dim, n_heads, backbone_layers, dropout
        )
        self.backbone = GNNBackbone(embedding_dim, backbone_layers)
        self.backbone = EquivariantTransformerBackbone(
            embedding_dim, n_heads, backbone_layers, dropout
        )

        self.estimate_value = EstimateValue(
            embedding_dim, n_heads, decoder_layers, dropout
        )
        self.value_query = nn.Parameter(torch.randn(embedding_dim))

    def dummy_input(
        self, board_height: int, board_width: int, device: str
    ) -> tuple[torch.Tensor]:
        tiles = torch.zeros(
            (1, N_SIDES, board_height, board_width),
            dtype=torch.long,
            device=device,
        )
        return (tiles,)

    def summary(self, board_height: int, board_width: int, device: str):
        """Torchinfo summary."""
        dummy_input = self.dummy_input(board_height, board_width, device)
        print("\nCritic summary:")
        summary(
            self,
            input_data=[*dummy_input],
            depth=1,
            device=device,
        )

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        """Predict the actions and value for the given game states.

        ---
        Args:
            tiles: The game state.
                Long tensor of shape [batch_size, N_SIDES, board_height, board_width].

        ---
        Returns:
            values: The predicted values.
                Shape of [batch_size,].
        """
        batch_size = tiles.shape[0]
        tiles = self.backbone(tiles)
        queries = repeat(self.value_query, "e -> b e", b=batch_size)
        values = self.estimate_value(tiles, queries)
        return values
