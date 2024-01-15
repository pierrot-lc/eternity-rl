import torch
import torch.nn as nn
from einops import repeat
from torchinfo import summary

from ..environment import N_SIDES
from .backbone import Backbone
from .heads import EstimateValue


class Critic(nn.Module):
    def __init__(
        self,
        board_width: int,
        board_height: int,
        embedding_dim: int,
        n_heads: int,
        backbone_layers: int,
        decoder_layers: int,
        dropout: float,
        n_memories: int,
    ):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.embedding_dim = embedding_dim
        self.n_memories = n_memories

        self.backbone = Backbone(
            embedding_dim,
            n_heads,
            backbone_layers,
            dropout,
        )

        self.estimate_value = EstimateValue(
            embedding_dim, n_heads, decoder_layers, dropout
        )
        self.value_query = nn.Parameter(torch.randn(embedding_dim))

    def init_memories(
        self, batch_size: int, device: torch.device | str
    ) -> torch.Tensor:
        return self.backbone.init_memories(batch_size, self.n_memories, device)

    def dummy_input(self, device: str) -> tuple[torch.Tensor]:
        tiles = torch.zeros(
            1,
            N_SIDES,
            self.board_height,
            self.board_width,
            dtype=torch.long,
            device=device,
        )
        memories = self.init_memories(1, device)
        return (tiles, memories)

    def summary(self, device: str):
        """Torchinfo summary."""
        dummy_input = self.dummy_input(device)
        print("\nCritic summary:")
        summary(
            self,
            input_data=[*dummy_input],
            depth=1,
            device=device,
        )

    def forward(
        self,
        tiles: torch.Tensor,
        memories: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the actions and value for the given game states.

        ---
        Args:
            tiles: The game state.
                Long tensor of shape [batch_size, N_SIDES, board_height, board_width].
            memories: The memories of the agent from the previous state.
                Tensor of shape [batch_size, n_memories, embedding_dim].
        ---
        Returns:
            values: The predicted values.
                Shape of [batch_size,].
            memories: The updated memories.
                Shape of [batch_size, n_memories, embedding_dim].
        """
        batch_size = tiles.shape[0]
        tiles, memories = self.backbone(tiles, memories)
        queries = repeat(self.value_query, "e -> b e", b=batch_size)
        values = self.estimate_value(tiles, queries)
        return values, memories
