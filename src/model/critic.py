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
    ):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.embedding_dim = embedding_dim

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

    def dummy_input(self, device: str) -> tuple[torch.Tensor]:
        tiles = torch.zeros(
            1,
            N_SIDES,
            self.board_height,
            self.board_width,
            dtype=torch.long,
            device=device,
        )
        n_steps = torch.zeros(1, dtype=torch.long, device=device)
        return (tiles, tiles, n_steps)

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
        best_tiles: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the actions and value for the given game states.

        ---
        Args:
            tiles: The game state.
                Long tensor of shape [batch_size, N_SIDES, board_height, board_width].
            best_tiles: The game best state.
                Long tensor of shape [batch_size, N_SIDES, board_height, board_width].
            n_steps: Number of steps of each game.
                Long tensor of shape [batch_size,].
            sampling_mode: The sampling mode of the actions.
                One of ["sample", "greedy"].
            sampled_actions: The already sampled actions, if any.
                To compute the logprobs and entropies of those actions.
                Long tensor of shape [batch_size, n_actions].
        ---
        Returns:
            actions: The predicted actions.
                Shape of [batch_size, n_actions].
            logprobs: The log probabilities of the predicted actions.
                Shape of [batch_size, n_actions].
            entropies: The entropies of the predicted actions.
                Shape of [batch_size, n_actions].
            values: The predicted values.
                Shape of [batch_size,].
        """
        batch_size = tiles.shape[0]
        tiles = self.backbone(tiles, best_tiles, n_steps)
        queries = repeat(self.value_query, "e -> b e", b=batch_size)
        values = self.estimate_value(tiles, queries)
        return values
