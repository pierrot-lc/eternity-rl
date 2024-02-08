import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from ...environment.constants import EAST, N_SIDES, NORTH, SOUTH, WEST
from ..class_encoding import ClassEncoding


class GNNExter(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.self_linear = nn.Linear(embedding_dim, embedding_dim)
        self.other_linear = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Share the embedding from the direct neighbour token of each tokens.

        ---
        Args:
            tokens: The current tokens embeddings.
                Shape of [batch_size, board_height, board_width, N_SIDES, embedding_dim].

        ---
        Returns:
            tokens: The updated tokens embeddings.
                Shape of [batch_size, board_height, board_width, N_SIDES, embedding_dim].
        """
        padded_tokens = nn.functional.pad(tokens, (0, 0, 0, 0, 1, 1, 1, 1), value=0.0)
        self_tokens = self.self_linear(tokens)
        other_tokens = self.other_linear(padded_tokens)
        messages = torch.zeros_like(tokens)

        messages[:, :, :, NORTH] = (
            self_tokens[:, :, :, NORTH] + other_tokens[:, 2:, 1:-1, SOUTH]
        )
        messages[:, :, :, SOUTH] = (
            self_tokens[:, :, :, SOUTH] + other_tokens[:, :-2, 1:-1, NORTH]
        )
        messages[:, :, :, EAST] = (
            self_tokens[:, :, :, EAST] + other_tokens[:, 1:-1, 2:, WEST]
        )
        messages[:, :, :, WEST] = (
            self_tokens[:, :, :, WEST] + other_tokens[:, 1:-1, :-2, EAST]
        )

        messages = self.activation(messages)
        self.norm(tokens + messages)

        return tokens


class GNNInter(nn.Module):
    def __init__(self, embedding_dim: int):
        assert embedding_dim % N_SIDES == 0
        super().__init__()

        self.self_linear = nn.Linear(embedding_dim, embedding_dim)
        self.other_linear = nn.Linear(embedding_dim, embedding_dim)
        self.mix = nn.Sequential(
            # nn.Linear(embedding_dim, embedding_dim // N_SIDES),
            Rearrange("b h w t e -> b h w (t e)"),
            nn.Linear(embedding_dim * N_SIDES, embedding_dim),
        )
        self.activation = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Do the message passing inter-tokens.

        ---
        Args:
            tokens: The current tokens embeddings.
                Shape of [batch_size, board_height, board_width, N_SIDES, embedding_dim].

        ---
        Returns:
            tokens: The updated tokens embeddings.
                Shape of [batch_size, board_height, board_width, N_SIDES, embedding_dim].
        """
        self_tokens = self.self_linear(tokens)
        other_tokens = self.other_linear(tokens)
        other_tokens = self.mix(other_tokens)
        other_tokens = einops.repeat(other_tokens, "b h w e -> b h w t e", t=N_SIDES)

        messages = self_tokens + other_tokens
        tokens = self.norm(tokens + self.activation(messages))

        return tokens


class GNNBackbone(nn.Module):
    def __init__(self, embedding_dim: int, n_layers: int):
        super().__init__()

        self.embed_tokens = nn.Sequential(
            ClassEncoding(embedding_dim),
            Rearrange("b t h w e -> b h w t e"),
        )

        self.neighbour_layers = nn.ModuleList(
            [GNNExter(embedding_dim) for _ in range(n_layers)]
        )
        self.inter_layers = nn.ModuleList(
            [GNNInter(embedding_dim) for _ in range(n_layers)]
        )
        self.to_sequence = nn.Sequential(
            Rearrange("b h w t e -> b h w (t e)"),
            nn.Linear(embedding_dim * N_SIDES, embedding_dim),
            Rearrange("b h w e -> (h w) b e"),
        )

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        # To shape [batch_size, board_height, board_width, N_SIDES, embedding_dim].
        tokens = self.embed_tokens(boards)

        for gnn_inter, gnn_exter in zip(self.inter_layers, self.neighbour_layers):
            tokens = gnn_inter(tokens)
            tokens = gnn_exter(tokens)

        return self.to_sequence(tokens)
