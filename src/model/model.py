from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchinfo import summary
from einops import repeat, rearrange

from .backbone import Backbone

N_SIDES, N_ACTIONS = 4, 4


class Policy(nn.Module):
    def __init__(
        self,
        n_classes: int,
        board_width: int,
        board_height: int,
        embedding_dim: int,
        backbone_layers: int,
        decoder_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.embedding_dim = embedding_dim

        self.backbone = Backbone(
            n_classes,
            embedding_dim,
            backbone_layers,
            dropout,
        )

        self.select_tile = SelectTile(embedding_dim, decoder_layers, dropout)
        self.select_side = SelectSide(embedding_dim, decoder_layers, dropout)
        self.node_query = nn.Parameter(torch.randn(embedding_dim))
        self.side_query = nn.Parameter(torch.randn(embedding_dim))
        self.node_selected_embeddings = nn.Parameter(torch.randn(2, embedding_dim))
        self.side_embeddings = nn.Parameter(torch.randn(N_SIDES, embedding_dim))

    def dummy_input(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        tiles = torch.zeros(
            1,
            N_SIDES,
            self.board_height,
            self.board_width,
            dtype=torch.long,
            device=device,
        )
        hidden_state = torch.zeros(
            1,
            self.embedding_dim,
            dtype=torch.float,
            device=device,
        )
        return tiles, hidden_state

    def summary(self, device: str):
        """Torchinfo summary."""
        dummy_input = self.dummy_input(device)
        summary(
            self,
            input_data=[*dummy_input],
            depth=1,
            device=device,
        )

    def forward(
        self,
        tiles: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        sampling_mode: str = "sample",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the actions and value for the given game states.

        ---
        Args:
            tiles: The game state.
                Tensor of shape [batch_size, 4, board_height, board_width].
            hidden_state: The previous hidden state of the model.
                Initialized to 0 if not provided.
                Tensor of shape [batch_size, embedding_dim].
            sampling_mode: The sampling mode of the actions.

        ---
        Returns:
            actions: The predicted actions.
                Shape of [batch_size, n_actions].
            logprobs: The log probabilities of the predicted actions.
                Shape of [batch_size, n_actions].
            entropies: The entropies of the predicted actions.
                Shape of [batch_size, n_actions].
        """
        batch_size = tiles.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros(
                (batch_size, self.embedding_dim),
                dtype=torch.float,
                device=tiles.device,
            )

        tiles, hidden_state = self.backbone(tiles, hidden_state)

        actions, logprobs, entropies = [], [], []

        # Node selections.
        for node_number in range(2):
            queries = repeat(self.node_query, "e -> b e", b=batch_size)
            probs = self.select_tile(tiles, queries)

            sampled_nodes = self.sample_actions(probs, sampling_mode)
            actions_logprob, actions_entropy = Policy.logprobs(probs, sampled_nodes)

            actions.append(sampled_nodes)
            logprobs.append(actions_logprob)
            entropies.append(actions_entropy)

            selected_embedding = self.node_selected_embeddings[node_number]
            selected_embedding = repeat(selected_embedding, "e -> b e", b=batch_size)
            tiles = Policy.selective_add(tiles, selected_embedding, sampled_nodes)

        # Side selections.
        for _ in range(2):
            queries = repeat(self.side_query, "e -> b e", b=batch_size)
            probs = self.select_side(tiles, queries)

            sampled_sides = self.sample_actions(probs, sampling_mode)
            actions_logprob, actions_entropy = Policy.logprobs(probs, sampled_sides)

            actions.append(sampled_sides)
            logprobs.append(actions_logprob)
            entropies.append(actions_entropy)

            side_embedding = self.side_embeddings[sampled_sides]
            tiles = Policy.selective_add(tiles, side_embedding, sampled_sides)

        actions = torch.stack(actions, dim=1)
        logprobs = torch.stack(logprobs, dim=1)
        entropies = torch.stack(entropies, dim=1)

        return actions, logprobs, entropies, hidden_state

    @staticmethod
    def sample_actions(probs: torch.Tensor, mode: str) -> torch.Tensor:
        match mode:
            case "sample":
                categorical = Categorical(probs=probs)
                action_ids = categorical.sample()
            case "argmax":
                action_ids = torch.argmax(probs, dim=-1)
            case _:
                raise ValueError(f"Invalid mode: {mode}")
        return action_ids

    @staticmethod
    def logprobs(
        probs: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from the given probabilities.
        Returns the sampled actions and their log-probilities.

        ---
        Args:
            probs: Probabilities of the actions.
                Shape of [batch_size, n_actions].
            action_ids: The actions take.

        ---
        Returns:
            log_probs: The log-probabilities of the sampled actions.
                Shape of [batch_size,].
            entropies: The entropy of the categorical distributions.
                The entropies are normalized by the log of the number of actions.
                Shape of [batch_size,].
        """
        categorical = Categorical(probs=probs)
        n_actions = probs.shape[-1]

        entropies = categorical.entropy() / np.log(n_actions)
        log_probs = categorical.log_prob(action_ids)
        return log_probs, entropies

    @staticmethod
    def selective_add(
        tiles: torch.Tensor, embeddings: torch.Tensor, tile_ids: torch.Tensor
    ) -> torch.Tensor:
        """Add the given embeddings only to the specified tiles.

        ---
        Args:
            tiles: The original tile embeddings.
                Shape of [n_tiles, batch_size, embedding_dim].
            embeddings: The embeddings to add.
                Shape of [batch_size, embedding_dim].
            tile_ids: The tile ids to add the embeddings to.
                Shape of [batch_size,].

        ---
        Returns:
            The new tile embeddings.
                Shape of [n_tiles, batch_size, embedding_dim].
        """
        n_tiles, batch_size, _ = tiles.shape
        device = tiles.device

        offsets = torch.arange(0, batch_size * n_tiles, n_tiles, device=device)
        tiles = rearrange(tiles, "t b e -> (b t) e")
        tiles[tile_ids + offsets] = tiles[tile_ids + offsets] + embeddings
        tiles = rearrange(tiles, "(b t) e -> t b e", b=batch_size)

        return tiles


class SelectTile(nn.Module):
    def __init__(self, embedding_dim: int, n_layers: int, dropout: float):
        super().__init__()

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                embedding_dim,
                nhead=1,
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
    def __init__(self, embedding_dim: int, n_layers: int, dropout: float):
        super().__init__()

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                embedding_dim,
                nhead=1,
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
            nn.Linear(embedding_dim, N_SIDES),
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
