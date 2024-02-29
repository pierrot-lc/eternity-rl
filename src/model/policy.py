import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.distributions import Categorical
from torchinfo import summary

from ..environment import N_SIDES
from ..sampling import (dirichlet_sampling, epsilon_greedy_sampling,
                        epsilon_sampling)
from .backbones import GNNBackbone
from .heads import SelectSide, SelectTile


class Policy(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        backbone_layers: int,
        decoder_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.backbone = GNNBackbone(embedding_dim, backbone_layers)

        self.select_tile = SelectTile(embedding_dim, n_heads, decoder_layers, dropout)
        self.select_side = SelectSide(embedding_dim, n_heads, decoder_layers, dropout)

        self.tile_query = nn.Parameter(torch.randn(embedding_dim))
        self.side_query = nn.Parameter(torch.randn(embedding_dim))

        self.tiles_embeddings = nn.Parameter(torch.randn(2, embedding_dim))
        self.sides_embeddings = nn.Parameter(torch.randn(N_SIDES, embedding_dim))

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
        print("\nPolicy summary:")
        summary(
            self,
            input_data=[*dummy_input],
            depth=1,
            device=device,
        )

    def forward(
        self,
        tiles: torch.Tensor,
        sampling_mode: str | None = "softmax",
        sampled_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the actions and value for the given game states.

        ---
        Args:
            tiles: The game state.
                Long tensor of shape [batch_size, N_SIDES, board_height, board_width].
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
        """
        assert not (
            sampling_mode is None and sampled_actions is None
        ), "Either sampling_mode or sampled_actions must be given."
        batch_size = tiles.shape[0]

        tiles = self.backbone(tiles)
        actions, logprobs, entropies = [], [], []

        # Node selections.
        for node_number in range(2):
            queries = repeat(self.tile_query, "e -> b e", b=batch_size)
            probs = self.select_tile(tiles, queries)

            if sampled_actions is None:
                sampled_tiles = self.sample_actions(probs, sampling_mode)
            else:
                sampled_tiles = sampled_actions[:, node_number]
            actions_logprob, actions_entropy = Policy.logprobs(probs, sampled_tiles)

            actions.append(sampled_tiles)
            logprobs.append(actions_logprob)
            entropies.append(actions_entropy)

            selected_embedding = self.tiles_embeddings[node_number]
            selected_embedding = repeat(selected_embedding, "e -> b e", b=batch_size)
            tiles = Policy.selective_add(tiles, selected_embedding, sampled_tiles)

        # Side selections.
        for side_number in range(2):
            queries = repeat(self.side_query, "e -> b e", b=batch_size)
            probs = self.select_side(tiles, queries)

            if sampled_actions is None:
                sampled_sides = self.sample_actions(probs, sampling_mode)
            else:
                sampled_sides = sampled_actions[:, side_number + 2]
            actions_logprob, actions_entropy = Policy.logprobs(probs, sampled_sides)

            actions.append(sampled_sides)
            logprobs.append(actions_logprob)
            entropies.append(actions_entropy)

            side_embedding = self.sides_embeddings[sampled_sides]
            tiles = Policy.selective_add(tiles, side_embedding, actions[side_number])

        actions = torch.stack(actions, dim=1)
        logprobs = torch.stack(logprobs, dim=1)
        entropies = torch.stack(entropies, dim=1)

        return actions, logprobs, entropies

    @staticmethod
    def sample_actions(probs: torch.Tensor, mode: str) -> torch.Tensor:
        match mode:
            case "softmax":
                action_ids = Categorical(probs=probs).sample()
            case "greedy":
                action_ids = torch.argmax(probs, dim=-1)
            case "epsilon":
                action_ids = epsilon_sampling(probs, epsilon=0.05)
            case "epsilon-greedy":
                action_ids = epsilon_greedy_sampling(probs, epsilon=0.05)
            case "tempered":
                logits = Categorical(probs=probs).logits
                logits = logits / 2.0
                action_ids = Categorical(logits=logits).sample()
            case "dirichlet":
                # Do as if we saw 10% of each action.
                # See https://stats.stackexchange.com/a/385139
                n_actions = probs.shape[-1]
                concentration = 0.1 * n_actions / n_actions
                action_ids = dirichlet_sampling(
                    probs, concentration=concentration, exploration=0.25
                )
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
                Shape of [batch_size,].

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
