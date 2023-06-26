import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions import Categorical
from torchinfo import summary

from .backbone import Backbone
from .decoder import Decoder

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

        self.embed_tile_ids = nn.Sequential(
            nn.Embedding(board_width * board_height, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.embed_roll_ids = nn.Sequential(
            nn.Embedding(4, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.backbone = Backbone(
            n_classes,
            embedding_dim,
            backbone_layers,
            dropout,
        )
        self.decoder = Decoder(
            embedding_dim,
            decoder_layers,
            dropout,
            n_queries=N_ACTIONS,
        )
        self.rnn = nn.GRUCell(embedding_dim, embedding_dim)
        self.predict_actions = nn.ModuleList(
            [
                nn.Linear(embedding_dim, board_width * board_height),
                nn.Linear(embedding_dim, board_width * board_height),
                nn.Linear(embedding_dim, N_SIDES),
                nn.Linear(embedding_dim, N_SIDES),
            ]
        )

    def dummy_input(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        tiles = torch.zeros(
            1,
            N_SIDES,
            self.board_height,
            self.board_width,
            dtype=torch.long,
            device=device,
        )
        timesteps = torch.zeros(
            1,
            dtype=torch.long,
            device=device,
        )
        return tiles, timesteps

    def summary(self, device: str):
        """Torchinfo summary."""
        dummy_input = self.dummy_input(device)
        summary(
            self,
            input_data=[*dummy_input],
            depth=2,
            device=device,
        )

    def forward(
        self,
        tiles: torch.Tensor,
        timesteps: torch.Tensor,
        sampling_mode: str = "sample",
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Predict the actions and value for the given game states.

        ---
        Args:
            tiles: The game state.
                Tensor of shape [batch_size, 4, board_height, board_width].
            timestep: The timestep of the game states.
                Tensor of shape [batch_size,].
            sampling_mode: The sampling mode of the actions.

        ---
        Returns:
            actions: The predicted actions.
                Shape of [batch_size, 4].
            probs: Distribution output of all heads.
                List of tensor of shape [batch_size, n_actions].
        """
        tiles = self.backbone(tiles, timesteps)
        queries = self.decoder(tiles)

        actions, probs = [], []
        hidden_state = None
        for query, predict_action in zip(queries, self.predict_actions):
            hidden_state = self.rnn(query, hidden_state)
            action_scores = predict_action(hidden_state)
            action_ids = self.sample_actions(action_scores, sampling_mode)

            actions.append(action_ids)
            probs.append(torch.softmax(action_scores, dim=-1))

        actions = torch.stack(actions, dim=1)
        return actions, probs

    @staticmethod
    def sample_actions(logits: torch.Tensor, mode: str) -> torch.Tensor:
        match mode:
            case "sample":
                distributions = torch.softmax(logits, dim=-1)
                categorical = Categorical(probs=distributions)
                action_ids = categorical.sample()
            case "argmax":
                action_ids = torch.argmax(logits, dim=-1)
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
