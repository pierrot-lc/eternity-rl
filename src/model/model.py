import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchinfo import summary

from .backbone import Backbone
from .head import Head


class Policy(nn.Module):
    def __init__(
        self,
        n_classes: int,
        board_width: int,
        board_height: int,
        tile_embedding_dim: int,
        embedding_dim: int,
        backbone_layers: int,
        head_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height

        self.embed_tile_ids = nn.Sequential(
            nn.Embedding(board_width * board_height, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.backbone = Backbone(
            n_classes,
            tile_embedding_dim,
            embedding_dim,
            backbone_layers,
            dropout,
        )

        self.predict_actions = nn.ModuleDict(
            {
                "tile-1": Head(embedding_dim, head_layers, board_width * board_height),
                "tile-2": nn.Sequential(
                    nn.Linear(2 * embedding_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    Head(embedding_dim, head_layers, board_width * board_height),
                ),
                "roll-1": nn.Sequential(
                    nn.Linear(3 * embedding_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    Head(embedding_dim, head_layers, 4),
                ),
                "roll-2": nn.Sequential(
                    nn.Linear(3 * embedding_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    Head(embedding_dim, head_layers, 4),
                ),
            }
        )

        # Init lazy layers.
        with torch.no_grad():
            dummy_input = self.dummy_input("cpu")
            self.forward(*dummy_input)

    def dummy_input(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        tiles = torch.zeros(
            1,
            4,
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
            values: The predicted values.
                Shape of [batch_size, 1].
        """
        embed = self.backbone(tiles, timesteps)

        # Compute action logits.
        tile_1 = self.predict_actions["tile-1"](embed)
        tile_1_id = self.sample_actions(tile_1, sampling_mode)
        tile_1_emb = self.embed_tile_ids(tile_1_id)

        # Shape of [batch_size, 2 * embedding_dim].
        embed = torch.concat([embed, tile_1_emb], dim=-1)

        tile_2 = self.predict_actions["tile-2"](embed)
        tile_2_id = self.sample_actions(tile_2, sampling_mode)
        tile_2_emb = self.embed_tile_ids(tile_2_id)

        # Shape of [batch_size, 3 * embedding_dim].
        embed = torch.concat([embed, tile_2_emb], dim=-1)

        roll_1 = self.predict_actions["roll-1"](embed)
        roll_1_id = self.sample_actions(roll_1, sampling_mode)

        roll_2 = self.predict_actions["roll-2"](embed)
        roll_2_id = self.sample_actions(roll_2, sampling_mode)

        actions = torch.stack([tile_1_id, roll_1_id, tile_2_id, roll_2_id], dim=1)
        logits = [tile_1, roll_1, tile_2, roll_2]
        probs = [torch.softmax(logit, dim=-1) for logit in logits]

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
