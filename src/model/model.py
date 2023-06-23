import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchinfo import summary

from .backbone import Backbone


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
        self.embed_roll_ids = nn.Sequential(
            nn.Embedding(4, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.backbone = Backbone(
            n_classes,
            tile_embedding_dim,
            embedding_dim,
            backbone_layers,
            dropout,
        )
        self.rnn_head = nn.GRU(
            embedding_dim,
            embedding_dim,
            num_layers=head_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.predict_actions = nn.ModuleDict(
            {
                "tile": nn.Linear(embedding_dim, board_width * board_height),
                "roll": nn.Linear(embedding_dim, 4),
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

        choosen_tiles, choosen_rolls = [], []
        head_hidden = None
        embed = embed.unsqueeze(1)

        # First, choose the selected tiles.
        for _ in range(2):
            embed, head_hidden = self.rnn_head(embed, head_hidden)

            # Sample the tile.
            distrib_tile = self.predict_actions["tile"](embed.squeeze(1))
            tile_id = self.sample_actions(distrib_tile, sampling_mode)

            # Update the embedding so that the model knows which tile is selected.
            embed = self.embed_tile_ids(tile_id).unsqueeze(1)

            choosen_tiles.append((distrib_tile, tile_id))

        # Then, choose the rolls.
        for _ in range(2):
            embed, head_hidden = self.rnn_head(embed, head_hidden)

            # Sample the roll.
            distrib_roll = self.predict_actions["roll"](embed.squeeze(1))
            roll_id = self.sample_actions(distrib_roll, sampling_mode)

            # Update the embedding so that the model knows which roll is selected.
            embed = self.embed_roll_ids(roll_id).unsqueeze(1)

            choosen_rolls.append((distrib_roll, roll_id))

        actions = torch.stack(
            [
                choosen_tiles[0][1],
                choosen_rolls[0][1],
                choosen_tiles[1][1],
                choosen_rolls[1][1],
            ],
            dim=1,
        )
        logits = [
            choosen_tiles[0][0],
            choosen_rolls[0][0],
            choosen_tiles[1][0],
            choosen_rolls[1][0],
        ]
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
