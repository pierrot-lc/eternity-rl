from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions import Categorical
from torchinfo import summary


class Head(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_head_layers: int,
        n_actions: int,
        zero_init_residuals: bool,
    ):
        super().__init__()

        self.residuals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.GELU(),
                    nn.LayerNorm(embedding_dim),
                )
                for _ in range(n_head_layers)
            ]
        )
        self.predict_actions = nn.Linear(embedding_dim, n_actions)

        if zero_init_residuals:
            self.init_residuals()

    def init_residuals(self):
        """Zero out the weights of the residual linear layers."""
        for module in self.residuals.modules():
            if isinstance(module, nn.Linear):
                for param in module.parameters():
                    param.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the actions for the given game states.

        ---
        Args:
            x: The game states.
                Shape of [batch_size, embedding_dim].

        ---
        Returns:
            The predicted logits actions.
                Shape of [batch_size, n_actions].
        """
        for layer in self.residuals:
            x = layer(x) + x
        return self.predict_actions(x)


class CNNPolicy(nn.Module):
    def __init__(
        self,
        n_classes: int,
        embedding_dim: int,
        n_res_layers: int,
        n_gru_layers: int,
        n_head_layers: int,
        maxpool_kernel: int,
        board_width: int,
        board_height: int,
        zero_init_residuals: bool,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.board_width = board_width
        self.board_height = board_height

        self.embed_classes = nn.Sequential(
            nn.Embedding(n_classes, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.embed_tile_ids = nn.Sequential(
            nn.Embedding(board_width * board_height, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.embed_board = nn.Sequential(
            nn.Conv2d(4 * embedding_dim, embedding_dim, 3, padding="same"),
            nn.GELU(),
            nn.LayerNorm([embedding_dim, board_height, board_width]),
        )
        self.residuals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embedding_dim, embedding_dim, 3, padding="same"),
                    nn.GELU(),
                    nn.LayerNorm([embedding_dim, board_height, board_width]),
                )
                for _ in range(n_res_layers)
            ]
        )
        self.project = nn.Sequential(
            nn.MaxPool2d(kernel_size=maxpool_kernel),
            nn.Flatten(),
            nn.LazyLinear(embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.gru = nn.GRU(
            embedding_dim, embedding_dim, num_layers=n_gru_layers, batch_first=False
        )

        self.predict_actions = nn.ModuleDict(
            {
                "tile-1": Head(
                    embedding_dim,
                    n_head_layers,
                    board_width * board_height,
                    zero_init_residuals,
                ),
                "tile-2": nn.Sequential(
                    nn.Linear(2 * embedding_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    Head(
                        embedding_dim,
                        n_head_layers,
                        board_width * board_height,
                        zero_init_residuals,
                    ),
                ),
                "roll-1": nn.Sequential(
                    nn.Linear(3 * embedding_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    Head(embedding_dim, n_head_layers, 4, zero_init_residuals),
                ),
                "roll-2": nn.Sequential(
                    nn.Linear(3 * embedding_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    Head(embedding_dim, n_head_layers, 4, zero_init_residuals),
                ),
            }
        )

        if zero_init_residuals:
            self.init_residuals()

        # Init lazy layers.
        with torch.no_grad():
            dummy_input = self.dummy_input("cpu")
            self.forward(dummy_input)

    def init_residuals(self):
        """Zero out the weights of the residual convolutional layers."""
        for module in self.residuals.modules():
            if isinstance(module, nn.Conv2d):
                for param in module.parameters():
                    param.data.zero_()

    def dummy_input(self, device: str) -> torch.Tensor:
        return torch.zeros(
            1,
            4,
            self.board_height,
            self.board_width,
            dtype=torch.long,
            device=device,
        )

    def summary(self, device: str):
        """Torchinfo summary."""
        dummy_input = self.dummy_input(device)
        summary(
            self,
            input_data=[dummy_input],
            device=device,
        )

    def forward(
        self, tiles: torch.Tensor, hidden_memory: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Predict the actions and value for the given game states.

        ---
        Args:
            tiles: The game state.
                Tensor of shape [batch_size, 4, board_height, board_width].
            hidden_memory: Memory of the GRU.
                Tensor of shape of [n_gru_layers, embedding_dim].

        ---
        Returns:
            actions: The predicted actions.
                Shape of [batch_size, 4].
            logprobs: The log probabilities of the predicted actions.
                Shape of [batch_size, 4].
            values: The value of the given game states.
                Shape [batch_size, 1].
            hidden_memory: Updated memory of the GRU.
                Shape of [n_gru_layers, embedding_dim].
            entropies: The entropy of the predicted actions.
                Shape of [batch_size, 4].
        """
        # Compute game embeddings.
        tiles = rearrange(tiles, "b t w h -> b h w t")
        embed = self.embed_classes(tiles)

        embed = rearrange(embed, "b h w t e -> b (t e) h w")
        embed = self.embed_board(embed)

        # Shape is of [batch_size, embedding_dim, height, width].
        for layer in self.residuals:
            embed = layer(embed) + embed

        embed = self.project(embed)
        # Shape is of [batch_size, embedding_dim].

        # Add and remove the sequence length dimension.
        embed = embed.unsqueeze(0)
        embed, hidden_memory = self.gru(embed, hidden_memory)
        embed = embed.squeeze(0)

        # Compute action logits.
        tile_1 = self.predict_actions["tile-1"](embed)
        tile_1_id, tile_1_logprob, entropies_tile_1 = self.select_actions(tile_1)
        tile_1_emb = self.embed_tile_ids(tile_1_id)

        # Shape of [batch_size, 2 * embedding_dim].
        embed = torch.concat([embed, tile_1_emb], dim=-1)

        tile_2 = self.predict_actions["tile-2"](embed)
        tile_2_id, tile_2_logprob, entropies_tile_2 = self.select_actions(tile_2)
        tile_2_emb = self.embed_tile_ids(tile_2_id)

        # Shape of [batch_size, 3 * embedding_dim].
        embed = torch.concat([embed, tile_2_emb], dim=-1)

        roll_1 = self.predict_actions["roll-1"](embed)
        roll_1_id, roll_1_logprob, entropies_roll_1 = self.select_actions(roll_1)
        roll_2 = self.predict_actions["roll-2"](embed)
        roll_2_id, roll_2_logprob, entropies_roll_2 = self.select_actions(roll_2)

        actions = torch.stack([tile_1_id, roll_1_id, tile_2_id, roll_2_id], dim=1)
        logprobs = torch.stack(
            [tile_1_logprob, roll_1_logprob, tile_2_logprob, roll_2_logprob], dim=1
        )
        entropies = torch.stack(
            [
                1.0 * entropies_tile_1,
                0.01 * entropies_roll_1,
                0.1 * entropies_tile_2,
                0.01 * entropies_roll_2,
            ],
            dim=1,
        )

        return actions, logprobs, hidden_memory, entropies

    def select_actions(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from the given logits.
        Returns the sampled actions and their log-probilities.

        ---
        Args:
            logits: The logits of the actions.
                Shape of [batch_size, n_actions].

        ---
        Returns:
            actions: The sampled actions.
                Shape [batch_size,].
            log_actions: The log-probabilities of the sampled actions.
                Shape of [batch_size,].
            entropies: The entropy of the categorical distributions.
                The entropies are normalized by the log of the number of actions.
                Shape of [batch_size,].
        """
        n_actions = logits.shape[-1]

        # Sample the actions using the nucleus sampling.
        distributions = torch.softmax(logits, dim=-1)
        action_ids = Categorical(probs=distributions).sample()

        # Compute the entropies of the true distribution.
        categorical = Categorical(probs=distributions)
        entropies = categorical.entropy() / np.log(n_actions)
        log_actions = categorical.log_prob(action_ids)
        return action_ids, log_actions, entropies
