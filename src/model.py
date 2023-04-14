from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch.distributions import Categorical


class Head(nn.Module):
    def __init__(self, embedding_dim: int, n_head_layers: int, n_actions: int):
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
        board_width: int,
        board_height: int,
        zero_init_residuals: bool,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.position_enc = PositionalEncoding1D(embedding_dim)
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
            nn.Flatten(),
            nn.Linear(embedding_dim * board_width * board_height, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.gru = nn.GRU(
            embedding_dim, embedding_dim, num_layers=n_gru_layers, batch_first=False
        )

        self.predict_actions = nn.ModuleDict(
            {
                "tile-1": Head(
                    embedding_dim, n_head_layers, board_width * board_height
                ),
                "tile-2": nn.Sequential(
                    nn.Linear(2 * embedding_dim, embedding_dim),
                    Head(embedding_dim, n_head_layers, board_width * board_height),
                ),
                "roll-1": nn.Sequential(
                    nn.Linear(3 * embedding_dim, embedding_dim),
                    Head(embedding_dim, n_head_layers, 4),
                ),
                "roll-2": nn.Sequential(
                    nn.Linear(3 * embedding_dim, embedding_dim),
                    Head(embedding_dim, n_head_layers, 4),
                ),
            }
        )

        self.predict_value = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Tanh(),
        )

        if zero_init_residuals:
            self.init_residuals()

    def init_residuals(self):
        """Zero out the weights of the residual convolutional layers."""
        for layer in self.residuals:
            for param in layer[0].parameters():
                param.data.zero_()

    def embed_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed the timesteps.

        ---
        Args:
            timesteps: The timesteps of the game states.
                Shape of [batch_size,].

        ---
        Returns:
            The positional encodings for the given timesteps.
                Shape of [batch_size, embedding_dim].
        """
        # Compute the positional encodings for the given timesteps.
        max_timesteps = int(timesteps.max().item())
        x = torch.zeros(
            1, max_timesteps + 1, self.embedding_dim, device=timesteps.device
        )
        encodings = self.position_enc(x)
        encodings = encodings.squeeze(0)  # Shape is [timesteps, embedding_dim].

        # Select the right encodings for the timesteps.
        encodings = repeat(encodings, "t e -> b t e", b=timesteps.shape[0])
        timesteps = repeat(timesteps, "b -> b t e", t=1, e=self.embedding_dim)
        encodings = torch.gather(encodings, dim=1, index=timesteps)

        encodings = encodings.squeeze(1)  # Shape is [batch_size, embedding_dim].
        return encodings

    def forward(
        self, tiles: torch.Tensor, hidden_memory: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Compute values.
        values = self.predict_value(embed)

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
            [entropies_tile_1, entropies_roll_1, entropies_tile_2, entropies_roll_2],
            dim=1,
        )

        return actions, logprobs, values, hidden_memory, entropies

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
                Shape of [batch_size,].
        """
        # Sample the actions using the nucleus sampling.
        distributions = torch.softmax(logits, dim=-1)
        action_ids = Categorical(probs=distributions).sample()

        # Compute the entropies of the true distribution.
        categorical = Categorical(probs=distributions)
        entropies = categorical.entropy()
        log_actions = categorical.log_prob(action_ids)
        return action_ids, log_actions, entropies
