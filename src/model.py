import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch.distributions import Categorical

from .environment.gym import EternityEnv


class CNNPolicy(nn.Module):
    def __init__(
        self,
        n_classes: int,
        embedding_dim: int,
        n_layers: int,
        board_width: int,
        board_height: int,
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
                for _ in range(n_layers)
            ]
        )
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim * board_width * board_height, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )

        self.predict_actions = nn.ModuleDict(
            {
                "tile-1": nn.Linear(embedding_dim, board_width * board_height),
                "tile-2": nn.Linear(embedding_dim, board_width * board_height),
                "roll-1": nn.Linear(embedding_dim, 4),
                "roll-2": nn.Linear(embedding_dim, 4),
            }
        )

        self.predict_value = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Tanh(),
        )

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
        self, tiles: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the actions and value for the given game states.

        ---
        Args:
            tiles: The game state.
                Tensor of shape [batch_size, 4, board_height, board_width].
            timesteps: The timesteps of the game states.
                Long tensor of shape of [batch_size,].

        ---
        Returns:
            actions: The predicted actions.
                Shape of [batch_size, 4].
            logprobs: The log probabilities of the predicted actions.
                Shape of [batch_size, 4].
            values: The value of the given game states.
                Shape [batch_size, 1].
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

        # Compute action logits.
        tile_1 = self.predict_actions["tile-1"](embed)
        tile_1_id, tile_1_logprob = self.select_actions(tile_1)
        tile_1_emb = self.embed_tile_ids(tile_1_id)

        tile_2 = self.predict_actions["tile-2"](embed + tile_1_emb)
        tile_2_id, tile_2_logprob = self.select_actions(tile_2)
        tile_2_emb = self.embed_tile_ids(tile_2_id)

        roll_1 = self.predict_actions["roll-1"](embed + tile_1_emb)
        roll_1_id, roll_1_logprob = self.select_actions(roll_1)
        roll_2 = self.predict_actions["roll-2"](embed + tile_2_emb)
        roll_2_id, roll_2_logprob = self.select_actions(roll_2)

        actions = torch.stack([tile_1_id, roll_1_id, tile_2_id, roll_2_id], dim=1)
        logprobs = torch.stack(
            [tile_1_logprob, roll_1_logprob, tile_2_logprob, roll_2_logprob], dim=1
        )

        # Compute value.
        timesteps = self.embed_timesteps(timesteps)
        values = self.predict_value(embed + timesteps)

        return actions, logprobs, values

    @torch.no_grad()
    def solve_env(
        self,
        env: EternityEnv,
        device: str,
        seed: int = 0,
        intermediate_images: bool = False,
    ) -> tuple[float, int, list[np.ndarray]]:
        self.eval()
        self.to(device)
        rng = np.random.default_rng(seed)

        state = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        images = []

        while not done:
            state = torch.LongTensor(state).to(device).unsqueeze(0)
            tile_1, tile_2 = self(state)

            act_1, _ = self.select_actions(tile_1, rng)
            act_2, _ = self.select_actions(tile_2, rng)

            action = np.concatenate((act_1, act_2), axis=0)
            state, reward, done, _ = env.step(action)

            total_reward += reward
            episode_length += 1
            if intermediate_images:
                images.append(env.render(mode="rgb_array"))

        return total_reward, episode_length, images

    @staticmethod
    def select_actions(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from the given logits.
        Returns the sampled actions and their log-probilities.

        ---
        Args:
            logits: The logits of the actions.

        ---
        Returns:
            actions: The sampled actions.
                Shape [batch_size,].
            log_actions: The log-probabilities of the sampled actions.
                Shape of [batch_size,].
        """
        distribution = Categorical(logits=logits)
        action_ids = distribution.sample()
        log_actions = distribution.log_prob(action_ids)
        return action_ids, log_actions
