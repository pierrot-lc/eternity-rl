import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange
from positional_encodings.torch_encodings import PositionalEncoding1D

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
        self.embed_tiles = nn.Sequential(
            Rearrange("b t h w -> b h w t"),
            nn.Embedding(n_classes, embedding_dim),
            nn.LayerNorm(embedding_dim),
            Rearrange("b h w t e -> b (t e) h w"),
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

        self.select_1 = nn.ModuleDict(
            {
                "tile": nn.Linear(embedding_dim, board_width * board_height),
                "roll": nn.Linear(embedding_dim, 4),
            }
        )
        self.select_2 = nn.ModuleDict(
            {
                "tile": nn.Linear(embedding_dim, board_width * board_height),
                "roll": nn.Linear(embedding_dim, 4),
            }
        )

        self.value = nn.Sequential(
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
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """Predict the actions and value for the given game states.

        ---
        Args:
            tiles: The game state.
                Tensor of shape [batch_size, 4, board_height, board_width].
            timesteps: The timesteps of the game states.
                Long tensor of shape of [batch_size,].

        ---
        Returns:
            tile_1: The logits for the first tile.
                Dict of shape {
                    "tile": [batch_size, board_width * board_height],
                    "roll": [batch_size, 4],
                }
            tile_2: The logits for the second tile.
                Dict of shape {
                    "tile": [batch_size, board_width * board_height],
                    "roll": [batch_size, 4],
                }
            value: The value of the given game state.
                Shape [batch_size, 1].
        """
        # Compute game embeddings.
        embed = self.embed_tiles(tiles)
        for layer in self.residuals:
            embed = layer(embed) + embed
        embed = self.project(embed)
        timesteps = self.embed_timesteps(timesteps)

        # Compute action logits.
        tile_1 = {key: layer(embed) for key, layer in self.select_1.items()}
        tile_2 = {key: layer(embed) for key, layer in self.select_2.items()}

        # Compute value.
        value = self.value(embed + timesteps)

        return tile_1, tile_2, value

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
    def select_actions(
        tile_logits: dict[str, torch.Tensor], rng: np.random.Generator
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Sample actions from the given tile and roll logits.
        Returns their numpy actions and the torch log probabilities.
        """
        actions, log_actions = [], []
        for action_name in ["tile", "roll"]:
            logits = tile_logits[action_name]
            distribution = torch.softmax(logits, dim=-1)
            action = rng.choice(
                np.arange(logits.shape[1]),
                size=(1,),
                p=distribution.cpu().detach().numpy()[0],
            )

            actions.append(action)
            log_actions.append(torch.log(distribution[0, action] + 1e-5))

        return (np.array(actions), torch.concat(log_actions))
