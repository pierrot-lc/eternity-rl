import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

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

        self.flatten = nn.Sequential(
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

    def forward(
        self, tiles: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """

        Args:
            tiles: The game state.
                Tensor of shape [batch_size, 4, board_height, board_width].

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
        """
        embed = self.embed_tiles(tiles)
        for layer in self.residuals:
            embed = layer(embed) + embed
        embed = self.flatten(embed)
        tile_1 = {key: layer(embed) for key, layer in self.select_1.items()}
        tile_2 = {key: layer(embed) for key, layer in self.select_2.items()}
        return tile_1, tile_2

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
