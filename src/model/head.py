import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        head_layers: int,
        n_actions: int,
    ):
        super().__init__()

        self.residuals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.GELU(),
                    nn.LayerNorm(embedding_dim),
                )
                for _ in range(head_layers)
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
