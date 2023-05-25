"""A simple time encoding module.
Giving this to the model makes the observation fully Markovian.
"""
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D
from einops import repeat

class TimeEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.position_enc = PositionalEncoding1D(embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
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
        encodings = self.position_enc(x).to(timesteps.device)
        encodings = encodings.squeeze(0)  # Shape is [timesteps, embedding_dim].

        # Select the right encodings for the timesteps.
        encodings = repeat(encodings, "t e -> b t e", b=timesteps.shape[0])
        timesteps = repeat(timesteps, "b -> b t e", t=1, e=self.embedding_dim)
        encodings = torch.gather(encodings, dim=1, index=timesteps)

        encodings = encodings.squeeze(1)  # Shape is [batch_size, embedding_dim].
        return encodings
