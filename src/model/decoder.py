"""Alternate between RNN and cross-attention layers.
The cross-attention layers are used to extract the embedding of the game state.
The RNN layers are used to know which action has been taken in the past.
"""
from typing import Optional

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self,
        tile_embedding_dim: int,
        embedding_dim: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.rnns = nn.ModuleList(
            [nn.GRUCell(embedding_dim, embedding_dim) for _ in range(n_layers)]
        )
        self.cross_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    kdim=tile_embedding_dim,
                    vdim=tile_embedding_dim,
                    num_heads=1,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        tiles: torch.Tensor,
        query: Optional[torch.Tensor],
        hidden_state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Extract information from the game state and update the internal hidden state.

        ---
        Args:
            tiles: The game state.
                Tensor of shape [batch_size, board_height x board_width, tile_embedding_dim].
            query: The query to use for the cross-attention.
                Tensor of shape [batch_size, embedding_dim].
            hidden_state: The current hidden state of the RNN.
                Tensor of shape [batch_size, embedding_dim].

        ---
        Returns:
            The new hidden state of the model.
                Shape of [batch_size, embedding_dim].
        """
        if query is None:
            query = torch.zeros(
                (tiles.shape[0], self.embedding_dim),
                device=tiles.device,
            )

        # Add token dimension.
        query = query.unsqueeze(1)

        for rnn, cross_attention in zip(self.rnns, self.cross_attentions):
            # Extract the embedding from the game state.
            embedding, _ = cross_attention(query, tiles, tiles)
            embedding = embedding.squeeze(1)

            # Update the hidden state.
            hidden_state = rnn(embedding, hidden_state)

            # The next query is the updated hidden state.
            query = hidden_state.unsqueeze(1)

        return hidden_state
