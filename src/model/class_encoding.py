"""Encode class with random orthogonal vectors.
The number of classes is not really defined nor important for the model. It should only
learn to identify which tile is properly placed and which it should swap.

Source: https://discuss.pytorch.org/t/how-to-efficiently-and-randomly-produce-an-orthonormal-matrix/153609.
"""
import torch
import torch.nn as nn


class ClassEncoding(nn.Module):
    """Encode class with random orthogonal vectors.

    ---
    Parameters:
        embedding_dim: The dimension of the embedding.
            Make sure that the embedding dimension is greater than the
            maximum number of classes. Otherwise you won't be able to
            have enough orthogonal vectors.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Instantiate a random orthogonal matrix.
        gaussian_matrix = torch.randn((embedding_dim, embedding_dim))
        svd = torch.linalg.svd(gaussian_matrix)
        ortho = svd.U @ svd.Vh

        self.class_enc = nn.Embedding(
            num_embeddings=embedding_dim,
            embedding_dim=embedding_dim,
            _weight=ortho,
            _freeze=True,
        )

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        """Encode the classes.

        ---
        Args:
            board: The timesteps of the game states.
                Shape of [...].
        ---
        Returns:
            The board with encoded classes.
                Shape of [..., embedding_dim].
        """
        assert (
            board.max().item() < self.embedding_dim
        ), f"Not enough orthogonal vectors {board.max().item()} vs {self.embedding_dim}."

        # Encode the classes of each tile.
        return self.class_enc(board)
