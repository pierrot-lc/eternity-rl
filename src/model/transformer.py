import torch
import torch.nn as nn


class Swish(nn.Module):
    """Activation function defined here: https://arxiv.org/abs/1710.05941."""

    def __init__(self, trainable: bool):
        super().__init__()

        self.beta = nn.Parameter(torch.ones(1), requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class SwiGLU(nn.Module):
    """Activation function defined here: https://arxiv.org/abs/2002.05202.

    This implements the first linear layer and activation function of a transformer.
    """

    def __init__(self, d_model: int, dim_feedforward: int, learn_swish: bool):
        super().__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(d_model, dim_feedforward)
        self.swish = Swish(learn_swish)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the original x1 in the `dim_feedforward` space and apply
        the GLU.

        ---
        Args:
            x: The input to the transformer FFN.
                Shape of [..., d_model].

        ---
        Returns:
            The output of the SwiGLU activation function.
                Shape of [..., dim_feedforward].
        """
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        return x1 * self.swish(x2)


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Implements a model with SwiGLU activation functions in the MLP.

    We replace the `nn.TransformerEncoderLayer.linear1` module by `SwiGLU`, and
    nullify the `nn.TransformerEncoderLayer.activation` module.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        learn_swish: bool = False,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            nn.Identity(),
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype,
        )

        self.swish = Swish(trainable=learn_swish)
        self.linear1 = SwiGLU(d_model, dim_feedforward, learn_swish)


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    """Implements a model with SwiGLU activation functions in the MLP.

    We replace the `nn.TransformerDecoderLayer.linear1` module by `SwiGLU`, and
    nullify the `nn.TransformerDecoderLayer.activation` module.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        learn_swish: bool = False,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            nn.Identity(),
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype,
        )

        self.swish = Swish(trainable=learn_swish)
        self.linear1 = SwiGLU(d_model, dim_feedforward, learn_swish)
