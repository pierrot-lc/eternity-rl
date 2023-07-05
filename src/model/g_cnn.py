from itertools import product
from typing import Optional

import torch
import torch.nn as nn
from einops import repeat


class GCNN(nn.Conv2d):
    def __init__(
        self, in_channels: int, out_channels: int, reduce_factor: int, *args, **kwargs
    ):
        assert out_channels % reduce_factor == 0
        super().__init__(in_channels, out_channels // reduce_factor, *args, **kwargs)

        self.group_params = [
            (rot, flip)
            for rot, flip in product(range(4), [(2,), (3,), (2, 3), tuple()])
        ]

        self.reduce = nn.Conv2d(
            len(self.group_params) * out_channels // reduce_factor,
            out_channels,
            kernel_size=1,
            padding="same",
        )

    @property
    def transformed_weight(self):
        group_transformed_weight = [
            torch.flip(torch.rot90(self.weight, k=rot, dims=(2, 3)), dims=flip)
            for rot, flip in self.group_params
        ]
        group_transformed_weight = torch.concat(group_transformed_weight, dim=0)
        return group_transformed_weight

    @property
    def transformed_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None

        group_transformed_bias = repeat(
            self.bias, "c -> (r c)", r=len(self.group_params)
        )
        return group_transformed_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_forward(x, self.transformed_weight, self.transformed_bias)
        x = self.reduce(x)
        return x
