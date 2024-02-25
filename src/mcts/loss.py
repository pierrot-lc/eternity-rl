import torch
import torch.nn as nn
from tensordict import TensorDictBase

from ..model import Critic, Policy


class MCTSLoss(nn.Module):
    def __init__(self, value_weight: float, entropy_weight: float, entropy_clip: float):
        super().__init__()

        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.entropy_clip = entropy_clip

    def forward(
        self, batch: TensorDictBase, policy: Policy, critic: Critic
    ) -> dict[str, torch.Tensor]:
        pass
