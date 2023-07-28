import torch
import torch.nn as nn
from tensordict import TensorDict

from ..model import Policy


class ReinforceLoss(nn.Module):
    def __init__(self, value_weight: float, entropy_weight: float, gamma: float):
        super().__init__()

        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.gamma = gamma

        self.mse_loss = nn.MSELoss()

    def forward(self, batch: TensorDict, model: Policy) -> dict[str, torch.Tensor]:
        losses = dict()

        _, batch["logprobs"], batch["entropies"], batch["values"] = model(
            batch["states"], "sample", batch["actions"]
        )

        # Compute the joint log probability of the actions.
        batch["logprobs"] = batch["logprobs"].sum(dim=1)

        batch["entropies"][:, 0] *= 1.0
        batch["entropies"][:, 1] *= 0.5
        batch["entropies"][:, 2] *= 0.1
        batch["entropies"][:, 3] *= 0.1
        batch["entropies"] = batch["entropies"].sum(dim=1)

        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        losses["policy"] = -(batch["logprobs"] * advantages).mean()
        losses["value"] = self.value_weight * self.mse_loss(
            batch["values"], batch["next-values"].detach()
        )
        losses["entropy"] = -self.entropy_weight * batch["entropies"].mean()
        losses["total"] = sum(losses.values())
        return losses
