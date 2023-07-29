import torch
import torch.nn as nn
from tensordict import TensorDictBase
from torchrl.objectives.value.functional import vec_generalized_advantage_estimate

from ..model import Policy


class ReinforceLoss(nn.Module):
    def __init__(
        self,
        value_weight: float,
        entropy_weight: float,
        gamma: float,
        gae_lambda: float,
    ):
        super().__init__()

        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.mse_loss = nn.MSELoss()

    def advantages(self, traces: TensorDictBase):
        advantages, value_targets = vec_generalized_advantage_estimate(
            self.gamma,
            self.gae_lambda,
            traces["values"].unsqueeze(-1),
            traces["next-values"].unsqueeze(-1),
            traces["rewards"].unsqueeze(-1),
            traces["dones"].unsqueeze(-1),
        )
        traces["advantages"] = advantages.squeeze(-1)
        traces["value-targets"] = value_targets.squeeze(-1)

    def forward(self, batch: TensorDictBase, model: Policy) -> dict[str, torch.Tensor]:
        losses = dict()

        _, logprobs, entropies, values = model(
            batch["states"], "sample", batch["actions"]
        )

        # Compute the joint log probability of the actions.
        logprobs = logprobs.sum(dim=1)

        entropies[:, 0] *= 1.0
        entropies[:, 1] *= 0.5
        entropies[:, 2] *= 0.1
        entropies[:, 3] *= 0.1
        entropies = entropies.sum(dim=1)

        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        losses["policy"] = -(logprobs * advantages).mean()
        losses["value"] = self.value_weight * self.mse_loss(
            values, batch["value-targets"]
        )
        losses["entropy"] = -self.entropy_weight * entropies.mean()
        losses["total"] = sum(losses.values())
        return losses
