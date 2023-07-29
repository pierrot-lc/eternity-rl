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
        ppo_clip: float,
    ):
        super().__init__()

        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip

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
        """
        ---
        Sources:
            PPO implementation: https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py
            Approximating KL Divergence: http://joschu.net/blog/kl-approx.html
        """
        metrics = dict()

        _, logprobs, entropies, values = model(
            batch["states"], "sample", batch["actions"]
        )

        # Compute the joint log probability of the actions.
        logprobs = logprobs.sum(dim=1)
        old_logprobs = batch["log-probs"].sum(dim=1)

        entropies[:, 0] *= 1.0
        entropies[:, 1] *= 0.5
        entropies[:, 2] *= 0.1
        entropies[:, 3] *= 0.1
        entropies = entropies.sum(dim=1)

        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        prob_ratios = (logprobs - old_logprobs).exp()
        clipped_ratios = torch.clamp(prob_ratios, 1 - self.ppo_clip, 1 + self.ppo_clip)
        gains = torch.stack(
            (
                prob_ratios * advantages,
                clipped_ratios * advantages,
            ),
            dim=-1,
        )
        metrics["policy"] = -gains.min(dim=-1).values.mean()

        old_values = batch["values"]
        clipped_values = old_values + torch.clamp(
            values - old_values, -self.ppo_clip, self.ppo_clip
        )
        value_losses = torch.stack(
            (
                (batch["value-targets"] - values).pow(2),
                (batch["value-targets"] - clipped_values).pow(2),
            ),
            dim=-1,
        )
        metrics["value"] = self.value_weight * value_losses.max(dim=-1).values.mean()

        metrics["entropy"] = -self.entropy_weight * entropies.mean()
        metrics["total"] = sum(metrics.values())

        # Some metrics to track, but that does not contribute to the loss.
        with torch.no_grad():
            metrics["approx-kl"] = (logprobs - old_logprobs).pow(2).mean() / 2
            metrics["clip-frac"] = (
                ((prob_ratios - 1.0).abs() > self.ppo_clip).float().mean()
            )

        return metrics
