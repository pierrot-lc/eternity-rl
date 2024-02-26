import torch
import torch.nn as nn
from einops import rearrange, repeat
from tensordict import TensorDictBase

from ..model import Critic, Policy


class MCTSLoss(nn.Module):
    def __init__(self, value_weight: float, entropy_weight: float, entropy_clip: float):
        super().__init__()

        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.entropy_clip = entropy_clip

        self.mse = nn.MSELoss()

    def forward(
        self, batch: TensorDictBase, policy: Policy, critic: Critic
    ) -> dict[str, torch.Tensor]:
        metrics = dict()
        n_actions = batch["actions"].shape[1]

        values = critic(batch["states"])
        metrics["loss/critic"] = self.mse(values, batch["values"])
        metrics["loss/weighted-critic"] = self.value_weight * metrics["loss/critic"]

        states = repeat(batch["states"], "b s h w -> b n s h w", n=n_actions)
        states = rearrange(states, "b n s h w -> (b n) s h w")
        actions = rearrange(batch["actions"], "b n a -> (b n) a")
        _, logprobs, entropies = policy(
            states, sampled_actions=actions, sampling_mode=None
        )
        logprobs = rearrange(logprobs, "(b n) a -> b n a", n=n_actions)

        metrics["loss/policy"] = -(batch["probs"] * logprobs.sum(dim=2)).mean()
        metrics["loss/weighted-policy"] = metrics["loss/policy"]

        metrics["loss/total"] = (
            metrics["loss/weighted-policy"] + metrics["loss/weighted-critic"]
        )

        return metrics
