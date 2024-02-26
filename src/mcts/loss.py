import torch
import torch.nn as nn
from tensordict import TensorDictBase
from einops import rearrange

from ..model import Critic, Policy
from ..environment import EternityEnv


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

        # WARNING: episode length should be set accordingly.
        env = EternityEnv(
            batch["states"], episode_length=float("+inf"), device=batch.device
        )
        env = EternityEnv.duplicate_interleave(env, n_duplicates=n_actions)

        actions = rearrange(batch["actions"], "b n a -> (b n) a")
        _, logprobs, entropies = policy(
            env.render(), sampled_actions=actions, sampling_mode=None
        )
        logprobs = rearrange(logprobs, "(b n) a -> b n a", n=n_actions)

        metrics["loss/policy"] = -(batch["probs"] * logprobs.sum(dim=2)).mean()
        metrics["loss/weighted-policy"] = metrics["loss/policy"]

        entropies = entropies.sum(dim=1)
        metrics["loss/entropy"] = -entropies.mean()
        metrics["loss/weighted-entropy"] = self.entropy_weight * metrics["loss/entropy"]
        entropy_penalty = torch.relu(self.entropy_clip - entropies).mean()
        metrics["loss/clipped-entropy"] = entropy_penalty

        states, *_ = env.step(actions)
        values = critic(states)
        values = rearrange(values, "(b n) -> b n", n=n_actions)
        metrics["loss/critic"] = self.mse(values, batch["values"])
        metrics["loss/weighted-critic"] = self.value_weight * metrics["loss/critic"]

        metrics["loss/total"] = (
            metrics["loss/weighted-policy"]
            + metrics["loss/weighted-critic"]
            + metrics["loss/weighted-entropy"]
            + metrics["loss/clipped-entropy"]
        )

        return metrics
