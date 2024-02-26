import torch
import torch.nn as nn
from tensordict import TensorDictBase
from torchrl.objectives.value.functional import vec_generalized_advantage_estimate

from ..model import Critic, Policy


class PPOLoss(nn.Module):
    """Compute the PPO loss.
    Also computes the advantages using the GAE algorithm.

    ---
    Parameters:
        value_weight: The weight of the value loss.
        entropy_weight: The weight of the entropy loss.
        gamma: The discount factor.
        gae_lambda: The GAE lambda parameter.
        ppo_clip_ac: The PPO action clipping parameter.
        ppo_clip_vf: The PPO value clipping parameter.
        no_value_function: Whether to deactivate the value function.

    ---
    Sources:
        PPO paper: https://arxiv.org/abs/1707.06347.
        PPO implementation: https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py.
        37 implementation details of PPO: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/.
        Approximating KL Divergence: http://joschu.net/blog/kl-approx.html.
    """

    def __init__(
        self,
        value_weight: float,
        entropy_weight: float,
        entropy_clip: float,
        gamma: float,
        gae_lambda: float,
        ppo_clip_ac: float,
        ppo_clip_vf: float,
    ):
        super().__init__()
        assert entropy_clip >= 0

        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.entropy_clip = entropy_clip
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip_ac = ppo_clip_ac
        self.ppo_clip_vf = ppo_clip_vf

        self.value_loss_fn = nn.HuberLoss(reduction="none")

    def advantages(self, traces: TensorDictBase):
        """Computes the advantages and value targets using the GAE algorithm.
        Adds the result to the `traces` dictionary.

        ---
        Args:
            traces: A dict containing the following entries:
                values: The rollout values of the given states.
                    Shape of [batch_size, steps].
                next-values: The rollout values of the next states.
                    Shape of [batch_size, steps].
                rewards: The rollout rewards of the given states.
                    Shape of [batch_size, steps].
                dones: The rollout dones of the given states.
                    Shape of [batch_size, steps].
        """
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

    def forward(
        self, batch: TensorDictBase, policy: Policy, critic: Critic
    ) -> dict[str, torch.Tensor]:
        """Computes the PPO loss for both actor and critic models.

        ---
        Args:
            batch: A dict containing the following entries:
                states: A batch of state observations.
                    Shape of [batch_size, N_SIDES, board_height, board_width].
                actions: The rollout actions taken in the given states.
                    Shape of [batch_size, n_actions].
                log-probs: The rollout log probabilities of the actions taken
                    in the given states.
                    Shape of [batch_size, n_actions].
                advantages: The rollout advantages of the actions taken in the
                    given states.
                    Shape of [batch_size,].
                value-targets: The rollout value targets of the given states.
                    Shape of [batch_size,].
            policy: The policy to evaluate.
            critic: The critic to evaluate.

        ---
        Returns:
            metrics: A dict containing the following entries:
                loss/policy: The PPO actor loss.
                loss/value: The PPO critic loss.
                loss/entropy: The entropy loss.
                loss/total: The sum of the policy, value and entropy losses.
                metrics/policy-approx-kl: The KL divergence between the old and new policies.
                metrics/policy-clip-frac: The clipped ratio between the old and new policies.
                metrics/value-clip-frac: The clipped ratio between the old and new values.
        """
        metrics = dict()

        _, logprobs, entropies = policy(
            batch["states"], sampling_mode=None, sampled_actions=batch["actions"]
        )
        values = critic(batch["states"])

        # Compute the joint log probability of the actions.
        logprobs = logprobs.sum(dim=1)
        old_logprobs = batch["log-probs"].sum(dim=1)

        advantages = batch["advantages"]
        mean, var = advantages.mean(), advantages.var()
        advantages = (advantages - mean) * torch.rsqrt(var + 1e-8)
        advantages = advantages + mean

        prob_ratios = (logprobs - old_logprobs).exp()
        clipped_ratios = torch.clamp(
            prob_ratios, 1 - self.ppo_clip_ac, 1 + self.ppo_clip_ac
        )
        gains = torch.stack(
            (
                prob_ratios * advantages,
                clipped_ratios * advantages,
            ),
            dim=-1,
        )
        metrics["loss/policy"] = -gains.min(dim=-1).values.mean()
        metrics["loss/weighted-policy"] = metrics["loss/policy"]

        old_values = batch["values"]
        clipped_values = torch.clamp(
            values, old_values - self.ppo_clip_vf, old_values + self.ppo_clip_vf
        )
        value_losses = torch.stack(
            (
                self.value_loss_fn(values, batch["value-targets"]),
                self.value_loss_fn(clipped_values, batch["value-targets"]),
            ),
            dim=-1,
        )
        metrics["loss/value"] = value_losses.max(dim=-1).values.mean()
        metrics["loss/weighted-value"] = self.value_weight * metrics["loss/value"]

        entropies = entropies.sum(dim=1)
        metrics["loss/entropy"] = -entropies.mean()
        metrics["loss/weighted-entropy"] = self.entropy_weight * metrics["loss/entropy"]
        entropy_penalty = torch.relu(self.entropy_clip - entropies).mean()
        metrics["loss/clipped-entropy"] = entropy_penalty

        metrics["loss/total"] = (
            metrics["loss/weighted-policy"]
            + metrics["loss/weighted-entropy"]
            + metrics["loss/clipped-entropy"]
            + metrics["loss/weighted-value"]
        )

        # Some metrics to track, but it does not contribute to the loss.
        with torch.no_grad():
            logr = logprobs - old_logprobs
            metrics["metrics/policy-approx-kl"] = ((logr.exp() - 1) - logr).mean()
            metrics["metrics/policy-clip-frac"] = (
                ((prob_ratios - 1.0).abs() > self.ppo_clip_ac).float().mean()
            )
            metrics["metrics/value-clip-frac"] = (
                ((values - old_values).abs() > self.ppo_clip_vf).float().mean()
            )
            metrics["metrics/entropy"] = entropies.mean()

        return metrics
