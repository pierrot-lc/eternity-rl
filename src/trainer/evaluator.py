from collections import defaultdict
from typing import Any

import einops
import torch
import wandb
from tensordict import TensorDictBase
from torch.optim.lr_scheduler import LRScheduler
from torchrl.data import ReplayBuffer
from wandb import Histogram

from ..environment import EternityEnv
from ..mcts import MCTSLoss
from ..model import Critic, Policy
from ..policy_gradient import PPOLoss


class Evaluator:
    def __init__(self):
        self.best_matches_found = 0

    @torch.inference_mode()
    def env_metrics(self, env: EternityEnv, suffix: str) -> dict[str, Any]:
        """Evaluate the metrics of the given env."""
        metrics = dict()

        matches = env.matches / env.best_possible_matches
        metrics[f"env-{suffix}/mean"] = matches.mean().item()
        metrics[f"env-{suffix}/best"] = (
            env.best_matches_ever / env.best_possible_matches
        )
        metrics[f"env-{suffix}/total-won"] = env.total_won
        metrics[f"env-{suffix}/n-steps"] = Histogram(env.n_steps.cpu())

        if self.best_matches_found < env.best_matches_ever:
            self.best_matches_found = env.best_matches_ever

            env.save_best_env("board.png")
            metrics["best-board"] = wandb.Image("board.png")

        return metrics

    @staticmethod
    def policy_rollouts_metrics(
        traces: TensorDictBase,
        replay_buffer: ReplayBuffer,
        loss: PPOLoss,
        policy: Policy,
        critic: Critic,
    ) -> dict[str, Any]:
        """Main metrics for the policy rollouts. Also estimate the gradients
        for the policy and the critic.
        """
        metrics = dict()
        masks = einops.rearrange(traces["masks"], "b d -> (b d)")
        value_targets = einops.rearrange(traces["value-targets"], "b d -> (b d)")
        advantages = einops.rearrange(traces["advantages"], "b d -> (b d)")
        actions = einops.rearrange(traces["actions"], "b d a -> (b d) a")

        value_targets = value_targets[masks]
        advantages = advantages[masks]
        actions = actions[masks]

        metrics["policy-rollouts/sum-rewards"] = Histogram(
            traces["rewards"].sum(dim=1).cpu()
        )
        metrics["policy-rollouts/value-targets"] = Histogram(value_targets.cpu())
        metrics["policy-rollouts/advantages"] = Histogram(advantages.cpu())
        for action_id in range(actions.shape[1]):
            metrics[f"policy-rollouts/action-{action_id+1}"] = Histogram(
                actions[:, action_id].cpu()
            )

        policy.train()
        critic.train()
        grads_policy, grads_critic = [], []
        losses = defaultdict(list)
        for batch in replay_buffer:
            policy.zero_grad()
            critic.zero_grad()

            loss_metrics = loss(batch.to(masks.device), policy, critic)

            loss_metrics["loss/total"].backward()
            for model, grads in [(policy, grads_policy), (critic, grads_critic)]:
                grads += [
                    param.grad.data.abs().mean().item()
                    for param in model.parameters()
                    if param.grad is not None
                ]

            for metric_name, metric_value in loss_metrics.items():
                metric_name = metric_name.replace("loss/", "ppo-loss/")
                metric_name = metric_name.replace("metrics/", "policy-rollouts/")
                losses[metric_name].append(metric_value.item())

        metrics["policy-rollouts/grads-policy"] = Histogram(grads_policy)
        metrics["policy-rollouts/grads-critic"] = Histogram(grads_critic)

        for metric_name, metric_values in losses.items():
            metrics[metric_name] = sum(metric_values) / len(metric_values)

        return metrics

    @staticmethod
    def mcts_rollouts_metrics(
        traces: TensorDictBase,
        replay_buffer: ReplayBuffer,
        loss: MCTSLoss,
        policy: Policy,
        critic: Critic,
    ) -> dict[str, Any]:
        """Main metrics for the MCTS rollouts. Also estimate the gradients
        for the policy and the critic.
        """
        metrics = dict()
        samples = {
            name: einops.rearrange(tensor, "b d ... -> (b d) ...")
            for name, tensor in traces.items()
        }

        metrics["mcts-rollouts/values"] = Histogram(samples["values"].cpu())
        metrics["mcts-rollouts/child-values"] = Histogram(
            samples["child-values"].flatten().cpu()
        )
        metrics["mcts-rollouts/probs"] = Histogram(samples["probs"].cpu())
        for action_id in range(samples["actions"].shape[2]):
            metrics[f"mcts-rollouts/action-{action_id+1}"] = Histogram(
                samples["actions"][:, :, action_id].flatten().cpu()
            )

        policy.train()
        critic.train()
        grads_policy, grads_critic = [], []
        losses = defaultdict(list)
        for batch in replay_buffer:
            policy.zero_grad()
            critic.zero_grad()

            loss_metrics = loss(batch.to(samples["values"].device), policy, critic)

            loss_metrics["loss/total"].backward()
            for model, grads in [(policy, grads_policy), (critic, grads_critic)]:
                grads += [
                    param.grad.data.abs().mean().item()
                    for param in model.parameters()
                    if param.grad is not None
                ]

            for metric_name, metric_value in loss_metrics.items():
                metric_name = metric_name.replace("loss/", "mcts-loss/")
                metric_name = metric_name.replace("metrics/", "mcts-rollouts/")
                losses[metric_name].append(metric_value.item())

        metrics["mcts-rollouts/grads-policy"] = Histogram(grads_policy)
        metrics["mcts-rollouts/grads-critic"] = Histogram(grads_critic)

        for metric_name, metric_values in losses.items():
            metrics[metric_name] = sum(metric_values) / len(metric_values)

        return metrics

    @staticmethod
    def models_metrics(
        model: torch.nn.Module,
        scheduler: LRScheduler,
        model_name: str,
    ) -> dict[str, Any]:
        """Compute the lr and weights of the model."""
        weights = [
            p.data.abs().mean().item() for p in model.parameters() if p.requires_grad
        ]

        return {
            f"{model_name}/lr": scheduler.get_last_lr()[0],
            f"{model_name}/weights": Histogram(weights),
        }
