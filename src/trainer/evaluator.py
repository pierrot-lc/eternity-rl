from typing import Any

import torch
import wandb
from tensordict import TensorDictBase
from torch.optim.lr_scheduler import LRScheduler
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
        metrics[f"env/{suffix}-mean"] = matches.mean().item()
        metrics[f"env/{suffix}-best"] = (
            env.best_matches_ever / env.best_possible_matches
        )
        metrics[f"env/{suffix}-total-won"] = env.total_won
        metrics[f"env/{suffix}-n-steps"] = Histogram(env.n_steps.cpu())

        if self.best_matches_found < env.best_matches_ever:
            self.best_matches_found = env.best_matches_ever

            env.save_best_env("board.png")
            metrics["best-board"] = wandb.Image("board.png")

        return metrics

    @staticmethod
    def rollout_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
        return {
            metric_name.replace("metrics/", f"{prefix}/"): metric_value
            for metric_name, metric_value in metrics.items()
        }

    @staticmethod
    def batch_metrics(
        batch: TensorDictBase,
        policy: Policy,
        critic: Critic,
        loss: PPOLoss | MCTSLoss,
        metrics_prefix: str,
    ) -> dict[str, Any]:
        metrics = dict()
        policy.zero_grad()
        critic.zero_grad()

        loss_metrics = loss(batch, policy, critic)
        loss_metrics["loss/total"].backward()
        for model, model_name in [(policy, "policy"), (critic, "critic")]:
            grads = [
                param.grad.data.abs().mean().item()
                for param in model.parameters()
                if param.grad is not None
            ]
            metrics[f"{model_name}/{metrics_prefix}-gradients"] = Histogram(grads)

        for metric_name, metric_value in loss_metrics.items():
            metric_name = metric_name.replace("loss/", f"{metrics_prefix}/loss-")
            metric_name = metric_name.replace("metrics/", f"{metrics_prefix}/")
            metrics[metric_name] = metric_value.item()

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
