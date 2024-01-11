from itertools import count
from pathlib import Path
from typing import Any

import einops
import torch
import torch.optim as optim
import wandb
from tensordict import TensorDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad
from torchrl.data import ReplayBuffer
from tqdm import tqdm

from ..environment import EternityEnv
from ..model import Critic, Policy
from .loss import PPOLoss
from .rollout import rollout, split_reset_rollouts


class Trainer:
    def __init__(
        self,
        env: EternityEnv,
        policy: Policy | DDP,
        critic: Critic | DDP,
        loss: PPOLoss,
        policy_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        replay_buffer: ReplayBuffer,
        clip_value: float,
        episodes: int,
        epochs: int,
        rollouts: int,
        reset_proportion: float,
    ):
        self.env = env
        self.policy = policy
        self.critic = critic
        self.loss = loss
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.replay_buffer = replay_buffer
        self.clip_value = clip_value
        self.episodes = episodes
        self.epochs = epochs
        self.rollouts = rollouts
        self.reset_proportion = reset_proportion

        self.device = env.device
        self.rng = self.env.rng

        # Dynamic infos.
        self.best_matches_found = 0

    @torch.inference_mode()
    def do_rollouts(self, disable_logs: bool):
        """Simulates a bunch of rollouts and adds them to the replay buffer."""
        total_resets = int(self.reset_proportion * self.env.batch_size)
        reset_ids = torch.randperm(
            self.env.batch_size, generator=self.rng, device=self.device
        )
        reset_ids = reset_ids[:total_resets]
        self.env.reset(reset_ids)

        traces = rollout(
            self.env,
            self.policy,
            self.critic,
            self.rollouts,
            disable_logs,
        )

        traces = split_reset_rollouts(traces)
        self.loss.advantages(traces)

        # assert torch.all(
        #     traces["rewards"].sum(dim=1) <= 1 + 1e-4  # Account for small FP errors.
        # ), f"Some returns are > 1 ({traces['rewards'].sum(dim=1).max().item()})."
        assert (
            self.replay_buffer._storage.max_size
            == traces["masks"].sum().item()
            == self.env.batch_size * self.rollouts
        ), "Some samples are missing."

        # Flatten the batch x steps dimensions and remove the masked steps.
        samples = dict()
        masks = einops.rearrange(traces["masks"], "b d -> (b d)")
        for name, tensor in traces.items():
            if name == "masks":
                continue

            tensor = einops.rearrange(tensor, "b d ... -> (b d) ...")
            samples[name] = tensor[masks]

        samples = TensorDict(
            samples, batch_size=samples["states"].shape[0], device=self.device
        )
        self.replay_buffer.extend(samples)

    def do_batch_update(
        self, batch: TensorDict, train_policy: bool, train_critic: bool
    ):
        """Performs a batch update on the model."""
        if train_policy:
            self.policy.train()
            self.policy_optimizer.zero_grad()

        if train_critic:
            self.critic.train()
            self.critic_optimizer.zero_grad()

        batch = batch.to(self.device)
        metrics = self.loss(batch, self.policy, self.critic)
        metrics["loss/total"].backward()

        if train_policy:
            clip_grad.clip_grad_norm_(self.policy.parameters(), self.clip_value)
            self.policy_optimizer.step()

        if train_critic:
            clip_grad.clip_grad_norm_(self.critic.parameters(), self.clip_value)
            self.critic_optimizer.step()

    def launch_training(
        self,
        group: str,
        config: dict[str, Any],
        mode: str = "online",
        save_every: int = 500,
    ):
        """Launches the training loop.

        ---
        Args:
            group: The name of the group for the run.
                Useful for grouping runs together.
            config: The configuration of the run.
                Only used for logging purposes.
            mode: The mode of the run. Can be:
                - "online": The run is logged online to W&B.
                - "offline": The run is logged offline to W&B.
                - "disabled": The run does not produce any output.
                    Useful for multi-GPU training.
            eval_every: The number of rollouts between each evaluation.
        """
        disable_logs = mode == "disabled"

        with wandb.init(
            project="eternity-rl",
            entity="pierrotlc",
            group=group,
            config=config,
            mode=mode,
        ) as run:
            self.policy.to(self.device)
            self.env.reset()
            self.best_matches_found = 0  # Reset.

            if not disable_logs:
                if isinstance(self.policy, DDP):
                    self.policy.module.summary(self.device)
                    self.critic.module.summary(self.device)
                else:
                    self.policy.summary(self.device)
                    self.critic.summary(self.device)

                print(f"\nLaunching training on device {self.device}.\n")

            # Log gradients and model parameters.
            run.watch(self.policy)
            run.watch(self.critic)

            # Infinite loop if n_batches is -1.
            iter = count(0) if self.episodes == -1 else range(self.episodes)

            for i in tqdm(iter, desc="Episode", disable=disable_logs):
                self.do_rollouts(disable_logs=disable_logs)

                for _ in tqdm(
                    range(self.epochs), desc="Epoch", leave=False, disable=disable_logs
                ):
                    for batch in tqdm(
                        self.replay_buffer,
                        total=len(self.replay_buffer) // self.replay_buffer._batch_size,
                        desc="Batch",
                        leave=False,
                        disable=disable_logs,
                    ):
                        self.do_batch_update(
                            batch, train_policy=True, train_critic=True
                        )

                metrics = self.evaluate()
                run.log(metrics)

                if i % save_every == 0 and not disable_logs:
                    self.save_checkpoint("checkpoint.pt")
                    self.env.save_sample("sample.gif")

    def evaluate(self) -> dict[str, Any]:
        """Evaluates the model and returns some computed metrics."""
        metrics = dict()

        self.policy.eval()
        self.critic.eval()

        matches = self.env.matches / self.env.best_possible_matches
        metrics["matches/mean"] = matches.mean()
        metrics["matches/best"] = (
            self.env.best_matches_ever / self.env.best_possible_matches
        )
        metrics["matches/rolling"] = (
            self.env.rolling_matches / self.env.best_possible_matches
        )
        metrics["matches/total-won"] = self.env.total_won

        # Compute losses.
        batch = self.replay_buffer.sample()
        batch = batch.to(self.device)
        metrics |= self.loss(batch, self.policy, self.critic)
        metrics["metrics/value-targets"] = wandb.Histogram(batch["value-targets"].cpu())
        metrics["metrics/n-steps"] = wandb.Histogram(self.env.n_steps.cpu())

        # Compute the gradient mean and maximum values.
        # Also computes the weight absolute mean and maximum values.
        self.policy.zero_grad()
        self.critic.zero_grad()

        metrics["loss/total"].backward()
        weights, grads = [], []
        for p in self.policy.parameters():
            weights.append(p.data.abs().mean().item())

            if p.grad is not None:
                grads.append(p.grad.data.abs().mean().item())

        metrics["global-weights/mean"] = sum(weights) / (len(weights) + 1)
        metrics["global-weights/max"] = max(weights)
        metrics["global-weights/hist"] = wandb.Histogram(weights)

        metrics["global-gradients/mean"] = sum(grads) / (len(grads) + 1)
        metrics["global-gradients/max"] = max(grads)
        metrics["global-gradients/hist"] = wandb.Histogram(grads)

        self.policy.zero_grad()
        self.critic.zero_grad()

        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[name] = value.item()

        if self.best_matches_found < self.env.best_matches_ever:
            self.best_matches_found = self.env.best_matches_ever

            self.env.save_best_env("board.png")
            metrics["best-board"] = wandb.Image("board.png")

        return metrics

    def save_checkpoint(self, filepath: Path | str):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "policy-optimizer": self.policy_optimizer.state_dict(),
                "critic-optimizer": self.critic_optimizer.state_dict(),
            },
            filepath,
        )
