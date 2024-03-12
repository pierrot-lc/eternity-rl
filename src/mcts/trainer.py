from itertools import chain, count
from pathlib import Path
from typing import Any

import einops
import torch
import torch.optim as optim
from tensordict import TensorDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad
from torch.optim.lr_scheduler import LRScheduler
from torchrl.data import ReplayBuffer
from tqdm import tqdm

import wandb

from ..environment import EternityEnv
from ..model import Critic, Policy
from ..policy_gradient.rollout import mcts_rollout, rollout, split_reset_rollouts
from .loss import MCTSLoss
from .tree import MCTSTree
from ..policy_gradient.loss import PPOLoss


class MCTSTrainer:
    def __init__(
        self,
        env: EternityEnv,
        policy: Policy | DDP,
        critic: Critic | DDP,
        mcts: MCTSTree,
        loss_ppo: PPOLoss,
        loss_mcts: MCTSLoss,
        policy_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        policy_scheduler: LRScheduler,
        critic_scheduler: LRScheduler,
        replay_buffer_ppo: ReplayBuffer,
        replay_buffer_mcts: ReplayBuffer,
        clip_value: float,
        episodes: int,
        epochs: int,
        rollouts: int,
        reset_proportion: float,
    ):
        self.env = env
        self.policy = policy
        self.critic = critic
        self.mcts = mcts
        self.loss_mcts = loss_mcts
        self.loss_ppo = loss_ppo
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.policy_scheduler = policy_scheduler
        self.critic_scheduler = critic_scheduler
        self.replay_buffer_ppo = replay_buffer_ppo
        self.replay_buffer_mcts = replay_buffer_mcts
        self.clip_value = clip_value
        self.episodes = episodes
        self.epochs = epochs
        self.rollouts = rollouts
        self.reset_proportion = reset_proportion

        self.policy_module = (
            self.policy.module if isinstance(self.policy, DDP) else self.policy
        )
        self.critic_module = (
            self.critic.module if isinstance(self.critic, DDP) else self.critic
        )

        self.device = env.device
        self.rng = self.env.rng

        # Dynamic infos.
        self.best_matches_found = 0

    @torch.inference_mode()
    def do_mcts_rollouts(self, disable_logs: bool, sampling_mode: str):
        """Simulates a bunch of rollouts and adds them to the replay buffer."""
        total_resets = int(self.reset_proportion * self.env.batch_size)
        reset_ids = torch.randperm(
            self.env.batch_size, generator=self.rng, device=self.device
        )
        reset_ids = reset_ids[:total_resets]
        self.env.reset(reset_ids)

        # We collect the rollouts with the current policy.
        self.policy_module.eval()
        self.critic_module.train()
        traces = mcts_rollout(
            self.env,
            self.policy_module,
            self.critic_module,
            self.mcts,
            self.rollouts,
            disable_logs,
            sampling_mode=sampling_mode,
        )

        # Flatten the batch x steps dimensions and remove the masked steps.
        samples = dict()
        for name, tensor in traces.items():
            samples[name] = einops.rearrange(tensor, "b d ... -> (b d) ...")

        samples = TensorDict(
            samples, batch_size=samples["states"].shape[0], device=self.device
        )
        self.replay_buffer_mcts.extend(samples)

    @torch.inference_mode()
    def do_ppo_rollouts(self, disable_logs: bool, sampling_mode: str):
        total_resets = int(self.reset_proportion * self.env.batch_size)
        reset_ids = torch.randperm(
            self.env.batch_size, generator=self.rng, device=self.device
        )
        reset_ids = reset_ids[:total_resets]
        self.env.reset(reset_ids)

        # We collect the rollouts with the current policy.
        self.policy_module.train()
        self.critic_module.train()
        traces = rollout(
            self.env,
            self.policy_module,
            self.critic_module,
            self.rollouts,
            disable_logs,
        )

        traces = split_reset_rollouts(traces)
        self.loss_ppo.advantages(traces)

        assert torch.all(
            traces["rewards"].sum(dim=1) <= 1 + 1e-4  # Account for small FP errors.
        ), f"Some returns are > 1 ({traces['rewards'].sum(dim=1).max().item()})."
        assert (
            self.replay_buffer_ppo._storage.max_size
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
        self.replay_buffer_ppo.extend(samples)

    def do_batch_update(
        self,
        batch: TensorDict,
        loss: torch.nn.Module,
        train_policy: bool,
        train_critic: bool,
    ):
        """Performs a batch update on the model."""
        if train_policy:
            self.policy.train()
            self.policy_optimizer.zero_grad()

        if train_critic:
            self.critic.train()
            self.critic_optimizer.zero_grad()

        batch = batch.to(self.device)
        metrics = loss(batch, self.policy, self.critic)
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
            self.critic.to(self.device)

            self.env.reset()
            self.best_matches_found = 0  # Reset.

            if not disable_logs:
                self.policy_module.summary(
                    self.env.board_size, self.env.board_size, self.device
                )
                self.critic_module.summary(
                    self.env.board_size, self.env.board_size, self.device
                )
                print(f"\nLaunching training on device {self.device}.\n")

            # Log gradients and model parameters.
            run.watch(self.policy)
            run.watch(self.critic)

            # Infinite loop if n_batches is -1.
            iter = count(0) if self.episodes == -1 else range(self.episodes)

            for i in tqdm(iter, desc="Episode", disable=disable_logs):
                self.do_mcts_rollouts(disable_logs=disable_logs, sampling_mode="greedy")
                print("\nFinal best matches:", self.env.best_matches, "\n\n")
                print("Final matches:", self.env.matches, "\n\n")
                print("\n\nBEST MATCHES:", self.env.best_matches_ever, "\n")
                return

                for _ in tqdm(
                    range(self.epochs * 5),
                    desc="Epoch",
                    leave=False,
                    disable=disable_logs,
                ):
                    for batch in tqdm(
                        self.replay_buffer_mcts,
                        total=len(self.replay_buffer_mcts)
                        // self.replay_buffer_mcts._batch_size,
                        desc="Batch",
                        leave=False,
                        disable=disable_logs,
                    ):
                        self.do_batch_update(
                            batch,
                            loss=self.loss_mcts,
                            train_policy=False,
                            train_critic=True,
                        )

                for _ in tqdm(
                    range(10), desc="PPO sub-epoch", leave=False, disable=disable_logs
                ):
                    self.do_ppo_rollouts(disable_logs, "softmax")

                    for _ in tqdm(
                        range(self.epochs),
                        desc="Epoch",
                        leave=False,
                        disable=disable_logs,
                    ):
                        for batch in tqdm(
                            self.replay_buffer_ppo,
                            total=len(self.replay_buffer_ppo)
                            // self.replay_buffer_ppo._batch_size,
                            desc="Batch",
                            leave=False,
                            disable=disable_logs,
                        ):
                            self.do_batch_update(
                                batch,
                                loss=self.loss_ppo,
                                train_policy=True,
                                train_critic=False,
                            )

                self.policy_scheduler.step()
                self.critic_scheduler.step()

                if not disable_logs:
                    metrics = self.evaluate()
                    run.log(metrics)

                    if i % save_every == 0:
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

        # Compute losses and other metrics.
        batch = self.replay_buffer_mcts.sample()
        batch = batch.to(self.device)
        metrics |= self.loss_mcts(batch, self.policy, self.critic)
        metrics["metrics/values"] = wandb.Histogram(batch["values"].cpu())
        metrics["metrics/n-steps"] = wandb.Histogram(self.env.n_steps.cpu())
        for action_id in range(batch["actions"].shape[2]):
            actions = batch["actions"][:, :, action_id].flatten().cpu()
            metrics[f"metrics/action-{action_id}"] = wandb.Histogram(actions)

        # Compute the gradient mean and maximum values.
        # Also computes the weight absolute mean and maximum values.
        self.policy.zero_grad()
        self.critic.zero_grad()

        metrics["loss/total"].backward()
        weights, grads = [], []
        for p in chain(self.policy.parameters(), self.critic.parameters()):
            weights.append(p.data.abs().mean().item())

            if p.grad is not None:
                grads.append(p.grad.data.abs().mean().item())

        metrics["lr/policy"] = self.policy_scheduler.get_last_lr()[0]
        metrics["lr/critic"] = self.critic_scheduler.get_last_lr()[0]

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
                "policy-scheduler": self.policy_scheduler.state_dict(),
                "critic-scheduler": self.critic_scheduler.state_dict(),
            },
            filepath,
        )
