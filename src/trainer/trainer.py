from itertools import count
from pathlib import Path
from typing import Any

import einops
import torch
import wandb
from tensordict import TensorDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from ..environment import EternityEnv
from ..mcts import MCTSConfig, MCTSLoss, MCTSTree
from ..model import Critic, Policy
from ..policy_gradient import PPOLoss
from ..rollouts import (
    mcts_rollouts,
    policy_rollouts,
    split_reset_rollouts,
)
from .config import TrainerConfig
from .evaluator import Evaluator


class Trainer:
    def __init__(
        self,
        policy: Policy | DDP,
        critic: Critic | DDP,
        policy_optimizer: Optimizer,
        critic_optimizer: Optimizer,
        policy_scheduler: LRScheduler,
        critic_scheduler: LRScheduler,
        ppo_trainer: TrainerConfig,
        mcts_trainer: TrainerConfig,
        mcts_config: MCTSConfig,
        episodes: int,
        clip_value: float,
        reset_proportion: float,
    ):
        self.policy = policy
        self.critic = critic
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.policy_scheduler = policy_scheduler
        self.critic_scheduler = critic_scheduler
        self.ppo_trainer = ppo_trainer
        self.mcts_trainer = mcts_trainer
        self.mcts_config = mcts_config
        self.episodes = episodes
        self.clip_value = clip_value
        self.reset_proportion = reset_proportion

        self.ppo_env = self.ppo_trainer.env
        self.mcts_env = self.mcts_trainer.env

        self.policy_module = (
            self.policy.module if isinstance(self.policy, DDP) else self.policy
        )
        self.critic_module = (
            self.critic.module if isinstance(self.critic, DDP) else self.critic
        )

        self.evaluator = Evaluator()
        self.device = ppo_trainer.env.device
        self.rng = ppo_trainer.env.rng

    def reset_envs(self, env: EternityEnv):
        """Randomly reset some of the environments."""
        total_resets = int(self.reset_proportion * env.batch_size)
        reset_ids = torch.randperm(
            env.batch_size, generator=self.rng, device=self.device
        )
        reset_ids = reset_ids[:total_resets]
        env.reset(reset_ids)

    @torch.inference_mode()
    def collect_ppo_rollouts(
        self, env: EternityEnv, sampling_mode: str, disable_logs: bool
    ) -> tuple[TensorDict, TensorDict]:
        """Simulates standard rollouts and adds them to the PPO replay buffer."""
        self.reset_envs(env)

        # Collect the rollouts.
        self.policy_module.train()
        self.critic_module.train()
        traces = policy_rollouts(
            env,
            self.policy_module,
            self.critic_module,
            self.ppo_trainer.rollouts,
            sampling_mode=sampling_mode,
            disable_logs=disable_logs,
        )

        traces = split_reset_rollouts(traces)
        self.ppo_trainer.loss.advantages(traces)

        assert torch.all(
            traces["rewards"].sum(dim=1) <= 1 + 1e-4  # Account for small FP errors.
        ), f"Some returns are > 1 ({traces['rewards'].sum(dim=1).max().item()})."
        assert (
            self.ppo_trainer.replay_buffer._storage.max_size
            == traces["masks"].sum().item()
            == self.ppo_env.batch_size * self.ppo_trainer.rollouts
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

        return traces, samples

    @torch.inference_mode()
    def collect_mcts_rollouts(
        self, env: EternityEnv, sampling_mode: str, disable_logs: bool
    ) -> tuple[TensorDict, TensorDict]:
        """Simulates MCTS rollouts and adds them to the MCTS replay buffer."""
        self.reset_envs(env)

        mcts_tree = MCTSTree(
            self.mcts_config.c_puct,
            self.mcts_config.gamma,
            self.mcts_config.simulations,
            self.mcts_config.childs,
            len(env.action_space),
            self.mcts_env.batch_size,
            self.device,
        )

        # Collect the rollouts with the current policy.
        self.policy_module.train()
        self.critic_module.train()
        traces = mcts_rollouts(
            env,
            self.policy_module,
            self.critic_module,
            mcts_tree,
            self.mcts_trainer.rollouts,
            sampling_mode=sampling_mode,
            disable_logs=disable_logs,
        )

        # Flatten the batch x steps dimensions and remove the masked steps.
        samples = dict()
        for name, tensor in traces.items():
            samples[name] = einops.rearrange(tensor, "b d ... -> (b d) ...")

        samples = TensorDict(
            samples, batch_size=samples["states"].shape[0], device=self.device
        )

        return traces, samples

    def do_batch_update(
        self,
        batch: TensorDict,
        loss: PPOLoss | MCTSLoss,
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
        evaluate_every: int = 1,
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

            self.ppo_env.reset()
            self.mcts_trainer.env.reset()

            self.best_matches_found = 0  # Reset.

            if not disable_logs:
                self.policy_module.summary(
                    self.ppo_env.board_size, self.ppo_env.board_size, self.device
                )
                self.critic_module.summary(
                    self.ppo_env.board_size, self.ppo_env.board_size, self.device
                )
                print(f"\nLaunching training on device {self.device}.\n")

            # Log gradients and model parameters.
            run.watch(self.policy)
            run.watch(self.critic)

            # Infinite loop if n_batches is -1.
            iter = count(0) if self.episodes == -1 else range(self.episodes)

            # Tuples of (method_name, method_config, method_rollout).
            methods = [
                (
                    "MCTS",
                    self.mcts_trainer,
                    lambda: self.collect_mcts_rollouts(
                        self.mcts_env,
                        sampling_mode="softmax",
                        disable_logs=disable_logs,
                    ),
                ),
                (
                    "PPO",
                    self.ppo_trainer,
                    lambda: self.collect_ppo_rollouts(
                        self.ppo_env,
                        sampling_mode="softmax",
                        disable_logs=disable_logs,
                    ),
                ),
            ]

            for i in tqdm(iter, desc="Episode", disable=disable_logs):
                if not disable_logs:
                    metrics = dict()

                for method_name, method_config, method_rollout in methods:
                    if method_config.epochs == 0:
                        continue  # Ignore the method.

                    # Fill the rollout buffer.
                    _, samples = method_rollout()
                    samples = samples.clone()  # Avoid in-place operations.
                    method_config.replay_buffer.extend(samples)

                    # Train the models.
                    for _ in tqdm(
                        range(method_config.epochs),
                        desc=f"Epoch {method_name}",
                        leave=False,
                        disable=disable_logs,
                    ):
                        for batch in tqdm(
                            method_config.replay_buffer,
                            total=len(method_config.replay_buffer)
                            // method_config.replay_buffer._batch_size,
                            desc="Batch",
                            leave=False,
                            disable=disable_logs,
                        ):
                            self.do_batch_update(
                                batch,
                                method_config.loss,
                                train_policy=method_config.train_policy,
                                train_critic=method_config.train_critic,
                            )

                    # Save some basic metrics.
                    if not disable_logs:
                        metrics |= self.evaluator.env_metrics(
                            method_config.env, method_name.lower()
                        )

                self.policy_scheduler.step()
                self.critic_scheduler.step()

                if not disable_logs and i % evaluate_every == 0 and i != 0:
                    metrics |= self.compute_metrics(disable_logs)

                if not disable_logs:
                    run.log(metrics)

                if not disable_logs and i % save_every == 0 and i != 0:
                    self.save_checkpoint("checkpoint.pt")
                    self.ppo_env.save_sample("ppo-sample.gif")
                    self.mcts_env.save_sample("mcts-sample.gif")

    def compute_metrics(self, disable_logs: bool) -> dict[str, Any]:
        """Compute all metrics for the current models and environments.

        This method is very slow and should be used sparingly.
        """
        metrics = dict()

        for model, scheduler, model_name in [
            (self.policy, self.policy_scheduler, "policy"),
            (self.critic, self.critic_scheduler, "critic"),
        ]:
            metrics |= self.evaluator.models_metrics(model, scheduler, model_name)

        # Normal rollouts, evaluate the losses as well as the envs.
        traces, _ = self.collect_ppo_rollouts(self.ppo_env, "softmax", disable_logs)
        metrics |= self.evaluator.policy_rollouts_metrics(
            traces,
            self.ppo_trainer.replay_buffer,
            self.ppo_trainer.loss,
            self.policy,
            self.critic,
        )
        metrics |= self.evaluator.env_metrics(self.ppo_env, "ppo")

        traces, _ = self.collect_mcts_rollouts(self.mcts_env, "softmax", disable_logs)
        metrics |= self.evaluator.mcts_rollouts_metrics(
            traces,
            self.mcts_trainer.replay_buffer,
            self.mcts_trainer.loss,
            self.policy,
            self.critic,
        )
        metrics |= self.evaluator.env_metrics(self.mcts_env, "mcts")

        # Greedy rollouts, only evaluate the envs.
        env = EternityEnv(
            self.ppo_env.instances,
            self.ppo_env.episode_length,
            self.device,
        )
        self.collect_ppo_rollouts(env, "greedy", disable_logs)
        metrics |= self.evaluator.env_metrics(env, "ppo-greedy")

        env = EternityEnv(
            self.mcts_env.instances,
            self.mcts_env.episode_length,
            self.device,
        )
        self.collect_mcts_rollouts(env, "greedy", disable_logs)
        metrics |= self.evaluator.env_metrics(env, "mcts-greedy")

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
