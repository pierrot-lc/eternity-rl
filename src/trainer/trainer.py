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
from wandb import Histogram

from ..environment import EternityEnv
from ..mcts import MCTSLoss, MCTSTree
from ..model import Critic, Policy
from ..policy_gradient import PPOLoss
from ..rollouts import mcts_rollouts, policy_rollouts, split_reset_rollouts
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
        self.episodes = episodes
        self.clip_value = clip_value
        self.reset_proportion = reset_proportion

        self.ppo_env = self.ppo_trainer.env
        self.ppo_greedy_env = EternityEnv.from_env(self.ppo_env)

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
    ) -> tuple[TensorDict, dict[str, Any]]:
        """Simulates standard rollouts and adds them to the PPO replay buffer."""
        metrics = dict()

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

        metrics["metrics/sum-rewards"] = Histogram(traces["rewards"].sum(dim=1).cpu())

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

        metrics["metrics/value-targets"] = Histogram(samples["value-targets"].cpu())
        metrics["metrics/advantages"] = Histogram(samples["advantages"].cpu())
        for action_id in range(samples["actions"].shape[1]):
            actions = samples["actions"][:, action_id].cpu()
            metrics[f"metrics/action-{action_id+1}"] = Histogram(actions)

        return samples, metrics

    @torch.inference_mode()
    def collect_mcts_rollouts(
        self, env: EternityEnv, sampling_mode: str, disable_logs: bool
    ) -> tuple[TensorDict, dict[str, Any]]:
        """Simulates MCTS rollouts and adds them to the MCTS replay buffer."""
        metrics = dict()

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

        metrics["metrics/values"] = Histogram(samples["values"].cpu())
        metrics["metrics/child-values"] = Histogram(
            samples["child-values"].flatten().cpu()
        )
        metrics["metrics/probs"] = Histogram(samples["probs"].cpu())

        for action_id in range(samples["actions"].shape[2]):
            actions = samples["actions"][:, :, action_id].flatten().cpu()
            metrics[f"metrics/action-{action_id+1}"] = Histogram(actions)

        return samples, metrics

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
        evaluate_every: int = 100,
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
            evaluate_every: The number of rollouts between each evaluation.
            save_every: The number of rollouts between each checkpoint.
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
            self.ppo_greedy_env.reset()

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

            for i in tqdm(iter, desc="Episode", disable=disable_logs):
                metrics = dict()

                # Fill the rollout buffer.
                samples, metrics_ = self.collect_ppo_rollouts(
                    self.ppo_env,
                    sampling_mode="softmax",
                    disable_logs=disable_logs,
                )
                samples = samples.clone()  # Avoid in-place operations.
                self.ppo_trainer.replay_buffer.extend(samples)

                # Train the models.
                for _ in tqdm(
                    range(self.ppo_trainer.epochs),
                    desc=f"Epoch {self.ppo_trainer.name}",
                    leave=False,
                    disable=disable_logs,
                ):
                    for batch in tqdm(
                        self.ppo_trainer.replay_buffer,
                        total=len(self.ppo_trainer.replay_buffer) // self.ppo_trainer.replay_buffer._batch_size,
                        desc="Batch",
                        leave=False,
                        disable=disable_logs,
                    ):
                        self.do_batch_update(
                            batch,
                            self.ppo_trainer.loss,
                            train_policy=self.ppo_trainer.train_policy,
                            train_critic=self.ppo_trainer.train_critic,
                        )

                if not disable_logs:
                    # It is important to evaluate the batch metrics after the epochs.
                    # Otherwise we would not be able to see the final approx-kl, and clip-frac.
                    metrics |= self.evaluator.rollout_metrics(
                        metrics_,
                        self.ppo_trainer.name,
                    )
                    metrics |= self.evaluator.env_metrics(
                        self.ppo_trainer.env,
                        self.ppo_trainer.name,
                    )
                    metrics |= self.evaluator.batch_metrics(
                        self.ppo_trainer.replay_buffer.sample(),
                        self.policy,
                        self.critic,
                        self.ppo_trainer.loss,
                        self.ppo_trainer.name,
                    )

                self.policy_scheduler.step()
                self.critic_scheduler.step()

                if not disable_logs and i % evaluate_every == 0:
                    metrics |= self.compute_metrics(disable_logs)

                if not disable_logs:
                    run.log(metrics)

                if not disable_logs and i % save_every == 0 and i != 0:
                    self.save_checkpoint("checkpoint.pt")
                    self.ppo_env.save_sample("ppo-sample.gif")

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
        samples, metrics_ = self.collect_ppo_rollouts(
            self.ppo_env, "softmax", disable_logs
        )
        self.ppo_trainer.replay_buffer.extend(samples.clone())
        metrics |= self.evaluator.rollout_metrics(metrics_, self.ppo_trainer.name)
        metrics |= self.evaluator.env_metrics(self.ppo_env, self.ppo_trainer.name)
        metrics |= self.evaluator.batch_metrics(
            self.ppo_trainer.replay_buffer.sample(),
            self.policy,
            self.critic,
            self.ppo_trainer.loss,
            self.ppo_trainer.name,
        )

        # Greedy rollouts, only evaluate the envs.
        _, metrics_ = self.collect_ppo_rollouts(
            self.ppo_greedy_env, "nucleus", disable_logs
        )
        metrics |= self.evaluator.rollout_metrics(
            metrics_, f"{self.ppo_trainer.name}-greedy"
        )
        metrics |= self.evaluator.env_metrics(
            self.ppo_greedy_env, f"{self.ppo_trainer.name}-greedy"
        )

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
