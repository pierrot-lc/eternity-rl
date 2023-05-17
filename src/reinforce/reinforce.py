from itertools import count
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad
from tqdm import tqdm

import wandb

from ..environment import BatchedEternityEnv
from ..model import CNNPolicy
from .rollout_buffer import RolloutBuffer


class Reinforce:
    def __init__(
        self,
        env: BatchedEternityEnv,
        model: CNNPolicy,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LinearLR,
        device: str,
        entropy_weight: float,
        gamma: float,
        clip_value: float,
        batch_size: int,
        n_batches_per_rollouts: int,
        n_total_rollouts: int,
        advantage: str,
        save_every: int,
    ):
        assert n_batches_per_rollouts > 0

        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.entropy_weight = entropy_weight
        self.gamma = gamma
        self.clip_value = clip_value
        self.batch_size = batch_size
        self.n_total_rollouts = n_total_rollouts
        self.n_batches_per_rollouts = n_batches_per_rollouts
        self.advantage = advantage
        self.save_every = save_every

    @torch.no_grad()
    def do_rollouts(self) -> RolloutBuffer:
        """Simulates a bunch of rollouts and returns a prepared rollout buffer."""
        rollout_buffer = RolloutBuffer(
            buffer_size=self.env.batch_size,
            max_steps=self.env.max_steps,
            board_size=self.env.board_size,
            n_classes=self.env.n_classes,
            device=self.device,
        )

        states, _ = self.env.reset()

        while not self.env.truncated and not torch.all(self.env.terminated):
            actions = self.model(states)
            states, rewards, _, _, infos = self.env.step(actions)
            rollout_buffer.store(
                states,
                actions,
                rewards,
                ~self.env.terminated | infos["just_won"],
            )

        rollout_buffer.finalize(self.advantage, self.gamma)

        return rollout_buffer

    def do_batch_update(self, rollout_buffer: RolloutBuffer):
        """Performs a batch update on the model."""
        self.model.train()
        self.optimizer.zero_grad()

        sample = rollout_buffer.sample(self.batch_size)
        logprobs, entropies = self.model.logprobs(
            tiles=sample["observations"], actions=sample["actions"]
        )
        loss = (-logprobs * sample["advantages"].unsqueeze(1)).mean()
        # loss = loss - self.entropy_weight * entropies.mean()

        loss.backward()
        clip_grad.clip_grad_norm_(self.model.parameters(), self.clip_value)
        self.optimizer.step()
        self.scheduler.step()

    def launch_training(self, group: str, config: dict[str, Any], mode: str = "online"):
        with wandb.init(
            project="eternity-rl",
            entity="pierrotlc",
            group=group,
            config=config,
            mode=mode,
        ) as run:
            self.model.to(self.device)

            if mode != "disabled":
                if isinstance(self.model, DDP):
                    self.model.module.summary(self.device)
                else:
                    self.model.summary(self.device)
                print(f"Launching training on device {self.device}.")

            # Log gradients and model parameters.
            run.watch(self.model)

            # Infinite loop if n_batches is -1.
            iter = (
                count(0)
                if self.n_total_rollouts == -1
                else range(self.n_total_rollouts)
            )

            for i in tqdm(
                iter,
                desc="Rollout",
                disable=mode == "disabled",
            ):
                self.model.train()
                rollout_buffer = self.do_rollouts()

                for _ in tqdm(
                    range(self.n_batches_per_rollouts),
                    desc="Batch",
                    disable=mode == "disabled",
                    leave=False,
                ):
                    self.do_batch_update(rollout_buffer)

                metrics = self.evaluate()
                run.log(metrics)

                if i % self.save_every == 0 and mode != "disabled":
                    self.save_model("model.pt")
                    self.env.save_best_env("board.png")
                    run.log(
                        {
                            "best board": wandb.Image("board.png"),
                        }
                    )

    def evaluate(self) -> dict[str, Any]:
        """Evaluates the model and returns some computed metrics."""
        metrics = dict()
        self.model.eval()

        rollout_buffer = self.do_rollouts()

        returns = RolloutBuffer.cumulative_decay_return(
            rollout_buffer.reward_buffer, rollout_buffer.mask_buffer, gamma=1.0
        )
        returns = returns[:, 0]
        metrics["return/mean"] = returns.mean()
        metrics["return/max"] = returns.max()
        metrics["return/std"] = returns.std()

        episodes_len = rollout_buffer.mask_buffer.float().sum(dim=1)
        metrics["ep-len/mean"] = episodes_len.mean()
        metrics["ep-len/min"] = episodes_len.min()
        metrics["ep-len/std"] = episodes_len.std()

        # Compute gradients and losses.
        batch_size = min(10 * self.batch_size, self.env.batch_size)
        sample = rollout_buffer.sample(batch_size)
        logprobs, entropies = self.model.logprobs(
            tiles=sample["observations"], actions=sample["actions"]
        )
        metrics["loss/policy"] = -(logprobs * sample["advantages"].unsqueeze(1)).mean()
        metrics["loss/entropy"] = -self.entropy_weight * entropies.mean()
        metrics["loss/total"] = metrics["loss/policy"] + metrics["loss/entropy"]
        metrics["loss/total"].backward()  # To compute gradients.
        metrics["loss/learning-rate"] = self.scheduler.get_last_lr()[0]

        # Compute the gradient mean and maximum values.
        mean_value, max_value, tot_params = 0, 0, 0
        for tot_params, p in enumerate(self.model.parameters()):
            if p.grad is not None:
                grad = p.grad.data.abs()
                mean_value += grad.mean().item()
                max_value = max(max_value, grad.max().item())

        metrics["gradients/mean"] = mean_value / (tot_params + 1)
        metrics["gradients/max"] = max_value

        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[name] = value.cpu().item()

        return metrics

    def save_model(self, filepath: Path | str):
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        torch.save(
            {
                "model": model_state,
                "optimizer": optimizer_state,
            },
            filepath,
        )
