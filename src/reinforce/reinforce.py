from itertools import count
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad
from tqdm import tqdm

import wandb

from ..environment import EternityEnv
from ..model import Policy
from .rollout_buffer import RolloutBuffer


class Reinforce:
    def __init__(
        self,
        env: EternityEnv,
        model: Policy | DDP,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LinearLR,
        device: str,
        entropy_weight: float,
        clip_value: float,
        batch_size: int,
        total_rollouts: int,
        advantage: str,
    ):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.entropy_weight = entropy_weight
        self.clip_value = clip_value
        self.batch_size = batch_size
        self.total_rollouts = total_rollouts
        self.advantage = advantage

        # Instantiate the rollout buffer once.
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.env.batch_size,
            max_steps=self.env.max_steps,
            board_size=self.env.board_size,
            n_classes=self.env.n_classes,
            device=self.device,
        )

    def do_rollouts(self, sampling_mode: str) -> RolloutBuffer:
        """Simulates a bunch of rollouts and returns a prepared rollout buffer."""
        self.rollout_buffer.reset()
        states, _ = self.env.reset()
        hidden_state = None

        while not self.env.truncated and not torch.all(self.env.terminated):
            actions, logprobs, entropies, hidden_state = self.model(
                states, hidden_state, sampling_mode
            )
            new_states, rewards, _, _, infos = self.env.step(actions)
            self.rollout_buffer.store(
                rewards,
                logprobs,
                entropies,
                ~self.env.terminated | infos["just_won"],
            )

            states = new_states

        self.rollout_buffer.finalize(self.advantage)

        return self.rollout_buffer

    def compute_loss(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        losses = dict()
        for key, value in sample.items():
            sample[key] = value.to(self.device)

        sample["entropies"][:, 0] *= 1.0
        sample["entropies"][:, 1] *= 0.5
        sample["entropies"][:, 2] *= 0.1
        sample["entropies"][:, 3] *= 0.1
        sample["entropies"] = sample["entropies"].sum(dim=1)

        losses["policy"] = -(
            sample["logprobs"] * sample["advantages"].unsqueeze(1).detach()
        ).mean()
        losses["entropy"] = -self.entropy_weight * sample["entropies"].mean()
        losses["total"] = sum(losses.values())
        return losses

    def do_batch_update(self, rollout_buffer: RolloutBuffer):
        """Performs a batch update on the model."""
        self.model.train()
        self.optimizer.zero_grad()

        sample = rollout_buffer.sample(self.batch_size)
        losses = self.compute_loss(sample)
        losses["total"].backward()

        clip_grad.clip_grad_norm_(self.model.parameters(), self.clip_value)
        self.optimizer.step()

    def launch_training(
        self,
        group: str,
        config: dict[str, Any],
        mode: str = "online",
        eval_every: int = 5,
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
            save_every: The number of rollouts between each checkpoint.
        """
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
            iter = count(0) if self.total_rollouts == -1 else range(self.total_rollouts)

            for i in tqdm(
                iter,
                desc="Rollout",
                disable=mode == "disabled",
            ):
                self.model.train()
                rollout_buffer = self.do_rollouts(sampling_mode="sample")
                self.do_batch_update(rollout_buffer)
                self.scheduler.step()

                if i % eval_every == 0 and mode != "disabled":
                    metrics = self.evaluate()
                    run.log(metrics)

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

        rollout_buffer = self.do_rollouts(sampling_mode="sample")

        matches = self.env.max_matches / self.env.best_matches
        metrics["matches/mean"] = matches.mean()
        metrics["matches/max"] = matches.max()
        metrics["matches/min"] = matches.min()
        metrics["matches/hist"] = wandb.Histogram(matches.cpu().numpy())

        episodes_len = rollout_buffer.mask_buffer.float().sum(dim=1)
        metrics["ep-len/mean"] = episodes_len.mean()
        metrics["ep-len/max"] = episodes_len.max()
        metrics["ep-len/min"] = episodes_len.min()
        metrics["ep-len/hist"] = wandb.Histogram(episodes_len.cpu().numpy())

        # Compute losses.
        sample = rollout_buffer.sample(self.batch_size)
        losses = self.compute_loss(sample)
        for k, v in losses.items():
            metrics[f"loss/{k}"] = v
        metrics["loss/learning-rate"] = self.scheduler.get_last_lr()[0]

        # Compute the gradient mean and maximum values.
        losses["total"].backward()
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.abs().mean().item())

        metrics["global-gradients/mean"] = sum(grads) / (len(grads) + 1)
        metrics["global-gradients/max"] = max(grads)
        metrics["global-gradients/hist"] = wandb.Histogram(grads)

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
