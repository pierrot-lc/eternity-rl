from itertools import count
from pathlib import Path
from typing import Any

import einops
import torch
import torch.optim as optim
from tensordict import TensorDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad
from torchrl.data import ReplayBuffer
from tqdm import tqdm

import wandb

from ..environment import EternityEnv
from ..model import Policy
from .loss import PPOLoss
from .rollout import rollout, split_reset_rollouts


class Trainer:
    def __init__(
        self,
        envs: list[EternityEnv],
        model: Policy | DDP,
        loss: PPOLoss,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LinearLR,
        replay_buffer: ReplayBuffer,
        clip_value: float,
        scramble_size: float,
        rollouts: int,
        epochs: int,
    ):
        self.envs = envs
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.replay_buffer = replay_buffer
        self.clip_value = clip_value
        self.rollouts = rollouts
        self.epochs = epochs

        self.scramble_size = int(scramble_size * self.envs[-1].batch_size)
        self.max_board_size = max(env.board_size for env in envs)
        self.rng = envs[0].rng
        self.device = envs[0].device

    @torch.inference_mode()
    def do_rollouts(self, env: EternityEnv, sampling_mode: str, disable_logs: bool):
        """Simulates a bunch of rollouts and adds them to the replay buffer."""
        if self.scramble_size > 0:
            scramble_ids = torch.randperm(
                env.batch_size, generator=self.rng, device=self.device
            )
            scramble_ids = scramble_ids[: self.scramble_size]
            env.reset(scramble_ids=scramble_ids)

        if env.terminated.sum() > 0:
            to_reset = torch.arange(env.batch_size, device=self.device)
            to_reset = to_reset[env.terminated]
            env.reset(scramble_ids=to_reset)

        traces = rollout(env, self.model, sampling_mode, self.rollouts, disable_logs)
        traces = split_reset_rollouts(traces)
        self.loss.advantages(traces)

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

        # Add padding to match the biggest env.
        states = samples["states"]
        batch_size, n_sides, board_size, board_size = states.shape

        states = torch.cat(
            (
                states,
                torch.zeros(
                    (batch_size, n_sides, self.max_board_size - board_size, board_size),
                    dtype=torch.long,
                    device=self.device,
                ),
            ),
            dim=2,
        )
        states = torch.cat(
            (
                states,
                torch.zeros(
                    (
                        batch_size,
                        n_sides,
                        self.max_board_size,
                        self.max_board_size - board_size,
                    ),
                    dtype=torch.long,
                    device=self.device,
                ),
            ),
            dim=3,
        )
        paddings = torch.zeros(
            (batch_size, self.max_board_size, self.max_board_size),
            dtype=torch.bool,
            device=self.device,
        )
        paddings[
            :, self.max_board_size - board_size :, self.max_board_size - board_size :
        ] = True

        samples["states"] = states
        samples["paddings"] = paddings

        self.replay_buffer.extend(samples)

    def do_batch_update(self, batch: TensorDict):
        """Performs a batch update on the model."""
        self.model.train()
        self.optimizer.zero_grad()

        batch = batch.to(self.device)
        metrics = self.loss(batch, self.model)
        metrics["loss/total"].backward()

        clip_grad.clip_grad_norm_(self.model.parameters(), self.clip_value)
        self.optimizer.step()

    def launch_training(
        self,
        group: str,
        config: dict[str, Any],
        mode: str = "online",
        save_every: int = 50,
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
            self.model.to(self.device)
            for env in self.envs:
                env.reset()

            if not disable_logs:
                if isinstance(self.model, DDP):
                    self.model.module.summary(self.device)
                else:
                    self.model.summary(self.device)
                print(f"Launching training on device {self.device}.")

            # Log gradients and model parameters.
            run.watch(self.model)

            # Infinite loop if n_batches is -1.
            iter = count(0) if self.epochs == -1 else range(self.epochs)

            for i in tqdm(iter, desc="Epoch", disable=disable_logs):
                self.model.train()

                for env in self.envs:
                    self.do_rollouts(
                        env, sampling_mode="softmax", disable_logs=disable_logs
                    )

                for batch in tqdm(
                    self.replay_buffer,
                    total=len(self.replay_buffer) // self.replay_buffer._batch_size,
                    desc="Batch",
                    leave=False,
                    disable=disable_logs,
                ):
                    self.do_batch_update(batch)

                self.scheduler.step()

                metrics = self.evaluate()
                run.log(metrics)

                if i % save_every == 0 and not disable_logs:
                    self.save_model("model.pt")
                    self.envs[-1].save_sample("sample.gif")

    def evaluate(self) -> dict[str, Any]:
        """Evaluates the model and returns some computed metrics."""
        metrics = dict()
        self.model.eval()

        matches = self.envs[-1].matches / self.envs[-1].best_matches_possible
        metrics["matches/mean"] = matches.mean()
        metrics["matches/hist"] = wandb.Histogram(matches.cpu())
        metrics["matches/best"] = (
            self.envs[-1].best_matches_found / self.envs[-1].best_matches_possible
        )
        metrics["matches/total-won"] = self.envs[-1].total_won

        # Compute losses.
        batch = self.replay_buffer.sample()
        batch = batch.to(self.device)
        metrics |= self.loss(batch, self.model)
        metrics["loss/learning-rate"] = self.scheduler.get_last_lr()[0]
        metrics["metrics/value-targets"] = wandb.Histogram(batch["value-targets"].cpu())
        metrics["metrics/n_steps"] = wandb.Histogram(self.envs[-1].n_steps.cpu())

        # Compute the gradient mean and maximum values.
        metrics["loss/total"].backward()
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

        self.envs[-1].save_best_env("board.png")
        metrics["best-board"] = wandb.Image("board.png")

        return metrics

    def save_model(self, filepath: Path | str):
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        scheduler_state = self.scheduler.state_dict()
        torch.save(
            {
                "model": model_state,
                "optimizer": optimizer_state,
                "scheduler": scheduler_state,
            },
            filepath,
        )
