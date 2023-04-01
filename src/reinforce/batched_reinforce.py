from itertools import count
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from einops import repeat
from tqdm import tqdm

import wandb

from ..environment import BatchedEternityEnv
from ..model import CNNPolicy


class Reinforce:
    def __init__(
        self,
        env: BatchedEternityEnv,
        model: CNNPolicy,
        device: str,
        optimizer: str,
        learning_rate: float,
        warmup_steps: int,
        value_weight: float,
        gamma: float,
        n_batches: int,
        advantage: str,
        save_every: int,
    ):
        self.env = env
        self.model = model
        self.device = device
        self.value_weight = value_weight
        self.gamma = gamma
        self.n_batches = n_batches
        self.advantage = advantage
        self.save_every = save_every

        if self.advantage != "learned":
            self.value_weight = 0

        match optimizer:
            case "adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            case "adamw":
                self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
            case "sgd":
                self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
            case "rmsprop":
                self.optimizer = optim.RMSprop(
                    self.model.parameters(), lr=learning_rate
                )
            case _:
                print(f"Unknown optimizer: {optimizer}.")
                print("Using Adam instead.")
                self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.scheduler = optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

    def rollout(
        self,
    ) -> dict[str, torch.Tensor]:
        """Simulates a batch of games and returns the cumulated rewards
        and the logit of the actions taken.

        ---
        Returns:
            returns: The cumulated rewards of the games.
                Shape of [batch_size, max_steps].
            log_probs: The logit of the actions taken.
                Shape of [batch_size, max_steps, 4].
            values: The value of the games state.
                Shape of [batch_size, max_steps].
            masks: The mask indicating whether the game is terminated.
                Shape of [batch_size, max_steps].
            returns: The cumulated rewards (without decay) of the games.
                Shape of [batch_size,].
        """
        states, _ = self.env.reset()
        rewards = torch.zeros(
            self.env.batch_size,
            self.env.max_steps,
            device=self.device,
        )
        log_actions = torch.zeros(
            self.env.batch_size,
            self.env.max_steps,
            4,
            device=self.device,
        )
        values = torch.zeros(
            self.env.batch_size,
            self.env.max_steps,
            device=self.device,
        )
        masks = torch.zeros(
            self.env.batch_size,
            self.env.max_steps,
            device=self.device,
            dtype=torch.bool,
        )
        timesteps = torch.zeros(
            self.env.batch_size,
            device=self.device,
            dtype=torch.long,
        )

        while not self.env.truncated and not torch.all(self.env.terminated):
            timesteps.fill_(self.env.step_id)
            actions, log_probs, value = self.model(states, timesteps)

            states, step_rewards, _, _, _ = self.env.step(actions)

            rewards[:, self.env.step_id - 1] = step_rewards
            log_actions[:, self.env.step_id - 1] = log_probs
            values[:, self.env.step_id - 1] = value.squeeze(1)
            masks[:, self.env.step_id - 1] = ~self.env.terminated

        # The last terminated state is not counted in the masks,
        # so we need to shift the masks by 1 to make sure we include id.
        masks = torch.roll(masks, shifts=1, dims=(1,))
        masks[:, 0] = True

        # Compute the cumulated rewards.
        decayed_returns = Reinforce.cumulative_decay_return(rewards, masks, self.gamma)
        total_returns = (rewards * masks).sum(dim=1)

        return {
            "decayed_returns": decayed_returns,
            "log_probs": log_actions,
            "values": values,
            "masks": masks,
            "returns": total_returns,
        }

    def compute_metrics(
        self,
        rollout: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute the metrics of the batched rollout.

        ---
        Args:
            rollout: The batched rollout outputs.

        ---
        Returns:
            A dictionary of metrics of the batched rollout.
        """
        metrics = dict()
        masks = rollout["masks"]
        decayed_returns = rollout["decayed_returns"]
        returns = rollout["returns"]
        values = rollout["values"]

        episodes_len = masks.sum(dim=1).long() - 1

        # Compute the advantage and policy loss.
        match self.advantage:
            case "learned":
                advantages = decayed_returns - values.detach()
            case "estimated":
                advantages = (decayed_returns - decayed_returns.mean()) / (
                    decayed_returns.std() + 1e-7
                )
            case "no-advantage":
                advantages = decayed_returns
            case _:
                raise ValueError(f"Unknown advantage: {self.advantage}.")

        masked_loss = (
            -rollout["log_probs"] * advantages.unsqueeze(2) * masks.unsqueeze(2)
        )

        metrics["loss/policy"] = masked_loss.sum() / masks.sum()
        metrics["loss/value"] = self.value_weight * F.mse_loss(
            values * masks, decayed_returns, reduction="mean"
        )
        metrics["loss/total"] = metrics["loss/policy"] + metrics["loss/value"]
        metrics["ep-len/mean"] = episodes_len.float().mean()
        metrics["ep-len/min"] = episodes_len.min()
        metrics["ep-len/std"] = episodes_len.float().std()
        metrics["return/mean"] = returns.mean()
        metrics["return/max"] = returns.max()
        metrics["return/std"] = returns.std()

        return metrics

    def launch_training(self, group: str, config: dict[str, Any]):
        self.model.to(self.device)
        with wandb.init(
            project="eternity-rl",
            entity="pierrotlc",
            group=group,
            config=config,
        ) as run:
            # Infinite loop if n_batches is -1.
            iter = count(0) if self.n_batches == -1 else range(self.n_batches)

            for i in tqdm(iter, desc="Batch"):
                self.model.train()

                rollout = self.rollout()
                metrics = self.compute_metrics(rollout)

                self.optimizer.zero_grad()
                metrics["loss/total"].backward()

                # Compute the gradient norm.
                grad_norms = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norms.append(p.grad.detach().data.norm())
                grad_norms = torch.stack(grad_norms)
                metrics["grad-norm/mean"] = grad_norms.mean()
                metrics["grad-norm/max"] = grad_norms.max()
                metrics["grad-norm/std"] = grad_norms.std()

                metrics = {k: v.cpu().item() for k, v in metrics.items()}
                metrics["learning-rate"] = self.scheduler.get_last_lr()
                run.log(metrics)

                self.optimizer.step()
                self.scheduler.step()

                if i % self.save_every == 0:
                    self.save_model("model.pt")
                    self.env.save_best_env("board.png")
                    run.log(
                        {
                            "best board": wandb.Image("board.png"),
                        }
                    )

    def save_model(self, filepath: Path | str):
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        torch.save({"model": model_state, "optimizer": optimizer_state}, filepath)

    @staticmethod
    def cumulative_decay_return(
        rewards: torch.Tensor, masks: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        """Compute the cumulative decayed return of a batch of games.
        It is efficiently implemented using tensor operations.
        Thanks to the kind stranger here: https://discuss.pytorch.org/t/cumulative-sum-with-decay-factor/69788/2.
        This function may not be numerically stable.

        ---
        Args:
            rewards: The rewards of the games.
                Shape of [batch_size, max_steps].
            masks: The mask indicating which steps are actual plays.
                Shape of [batch_size, max_steps].
            gamma: The discount factor.

        ---
        Returns:
            The cumulative decayed return of the games.
                Shape of [batch_size, max_steps].
        """
        # Compute the gamma powers.
        powers = (rewards.shape[1] - 1) - torch.arange(
            rewards.shape[1], device=rewards.device
        )
        powers = gamma**powers
        powers = repeat(powers, "t -> b t", b=rewards.shape[0])

        # Compute the cumulative decayed return.
        rewards = torch.flip(rewards, dims=(1,))
        masks = torch.flip(masks, dims=(1,))
        returns = torch.cumsum(masks * rewards * powers, dim=1) / powers
        returns = torch.flip(returns, dims=(1,))

        return returns
