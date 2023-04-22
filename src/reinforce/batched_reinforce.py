from itertools import count
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from einops import repeat
from torch.nn.utils import clip_grad
from tqdm import tqdm

import wandb

from ..environment import BatchedEternityEnv
from ..model import CNNPolicy


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
        n_batches_per_iteration: int,
        n_total_iterations: int,
        advantage: str,
        save_every: int,
    ):
        assert n_batches_per_iteration > 0

        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.entropy_weight = entropy_weight
        self.gamma = gamma
        self.clip_value = clip_value
        self.n_total_iterations = n_total_iterations
        self.n_batches_per_iteration = n_batches_per_iteration
        self.advantage = advantage
        self.save_every = save_every

    def rollout(
        self,
    ) -> dict[str, torch.Tensor]:
        """Simulates a batch of games and returns the cumulated rewards
        and the logit of the actions taken.

        ---
        Returns:
            decayed-returns: The cumulated rewards of the games.
                Shape of [batch_size, max_steps].
            log-probs: The logit of the actions taken.
                Shape of [batch_size, max_steps, 4].
            entropies: The entropy of the actions distributions.
                Shape of [batch_size, max_steps, 4].
            values: The value of the games state.
                Shape of [batch_size, max_steps].
            masks: The mask indicating whether the game is terminated.
                Shape of [batch_size, max_steps].
            returns: The cumulated rewards (without decay) of the games.
                Shape of [batch_size,].
        """
        rollout_infos = {
            "rewards": torch.zeros(
                self.env.batch_size,
                self.env.max_steps,
                device=self.device,
            ),
            "log-probs": torch.zeros(
                self.env.batch_size,
                self.env.max_steps,
                4,
                device=self.device,
            ),
            "entropies": torch.zeros(
                self.env.batch_size,
                self.env.max_steps,
                4,
                device=self.device,
            ),
            "masks": torch.zeros(
                self.env.batch_size,
                self.env.max_steps,
                device=self.device,
                dtype=torch.bool,
            ),
        }
        states, _ = self.env.reset()
        gru_memory = None

        while not self.env.truncated and not torch.all(self.env.terminated):
            actions, log_probs, gru_memory, entropies = self.model(states, gru_memory)

            states, rewards, _, _, _ = self.env.step(actions)

            rollout_infos["rewards"][:, self.env.step_id - 1] = rewards
            rollout_infos["log-probs"][:, self.env.step_id - 1] = log_probs
            rollout_infos["masks"][:, self.env.step_id - 1] = ~self.env.terminated
            rollout_infos["entropies"][:, self.env.step_id - 1] = entropies

        # The last terminated state is not counted in the masks,
        # so we need to shift the masks by 1 to make sure we include id.
        masks = rollout_infos["masks"]
        masks = torch.roll(masks, shifts=1, dims=(1,))
        masks[:, 0] = True
        rollout_infos["masks"] = masks

        # Compute the cumulated rewards.
        rewards = rollout_infos["rewards"]
        rollout_infos["decayed-returns"] = Reinforce.cumulative_decay_return(
            rewards, masks, self.gamma
        )
        rollout_infos["returns"] = (rewards * masks).sum(dim=1)

        return rollout_infos

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
        decayed_returns = rollout["decayed-returns"]
        returns = rollout["returns"]

        episodes_len = masks.sum(dim=1).long()

        # Compute the advantage and policy loss.
        match self.advantage:
            case "estimated":
                advantages = (
                    decayed_returns - decayed_returns.mean(dim=0, keepdim=True)
                ) / (decayed_returns.std(dim=0, keepdim=True) + 1e-7)
            case "no-advantage":
                advantages = decayed_returns
            case _:
                raise ValueError(f"Unknown advantage: {self.advantage}.")

        masked_loss = (
            -rollout["log-probs"] * advantages.unsqueeze(2) * masks.unsqueeze(2)
        )

        metrics["loss/policy"] = masked_loss.sum() / masks.sum()
        metrics["loss/entropy"] = (
            -self.entropy_weight
            * (rollout["entropies"] * masks.unsqueeze(2)).sum()
            / masks.sum()
        )
        metrics["loss/total"] = metrics["loss/policy"] + metrics["loss/entropy"]
        metrics["ep-len/mean"] = episodes_len.float().mean()
        metrics["ep-len/min"] = episodes_len.min()
        metrics["ep-len/std"] = episodes_len.float().std()
        metrics["return/mean"] = returns.mean()
        metrics["return/max"] = returns.max()
        metrics["return/std"] = returns.std()

        return metrics

    def launch_training(self, group: str, config: dict[str, Any], mode: str = "online"):
        with wandb.init(
            project="eternity-rl",
            entity="pierrotlc",
            group=group,
            config=config,
            mode=mode,
        ) as run:
            metrics = dict()
            tot_params = 0

            self.model.to(self.device)

            if mode != "disabled":
                self.model.summary(self.device)
                print(f"Launching training on device {self.device}.")

            # Log gradients and model parameters.
            run.watch(self.model)

            # Infinite loop if n_batches is -1.
            iter = (
                count(0)
                if self.n_total_iterations == -1
                else range(self.n_total_iterations)
            )

            for i in tqdm(iter, desc="Batch", disable=mode == "disabled"):
                self.model.train()

                for _ in range(self.n_batches_per_iteration):
                    rollout = self.rollout()
                    metrics = self.compute_metrics(rollout)

                    self.optimizer.zero_grad()
                    metrics["loss/total"].backward()

                # Log the metrics.
                metrics = {k: v.cpu().item() for k, v in metrics.items()}

                # Compute the gradient mean and maximum values.
                mean_value, max_value = 0, 0
                for tot_params, p in enumerate(self.model.parameters()):
                    if p.grad is not None:
                        grad = p.grad.data.abs()
                        mean_value += grad.mean().item()
                        max_value = max(max_value, grad.max().item())

                metrics["gradients/mean"] = mean_value / (tot_params + 1)
                metrics["gradients/max"] = max_value

                metrics["loss/learning-rate"] = self.scheduler.get_last_lr()[0]
                run.log(metrics)

                clip_grad.clip_grad_value_(self.model.parameters(), self.clip_value)

                self.optimizer.step()
                self.scheduler.step()

                if i % self.save_every == 0 and mode != "disabled":
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
        if gamma == 1:
            rewards = torch.flip(rewards, dims=(1,))
            masks = torch.flip(masks, dims=(1,))
            returns = torch.cumsum(masks * rewards, dim=1)
            returns = torch.flip(returns, dims=(1,))
            return returns

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
