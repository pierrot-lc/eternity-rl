from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from einops import repeat
from torch.distributions import Categorical
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
        learning_rate: float,
        gamma: float,
        n_batches: int,
    ):
        self.env = env
        self.model = model
        self.device = device
        self.gamma = gamma
        self.n_batches = n_batches

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

    def rollout(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulates a batch of games and returns the cumulated rewards
        and the logit of the actions taken.

        ---
        Returns:
            returns: The cumulated rewards of the games.
                Shape of [batch_size, max_steps].
            log_actions: The logit of the actions taken.
                Shape of [batch_size, max_steps, 4].
            values: The value of the games state.
            masks: The mask indicating whether the game is terminated.
                Shape of [batch_size, max_steps].
            total_returns: The cumulated rewards (without decay) of the games.
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

        while not self.env.truncated and not torch.all(self.env.terminated):
            tile_1, tile_2, value = self.model(states)
            actions_1, log_probs_1 = Reinforce.sample_action(tile_1)
            actions_2, log_probs_2 = Reinforce.sample_action(tile_2)

            actions = torch.concat([actions_1, actions_2], dim=1)
            log_probs = torch.concat([log_probs_1, log_probs_2], dim=1)

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
        return decayed_returns, log_actions, values, masks, total_returns

    def compute_metrics(
        self,
        returns: torch.Tensor,
        log_actions: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        total_returns: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the metrics of the batched rollout.

        ---
        Args:
            returns: The cumulated rewards of the games (with decay).
                Shape of [batch_size, max_steps].
            log_actions: The logit of the actions taken.
                Shape of [batch_size, max_steps, 4].
            values: The value of the games state.
                Shape of [batch_size, max_steps].
            masks: The mask indicating whether the game is terminated.
                Shape of [batch_size, max_steps].
            total_returns: The total rewards of the games (without decay).
                Shape of [batch_size,].

        ---
        Returns:
            A dictionary of metrics of the batched rollout.
        """
        metrics = dict()

        episodes_len = masks.sum(dim=1).long() - 1

        # Compute the advantage.
        advantages = returns - values
        masked_loss = -log_actions * masks.unsqueeze(2) * advantages.unsqueeze(2)

        metrics["loss/policy"] = masked_loss.sum() / masks.sum()
        metrics["loss/value"] = F.mse_loss(values, returns, reduction="mean")
        metrics["loss/total"] = metrics["loss/value"] + metrics["loss/value"]
        metrics["ep-len/mean"] = episodes_len.float().mean()
        metrics["ep-len/min"] = episodes_len.min()
        metrics["ep-len/std"] = episodes_len.float().std()
        metrics["return/max"] = total_returns.max()
        metrics["return/mean"] = total_returns.mean()
        metrics["return/std"] = total_returns.std()

        return metrics

    def launch_training(self, config: dict[str, Any]):
        self.model.to(self.device)
        with wandb.init(
            project="eternity-rl",
            entity="pierrotlc",
            group=f"batched-reinforce/{self.env.size}x{self.env.size}",
            config=config,
        ) as run:
            for _ in tqdm(range(self.n_batches)):
                self.model.train()

                (
                    decayed_returns,
                    log_actions,
                    values,
                    masks,
                    total_returns,
                ) = self.rollout()
                metrics = self.compute_metrics(
                    decayed_returns, log_actions, values, masks, total_returns
                )

                self.optimizer.zero_grad()
                metrics["loss/total"].backward()
                self.optimizer.step()

                metrics = {k: v.cpu().item() for k, v in metrics.items()}
                run.log(metrics)

    @staticmethod
    def sample_action(
        tile_logits: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the tile actions using the policy prediction.

        ---
        Args:
            tile_logits: The logits of the tile and roll actions.

        ---
        Returns:
            actions: The sampled tile ids and roll values.
                Shape of [batch_size, 2].
            log_probs: The logit of the sampled actions.
                Shape of [batch_size, 2].
        """
        # Sample the tile ids.
        tile_distribution = Categorical(logits=tile_logits["tile"])
        tile_ids = tile_distribution.sample()
        tile_id_log_probs = tile_distribution.log_prob(tile_ids)

        # Sample the roll values.
        roll_distribution = Categorical(logits=tile_logits["roll"])
        rolls = roll_distribution.sample()
        roll_log_probs = roll_distribution.log_prob(rolls)

        actions = torch.stack([tile_ids, rolls], dim=1)
        log_probs = torch.stack([tile_id_log_probs, roll_log_probs], dim=1)
        return actions, log_probs

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
