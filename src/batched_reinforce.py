from typing import Any

import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
from tqdm import tqdm

import wandb

from .environment import BatchedEternityEnv
from .model import CNNPolicy


class BatchedReinforce:
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
        self.gamma = gamma  # TODO: Use this.
        self.n_batches = n_batches

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

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

    def rollout(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulates a batch of games and returns the cumulated rewards
        and the logit of the actions taken.

        ---
        Returns:
            returns: The cumulated rewards of the games.
                Shape of [batch_size, max_steps].
            log_actions: The logit of the actions taken.
                Shape of [batch_size, max_steps, 4].
            masks: The mask indicating whether the game is terminated.
                Shape of [batch_size, max_steps].
        """
        states, _ = self.env.reset()
        returns = torch.zeros(
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
        masks = torch.zeros(
            self.env.batch_size,
            self.env.max_steps,
            device=self.device,
            dtype=torch.bool,
        )

        while not self.env.truncated and not torch.all(self.env.terminated):
            tile_1, tile_2 = self.model(states)
            actions_1, log_probs_1 = self.sample_action(tile_1)
            actions_2, log_probs_2 = self.sample_action(tile_2)

            actions = torch.concat([actions_1, actions_2], dim=1)
            log_probs = torch.concat([log_probs_1, log_probs_2], dim=1)

            states, rewards, _, _, _ = self.env.step(actions)

            returns[:, self.env.step_id - 1] = rewards
            log_actions[:, self.env.step_id - 1] = log_probs
            masks[:, self.env.step_id - 1] = ~self.env.terminated

        # The last terminated state is not counted in the masks,
        # so we need to shift the masks by 1 to make sure we include id.
        masks = torch.roll(masks, shifts=1, dims=(1,))
        masks[:, 0] = True

        # Compute the cumulated rewards.
        returns = torch.flip(returns, dims=(1,))
        masks = torch.flip(masks, dims=(1,))

        returns = torch.cumsum(returns * masks, dim=1)  # Ignore masked rewards.

        returns = torch.flip(returns, dims=(1,))
        masks = torch.flip(masks, dims=(1,))

        return returns, log_actions, masks

    def compute_metrics(
        self, returns: torch.Tensor, log_actions: torch.Tensor, masks: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute the metrics of the batched rollout.

        ---
        Args:
            returns: The cumulated rewards of the games.
                Shape of [batch_size, max_steps].
            log_actions: The logit of the actions taken.
                Shape of [batch_size, max_steps, 4].
            masks: The mask indicating whether the game is terminated.
                Shape of [batch_size, max_steps].

        ---
        Returns:
            A dictionary of metrics of the batched rollout.
        """
        metrics = dict()

        end_game = masks.sum(dim=1).long() - 1
        end_return = torch.gather(returns, dim=1, index=end_game.unsqueeze(1))
        end_return = end_return.squeeze(1)
        mean_return, std_return = end_return.mean(), end_return.std()
        returns = (returns - mean_return) / (std_return + 1e-7)
        masked_loss = -(log_actions * masks.unsqueeze(2) * returns.unsqueeze(2))

        metrics["loss"] = masked_loss.sum() / masks.sum()
        metrics["ep-len/mean"] = end_game.float().mean()
        metrics["ep-len/min"] = end_game.min()
        metrics["return/max"] = end_return.max()
        metrics["return/mean"] = mean_return
        metrics["return/std"] = std_return

        return metrics

    def launch_training(self, config: dict[str, Any]):
        self.model.to(self.device)
        with wandb.init(
            project="eternity-rl",
            entity="pierrotlc",
            group="batched-reinforce",
            config=config,
        ) as run:
            for _ in tqdm(range(self.n_batches)):
                self.model.train()

                returns, log_actions, masks = self.rollout()
                metrics = self.compute_metrics(returns, log_actions, masks)

                self.optimizer.zero_grad()
                metrics["loss"].backward()
                # clip_grad_value_(self.model.parameters(), clip_value=1)
                clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()

                metrics = {k: v.cpu().item() for k, v in metrics.items()}
                run.log(metrics)
