from itertools import accumulate

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from .environment.gym import EternityEnv
from .model import CNNPolicy


class Reinforce:
    def __init__(
        self,
        env: EternityEnv,
        model: CNNPolicy,
        device: str,
    ):
        self.env = env
        self.model = model
        self.device = device
        self.rng = np.random.default_rng()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99

        self.episodes_history = []

    def select_tile(
        self, tile_logits: dict[str, torch.Tensor]
    ) -> tuple[np.ndarray, torch.Tensor]:
        actions = []
        log_actions = []
        for action_name in ["tile", "roll"]:
            logits = tile_logits[action_name]
            distribution = torch.softmax(logits, dim=-1)
            action = self.rng.choice(
                np.arange(logits.shape[1]),
                size=(1,),
                p=distribution.cpu().detach().numpy()[0],
            )

            actions.append(action)
            log_actions.append(torch.log(distribution[0, action] + 1e-5))

        return (np.array(actions), torch.concat(log_actions))

    def rollout(self):
        """Do a simple episode rollout using the current model's policy.

        It saves the history so that the gradient can be later computed.
        """
        env, model, device = self.env, self.model, self.device

        state = env.reset()
        done = False

        rewards, log_actions = [], []
        while not done:
            state = torch.LongTensor(state).to(device).unsqueeze(0)
            tile_1, tile_2 = model(state)
            act_1, log_act_1 = self.select_tile(tile_1)
            act_2, log_act_2 = self.select_tile(tile_2)

            action = np.concatenate((act_1, act_2), axis=0)
            log_actions.append(torch.concat((log_act_1, log_act_2), dim=0))

            state, reward, done, _ = env.step(action)
            rewards.append(reward)

        rewards = torch.tensor(rewards, device=device)  # Shape of [ep_len,].
        log_actions = torch.stack(log_actions, dim=0)  # Shape of [ep_len, 4].

        # Compute the cumulated returns.
        rewards = torch.flip(rewards, dims=(0,))
        returns = list(accumulate(rewards, lambda R, r: r + self.gamma * R))
        returns = torch.tensor(returns, device=device)
        returns = torch.flip(returns, dims=(0,))

        self.episodes_history.append((log_actions, returns))

    def compute_metrics(self) -> dict[str, torch.Tensor]:
        loss = torch.tensor(0.0, device=self.device)
        history_returns = torch.zeros(len(self.episodes_history), device=self.device)

        mean_returns = sum(returns.sum() for _, returns in self.episodes_history)
        mean_returns = mean_returns / sum(
            len(returns) for _, returns in self.episodes_history
        )

        for ep_id, (log_actions, returns) in enumerate(self.episodes_history):
            history_returns[ep_id] = returns[0]
            returns = returns - mean_returns
            loss += -(log_actions * returns.unsqueeze(1)).mean()

        metrics = {
            "loss": loss,
            "return": history_returns.mean(),
        }
        return metrics

    def launch_training(self):
        optim = self.optimizer
        self.model.to(self.device)

        n_batches = 2000
        n_rollouts = 200

        for _ in range(n_batches):
            self.episodes_history = []

            for _ in range(n_rollouts):
                self.rollout()

            metrics = self.compute_metrics()
            optim.zero_grad()
            metrics["loss"].backward()
            clip_grad_norm_(self.model.parameters(), 1)
            optim.step()

            for metric_name in ["loss", "return"]:
                value = metrics[metric_name].cpu().item()
                print(f"{metric_name}: {value:.3f}", end="\t")
            print("")
