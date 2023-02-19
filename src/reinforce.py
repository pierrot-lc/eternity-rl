from itertools import accumulate

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

import wandb

from .environment.gym import EternityEnv
from .model import CNNPolicy


class Reinforce:
    def __init__(
        self,
        env: EternityEnv,
        model: CNNPolicy,
        device: str,
        learning_rate: float,
        gamma: float,
        n_batches: int,
        batch_size: int,
        use_standardized_returns: bool,
    ):
        self.env = env
        self.model = model
        self.device = device
        self.rng = np.random.default_rng()
        self.gamma = gamma
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.use_standardized_returns = use_standardized_returns

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.episodes_history = []

    def select_tile(
        self, tile_logits: dict[str, torch.Tensor]
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Sample the tile actions using the policy prediction."""
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
        """Compute the loss and the metrics to log."""
        # Compute some basic metrics.
        episodes_lengths, episodes_return = [], []
        for _, returns in self.episodes_history:
            episodes_lengths.append(returns.shape[0])
            episodes_return.append(returns[-1])

        episodes_lengths = torch.FloatTensor(episodes_lengths)
        episodes_return = torch.FloatTensor(episodes_return)

        # Compute the mean and std of the returns.
        mean_return = episodes_return.mean()
        std_return = episodes_return.std()

        loss = torch.tensor(0.0, device=self.device)
        for log_actions, returns in self.episodes_history:
            if self.use_standardized_returns:
                returns = (returns - mean_return) / (std_return + 1e-5)
            loss += -(log_actions * returns.unsqueeze(1)).mean()

        metrics = {
            "loss": loss,
            "return": mean_return,
            "return_std": std_return,
            "ep_len": episodes_lengths.mean(),
        }
        return metrics

    def launch_training(self, config: dict[str, any]):
        """Train the model and log everything to wandb."""
        optim = self.optimizer
        self.model.to(self.device)
        with wandb.init(
            project="eternity-rl",
            entity="pierrotlc",
            config=config,
        ) as run:
            for epoch_id in tqdm(range(self.n_batches)):
                self.episodes_history = []

                for _ in range(self.batch_size):
                    self.rollout()

                metrics = self.compute_metrics()
                optim.zero_grad()
                metrics["loss"].backward()
                clip_grad_norm_(self.model.parameters(), 1)
                optim.step()

                # Log the metrics.
                metrics = {
                    metric_name: metric_value.cpu().item()
                    for metric_name, metric_value in metrics.items()
                }
                run.log(metrics)

                if epoch_id % 100 == 0:
                    # Log a sample of the current policy.
                    self.make_gif("./logs/sample.gif")
                    run.log({"sample": wandb.Video("./logs/sample.gif", fps=1)})

    def make_gif(self, path: str):
        env, model = self.env, self.model
        device = self.device

        model.to(device)
        state = env.reset()
        done = False

        images = []
        while not done:
            state = torch.LongTensor(state).to(device).unsqueeze(0)
            tile_1, tile_2 = model(state)
            act_1, _ = self.select_tile(tile_1)
            act_2, _ = self.select_tile(tile_2)

            action = np.concatenate((act_1, act_2), axis=0)

            state, _, done, _ = env.step(action)
            image = env.render(mode="rgb_array")
            images.append(image)

        # Make a gif of the episode based on the images.
        import imageio

        imageio.mimwrite(path, images, fps=1)
