import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

import wandb

from ..environment.gym import EternityEnv
from ..model import CNNPolicy
from .dataset import EternityDataset


class EternityTrainer:
    def __init__(
        self,
        model: CNNPolicy,
        train_dataset: EternityDataset,
        test_dataset: EternityDataset,
        env: EternityEnv,
        lr: float,
        batch_size: int,
        epoch_size: int,
        n_epochs: int,
        device: str,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=RandomSampler(
                self.train_dataset, replacement=True, num_samples=epoch_size
            ),
            drop_last=True,
        )
        self.env = env
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.n_epochs = n_epochs
        self.device = device

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def compute_metrics(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        metrics = dict()
        # Shapes of [batch_size,].
        true_tiles = torch.concat((batch["tile_1"], batch["tile_2"]), dim=0)
        true_rolls = torch.concat((batch["roll_1"], batch["roll_2"]), dim=0)

        logits_1, logits_2 = self.model(batch["instance"])
        # Shapes of [batch_size, n_classes].
        tile_logits = torch.concat((logits_1["tile"], logits_2["tile"]), dim=0)
        roll_logits = torch.concat((logits_1["roll"], logits_2["roll"]), dim=0)

        metrics["loss"] = self.loss_fn(tile_logits, true_tiles) + self.loss_fn(
            roll_logits, true_rolls
        )
        metrics["tile-accuracy"] = (
            (tile_logits.argmax(dim=1) == true_tiles).float().mean()
        )
        metrics["roll-accuracy"] = (
            (roll_logits.argmax(dim=1) == true_rolls).float().mean()
        )

        return metrics

    def train_one_epoch(self):
        device = self.device
        self.model.train()
        for batch in self.train_loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            metrics = self.compute_metrics(batch)

            self.optimizer.zero_grad()
            metrics["loss"].backward()
            self.optimizer.step()

    @torch.no_grad()
    def eval_dataset(self, dataset: EternityDataset) -> dict[str, float]:
        metrics = defaultdict(list)
        device = self.device
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            sampler=RandomSampler(
                dataset, replacement=True, num_samples=self.epoch_size
            ),
            drop_last=True,
        )
        self.model.eval()

        for batch in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            for metric_name, metric_value in self.compute_metrics(batch).items():
                metrics[metric_name].append(metric_value.cpu().item())

        return {
            metric_name: float(np.mean(metric_values))
            for metric_name, metric_values in metrics.items()
        }

    def eval_env(self, n_games: int, gif_path: str) -> dict[str, float]:
        metrics = defaultdict(list)
        for _ in range(n_games):
            tot_rewards, ep_len, _ = self.model.solve_env(
                self.env, self.device, intermediate_images=False
            )
            metrics["return"].append(tot_rewards)
            metrics["episode-length"].append(ep_len)

        # Save a gif of an episode.
        import imageio

        _, _, images = self.model.solve_env(
            self.env, self.device, intermediate_images=True
        )
        imageio.mimwrite(gif_path, images, fps=1)

        return {
            metric_name: float(np.mean(metric_values))
            for metric_name, metric_values in metrics.items()
        }

    def save_model(self, config: dict[str, Any]):
        env_name = os.path.basename(config["env"]["path"]).replace(".txt", "")
        torch.save(
            {
                "model": self.model.state_dict(),
                "embedding_dim": config["model"]["embedding_dim"],
                "n_layers": config["model"]["n_layers"],
            },
            f"./logs/{env_name}-supervised-model.pt",
        )

    def launch_training(self, config: dict[str, Any]):
        self.model.to(self.device)
        with wandb.init(
            project="eternity-rl",
            entity="pierrotlc",
            config=config,
            group=config["group"],
        ) as run:
            for _ in tqdm(range(self.n_epochs)):
                self.train_one_epoch()

                logs = dict()
                for dataset, mode in zip(
                    [self.train_dataset, self.test_dataset],
                    ["train", "test"],
                ):
                    metrics = self.eval_dataset(dataset)
                    for metric_name, metric_value in metrics.items():
                        logs[f"{mode}/{metric_name}"] = metric_value

                metrics = self.eval_env(50, "./logs/sample.gif")
                for metric_name, metric_value in metrics.items():
                    logs[f"env/{metric_name}"] = metric_value
                logs["env/sample"] = wandb.Video("./logs/sample.gif", fps=1)

                run.log(logs)
            self.save_model(config)


def main(config: dict[str, Any]):
    """Train a supervised model."""
    from pathlib import Path

    from torchinfo import summary

    # Load data.
    filename = os.path.basename(config["env"]["path"]).replace(".txt", ".npz")
    filepath = Path("./data") / filename
    train_dataset, test_dataset = EternityDataset.from_file(
        filepath,
        test_size=config["supervised"]["test_size"],
        seed=config["seed"],
    )

    # Get environment infos.
    env = EternityEnv(
        instance_path=config["env"]["path"],
        reward_type=config["env"]["reward_type"],
    )

    # Initialize model.
    model = CNNPolicy(
        env.n_class,
        embedding_dim=config["model"]["embedding_dim"],
        n_layers=config["model"]["n_layers"],
        board_width=env.size,
        board_height=env.size,
    )
    summary(
        model,
        input_size=(4, env.size, env.size),
        batch_dim=0,
        dtypes=[
            torch.long,
        ],
        device="cpu",
    )

    trainer = EternityTrainer(
        model,
        train_dataset,
        test_dataset,
        env,
        config["supervised"]["learning_rate"],
        config["supervised"]["batch_size"],
        config["supervised"]["epoch_size"],
        config["supervised"]["n_epochs"],
        config["device"],
    )
    trainer.launch_training(config)
