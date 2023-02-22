from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

import wandb

from ..model import CNNPolicy
from .dataset import EternityDataset


class EternityTrainer:
    def __init__(
        self,
        model: CNNPolicy,
        train_dataset: EternityDataset,
        test_dataset: EternityDataset,
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

        for batch in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            for metric_name, metric_value in self.compute_metrics(batch).items():
                metrics[metric_name].append(metric_value.cpu().item())

        return {
            metric_name: np.mean(metric_values)
            for metric_name, metric_values in metrics.items()
        }

    def launch_training(self, config: dict[str, Any]):
        self.model.to(self.device)
        with wandb.init(
            project="eternity-rl",
            entity="pierrotlc",
            config=config,
        ) as run:
            for _ in tqdm(range(self.n_epochs)):
                self.train_one_epoch()

                for dataset, mode in zip(
                    [self.train_dataset, self.test_dataset],
                    ["train", "test"],
                ):
                    metrics = self.eval_dataset(dataset)
                    print(mode, metrics)
