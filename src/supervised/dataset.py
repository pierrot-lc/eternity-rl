from pathlib import Path
from typing import Union

import einops
import numpy as np
import torch
from torch.utils.data import Dataset


class EternityDataset(Dataset):
    def __init__(self, instances: list[np.ndarray], actions: list[np.ndarray]):
        self.instances = instances
        self.actions = actions

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        instance = torch.LongTensor(self.instances[idx])
        action = torch.LongTensor(self.actions[idx])
        return instance, action

    @classmethod
    def from_file(
        cls, filepath: Union[str, Path], test_size: float = 0.2, seed: int = 42
    ):
        """Load data from a file and split it into a training and a test set."""
        # Load the data.
        # `instances` is of shape (n_samples, max_steps, 4, size, size).
        # `actions` is of shape (n_samples, max_steps, 4).
        data = np.load(filepath)
        instances, actions = data["instances"], data["actions"]

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(instances))
        n_test = int(len(instances) * test_size)
        train_indices, test_indices = indices[n_test:], indices[:n_test]

        train_instances, train_actions = (
            instances[train_indices],
            actions[train_indices],
        )
        test_instances, test_actions = instances[test_indices], actions[test_indices]

        # Flatten instances and actions.
        train_instances, test_instances = [
            einops.rearrange(x, "b s c h w -> (b s) c h w")
            for x in (train_instances, test_instances)
        ]
        train_actions, test_actions = [
            einops.rearrange(x, "b s c -> (b s) c")
            for x in (train_actions, test_actions)
        ]

        train_dataset = cls(train_instances, train_actions)
        test_dataset = cls(test_instances, test_actions)

        return train_dataset, test_dataset
