from ..model import CNNPolicy
from .dataset import EternityDataset


class EternityTrainer:
    def __init__(
        self,
        model: CNNPolicy,
        train_dataset: EternityDataset,
        test_dataset: EternityDataset,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
