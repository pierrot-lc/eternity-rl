import torch
from torchinfo import summary

from src.model import CNNPolicy

model = CNNPolicy(10, 64, 4, 4)
summary(
    model,
    input_size=(4, 4, 4),
    batch_dim=0,
    dtypes=[
        torch.long,
    ],
)
