import torch
from torchinfo import summary

from src.environment.gym import EternityEnv
from src.model import CNNPolicy
from src.reinforce import Reinforce

env = EternityEnv(
    "./instances/eternity_A.txt", manual_orient=True, reward_type="win_ratio"
)
model = CNNPolicy(
    env.n_class,
    embedding_dim=32,
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

trainer = Reinforce(env, model, "cpu")
trainer.rollout()
