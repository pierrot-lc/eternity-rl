from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary

from src.environment import BatchedEternityEnv
from src.model import CNNPolicy
from src.reinforce import Reinforce


@hydra.main(version_base="1.3", config_path="configs", config_name="normal")
def reinforce(config: DictConfig):
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    env = BatchedEternityEnv.from_file(
        Path(config.env.path),
        config.reinforce.batch_size,
        config.env.reward,
        config.device,
        config.seed,
    )
    model = CNNPolicy(
        int(env.n_class),
        embedding_dim=config.model.embedding_dim,
        n_layers=config.model.n_layers,
        board_width=env.size,
        board_height=env.size,
    )
    summary(
        model,
        input_data=[
            torch.zeros(1, 4, env.size, env.size, dtype=torch.long),
            torch.randint(0, 10, size=(1,)),
        ],
        device="cpu",
    )
    trainer = Reinforce(
        env,
        model,
        config.device,
        config.reinforce.learning_rate,
        config.reinforce.value_weight,
        config.reinforce.gamma,
        config.reinforce.n_batches,
        config.reinforce.advantage,
    )
    trainer.launch_training(OmegaConf.to_container(config))


if __name__ == "__main__":
    # Launch with hydra.
    reinforce()
