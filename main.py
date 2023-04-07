from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary

from src.environment import BatchedEternityEnv
from src.model import CNNPolicy
from src.reinforce import Reinforce


@hydra.main(version_base="1.3", config_path="configs", config_name="trivial_B")
def reinforce(config: DictConfig):
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    env_path = Path(to_absolute_path(config.env.path))
    env = BatchedEternityEnv.from_file(
        env_path,
        config.reinforce.batch_size,
        config.env.reward,
        config.device,
        config.seed,
    )
    model = CNNPolicy(
        int(env.n_class),
        embedding_dim=config.model.embedding_dim,
        n_res_layers=config.model.n_res_layers,
        n_gru_layers=config.model.n_gru_layers,
        board_width=env.size,
        board_height=env.size,
        zero_init_residuals=config.model.zero_init_residuals,
    )
    summary(
        model,
        input_data=[
            torch.zeros(1, 4, env.size, env.size, dtype=torch.long),
        ],
        device="cpu",
    )
    trainer = Reinforce(
        env,
        model,
        config.device,
        config.reinforce.optimizer,
        config.reinforce.learning_rate,
        config.reinforce.warmup_steps,
        config.reinforce.value_weight,
        config.reinforce.entropy_weight,
        config.reinforce.gamma,
        config.reinforce.n_batches_per_iteration,
        config.reinforce.n_total_iterations,
        config.reinforce.advantage,
        config.reinforce.save_every,
    )
    trainer.launch_training(config.group, OmegaConf.to_container(config))


if __name__ == "__main__":
    # Launch with hydra.
    reinforce()
