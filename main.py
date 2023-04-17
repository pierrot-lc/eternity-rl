from pathlib import Path

import hydra
import torch
import torch.optim as optim
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary

from src.environment import BatchedEternityEnv
from src.model import CNNPolicy
from src.reinforce import Reinforce


def init_env(config: DictConfig) -> BatchedEternityEnv:
    env_path = Path(to_absolute_path(config.env.path))
    env = BatchedEternityEnv.from_file(
        env_path,
        config.reinforce.batch_size,
        config.env.reward,
        config.device,
        config.seed,
    )
    return env


def init_model(config: DictConfig, env: BatchedEternityEnv) -> CNNPolicy:
    model = CNNPolicy(
        int(env.n_class),
        embedding_dim=config.model.embedding_dim,
        n_res_layers=config.model.n_res_layers,
        n_gru_layers=config.model.n_gru_layers,
        n_head_layers=config.model.n_head_layers,
        kernel_maxpool=config.model.kernel_maxpool,
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
    return model


def init_optimizer(config: DictConfig, model: CNNPolicy) -> optim.Optimizer:
    optimizer_name = config.optimizer.optimizer
    lr = config.optimizer.learning_rate
    weight_decay = config.optimizer.weight_decay
    optimizers = {
        "adamw": optim.AdamW,
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
    }

    if optimizer_name not in optimizers:
        print(f"Unknown optimizer: {optimizer_name}.")
        print("Using AdamW instead.")
        optimizer_name = "adamw"

    optimizer = optimizers[optimizer_name](
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    return optimizer


def init_scheduler(
    config: DictConfig, optimizer: optim.Optimizer
) -> optim.lr_scheduler.LinearLR:
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=0.001,
        end_factor=1.0,
        total_iters=config.scheduler.warmup_steps,
    )
    return scheduler


def init_trainer(
    config: DictConfig,
    env: BatchedEternityEnv,
    model: CNNPolicy,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LinearLR,
) -> Reinforce:
    trainer = Reinforce(
        env,
        model,
        optimizer,
        scheduler,
        config.device,
        config.reinforce.value_weight,
        config.reinforce.entropy_weight,
        config.reinforce.gamma,
        config.reinforce.clip_value,
        config.reinforce.n_batches_per_iteration,
        config.reinforce.n_total_iterations,
        config.reinforce.advantage,
        config.reinforce.save_every,
    )
    return trainer


@hydra.main(version_base="1.3", config_path="configs", config_name="trivial_B")
def reinforce(config: DictConfig):
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    env = init_env(config)
    model = init_model(config, env)
    optimizer = init_optimizer(config, model)
    scheduler = init_scheduler(config, optimizer)
    trainer = init_trainer(config, env, model, optimizer, scheduler)
    trainer.launch_training(config.group, OmegaConf.to_container(config))


if __name__ == "__main__":
    # Launch with hydra.
    reinforce()
