import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_optimizer import Lamb

from src.environment import EternityEnv
from src.model import Policy
from src.reinforce import Reinforce


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def init_env(config: DictConfig) -> EternityEnv:
    """Initialize the environment."""
    env = EternityEnv.from_file(
        config.exp.env.path,
        config.exp.rollout_buffer.buffer_size,
        config.exp.env.max_steps,
        config.device,
        config.seed,
    )
    return env


def init_model(config: DictConfig, env: EternityEnv) -> Policy:
    """Initialize the model."""
    model = Policy(
        n_classes=env.n_classes,
        board_width=env.board_size,
        board_height=env.board_size,
        tile_embedding_dim=config.exp.model.tile_embedding_dim,
        embedding_dim=config.exp.model.embedding_dim,
        backbone_layers=config.exp.model.backbone_layers,
        head_layers=config.exp.model.head_layers,
        dropout=config.exp.model.dropout,
    )
    return model


def init_optimizer(config: DictConfig, model: nn.Module) -> optim.Optimizer:
    """Initialize the optimizer."""
    optimizer_name = config.exp.optimizer.optimizer
    lr = config.exp.optimizer.learning_rate
    weight_decay = config.exp.optimizer.weight_decay
    optimizers = {
        "adamw": optim.AdamW,
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
        "lamb": Lamb,
    }

    optimizer = optimizers[optimizer_name](
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    return optimizer


def init_scheduler(
    config: DictConfig, optimizer: optim.Optimizer
) -> optim.lr_scheduler.LinearLR:
    """Initialize the scheduler."""
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=0.001,
        end_factor=1.0,
        total_iters=config.exp.scheduler.warmup_steps,
    )
    return scheduler


def init_trainer(
    config: DictConfig,
    env: EternityEnv,
    model: Policy | DDP,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LinearLR,
) -> Reinforce:
    """Initialize the trainer."""
    trainer = Reinforce(
        env,
        model,
        optimizer,
        scheduler,
        config.device,
        config.exp.reinforce.entropy_weight,
        config.exp.reinforce.clip_value,
        config.exp.rollout_buffer.batch_size,
        config.exp.rollout_buffer.batches_per_rollouts,
        config.exp.reinforce.total_rollouts,
        config.exp.reinforce.advantage,
    )
    return trainer


def reload_checkpoint(config: DictConfig, trainer: Reinforce):
    """Reload a checkpoint."""
    if config.exp.checkpoint is None:
        return

    checkpoint_path = config.exp.checkpoint
    state_dict = torch.load(checkpoint_path, map_location=config.device)
    trainer.model.load_state_dict(state_dict["model"])
    # HACK: The training seems to not be stable when loading the optimizer state.
    # trainer.optimizer.load_state_dict(state_dict["optimizer"])
    print(f"Checkpoint from {checkpoint_path} loaded.")


def run_trainer_ddp(rank: int, world_size: int, config: DictConfig):
    """Run the trainer in distributed mode."""
    setup_distributed(rank, world_size)

    # Make sure we log training info only for the rank 0 process.
    if rank != 0:
        config.mode = "disabled"

    config.device = config.distributed[rank]
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    env = init_env(config)
    model = init_model(config, env)
    model = model.to(config.device)
    model = DDP(model, device_ids=[config.device], output_device=config.device)
    optimizer = init_optimizer(config, model)
    scheduler = init_scheduler(config, optimizer)
    trainer = init_trainer(config, env, model, optimizer, scheduler)
    reload_checkpoint(config, trainer)

    try:
        trainer.launch_training(
            config.exp.group, OmegaConf.to_container(config), config.mode
        )
    except KeyboardInterrupt:
        # Capture a potential CTRL+C to make sure we clean up distributed processes.
        print("Caught KeyboardInterrupt. Cleaning up distributed processes...")
    finally:
        cleanup_distributed()


def run_trainer_single_gpu(config: DictConfig):
    """Run the trainer in single GPU or CPU mode."""
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    env = init_env(config)
    model = init_model(config, env)
    model = model.to(config.device)
    optimizer = init_optimizer(config, model)
    scheduler = init_scheduler(config, optimizer)
    trainer = init_trainer(config, env, model, optimizer, scheduler)
    reload_checkpoint(config, trainer)

    trainer.launch_training(
        config.exp.group, OmegaConf.to_container(config), config.mode
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(config: DictConfig):
    config.exp.env.path = Path(to_absolute_path(config.exp.env.path))
    if config.exp.checkpoint is not None:
        config.exp.checkpoint = Path(to_absolute_path(config.exp.checkpoint))
    world_size = len(config.distributed)
    if world_size > 1:
        mp.spawn(run_trainer_ddp, nprocs=world_size, args=(world_size, config))
    else:
        run_trainer_single_gpu(config)


if __name__ == "__main__":
    # Launch with hydra.
    main()
