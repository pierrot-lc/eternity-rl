import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from instantiate import (
    init_env,
    init_mcts,
    init_mcts_loss,
    init_mcts_trainer,
    init_models,
    init_optimizer,
    init_ppo_loss,
    init_ppo_trainer,
    init_replay_buffer,
    init_scheduler,
)
from src.mcts import MCTSTrainer
from src.policy_gradient import PPOTrainer


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def reload_checkpoint(config: DictConfig, trainer: PPOTrainer | MCTSTrainer):
    """Reload a checkpoint."""
    if config.exp.checkpoint is None:
        return

    checkpoint_path = config.exp.checkpoint
    state_dict = torch.load(checkpoint_path, map_location=config.device)

    trainer.policy.load_state_dict(state_dict["policy"])
    trainer.critic.load_state_dict(state_dict["critic"])

    trainer.policy_optimizer.load_state_dict(state_dict["policy-optimizer"])
    trainer.critic_optimizer.load_state_dict(state_dict["critic-optimizer"])

    trainer.policy_scheduler.load_state_dict(state_dict["policy-scheduler"])
    trainer.critic_scheduler.load_state_dict(state_dict["critic-scheduler"])

    print(f"Checkpoint from {checkpoint_path} loaded.")


def run_trainer(rank: int, world_size: int, config: DictConfig):
    """Run the trainer in distributed mode."""
    use_ddp = world_size > 1

    if use_ddp:
        setup_distributed(rank, world_size)
        config.device = config.distributed[rank]

    # Make sure we log training info only for the rank 0 process.
    if rank != 0:
        config.mode = "disabled"

    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    env = init_env(config)

    policy, critic = init_models(config)
    policy, critic = policy.to(config.device), critic.to(config.device)
    if use_ddp:
        policy = DDP(policy, device_ids=[config.device], output_device=config.device)
        critic = DDP(critic, device_ids=[config.device], output_device=config.device)

    policy_optimizer = init_optimizer(config, policy)
    critic_optimizer = init_optimizer(config, critic)

    policy_scheduler = init_scheduler(config, policy_optimizer)
    critic_scheduler = init_scheduler(config, critic_optimizer)

    replay_buffer = init_replay_buffer(config)

    match config.training_mode:
        case "ppo":
            loss = init_ppo_loss(config)
            trainer = init_ppo_trainer(
                config,
                env,
                policy,
                critic,
                loss,
                policy_optimizer,
                critic_optimizer,
                policy_scheduler,
                critic_scheduler,
                replay_buffer,
            )
        case "mcts":
            mcts = init_mcts(config, env)
            loss = init_mcts_loss(config)
            trainer = init_mcts_trainer(
                config,
                env,
                policy,
                critic,
                mcts,
                loss,
                policy_optimizer,
                critic_optimizer,
                policy_scheduler,
                critic_scheduler,
                replay_buffer,
            )
        case _:
            raise ValueError(f"Unknown training mode: {config.training_mode}")

    reload_checkpoint(config, trainer)

    try:
        trainer.launch_training(
            config.exp.group, OmegaConf.to_container(config), config.mode
        )
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt.")
    finally:
        if use_ddp:
            print("Cleaning up distributed processes...")
            cleanup_distributed()


@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(config: DictConfig):
    config.exp.env.path = Path(to_absolute_path(config.exp.env.path))
    if config.exp.checkpoint is not None:
        config.exp.checkpoint = Path(to_absolute_path(config.exp.checkpoint))

    assert config.training_mode in {
        "ppo",
        "mcts",
    }, f"Unknown training mode: {config.training_mode}"

    world_size = len(config.distributed)
    if world_size > 1:
        print(f"Training on {world_size} GPUs.")
        mp.spawn(run_trainer, nprocs=world_size, args=(world_size, config))
    else:
        run_trainer(rank=0, world_size=1, config=config)


if __name__ == "__main__":
    # Launch with hydra.
    main()
