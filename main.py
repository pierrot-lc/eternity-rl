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
from pytorch_optimizer import Lamb, Lion
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from src.environment import EternityEnv
from src.mcts import MCTSTree
from src.model import Critic, Policy
from src.policy_gradient import PPOLoss, Trainer


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
    env = config.exp.env
    if not isinstance(env.episode_length, int):
        assert env.episode_length in {
            "inf",
            "+inf",
        }, "Provide either an integer or 'inf'."
        env.episode_length = float(env.episode_length)

    return EternityEnv.from_file(
        env.path,
        env.episode_length,
        env.batch_size,
        config.device,
        config.seed,
    )


def init_models(config: DictConfig, env: EternityEnv) -> tuple[Policy, Critic]:
    """Initialize the model."""
    model = config.model
    policy = Policy(
        board_width=env.board_size,
        board_height=env.board_size,
        embedding_dim=model.embedding_dim,
        n_heads=model.n_heads,
        backbone_layers=model.backbone_layers,
        decoder_layers=model.decoder_layers,
        dropout=model.dropout,
    )
    critic = Critic(
        board_width=env.board_size,
        board_height=env.board_size,
        embedding_dim=model.embedding_dim,
        n_heads=model.n_heads,
        backbone_layers=model.backbone_layers,
        decoder_layers=model.decoder_layers,
        dropout=model.dropout,
    )
    return policy, critic


def init_mcts(
    config: DictConfig, env: EternityEnv, policy: Policy, critic: Critic
) -> MCTSTree:
    mcts = config.exp.mcts
    return MCTSTree(env, policy, critic, mcts.simulations, mcts.childs)


def init_loss(config: DictConfig) -> PPOLoss:
    loss = config.exp.loss
    return PPOLoss(
        loss.value_weight,
        loss.entropy_weight,
        loss.entropy_clip,
        loss.gamma,
        loss.gae_lambda,
        loss.ppo_clip_ac,
        loss.ppo_clip_vf,
    )


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
        "lion": Lion,
    }

    return optimizers[optimizer_name](
        model.parameters(), lr=lr, weight_decay=weight_decay
    )


def init_scheduler(
    config: DictConfig, optimizer: optim.Optimizer
) -> optim.lr_scheduler.LRScheduler:
    """Initialize the scheduler."""
    scheduler = config.exp.scheduler
    schedulers = []

    if scheduler.warmup_steps > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=scheduler.warmup_steps,
        )
        schedulers.append(warmup_scheduler)

    if scheduler.cosine_t0 > 0:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=scheduler.cosine_t0,
            T_mult=scheduler.cosine_tmult,
            eta_min=scheduler.eta_min,
        )
        schedulers.append(lr_scheduler)

    # To make sure the schedulers isn't an empty list.
    identity_scheduler = optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1.0,
        end_factor=1.0,
        total_iters=1,
    )
    schedulers.append(identity_scheduler)

    return optim.lr_scheduler.ChainedScheduler(schedulers)


def init_replay_buffer(config: DictConfig) -> ReplayBuffer:
    exp = config.exp
    max_size = exp.env.batch_size * exp.trainer.rollouts
    return TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=max_size, device=config.device),
        sampler=SamplerWithoutReplacement(drop_last=True),
        batch_size=exp.trainer.batch_size,
        pin_memory=True,
    )


def init_trainer(
    config: DictConfig,
    env: EternityEnv,
    policy: Policy | DDP,
    critic: Critic | DDP,
    loss: PPOLoss,
    policy_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    policy_scheduler: optim.lr_scheduler.LRScheduler,
    critic_scheduler: optim.lr_scheduler.LRScheduler,
    replay_buffer: ReplayBuffer,
) -> Trainer:
    """Initialize the trainer."""
    trainer = config.exp.trainer
    return Trainer(
        env,
        policy,
        critic,
        loss,
        policy_optimizer,
        critic_optimizer,
        policy_scheduler,
        critic_scheduler,
        replay_buffer,
        trainer.clip_value,
        trainer.episodes,
        trainer.epochs,
        trainer.rollouts,
        trainer.reset_proportion,
    )


def reload_checkpoint(config: DictConfig, trainer: Trainer):
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
    policy, critic = init_models(config, env)
    policy, critic = policy.to(config.device), critic.to(config.device)
    policy = DDP(policy, device_ids=[config.device], output_device=config.device)
    critic = DDP(critic, device_ids=[config.device], output_device=config.device)
    loss = init_loss(config)
    policy_optimizer = init_optimizer(config, policy)
    critic_optimizer = init_optimizer(config, critic)
    policy_scheduler = init_scheduler(config, policy_optimizer)
    critic_scheduler = init_scheduler(config, critic_optimizer)
    replay_buffer = init_replay_buffer(config)
    trainer = init_trainer(
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
    reload_checkpoint(config, trainer)

    try:
        trainer.launch_training(
            config.exp.group, OmegaConf.to_container(config), config.mode
        )
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt. Cleaning up distributed processes...")
    finally:
        cleanup_distributed()


def run_trainer_single_gpu(config: DictConfig):
    """Run the trainer in single GPU or CPU mode."""
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    env = init_env(config)
    policy, critic = init_models(config, env)
    policy, critic = policy.to(config.device), critic.to(config.device)
    loss = init_loss(config)
    policy_optimizer = init_optimizer(config, policy)
    critic_optimizer = init_optimizer(config, critic)
    policy_scheduler = init_scheduler(config, policy_optimizer)
    critic_scheduler = init_scheduler(config, critic_optimizer)
    replay_buffer = init_replay_buffer(config)
    trainer = init_trainer(
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
        print(f"Training on {world_size} GPUs.")
        mp.spawn(run_trainer_ddp, nprocs=world_size, args=(world_size, config))
    else:
        run_trainer_single_gpu(config)


if __name__ == "__main__":
    # Launch with hydra.
    main()
