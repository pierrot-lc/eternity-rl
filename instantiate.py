import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_optimizer import Lamb, Lion
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from src.environment import EternityEnv
from src.mcts import MCTSConfig, MCTSLoss
from src.model import Critic, Policy
from src.policy_gradient import PPOLoss
from src.trainer import Trainer, TrainerConfig


def init_env(config: DictConfig, batch_size: int) -> EternityEnv:
    """Initialize the environment."""
    env = config.env
    if not isinstance(env.episode_length, int):
        assert env.episode_length in {
            "inf",
            "+inf",
        }, "Provide either an integer or 'inf'."
        episode_length = float(env.episode_length)

    return EternityEnv.from_file(
        env.path,
        episode_length,
        batch_size,
        config.device,
        config.seed,
    )


def init_models(config: DictConfig) -> tuple[Policy, Critic]:
    """Initialize the model."""
    model = config.model
    policy = Policy(
        embedding_dim=model.embedding_dim,
        n_heads=model.n_heads,
        backbone_layers=model.backbone_layers,
        decoder_layers=model.decoder_layers,
        dropout=model.dropout,
    )
    critic = Critic(
        embedding_dim=model.embedding_dim,
        n_heads=model.n_heads,
        backbone_layers=model.backbone_layers,
        decoder_layers=model.decoder_layers,
        dropout=model.dropout,
    )
    return policy, critic


def init_mcts_config(config: DictConfig) -> MCTSConfig:
    mcts = config.mcts
    return MCTSConfig(
        mcts.search.c_puct, config.gamma, mcts.search.simulations, mcts.search.childs
    )


def init_ppo_loss(config: DictConfig) -> PPOLoss:
    loss = config.ppo.loss
    return PPOLoss(
        loss.value_weight,
        loss.entropy_weight,
        loss.entropy_clip,
        config.gamma,
        loss.gae_lambda,
        loss.ppo_clip_ac,
        loss.ppo_clip_vf,
    )


def init_mcts_loss(config: DictConfig) -> MCTSLoss:
    loss = config.mcts.loss
    return MCTSLoss(loss.value_weight, loss.entropy_weight)


def init_optimizer(config: DictConfig, model: nn.Module) -> optim.Optimizer:
    """Initialize the optimizer."""
    optimizer = config.optimizer
    optimizer_name = optimizer.optimizer
    lr = optimizer.learning_rate
    weight_decay = optimizer.weight_decay
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
    scheduler = config.scheduler
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


def init_replay_buffer(
    config: DictConfig, batch_size: int, max_size: int
) -> ReplayBuffer:
    return TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=max_size, device=config.device),
        sampler=SamplerWithoutReplacement(drop_last=True),
        batch_size=batch_size,
        pin_memory=True if config.device != "cpu" else False,
    )


def init_trainer_config(
    trainer_config: DictConfig,
    env: EternityEnv,
    loss: PPOLoss | MCTSLoss,
    replay_buffer: ReplayBuffer,
) -> TrainerConfig:
    return TrainerConfig(
        env=env,
        loss=loss,
        replay_buffer=replay_buffer,
        epochs=trainer_config.epochs,
        rollouts=trainer_config.rollouts,
        train_policy=trainer_config.train_policy,
        train_critic=trainer_config.train_critic,
    )


def init_trainer(
    config: DictConfig,
    policy: Policy | DDP,
    critic: Critic | DDP,
    policy_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    policy_scheduler: optim.lr_scheduler.LRScheduler,
    critic_scheduler: optim.lr_scheduler.LRScheduler,
    ppo_trainer: TrainerConfig,
    mcts_trainer: TrainerConfig,
    mcts_config: MCTSConfig,
) -> Trainer:
    """Initialize the trainer."""
    trainer = config.trainer
    return Trainer(
        policy,
        critic,
        policy_optimizer,
        critic_optimizer,
        policy_scheduler,
        critic_scheduler,
        ppo_trainer,
        mcts_trainer,
        mcts_config,
        trainer.episodes,
        trainer.clip_value,
        trainer.reset_proportion,
    )
