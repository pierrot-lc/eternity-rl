import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_optimizer import Lamb, Lion
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from src.environment import EternityEnv
from src.mcts import MCTSTree, MCTSTrainer, MCTSLoss
from src.model import Critic, Policy
from src.policy_gradient import PPOLoss, PPOTrainer


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


def init_mcts(config: DictConfig, env: EternityEnv) -> MCTSTree:
    mcts = config.mcts
    return MCTSTree(
        config.exp.loss.gamma,
        mcts.simulations,
        mcts.childs,
        len(env.action_space),
        env.batch_size,
        env.device,
    )


def init_ppo_loss(config: DictConfig) -> PPOLoss:
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
        pin_memory=True if config.device != "cpu" else False,
    )


def init_ppo_trainer(
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
) -> PPOTrainer:
    """Initialize the trainer."""
    trainer = config.exp.trainer
    return PPOTrainer(
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


def init_mcts_trainer(
    config: DictConfig,
    env: EternityEnv,
    policy: Policy | DDP,
    critic: Critic | DDP,
    mcts: MCTSTree,
    policy_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    policy_scheduler: optim.lr_scheduler.LRScheduler,
    critic_scheduler: optim.lr_scheduler.LRScheduler,
    replay_buffer: ReplayBuffer,
) -> MCTSTrainer:
    trainer = config.exp.trainer
    return MCTSTrainer(
        env,
        policy,
        critic,
        mcts,
        MCTSLoss(1.0, 0.0, 0.0),
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
