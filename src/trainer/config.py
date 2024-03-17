import dataclasses

from torchrl.data import ReplayBuffer

from ..mcts import MCTSLoss
from ..policy_gradient import PPOLoss
from ..environment import EternityEnv


@dataclasses.dataclass
class TrainerConfig:
    env: EternityEnv
    loss: PPOLoss | MCTSLoss
    replay_buffer: ReplayBuffer
    epochs: int
    rollouts: int
    train_policy: bool
    train_critic: bool
    name: str
