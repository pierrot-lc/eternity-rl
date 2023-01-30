import numpy as np
import torch
from torch.distributions import Categorical

from .environment.gym import EternityEnv
from .model import CNNPolicy


class Reinforce:
    def __init__(
        self,
        env: EternityEnv,
        model: CNNPolicy,
        device: str,
    ):
        self.env = env
        self.model = model
        self.device = device

        self.episodes_history = []

    @staticmethod
    def select_tile(tile_1: dict[str, torch.Tensor]) -> tuple[np.ndarray, torch.Tensor]:
        actions = []
        log_actions = []
        for action_name in ["tile", "roll"]:
            logits = tile_1[action_name]
            print(logits.shape)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            print(action)
            actions.append(action.cpu().item())
            log_actions.append(distribution.log_prob(action))

        return (np.array(actions), torch.concat(log_actions))

    def rollout(self):
        """Do a simple episode rollout using the current model's policy.

        It saves the history so that the gradient can be later computed.
        """
        env, model, device = self.env, self.model, self.device

        state = env.reset()
        done = False

        rewards, log_actions = [], []
        while not done:
            state = torch.LongTensor(state).to(device).unsqueeze(0)
            tile_1, tile_2 = model(state)
            act_1, log_act_1 = Reinforce.select_tile(tile_1)
            act_2, log_act_2 = Reinforce.select_tile(tile_2)

            action = np.concatenate((act_1, act_2), axis=0)
            log_actions.append(torch.concat((log_act_1, log_act_2), dim=0))

            state, reward, done, _ = env.step(action)
            rewards.append(reward)

        rewards = torch.tensor(rewards)  # Shape of [ep_len,].
        log_actions = torch.stack(log_actions, dim=0)  # Shape of [ep_len, 4].

        # Compute the cumulated returns without discount factor.
        rewards = torch.flip(rewards, dims=(0,))
        returns = torch.cumsum(rewards, dim=0)

        self.episodes_history.append((log_actions, returns))
