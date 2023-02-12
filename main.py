from pathlib import Path
from typing import Any

import torch
import yaml
from torchinfo import summary

from src.environment.gym import EternityEnv
from src.model import CNNPolicy
from src.reinforce import Reinforce


def read_config(yaml_path: Path) -> dict:
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main(config: dict[str, Any]):
    env = EternityEnv(
        config["env"]["path"],
        manual_orient=True,
        reward_type=config["env"]["reward_type"],
    )
    model = CNNPolicy(
        env.n_class,
        embedding_dim=config["model"]["embedding_dim"],
        board_width=env.size,
        board_height=env.size,
    )
    summary(
        model,
        input_size=(4, env.size, env.size),
        batch_dim=0,
        dtypes=[
            torch.long,
        ],
        device="cpu",
    )

    trainer = Reinforce(
        env,
        model,
        "cuda",
        config["training"]["learning_rate"],
        config["training"]["gamma"],
        config["training"]["n_batches"],
        config["training"]["batch_size"],
    )
    trainer.launch_training()
    trainer.make_gif("test.gif")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default="configs/trivial.yaml")
    args = parser.parse_args()

    config = read_config(args.config)
    main(config)
