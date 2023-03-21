from pathlib import Path
from typing import Any

import torch
import yaml
from torchinfo import summary

from src.environment import BatchedEternityEnv
from src.model import CNNPolicy
from src.reinforce import Reinforce


def read_config(yaml_path: Path) -> dict:
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config


def reinforce(config: dict[str, Any]):
    env = BatchedEternityEnv.from_file(
        config["env"],
        config["reinforce"]["batch_size"],
        config["device"],
        config["seed"],
    )
    model = CNNPolicy(
        int(env.n_class),
        embedding_dim=config["model"]["embedding_dim"],
        n_layers=config["model"]["n_layers"],
        board_width=env.size,
        board_height=env.size,
    )
    summary(
        model,
        input_data=[
            torch.zeros(1, 4, env.size, env.size, dtype=torch.long),
            torch.randint(0, 10, size=(1,)),
        ],
        device="cpu",
    )
    trainer = Reinforce(
        env,
        model,
        config["device"],
        config["reinforce"]["learning_rate"],
        config["reinforce"]["value_weight"],
        config["reinforce"]["gamma"],
        config["reinforce"]["n_batches"],
        config["reinforce"]["advantage"],
    )
    trainer.launch_training(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default="configs/trivial.yaml")
    args = parser.parse_args()

    config = read_config(args.config)
    reinforce(config)
