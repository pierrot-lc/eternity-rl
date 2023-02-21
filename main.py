import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torchinfo import summary
from tqdm import tqdm

from src.environment.data_generation import generate_sample
from src.environment.gym import EternityEnv
from src.model import CNNPolicy
from src.reinforce import Reinforce


def read_config(yaml_path: Path) -> dict:
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main(config: dict[str, Any]):
    env = EternityEnv(
        instance_path=config["env"]["path"],
        reward_type=config["env"]["reward_type"],
    )
    model = CNNPolicy(
        env.n_class,
        embedding_dim=config["model"]["embedding_dim"],
        n_layers=config["model"]["n_layers"],
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
    trainer.launch_training(config)
    trainer.make_gif("test.gif")


def generate_data(config: dict[str, Any]):
    """Generate data to be used to train a model."""
    # Get environment infos.
    env = EternityEnv(
        instance_path=config["env"]["path"],
        reward_type=config["env"]["reward_type"],
    )

    # Generate data.
    instances = np.zeros(
        (config["data_generation"]["n_samples"], env.max_steps, 4, env.size, env.size),
        dtype=np.int32,
    )
    actions = np.zeros(
        (config["data_generation"]["n_samples"], env.max_steps, 4),
        dtype=np.int32,
    )
    for sample_id in tqdm(range(config["data_generation"]["n_samples"])):
        inst, act = generate_sample(
            env.size,
            env.n_class,
            env.max_steps,
            config["data_generation"]["seed"],
        )

        instances[sample_id] = inst
        actions[sample_id] = act

    # Save data.
    os.makedirs("./data", exist_ok=True)
    filename = os.path.basename(config["env"]["path"])
    filename = os.path.splitext(filename)[0]
    filepath = Path("./data") / f"{filename}.npz"
    np.savez_compressed(filepath, instances=instances, actions=actions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["reinforce", "generate-data"],
        default="reinforce",
    )
    parser.add_argument("--config", type=Path, default="configs/trivial.yaml")
    args = parser.parse_args()

    config = read_config(args.config)

    match args.mode:
        case "reinforce":
            main(config)
        case "generate-data":
            generate_data(config)
