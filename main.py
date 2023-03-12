import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torchinfo import summary
from tqdm import tqdm

from src.batched_reinforce import BatchedReinforce
from src.environment import BatchedEternityEnv, EternityEnv
from src.model import CNNPolicy
from src.monte_carlo import MonteCarloTreeSearch
from src.reinforce import Reinforce
from src.supervised import EternityDataset, EternityTrainer
from src.supervised.data_generation import generate_sample


def read_config(yaml_path: Path) -> dict:
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config


def reinforce(config: dict[str, Any]):
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
    if config["reinforce"]["model_path"] != "":
        model.load_state_dict(torch.load(config["reinforce"]["model_path"])["model"])

    summary(
        model,
        input_size=(4, env.size, env.size),
        batch_dim=0,
        dtypes=[
            torch.long,
        ],
        device=config["device"],
    )

    trainer = Reinforce(
        env,
        model,
        config["device"],
        config["reinforce"]["learning_rate"],
        config["reinforce"]["gamma"],
        config["reinforce"]["n_batches"],
        config["reinforce"]["batch_size"],
    )
    trainer.launch_training(config)


def batched_reinforce(config: dict[str, Any]):
    env = EternityEnv(
        instance_path=config["env"]["path"],
    )
    instances = torch.stack(
        [
            torch.LongTensor(env.render())
            for _ in range(config["reinforce"]["batch_size"])
        ]
    )
    env = BatchedEternityEnv(instances, config["device"], seed=config["seed"])
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
        device=config["device"],
    )
    trainer = BatchedReinforce(
        env,
        model,
        config["device"],
        config["reinforce"]["learning_rate"],
        config["reinforce"]["gamma"],
        config["reinforce"]["n_batches"],
    )
    trainer.launch_training(config)


def generate_data(config: dict[str, Any]):
    """Generate data to be used to train a model."""
    # Get environment infos.
    env = EternityEnv(
        instance_path=config["env"]["path"],
        reward_type=config["env"]["reward_type"],
    )

    # Generate data.
    n_steps = env.max_steps * 2
    instances = np.zeros(
        (config["data_generation"]["n_samples"], n_steps, 4, env.size, env.size),
        dtype=np.int32,
    )
    actions = np.zeros(
        (config["data_generation"]["n_samples"], n_steps, 4),
        dtype=np.int32,
    )
    for sample_id in tqdm(range(config["data_generation"]["n_samples"])):
        inst, act = generate_sample(
            env.size,
            env.n_class,
            n_steps,
            config["seed"],
        )

        instances[sample_id] = inst
        actions[sample_id] = act

    # Save data.
    os.makedirs("./data", exist_ok=True)
    filename = os.path.basename(config["env"]["path"]).replace(".txt", ".npz")
    filepath = Path("./data") / filename
    np.savez_compressed(filepath, instances=instances, actions=actions)


def supervised(config: dict[str, Any]):
    """Train a supervised model."""
    # Load data.
    filename = os.path.basename(config["env"]["path"]).replace(".txt", ".npz")
    filepath = Path("./data") / filename
    train_dataset, test_dataset = EternityDataset.from_file(
        filepath,
        test_size=config["supervised"]["test_size"],
        seed=config["seed"],
    )

    # Get environment infos.
    env = EternityEnv(
        instance_path=config["env"]["path"],
        reward_type=config["env"]["reward_type"],
    )

    # Initialize model.
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

    trainer = EternityTrainer(
        model,
        train_dataset,
        test_dataset,
        env,
        config["supervised"]["learning_rate"],
        config["supervised"]["batch_size"],
        config["supervised"]["epoch_size"],
        config["supervised"]["n_epochs"],
        config["device"],
    )
    trainer.launch_training(config)


def monte_carlo():
    env = EternityEnv(instance_path="./instances/eternity_trivial_B.txt")
    env.reset()
    mcts = MonteCarloTreeSearch(env)
    root = mcts.root
    while not root.terminal:
        root = mcts.search(root, 2000)
        root.parent = None  # Cut the tree (to avoid useless backpropagations).

    env.reset(root.state)
    env.render("rgb_array", "test.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["reinforce", "generate-data", "supervised", "batched-reinforce"],
        default="reinforce",
    )
    parser.add_argument("--config", type=Path, default="configs/trivial.yaml")
    args = parser.parse_args()

    config = read_config(args.config)

    match args.mode:
        case "reinforce":
            reinforce(config)
        case "generate-data":
            generate_data(config)
        case "supervised":
            supervised(config)
        case "batched-reinforce":
            batched_reinforce(config)
