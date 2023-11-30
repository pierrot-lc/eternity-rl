import pytest
import torch

from ..environment import EternityEnv
from ..model import Policy
from .tree import MCTSTree


def env_mockup(instance_path: str = "./instances/eternity_A.txt") -> EternityEnv:
    return EternityEnv.from_file(
        instance_path,
        episode_length=10,
        batch_size=2,
        scramble_size=1.0,
        device="cpu",
        seed=0,
    )


def policy_mockup(env: EternityEnv) -> Policy:
    return Policy(
        board_width=env.board_size,
        board_height=env.board_size,
        embedding_dim=20,
        n_heads=1,
        backbone_layers=1,
        decoder_layers=1,
        dropout=0.0,
    )


def tree_mockup() -> MCTSTree:
    env = env_mockup()
    policy = policy_mockup(env)
    tree = MCTSTree(env, policy, n_simulations=2, n_childs=3)
    assert tree.n_nodes == 7
    tree.childs = torch.LongTensor(
        [
            [
                [1, 2, 3],
                [4, 5, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [1, 2, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ]
    )
    tree.parents = torch.LongTensor(
        [
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    tree.actions = torch.LongTensor(
        [
            [
                [0, 0, 0, 0],
                [6, 5, 3, 2],
                [4, 5, 0, 1],
                [0, 0, 0, 0],
                [2, 7, 3, 1],
                [1, 3, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [3, 4, 2, 2],
                [4, 2, 2, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
    )
    tree.visits = torch.LongTensor(
        [
            [4, 2, 1, 1, 1, 1, 0],
            [2, 1, 1, 0, 0, 0, 0],
        ]
    )
    tree.sum_scores = torch.FloatTensor(
        [
            [3.0, 2.0, 0.6, 0.4, 1.0, 1.0, 0.0],
            [2.0, 1.1, 0.9, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    return tree


def tree_mockup_small() -> MCTSTree:
    env = env_mockup()
    policy = policy_mockup(env)
    tree = MCTSTree(env, policy, n_simulations=2, n_childs=3)
    assert tree.n_nodes == 7
    tree.childs = torch.LongTensor(
        [
            [
                [1, 2, 3],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ]
    )
    tree.parents = torch.LongTensor(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    tree.actions = torch.LongTensor(
        [
            [
                [0, 0, 0, 0],
                [6, 5, 3, 2],
                [4, 5, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
    )
    tree.visits = torch.LongTensor(
        [
            [4, 2, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    tree.sum_scores = torch.FloatTensor(
        [
            [3.0, 2.0, 0.6, 0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    return tree


@pytest.mark.parametrize(
    "nodes",
    [
        torch.LongTensor(
            [
                [4, 5, 6],
                [0, 1, 2],
            ]
        ),
        torch.LongTensor(
            [
                [1, 1, 6],
                [0, 0, 0],
            ]
        ),
    ],
)
def test_ucb(nodes: torch.Tensor):
    tree = tree_mockup()
    c = torch.sqrt(torch.Tensor([2]))
    ucb = torch.zeros_like(nodes, dtype=torch.float)
    for batch_id in range(nodes.shape[0]):
        for ucb_index, node_id in enumerate(nodes[batch_id]):
            node_visits = tree.visits[batch_id, node_id]

            if node_visits == 0:
                ucb[batch_id, ucb_index] = torch.inf
                continue

            parent_id = tree.parents[batch_id, node_id]
            parent_visits = tree.visits[batch_id, parent_id]
            node_score = tree.sum_scores[batch_id, node_id] / node_visits
            ucb[batch_id, ucb_index] = node_score + c * torch.sqrt(
                torch.log(parent_visits) / node_visits
            )

    assert torch.allclose(ucb, tree.ucb_scores(nodes)), "Wrong UCB scores"


@pytest.mark.parametrize(
    "nodes",
    [
        torch.LongTensor([1, 0]),
        torch.LongTensor([6, 1]),
    ],
)
def test_select_leaf(nodes: torch.Tensor):
    tree = tree_mockup()

    childs = []
    for batch_id, node_id in enumerate(nodes):
        childs.append(tree.childs[batch_id, node_id])
    childs = torch.stack(childs, dim=0)

    ucb = tree.ucb_scores(childs)
    best_childs = torch.argmax(ucb, dim=1)

    # Make sure we do not change the value of a leaf node.
    for batch_id, child_ids in enumerate(childs):
        if torch.all(child_ids == 0):
            # No childs !
            best_childs[batch_id] = nodes[batch_id]

    assert torch.all(best_childs == tree.select_leafs(nodes))


@pytest.mark.parametrize(
    "nodes",
    [
        torch.LongTensor([3, 0]),
        torch.LongTensor([1, 0]),
    ],
)
def test_expand_nodes(nodes: torch.Tensor):
    tree = tree_mockup_small()

    actions = tree.sample_actions(tree.envs)
    assert actions.shape == torch.Size(
        (tree.batch_size, tree.n_childs, 4)
    ), "Wrong actions shape"

    original_tree_nodes = tree.tree_nodes.clone()

    tree.expand_nodes(nodes, actions)

    for batch_id, node_id in enumerate(nodes):
        childs = tree.childs[batch_id, node_id]
        for child_number, child_id in enumerate(childs):
            assert (
                child_number + original_tree_nodes[batch_id] == child_id
            ), "Wrong child id"

            assert tree.parents[batch_id, child_id] == node_id, "Wrong parent id"

            assert torch.all(
                tree.actions[batch_id, child_id] == actions[batch_id, child_number]
            ), "Wrong child actions"

    assert torch.all(tree.tree_nodes == original_tree_nodes + tree.n_childs)
