from copy import deepcopy

import pytest
import torch
from einops import rearrange, repeat

from ..environment import EternityEnv
from ..model import Critic, Policy
from .tree import MCTSTree


def env_mockup(instance_path: str = "./instances/eternity_A.txt") -> EternityEnv:
    return EternityEnv.from_file(
        instance_path,
        episode_length=10,
        batch_size=2,
        device="cpu",
        seed=0,
    )


def models_mockup() -> tuple[Policy, Critic]:
    policy = Policy(
        embedding_dim=20,
        n_heads=1,
        backbone_layers=1,
        decoder_layers=1,
        dropout=0.0,
    )
    critic = Critic(
        embedding_dim=20,
        n_heads=1,
        backbone_layers=1,
        decoder_layers=1,
        dropout=0.0,
    )
    return policy, critic


def tree_mockup() -> MCTSTree:
    """A fake tree to make some test on it.

    Tree 1:
        0
        ├── 1
        │   ├── 4
        │   └── 5
        ├── 2
        └── 3

    Tree 2:
        0
        ├── 1
        └── 2
    """
    env = env_mockup()
    tree = MCTSTree(
        c_puct=1.5,
        gamma=0.9,
        n_simulations=2,
        n_childs=3,
        n_actions=len(env.action_space),
        batch_size=env.batch_size,
        device=env.device,
    )
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
    tree.rewards = torch.FloatTensor(
        [
            [0.0, 1.0, 0.4, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.4, 0.9, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    tree.values = torch.FloatTensor(
        [
            [0.0, 1.1, 0.8, 0.1, 0.1, 2.0, 0.0],
            [0.0, 0.4, 1.2, 0.2, 1.0, 1.0, 0.0],
        ]
    )
    tree.priors = torch.FloatTensor(
        [
            [0.0, 1.0, 0.8, 0.1, 0.1, 0.4, 0.3],
            [0.0, 0.4, 0.7, 0.2, 1.0, 1.0, 0.0],
        ]
    )
    tree.sum_scores = torch.FloatTensor(
        [
            [3.0, 2.0, 0.6, 0.4, 1.0, 1.0, 0.0],
            [2.0, 1.1, 0.9, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    tree.terminated = torch.BoolTensor(
        [
            [False, False, False, False, True, False, False],
            [False, True, True, False, False, False, False],
        ]
    )
    return tree


def tree_mockup_small() -> MCTSTree:
    """A fake tree to make some test on it.
    Here's its schema with node ids:

    Tree 1:
        0
        ├── 1
        ├── 2
        └── 3

    Tree 2:
        0
        └
    """
    env = env_mockup()
    tree = MCTSTree(
        c_puct=1.5,
        gamma=1.0,
        n_simulations=2,
        n_childs=3,
        n_actions=len(env.action_space),
        batch_size=env.batch_size,
        device=env.device,
    )
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
    tree.terminated = torch.BoolTensor(
        [
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
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
    # Backprop some q minmax estimates to avoid having qmin/qmax = 0.
    tree.update_q_minmax_estimates(nodes[:, 0])

    c = tree.c_puct
    ucb = torch.zeros_like(nodes, dtype=torch.float)
    for batch_id in range(nodes.shape[0]):
        q_min = tree.q_estimate_min[batch_id]
        q_max = tree.q_estimate_max[batch_id]

        for ucb_index, node_id in enumerate(nodes[batch_id]):
            node_visits = tree.visits[batch_id, node_id]

            parent_id = tree.parents[batch_id, node_id]
            parent_visits = tree.visits[batch_id, parent_id]
            node_score = tree.sum_scores[batch_id, node_id] / (node_visits + 1)
            node_score = (node_score - q_min) / (q_max - q_min + 1e-8)
            ucb[batch_id, ucb_index] = node_score + c * torch.sqrt(
                parent_visits + 1
            ) / (node_visits + 1)

    assert torch.allclose(ucb, tree.ucb_scores(nodes)), "Wrong UCB scores"


@pytest.mark.parametrize(
    "nodes",
    [
        torch.LongTensor([1, 0]),
        torch.LongTensor([6, 1]),
    ],
)
def test_select_childs(nodes: torch.Tensor):
    tree = tree_mockup()

    childs = []
    terminated = []
    for batch_id, node_id in enumerate(nodes):
        childs.append(tree.childs[batch_id, node_id])

        terminated.append([])
        for child_id in childs[-1]:
            terminated[-1].append(tree.terminated[batch_id, child_id])
        terminated[-1] = torch.BoolTensor(terminated[-1])

    childs = torch.stack(childs, dim=0)
    terminated = torch.stack(terminated, dim=0)

    ucb = tree.ucb_scores(childs)
    ucb[childs == 0] = -torch.inf
    best_childs_ids = torch.argmax(ucb, dim=1)
    best_childs = torch.stack(
        [
            childs[batch_id, best_childs_id]
            for batch_id, best_childs_id in enumerate(best_childs_ids)
        ],
        dim=0,
    )

    # Make sure we do not change the value of a leaf node.
    for batch_id, child_ids in enumerate(childs):
        if torch.all(child_ids == 0):
            # No childs !
            best_childs[batch_id] = nodes[batch_id]

    assert torch.all(best_childs == tree.select_childs(nodes))


@pytest.mark.parametrize(
    "tree",
    [
        tree_mockup(),
        tree_mockup_small(),
    ],
)
def test_select_leafs(tree: MCTSTree):
    env = env_mockup()
    policy, critic = models_mockup()
    tree.envs, tree.policy, tree.critic = env, policy, critic
    leafs, envs = tree.select_leafs()

    assert torch.all(
        tree.childs[tree.batch_range, leafs] == 0
    ), "Some leafs have a child"

    assert torch.any(
        envs.instances != tree.envs.instances
    ), "Tree instances have changed"

    for batch_id in range(tree.batch_size):
        actions = []
        copy_envs = EternityEnv.from_env(tree.envs)

        current_node = leafs[batch_id]
        while current_node != 0:
            actions.append(tree.actions[batch_id, current_node])
            current_node = tree.parents[batch_id, current_node]

        if len(actions) == 0:
            assert torch.all(
                copy_envs.instances[batch_id] == envs.instances[batch_id]
            ), "Replayed instances differ"
            continue

        # Shape of [n_actions, action_shape].
        actions = torch.stack(list(reversed(actions)))

        # Build the fictive actions for the other envs.
        all_actions = torch.zeros(
            (copy_envs.batch_size, actions.shape[0], actions.shape[1]),
            dtype=torch.long,
            device=copy_envs.device,
        )
        all_actions[batch_id] = actions

        # Simulate all actions and compare the final env with the given envs.
        for step_id in range(all_actions.shape[1]):
            copy_envs.step(all_actions[:, step_id])

        assert torch.all(
            copy_envs.instances[batch_id] == envs.instances[batch_id]
        ), "Replayed instances differ"


@pytest.mark.parametrize(
    "nodes",
    [
        torch.LongTensor([3, 0]),
        torch.LongTensor([1, 0]),
    ],
)
def test_expand_nodes(nodes: torch.Tensor):
    tree = tree_mockup_small()
    env = env_mockup()
    policy, critic = models_mockup()
    tree.envs, tree.policy, tree.critic = env, policy, critic

    actions, priors, rewards, values, terminated = tree.sample_nodes(
        tree.envs, sampling_mode="dirichlet"
    )
    assert actions.shape == torch.Size(
        (tree.batch_size, tree.n_childs, 4)
    ), "Wrong actions shape"
    assert values.shape == torch.Size(
        (tree.batch_size, tree.n_childs)
    ), "Wrong values shape"

    original_tree_nodes = tree.tree_nodes.clone()

    to_ignore = torch.zeros(tree.batch_size, dtype=torch.bool)
    tree.expand_nodes(nodes, actions, priors, rewards, values, terminated, to_ignore)

    for batch_id, node_id in enumerate(nodes):
        childs = tree.childs[batch_id, node_id]
        for child_number, child_id in enumerate(childs):
            assert (
                child_number + original_tree_nodes[batch_id] == child_id
            ), "Wrong child id"

            assert tree.parents[batch_id, child_id] == node_id, "Wrong parent id"

            assert torch.all(
                tree.actions[batch_id, child_id] == actions[batch_id, child_number]
            ), "Wrong children actions"
            assert torch.all(
                tree.priors[batch_id, child_id] == priors[batch_id, child_number]
            ), "Wrong children priors"
            assert torch.all(
                tree.rewards[batch_id, child_id] == rewards[batch_id, child_number]
            ), "Wrong children rewards"
            assert torch.all(
                tree.values[batch_id, child_id] == values[batch_id, child_number]
            ), "Wrong children values"
            assert torch.all(
                tree.sum_scores[batch_id, child_id] == values[batch_id, child_number]
            ), "Wrong children values"
            assert torch.all(
                tree.visits[batch_id, child_id] == 0
            ), "Wrong children visits"
            assert torch.all(
                tree.terminated[batch_id, child_id]
                == terminated[batch_id, child_number]
            ), "Wrong children terminated"

    assert torch.all(tree.tree_nodes == original_tree_nodes + tree.n_childs)


def test_repeat_interleave():
    """Mimic the way the inputs are duplicated in the `MCTSTree.sample_actions`."""
    n_repeats = 3
    batch_size = 10
    tensor = torch.randn((batch_size, 5, 5))
    tensor_interleave = repeat(tensor, "b ... -> b c ...", c=n_repeats)
    tensor_interleave = rearrange(tensor_interleave, "b c ... -> (b c) ...")
    tensor_interleave = rearrange(
        tensor_interleave, "(b c) ... -> b c ...", c=n_repeats
    )

    for b in range(batch_size):
        for i in range(n_repeats):
            assert torch.all(
                tensor[b] == tensor_interleave[b, i]
            ), "Tensor interleave not working!"


@pytest.mark.parametrize(
    "nodes",
    [
        torch.LongTensor([0, 1]),
        torch.LongTensor([5, 1]),
    ],
)
def test_backpropagate(nodes: torch.Tensor):
    tree = tree_mockup()

    sum_scores = tree.sum_scores.clone()
    visits = tree.visits.clone()
    terminated = tree.terminated.clone()

    for batch_id in range(tree.batch_size):
        node_id = nodes[batch_id]
        scores = tree.values[batch_id, node_id]

        while node_id != 0:
            rewards = tree.rewards[batch_id, node_id]
            scores = rewards + tree.gamma * scores

            node_id = tree.parents[batch_id, node_id]

            # Terminated is True if all non-fictive childs are terminated.
            all_terminated = True
            for child_id in tree.childs[batch_id, node_id]:
                if child_id != 0 and not tree.terminated[batch_id, child_id]:
                    all_terminated = False

            sum_scores[batch_id, node_id] += scores
            visits[batch_id, node_id] += 1
            terminated[batch_id, node_id] = all_terminated

    tree.backpropagate(nodes)

    assert torch.allclose(sum_scores, tree.sum_scores)
    assert torch.allclose(visits, tree.visits)
    assert torch.allclose(terminated, tree.terminated)


@pytest.mark.parametrize(
    "nodes",
    [
        torch.LongTensor([3, 0]),
        torch.LongTensor([1, 0]),
    ],
)
def test_update_q_minmax_estimates(nodes: torch.Tensor):
    tree = tree_mockup()
    q_max = tree.q_estimate_max.clone()
    q_min = tree.q_estimate_min.clone()

    tree.update_q_minmax_estimates(nodes)

    for batch_id in range(tree.batch_size):
        node_id = nodes[batch_id]
        while node_id != 0:
            q_estimate = tree.sum_scores[batch_id, node_id] / (
                tree.visits[batch_id, node_id] + 1
            )

            q_max[batch_id] = max(q_max[batch_id], q_estimate)
            q_min[batch_id] = min(q_min[batch_id], q_estimate)

            node_id = tree.parents[batch_id, node_id]

        # Do a step for the root node as well.
        q_estimate = tree.sum_scores[batch_id, node_id] / (
            tree.visits[batch_id, node_id] + 1
        )

        q_max[batch_id] = max(q_max[batch_id], q_estimate)
        q_min[batch_id] = min(q_min[batch_id], q_estimate)

    assert torch.allclose(q_max, tree.q_estimate_max)
    assert torch.allclose(q_min, tree.q_estimate_min)


def test_all_terminated():
    """When all the tree is terminated, the tree.step() should not break."""
    env = env_mockup()
    policy, critic = models_mockup()

    tree = tree_mockup_small()
    tree.envs, tree.policy, tree.critic = env, policy, critic
    tree.terminated[0, ...] = True

    actions_ = tree.actions.clone()
    childs_ = tree.childs.clone()
    parents_ = tree.parents.clone()
    priors_ = tree.priors.clone()
    rewards_ = tree.rewards.clone()
    sum_scores_ = tree.sum_scores.clone()
    terminated_ = tree.terminated.clone()
    values_ = tree.values.clone()
    visits_ = tree.visits.clone()

    leafs, envs = tree.select_leafs()
    actions, priors, rewards, values, terminated = tree.sample_nodes(
        envs, sampling_mode="dirichlet"
    )
    to_ignore = torch.zeros(env.batch_size, dtype=torch.bool)
    to_ignore[0] = True
    tree.expand_nodes(leafs, actions, priors, rewards, values, terminated, to_ignore)

    assert torch.all(actions_[to_ignore] == tree.actions[to_ignore])
    assert torch.all(childs_[to_ignore] == tree.childs[to_ignore])
    assert torch.all(parents_[to_ignore] == tree.parents[to_ignore])
    assert torch.all(priors_[to_ignore] == tree.priors[to_ignore])
    assert torch.all(rewards_[to_ignore] == tree.rewards[to_ignore])
    assert torch.all(sum_scores_[to_ignore] == tree.sum_scores[to_ignore])
    assert torch.all(terminated_[to_ignore] == tree.terminated[to_ignore])
    assert torch.all(values_[to_ignore] == tree.values[to_ignore])
    assert torch.all(visits_[to_ignore] == tree.visits[to_ignore])

    # Make sure the step does not break.
    tree.step()
