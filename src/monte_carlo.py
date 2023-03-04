import random
from itertools import product
from math import log, sqrt
from typing import Optional

import numpy as np

from .environment import EternityEnv


class Node:
    def __init__(
        self,
        state: np.ndarray,
        steps: int,
        action: np.ndarray,
        parent: Optional["Node"] = None,
        terminal: bool = False,
    ):
        """A node in the MCTS tree.

        ---
        Args:
            state: The state of the environment at this node.
                Shape of [4, size, size].
            steps: The number of steps taken to reach the state.
            action: The action taken to reach the state from the parent node.
            parent: The parent of this node.
                Is None if this node is the root.
            terminal: Whether this node is a terminal node.
        """
        self.state = state
        self.steps = steps
        self.action = action
        self.parent = parent
        self.terminal = terminal
        self.children = []
        self.visits = 0
        self.rewards = 0.0

    @property
    def exploitation(self) -> float:
        """The exploitation value of this node."""
        if self.visits == 0:
            return float("+inf")
        return self.rewards / self.visits

    @property
    def exploration(self) -> float:
        """The exploration value of this node."""
        if self.visits == 0:
            return float("+inf")
        if self.parent is None:
            return 0.0
        return sqrt(2 * log(self.parent.visits) / self.visits + 1e-8)

    def ucb(self, c: float = 2.0) -> float:
        """The value of this node.

        ---
        Args:
            c: The exploration constant.

        ---
        Returns:
            The value of this node.
        """
        return self.exploitation + c * self.exploration

    def __repr__(self) -> str:
        """Return the string representation of this node."""
        return f"Node(visits={self.visits}, rewards={self.rewards}, steps={self.steps}, terminal={self.terminal})"


class MonteCarloTreeSearch:
    def __init__(self, env: EternityEnv):
        """A Monte Carlo Tree Search algorithm.

        ---
        Args:
            state: The initial state of the environment.
        """
        self.env = env
        self.root = Node(env.instance.copy(), env.tot_steps, np.array([0, 0, 0, 0]))

    def search(self, root: Node, n: int = 100) -> Node:
        """Search the tree for the best node.

        ---
        Args:
            root: The root of the tree.
            n: The number of iterations to run.

        ---
        Returns:
            The best node.
        """
        for i in range(n):
            node = self.select(root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)

        return self.best_child(root, search=False)

    def select(self, node: Node) -> Node:
        """Select a node to expand.

        ---
        Args:
            node: The node to start from.

        ---
        Returns:
            The selected node.
        """
        while node.children:  # Until the node is a leaf.
            node = self.best_child(node, search=True)

        if not node.terminal:  # We can expand the node.
            self.expand(node)
            node = self.best_child(node, search=True)

        return node

    def expand(self, node: Node):
        """Expand the given node.

        ---
        Args:
            node: The node to expand.
        """
        assert not node.children, "The children of the node must be empty."

        env = self.env
        for tile_1, tile_2, roll_1, roll_2 in product(
            range(env.n_pieces), range(env.n_pieces), range(4), range(4)
        ):
            action = np.array([tile_1, roll_1, tile_2, roll_2])
            env.reset(node.state.copy(), node.steps)
            _, _, done, _ = env.step(action)

            child = Node(
                env.instance.copy(),
                env.tot_steps,
                action,
                node,
                terminal=done,
            )
            if done:
                child.visits = 1
                child.rewards = env.matches
            node.children.append(child)

    def simulate(self, node: Node) -> float:
        """Simulate a game from the given node.

        ---
        Args:
            node: The node to start from.

        ---
        Returns:
            The reward of the game.
        """
        if node.terminal:
            return node.exploitation

        env = self.env
        env.reset(node.state.copy(), node.steps)
        done = False
        while not done:
            tile_1 = random.randint(0, env.n_pieces - 1)
            tile_2 = random.randint(0, env.n_pieces - 1)
            roll_1 = random.randint(0, 3)
            roll_2 = random.randint(0, 3)

            action = np.array([tile_1, roll_1, tile_2, roll_2])
            _, _, done, _ = env.step(action)

        return env.matches

    def backpropagate(self, node: Node, reward: float):
        """Backpropagate the reward.

        ---
        Args:
            node: The node to start from.
            reward: The reward to backpropagate.
        """
        while node is not None:
            node.visits += 1
            node.rewards += reward
            node = node.parent

    def best_child(self, node: Node, search: bool) -> Node:
        """Find the best child of the given node.

        ---
        Args:
            node: The node to start from.
            search: Whether to use the UCB value or the exploitation value.

        ---
        Returns:
            The best child.
        """
        if search:
            return max(node.children, key=lambda child: child.ucb())
        else:
            return max(node.children, key=lambda child: child.exploitation)
