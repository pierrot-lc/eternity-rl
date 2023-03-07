import numpy as np
import pytest

from .monte_carlo import Node


def test_node_values():
    state = np.zeros((4, 10, 10))
    action = np.zeros(4)

    node = Node(state=state, steps=0, action=action, parent=None, terminal=False)

    # Parent = None, visits = 0.
    assert node.exploration == float("+inf")
    assert node.exploitation == float("+inf")

    # Parent = None, visits != 0
    node.visits = 1
    assert node.exploration == 0.0
    assert node.exploitation != float("+inf")

    # Parent is a node visited 3 times.
    node.visits = 3
    node = Node(state=state, steps=0, action=action, parent=node, terminal=False)

    # Parent != None, visits != 0
    node.visits = 3
    node.rewards = 1
    assert node.exploration != 0.0 and node.exploration != float("+inf")
    assert node.exploitation != float("+inf")
    assert node.ucb(c=0) == node.exploitation

    node = Node(state=state, steps=0, action=action, parent=node, terminal=True)

    # Terminal = True, visits = 0, parent != None
    assert node.exploration == float("-inf")
