import torch

from ..environment import EternityEnv
from ..model import Policy


class MCTSTree:
    def __init__(
        self, envs: EternityEnv, model: Policy, n_simulations: int, n_childs: int
    ):
        self.envs = envs
        self.model = model
        self.n_simulations = n_simulations
        self.n_childs = n_childs
        self.n_nodes = (
            self.n_simulations * self.n_childs
        ) + 1  # Add the root node (id '0').
        self.batch_size = self.envs.batch_size
        self.device = self.envs.device

        self.current_node_envs = EternityEnv.from_env(self.envs)
        self.batch_range = torch.arange(self.batch_size, device=self.device)

        self.nodes = torch.zeros(
            (self.batch_size, self.n_nodes),
            dtype=torch.long,
            device=self.device,
        )
        self.childs = torch.zeros(
            (self.batch_size, self.n_nodes, self.n_childs),
            dtype=torch.long,
            device=self.device,
        )
        self.parents = torch.zeros(
            (self.batch_size, self.n_nodes),
            dtype=torch.long,
            device=self.device,
        )
        self.actions = torch.zeros(
            (self.batch_size, self.n_nodes, len(self.envs.action_space)),
            dtype=torch.long,
            device=self.device,
        )
        self.visits = torch.zeros(
            (self.batch_size, self.n_nodes),
            dtype=torch.long,
            device=self.device,
        )
        self.sum_scores = torch.zeros(
            (self.batch_size, self.n_nodes),
            dtype=torch.float,
            device=self.device,
        )

    def ucb_scores(self, nodes: torch.Tensor) -> torch.Tensor:
        """Compute the UCB score. of the given nodes.

        ---
        Args:
            nodes: Id of the node for each batch sample.
                Shape of [batch_size, n_nodes].

        ---
        Returns:
            The UCB score of the nodes. If a node has never
            been visited, its score is '+inf'.
                Shape of [batch_size, n_nodes].
        """
        c = torch.sqrt(torch.Tensor([2]))
        node_visits = torch.gather(self.visits, dim=1, index=nodes)
        parent_nodes = torch.gather(self.parents, dim=1, index=nodes)
        parent_visits = torch.gather(self.visits, dim=1, index=parent_nodes)
        sum_scores = torch.gather(self.sum_scores, dim=1, index=nodes)

        corrected_node_visits = node_visits.clone()
        corrected_node_visits[node_visits == 0] = 1  # Avoid division by 0.

        ucb = sum_scores / node_visits + c * torch.sqrt(
            torch.log(parent_visits) / node_visits
        )
        ucb[node_visits == 0] = torch.inf
        return ucb

    def select_leafs(self, nodes: torch.Tensor) -> torch.Tensor:
        """Dive one step into the tree following the UCB score.
        Do not change the id of a node that has no child.

        ---
        Args:
            nodes: Id of the node for each batch sample.
                Shape of [batch_size,].

        ---
        Returns:
            The id of the leaf nodes.
                Shape of [batch_size,].
        """
        # Shape of [batch_size, n_childs].
        childs = self.childs[self.batch_range, nodes]
        ucb = self.ucb_scores(childs)
        best_childs = torch.argmax(ucb, dim=1)

        # If a node has no child, it remains unchanged.
        no_child = (childs != 0).sum(dim=1) == 0
        best_childs[no_child] = nodes

        return best_childs

    def expand_nodes(self, nodes: torch.Tensor, envs: EternityEnv):
        """Sample childs from the current nodes and add them
        to the trees.

        ---
        Args:
            nodes: Id of the node for each batch sample.
                Shape of [batch_size,].
            envs: The environments to use to sample the childs.
                They correspond to the state represented by the nodes.
        """
        actions = self.model.forward_multi_sample(
            envs.render(), envs.best_boards, envs.n_steps, self.n_childs
        )
