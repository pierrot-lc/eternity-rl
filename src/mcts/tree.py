import torch
from einops import repeat

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
        best_childs[no_child] = nodes[no_child]

        return best_childs

    def sample_actions(self, envs: EternityEnv) -> torch.Tensor:
        """Use the policy to sample new actions from the given environments.

        ---
        Args:
            envs: The environments to use to sample the childs.

        ---
        Returns:
            The sampled actions.
                Shape of [batch_size, n_childs, n_actions].
        """
        return self.model.forward_multi_sample(
            envs.render(), envs.best_boards, envs.n_steps, self.n_childs
        )

    def expand_nodes(self, nodes: torch.Tensor, actions: torch.Tensor):
        """Sample childs from the current nodes and add them
        to the trees.

        To be added to a tree, a child needs to:
        - Be added to the list of childs of its father.
        - Have its parent set.
        - Have its actions set.
        - Add one to the number of nodes in the tree.

        ---
        Args:
            nodes: Id of the node for each batch sample.
                Shape of [batch_size,].
            actions: The actions to add to the tree.
                Shape of [batch_size, n_childs, n_actions].
        """
        assert torch.all(
            self.tree_nodes + self.n_childs <= self.n_nodes
        ), "A tree has run out of nodes!"

        assert torch.all(
            self.childs[self.batch_range, nodes] == 0
        ), "A node already has childs!"

        # Compute the node id of each child.
        arange = torch.arange(self.n_childs, device=self.device)
        arange = repeat(arange, "c -> b c", b=self.batch_size)
        childs_node_id = self.tree_nodes.unsqueeze(1) + arange
        # Shape of [batch_size, n_childs].

        # Add the childs to their parent childs.
        self.childs[self.batch_range, nodes] = childs_node_id

        # Add the parents.
        parents_id = repeat(nodes, "b -> b c", c=self.n_childs)
        self.parents.scatter_(dim=1, index=childs_node_id, src=parents_id)

        # Add the actions.
        childs_node_id = repeat(childs_node_id, "b c -> b c a", a=actions.shape[2])
        self.actions.scatter_(dim=1, index=childs_node_id, src=actions)


    @property
    def tree_nodes(self) -> torch.Tensor:
        """Count the total number of nodes in each tree.

        ---
        Returns:
            The number of nodes in each tree.
                Shape of [batch_size,].
        """
        return (self.childs != 0).sum(dim=(1, 2)) + 1
