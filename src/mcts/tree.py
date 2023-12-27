import torch
from einops import rearrange, repeat
from tqdm import tqdm

from ..environment import EternityEnv
from ..model import Critic, Policy


class MCTSTree:
    def __init__(
        self,
        envs: EternityEnv,
        policy: Policy,
        critic: Critic,
        simulations: int,
        childs: int,
    ):
        self.envs = envs
        self.policy = policy
        self.critic = critic
        self.n_simulations = simulations
        self.n_childs = childs
        self.n_nodes = (
            self.n_simulations * self.n_childs
        ) + 1  # Add the root node (id '0').
        self.batch_size = self.envs.batch_size
        self.device = self.envs.device

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
        self.terminated = torch.zeros(
            (self.batch_size, self.n_nodes),
            dtype=torch.bool,
            device=self.device,
        )

    @torch.inference_mode()
    def step(self):
        """Do a one step of the MCTS algorithm."""
        # 1. Dive until we find a leaf.
        leafs, envs = self.select_leafs()

        # 2. Sample new nodes to add to the tree.
        actions, values, terminated = self.sample_nodes(envs)
        self.expand_nodes(leafs, actions, values, terminated)

        # 3. Backpropagate child values.
        # NOTE: The new child is upadated here as well.
        # Hence it will be visited two times already and have its score
        # doubled.
        childs = self.select_childs(leafs)
        values = self.sum_scores[self.batch_range, childs]
        self.backpropagate(childs, values)

    def best_actions(self) -> torch.Tensor:
        """Return the actions corresponding to the best child
        of the root node.

        ---
        Returns:
            The best actions to take from the root node based on the MCTS
            simulations.
                Shape of [batch_size, n_actions].
        """
        roots = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        childs = self.childs[self.batch_range, roots]
        scores = self.scores(childs)
        best_childs = torch.argmax(scores, dim=1)
        return self.actions[self.batch_range, best_childs]

    @torch.inference_mode()
    def evaluate(self, disable_logs: bool) -> torch.Tensor:
        """Do all the simulation steps of the MCTS algorithm.

        ---
        Returns:
            The best actions to take from the root node based on the MCTS
            simulations.
                Shape of [batch_size, n_actions].
        """
        assert torch.all(
            self.tree_nodes == 1
        ), "All trees should have the root node only."

        # An env is terminated if its root node is marked as terminated.
        terminated_envs = torch.zeros(
            self.batch_size, dtype=torch.bool, device=self.device
        )
        best_actions = torch.zeros(
            (self.batch_size, self.actions.shape[2]),
            dtype=torch.long,
            device=self.device,
        )

        for _ in tqdm(
            range(self.n_simulations),
            desc="MCTS simulations",
            leave=False,
            disable=disable_logs,
        ):
            self.step()

            # Save the actions of the new terminated envs if there are some.
            newly_terminated_envs = (~terminated_envs) & self.terminated[:, 0]
            best_actions[newly_terminated_envs] = self.best_actions()[
                newly_terminated_envs
            ]

            terminated_envs = self.terminated[:, 0]

        # Save the actions of the remaining envs.
        best_actions = self.best_actions()[~terminated_envs]

        return best_actions

    def ucb_scores(self, nodes: torch.Tensor) -> torch.Tensor:
        """Compute the UCB score of the given nodes.

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

        ucb = sum_scores / corrected_node_visits + c * torch.sqrt(
            torch.log(parent_visits) / corrected_node_visits
        )
        ucb[node_visits == 0] = torch.inf
        return ucb

    def scores(self, nodes: torch.Tensor) -> torch.Tensor:
        """Compute the mean score of the given nodes."""
        node_visits = torch.gather(self.visits, dim=1, index=nodes)
        sum_scores = torch.gather(self.sum_scores, dim=1, index=nodes)

        corrected_node_visits = node_visits.clone()
        corrected_node_visits[node_visits == 0] = 1  # Avoid division by 0.

        scores = sum_scores / corrected_node_visits
        scores[node_visits == 0] = torch.inf
        return scores

    def select_childs(self, nodes: torch.Tensor) -> torch.Tensor:
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

        # Ignore terminated or fictive childs.
        terminated = torch.gather(self.terminated, dim=1, index=childs)
        ucb[terminated | (childs == 0)] = -torch.inf

        best_childs = childs[self.batch_range, torch.argmax(ucb, dim=1)]

        # If a node has no child, it remains unchanged.
        no_child = (childs != 0).sum(dim=1) == 0
        best_childs[no_child] = nodes[no_child]

        return best_childs

    def select_leafs(self) -> tuple[torch.Tensor, EternityEnv]:
        """Iteratively dive to select the leaf to expand in each environment.

        ---
        Returns:
            leafs: The id of the leaf nodes.
                Shape of [batch_size,].
            envs: The environments corresponding to the leafs.
        """
        # Start with the root nodes.
        nodes = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        envs = EternityEnv.from_env(self.envs)

        # Iterate until all selected nodes have no childs.
        leafs = (self.childs[self.batch_range, nodes] != 0).sum(dim=1) == 0
        while not torch.all(leafs):
            nodes = self.select_childs(nodes)
            actions = self.actions[self.batch_range, nodes]

            # Some of the actions may come from leafs.
            # Those actions have to be ignored so that its env is unchanged.
            # WARNING: This should be modified appropriately with the env.
            actions[leafs] = 0  # If actions are null, the puzzle will not change.
            envs.n_steps[leafs] -= 1
            envs.step(actions)

            leafs = (self.childs[self.batch_range, nodes] != 0).sum(dim=1) == 0

        return nodes, envs

    def sample_nodes(self, envs: EternityEnv) -> torch.Tensor:
        """Use the policy to sample new actions from the given environments.
        Also evaluate the new childs directly using the critic.

        Duplicate the envs for each child and sample independently all of them.
        This means that some actions could be sampled multiple times.

        ---
        Args:
            envs: The environments to use to sample the childs.

        ---
        Returns:
            actions: The sampled actions.
                Shape of [batch_size, n_childs, n_actions].
            values: The estimated values of the corresponding childs.
                Shape of [batch_size, n_childs].
            terminated: The terminated childs.
                Shape of [batch_size, n_childs].
        """
        envs = EternityEnv.duplicate_interleave(envs, self.n_childs)
        actions, *_ = self.policy(
            envs.render(), envs.best_boards, envs.n_steps, sampling_mode="softmax"
        )
        boards, _, dones, truncated, _ = envs.step(actions)
        values = self.critic(boards, envs.best_boards, envs.n_steps)

        # If an env is done, we take the value of the final board instead of
        # the prediction of the critic.
        values = dones * envs.matches / envs.best_possible_matches + ~dones * values

        actions = rearrange(actions, "(b c) a -> b c a", c=self.n_childs)
        values = rearrange(values, "(b c) -> b c", c=self.n_childs)
        terminated = rearrange(dones | truncated, "(b c) -> b c", c=self.n_childs)

        # Save the best board if necessary.
        if self.envs.best_matches_ever < envs.best_matches_ever:
            self.envs.best_matches_ever = envs.best_matches_ever
            self.envs.best_board_ever = envs.best_board_ever

        return actions, values, terminated

    def expand_nodes(
        self,
        nodes: torch.Tensor,
        actions: torch.Tensor,
        values: torch.Tensor,
        terminated: torch.Tensor,
    ):
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
            values: Initial values of the childs.
                Shape of [batch_size, n_childs].
            terminated: Whether the childs are terminated.
                Shape of [batch_size, n_childs].
        """
        assert torch.any(
            self.tree_nodes + self.n_childs <= self.n_nodes
        ), "A tree has run out of nodes!"

        assert torch.any(
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

        # Add the values, initial visits and terminated states.
        self.sum_scores.scatter_(dim=1, index=childs_node_id, src=values)
        ones = torch.ones_like(childs_node_id, device=self.device, dtype=torch.long)
        self.visits.scatter_(dim=1, index=childs_node_id, src=ones)
        self.terminated.scatter_(dim=1, index=childs_node_id, src=terminated)

        # Add the actions.
        childs_node_id = repeat(childs_node_id, "b c -> b c a", a=actions.shape[2])
        self.actions.scatter_(dim=1, index=childs_node_id, src=actions)

    def update_nodes_info(
        self, nodes: torch.Tensor, values: torch.Tensor, filters: torch.Tensor
    ):
        """Update the information of the given nodes.

        If a node is masked (filter is False), its value is not updated.
        """
        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        self.visits[self.batch_range, nodes] += ones * filters
        self.sum_scores[self.batch_range, nodes] += values * filters

        # A parent is terminated if all its childs are terminated.
        childs = self.childs[self.batch_range, nodes]
        terminated = torch.gather(self.terminated, dim=1, index=childs)
        terminated[childs == 0] = False  # Ignore fictive childs.
        terminated = terminated.sum(dim=1) == (childs != 0).sum(dim=1)
        self.terminated[self.batch_range, nodes] = terminated

    def backpropagate(self, nodes: torch.Tensor, values: torch.Tensor):
        """Backpropagate the given values to the given nodes and their parents."""
        filters = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        self.update_nodes_info(nodes, values, filters)

        # Do not update twice a root node.
        filters = nodes != 0

        # While we did not update all root nodes.
        while not torch.all(~filters):
            # Root nodes are their own parents.
            nodes = self.parents[self.batch_range, nodes]
            self.update_nodes_info(nodes, values, filters)

            # Do not update twice a root node.
            filters = nodes != 0

    @property
    def tree_nodes(self) -> torch.Tensor:
        """Count the total number of nodes in each tree.

        ---
        Returns:
            The number of nodes in each tree.
                Shape of [batch_size,].
        """
        return (self.childs != 0).sum(dim=(1, 2)) + 1
