"""A batched implementation of the MCTS algorithm.

Thanks to:
- https://github.com/tmoer/alphazero_singleplayer.
- https://arxiv.org/pdf/1911.08265.pdf (notably, page 12).
- royale for the fruitful conversations.
"""
import torch
from einops import rearrange, repeat
from tqdm import tqdm

from ..environment import EternityEnv
from ..model import Critic, Policy


class MCTSTree:
    def __init__(
        self,
        gamma: float,
        n_simulations: int,
        n_childs: int,
        n_actions: int,
        batch_size: int,
        device: torch.device,
    ):
        self.gamma = gamma
        self.n_simulations = n_simulations
        self.n_childs = n_childs
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.device = device

        self.n_nodes = (
            self.n_simulations * self.n_childs
        ) + 1  # Add the root node (id '0').

        self.batch_range = torch.arange(self.batch_size, device=self.device)
        self.c_puct = torch.FloatTensor([0.01]).to(self.device)

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
            (self.batch_size, self.n_nodes, n_actions),
            dtype=torch.long,
            device=self.device,
        )
        self.visits = torch.zeros(
            (self.batch_size, self.n_nodes),
            dtype=torch.long,
            device=self.device,
        )
        self.rewards = torch.zeros(
            (self.batch_size, self.n_nodes),
            dtype=torch.float,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.batch_size, self.n_nodes),
            dtype=torch.float,
            device=self.device,
        )
        self.sum_scores = torch.zeros(
            (self.batch_size, self.n_nodes),
            dtype=torch.float,
            device=self.device,
        )
        self.priors = torch.zeros(
            (self.batch_size, self.n_nodes),
            dtype=torch.float,
            device=self.device,
        )
        self.terminated = torch.zeros(
            (self.batch_size, self.n_nodes),
            dtype=torch.bool,
            device=self.device,
        )

    def reset(self, envs: EternityEnv, policy: Policy, critic: Critic):
        """Reset the overall object state.
        The MCTSTree can then be used again.

        The given envs should be of the same batch size as the initial envs.
        """
        assert envs.batch_size == self.batch_size
        assert envs.device == self.device

        self.envs = envs
        self.policy = policy
        self.critic = critic

        self.childs.zero_()
        self.parents.zero_()
        self.actions.zero_()
        self.visits.zero_()
        self.rewards.zero_()
        self.values.zero_()
        self.sum_scores.zero_()
        self.priors.zero_()
        self.terminated.zero_()

    @torch.inference_mode()
    def evaluate(self, disable_logs: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Do all the simulation steps of the MCTS algorithm.

        ---
        Returns:
            probs: The target policy to learn from the MCTS simulations.
                Shape of [batch_size, n_childs].
            values: The target value to learn from the MCTS simulations.
                Shape of [batch_size,].
            actions: The corresponding actions sampled from the MCTS simulations.
                Shape of [batch_size, n_childs, n_actions].
        """
        assert torch.all(
            self.tree_nodes == 1
        ), "All trees should have the root node only."
        assert (
            hasattr(self, "envs")
            and hasattr(self, "policy")
            and hasattr(self, "critic")
        ), "You should reset the MCTS before using it."

        # Init roots value.
        self.values[:, 0] = self.critic(self.envs.render())
        self.sum_scores[:, 0] = self.values[:, 0]

        for _ in tqdm(
            range(self.n_simulations),
            desc="MCTS simulations",
            leave=False,
            disable=disable_logs,
        ):
            self.step()

        childs = self.childs[:, 0]  # Root's children.
        visits = torch.gather(self.visits, dim=1, index=childs)
        scores = self.scores(childs)
        rewards = torch.gather(self.rewards, dim=1, index=childs)

        probs = visits / torch.max(visits, dim=1, keepdim=True).values
        probs = probs / probs.sum(dim=1, keepdim=True)
        values = rewards + self.gamma * scores
        values = visits * values / visits.sum(dim=1, keepdim=True)
        values = values.sum(dim=1)

        childs = repeat(childs, "b c -> b c a", a=self.n_actions)
        actions = torch.gather(self.actions, dim=1, index=childs)

        return probs, values, actions

    @torch.inference_mode()
    def step(self):
        """Do a one step of the MCTS algorithm."""
        # 1. Dive until we find a leaf.
        leafs, envs = self.select_leafs()
        terminated_leafs = self.terminated[self.batch_range, leafs]

        # 2. Sample new nodes to add to the tree.
        actions, priors, rewards, values, terminated = self.sample_nodes(
            envs, sampling_mode="uniform"
        )

        # 3. Add and expand the new nodes.
        self.expand_nodes(
            leafs,
            actions,
            priors,
            rewards,
            values,
            terminated,
            to_ignore=terminated_leafs,
        )

        # 3.5 Add untouched values to terminated_leafs.
        if torch.any(terminated_leafs):
            self.visits[terminated_leafs, leafs[terminated_leafs]] += 1
            self.sum_scores[terminated_leafs, leafs[terminated_leafs]] += self.values[
                terminated_leafs, leafs[terminated_leafs]
            ]

        # 4. Backpropagate best child values.
        childs = self.select_childs(leafs)
        self.backpropagate(childs)

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
        node_visits = torch.gather(self.visits, dim=1, index=nodes)
        parent_nodes = torch.gather(self.parents, dim=1, index=nodes)
        parent_visits = torch.gather(self.visits, dim=1, index=parent_nodes)
        sum_scores = torch.gather(self.sum_scores, dim=1, index=nodes)
        priors = torch.gather(self.priors, dim=1, index=nodes)

        q_estimate = sum_scores / (node_visits + 1)
        # u_estimate = (
        #     priors * self.c_puct * torch.sqrt(parent_visits + 1) / (node_visits + 1)
        # )
        u_estimate = (
            self.c_puct * torch.sqrt(parent_visits + 1) / (node_visits + 1)
        )
        ucb = q_estimate + u_estimate
        return ucb

    def scores(self, nodes: torch.Tensor) -> torch.Tensor:
        """Compute the mean score of the given nodes.

        This is used to evaluate the best node.
        Do not use it to explore the tree.
        """
        node_visits = torch.gather(self.visits, dim=1, index=nodes)
        sum_scores = torch.gather(self.sum_scores, dim=1, index=nodes)

        scores = sum_scores / (node_visits + 1)
        return scores

    def select_childs(self, nodes: torch.Tensor) -> torch.Tensor:
        """Dive one step into the tree following the UCB score.
        Do not change the id of a node that has no child.

        NOTE: It is possible to select a terminated node!

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

        # Ignore fictive childs.
        ucb[childs == 0] = -torch.inf

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

    def sample_nodes(
        self, envs: EternityEnv, sampling_mode: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use the policy to sample new actions from the given environments.
        Also evaluate the new childs directly using the critic.

        Duplicate the envs for each child and sample independently all of them.
        This means that some actions could be sampled multiple times.

        ---
        Args:
            envs: The environments to use to sample the childs.
            sampling_mode: The sampling mode of the actions.

        ---
        Returns:
            actions: The sampled actions.
                Shape of [batch_size, n_childs, n_actions].
            priors: The priors of the corresponding childs.
                Shape of [batch_size, n_childs].
            rewards: The rewards obtained when arriving in
                the corresponding child states.
                Shape of [batch_size, n_childs].
            values: The estimated values of the corresponding childs.
                Shape of [batch_size, n_childs].
            terminated: The terminated childs.
                Shape of [batch_size, n_childs].
        """
        envs = EternityEnv.duplicate_interleave(envs, self.n_childs)
        actions, logprobs, _ = self.policy(envs.render(), sampling_mode=sampling_mode)

        # NOTE: We should technically ignore the potentially already terminated envs
        # here. But it is not a big deal as the envs won't crash if we try to step them.
        boards, rewards, dones, truncated, _ = envs.step(actions)
        values = self.critic(boards)

        # If an env is done, we ignore the predicted value from the critic.
        values *= (~dones).float()

        # Compute priors.
        logprobs = logprobs.sum(dim=1)
        priors = torch.exp(logprobs)

        # Reshape outputs.
        actions = rearrange(actions, "(b c) a -> b c a", c=self.n_childs)
        priors = rearrange(priors, "(b c) -> b c", c=self.n_childs)
        rewards = rearrange(rewards, "(b c) -> b c", c=self.n_childs)
        values = rearrange(values, "(b c) -> b c", c=self.n_childs)
        terminated = rearrange(dones | truncated, "(b c) -> b c", c=self.n_childs)

        # Save the best board if any.
        if self.envs.best_matches_ever < envs.best_matches_ever:
            self.envs.best_matches_ever = envs.best_matches_ever
            self.envs.best_board_ever = envs.best_board_ever

        return actions, priors, rewards, values, terminated

    def expand_nodes(
        self,
        nodes: torch.Tensor,
        actions: torch.Tensor,
        priors: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        terminated: torch.Tensor,
        to_ignore: torch.Tensor,
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
            priors: The priors of the corresponding childs.
                Shape of [batch_size, n_childs].
            rewards: The rewards obtained when arriving in
                the corresponding child states.
                Shape of [batch_size, n_childs].
            values: Initial values of the childs.
                Shape of [batch_size, n_childs].
            terminated: Whether the childs are terminated.
                Shape of [batch_size, n_childs].
            to_ignore: Whether to ignore the given nodes (e.g.
                terminal nodes).
                Shape of [batch_size, ].
        """
        assert torch.any(
            self.tree_nodes + self.n_childs <= self.n_nodes
        ), "Some tree to expand has run out of nodes!"

        assert torch.any(
            self.childs[self.batch_range, nodes] == 0
        ), "Some node to expand already has childs!"

        # Compute the node id of each child.
        arange = torch.arange(self.n_childs, device=self.device)
        arange = repeat(arange, "c -> b c", b=self.batch_size)
        childs_node_id = self.tree_nodes.unsqueeze(1) + arange
        # Shape of [batch_size, n_childs].

        parents_id = repeat(nodes, "b -> b c", c=self.n_childs)
        visits = torch.zeros_like(childs_node_id, device=self.device, dtype=torch.long)

        # This is a filter to modify only the non-terminated nodes.
        to_modify = ~to_ignore

        # Add the childs to their parent childs.
        self.childs[self.batch_range[to_modify], nodes[to_modify]] = childs_node_id[
            to_modify
        ]

        # Add the parents.
        parents = self.parents.scatter(dim=1, index=childs_node_id, src=parents_id)
        self.parents[to_modify] = parents[to_modify]

        # Add the values, initial visits and terminated states.
        priors = self.priors.scatter(dim=1, index=childs_node_id, src=priors)
        self.priors[to_modify] = priors[to_modify]

        rewards = self.rewards.scatter(dim=1, index=childs_node_id, src=rewards)
        self.rewards[to_modify] = rewards[to_modify]

        values = self.values.scatter(dim=1, index=childs_node_id, src=values)
        self.values[to_modify] = values[to_modify]
        self.sum_scores[to_modify] = values[to_modify]

        visits = self.visits.scatter(dim=1, index=childs_node_id, src=visits)
        self.visits[to_modify] = visits[to_modify]

        terminated = self.terminated.scatter(
            dim=1, index=childs_node_id, src=terminated
        )
        self.terminated[to_modify] = terminated[to_modify]

        childs_node_id = repeat(childs_node_id, "b c -> b c a", a=actions.shape[2])
        actions = self.actions.scatter(dim=1, index=childs_node_id, src=actions)
        self.actions[to_modify] = actions[to_modify]

    def update_nodes_info(
        self, nodes: torch.Tensor, scores: torch.Tensor, filters: torch.Tensor
    ):
        """Update the information of the given nodes.

        Do not update the nodes that are marked as filtered (when True).

        ---
        Args:
            nodes: Id of the node for each batch sample.
                Shape of [batch_size,].
            scores: The scores to add to the sum_scores.
                Shape of [batch_size,].
            filters: Whether to ignore the given nodes.
                Shape of [batch_size, ].
        """
        # A parent is terminated if all its childs are terminated.
        childs = self.childs[self.batch_range, nodes]
        terminated = torch.gather(self.terminated, dim=1, index=childs)
        terminated[childs == 0] = False  # Ignore fictive childs.
        terminated = terminated.sum(dim=1) == (childs != 0).sum(dim=1)
        self.terminated[self.batch_range, nodes] = terminated

        # Increment the number of visits.
        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        ones[filters] = 0
        self.visits[self.batch_range, nodes] += ones

        # Add scores to `sum_scores`.
        scores[filters] = 0
        self.sum_scores[self.batch_range, nodes] += scores

    def backpropagate(self, leafs: torch.Tensor):
        """Backpropagate the new value estimate from the given leafs to their parents.

        The leafs themselves are not updated as their initial expansion has already
        initialized the `sum_scores` and `terminated` states.
        """
        scores = self.values[self.batch_range, leafs]
        nodes = leafs

        # We maintain a list of node to exclude so that we do not update
        # twice a root node.
        filters = nodes == 0

        # While we did not update all root nodes.
        while not torch.all(filters):
            rewards = self.rewards[self.batch_range, nodes]
            scores = rewards + self.gamma * scores

            # NOTE: root nodes are their own parents.
            nodes = self.parents[self.batch_range, nodes]
            self.update_nodes_info(nodes, scores, filters)

            filters = nodes == 0

    @property
    def tree_nodes(self) -> torch.Tensor:
        """Count the total number of nodes in each tree.

        ---
        Returns:
            The number of nodes in each tree.
                Shape of [batch_size,].
        """
        return (self.childs != 0).sum(dim=(1, 2)) + 1
