from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from jax import numpy as jnp

from src.game_logic import ACTION_COUNT, GameState
from src.mcts.state_processors import PolicyStateProcessor, StateProcessor, ValueStateProcessor
from src.ml.neural_networks import AlphaZeroNNs
from src.ml.policy_net import call_policy_network_batched
from src.ml.value_net import call_value_network_batched

# MCTS constants
ROOT_PRIOR = 1.0
LOSER_VALUE = -1.0


class McNode:
    """A node in the Monte Carlo Tree Search tree."""

    def __init__(self, prior: float, state: GameState) -> None:
        """
        Initialize a node with prior probability and game state.

        Args:
            prior: Prior probability from policy network.
            state: Game state at this node.
        """
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children: dict[int, McNode] = {}
        self.state: GameState = state
        self.uct_scores = -np.ones((ACTION_COUNT,)) * np.inf

    @staticmethod
    def puct_score(parent: 'McNode', child: 'McNode', c_puct_value: float) -> float:
        """
        Calculate the PUCT score for child selection.

        Args:
            parent: Parent node.
            child: Child node to score.
            c_puct_value: Exploration constant.

        Returns:
            PUCT score combining exploration and exploitation.
        """
        u_value = c_puct_value * child.prior * np.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
        q_value = child.value_sum / child.visit_count if child.visit_count > 0 else 0
        return u_value + q_value

    def expanded(self) -> bool:
        """Check if this node has been expanded (has children)."""
        return len(self.children) > 0

    def is_terminal(self) -> bool:
        """Check if this node represents a terminal game state."""
        return sum(self.state.is_done_array) >= self.state.no_players - 1

    def expand(self, action_probs: np.ndarray, actions_list: list[int], c_puct_value: float) -> None:
        """
        Expand this node by creating children for all legal actions.

        Args:
            action_probs: Action probability distribution from policy network.
            actions_list: List of legal action indices.
            c_puct_value: Exploration constant for UCT scoring.
        """
        for legal_action in actions_list:
            new_state = deepcopy(self.state)
            new_state.execute_action(legal_action)
            child = McNode(prior=float(action_probs[legal_action]), state=new_state)
            self.children[legal_action] = child
            self.uct_scores[legal_action] = McNode.puct_score(self, child, c_puct_value)

    def select_child(self) -> tuple['McNode', int]:
        """Select the child with the highest UCT score."""
        action_index = int(np.argmax(self.uct_scores))
        return self.children[action_index], action_index

    def compute_visit_counts(self) -> np.ndarray:
        """Return visit counts for each action as an array."""
        visits = [(action, child.visit_count) for action, child in self.children.items()]
        visit_counts = np.zeros((ACTION_COUNT,))
        for action, count in visits:
            visit_counts[action] = count
        return visit_counts

    def is_player_finished(self, player_number: int) -> bool:
        """
        Check if the specified player has finished the game.

        Args:
            player_number: Player index to check.

        Returns:
            True if player has finished.
        """
        return self.state.is_done_array[player_number]


@dataclass
class MctsSearchState:
    """Tracks the state of a single MCTS search path during batched simulation."""

    world_idx: int
    leaf: McNode
    path: list[tuple[McNode, int]]


class MCTS:
    """Monte Carlo Tree Search implementation with neural network guidance."""

    def __init__(self, networks: AlphaZeroNNs, num_worlds: int, num_simulations: int, c_puct_value: float = 1.0, policy_temp: float = 1.0) -> None:
        """
        Initialize MCTS with neural networks and search parameters.

        Args:
            networks: Neural networks for policy and value estimation.
            num_worlds: Number of parallel search trees for variance reduction.
            num_simulations: Number of simulations per move.
            c_puct_value: Exploration constant for PUCT formula.
            policy_temp: Temperature for action probability scaling.
        """
        self.networks = networks
        self._num_worlds = num_worlds
        self._num_simulations = num_simulations
        self._c_puct_value = c_puct_value
        self._policy_temp = policy_temp

    @staticmethod
    def compute_action_probs(visit_counts: np.ndarray, temperature: float) -> np.ndarray:
        """
        Convert visit counts to action probabilities using temperature scaling.

        Args:
            visit_counts: Array of visit counts per action.
            temperature: Temperature parameter (0 = greedy, higher = more random).

        Returns:
            Normalized action probability distribution.
        """
        if temperature == 0:
            best_action = np.argmax(visit_counts)
            action_probs = np.zeros_like(visit_counts)
            action_probs[best_action] = 1.0
            return action_probs
        else:
            visit_counts_temp = visit_counts ** (1 / temperature)  # TODO: check if correct, maybe softmax?
            total_counts = np.sum(visit_counts_temp)
            if total_counts == 0:
                return np.ones_like(visit_counts) / len(visit_counts)
            return visit_counts_temp / total_counts

    def _run_batched_simulations(self, roots: list[McNode]) -> list[np.ndarray]:
        """Run multiple MCTS simulations in batched mode for efficiency."""
        accumulated_values = [np.zeros(roots[0].state.no_players) for _ in roots]

        for _ in range(self._num_simulations):
            # 1. Selection Phase
            leaves: list[MctsSearchState] = []
            for world_idx, root in enumerate(roots):
                root.visit_count += 1
                path, leaf = self.explore(root)
                leaves.append(MctsSearchState(world_idx=world_idx, leaf=leaf, path=path))

            # 2. Batch Preparation
            (
                prepared_knowledge_batch,
                table_state_batch_policy,
                actions_mask_batch,
                prepared_player_hands_batch,
                table_state_batch_value,
                expansion_contexts,
            ) = self._prepare_batch(leaves)

            # 3. Inference
            policy_outputs, value_outputs = self._run_inference(
                prepared_knowledge_batch,
                table_state_batch_policy,
                actions_mask_batch,
                prepared_player_hands_batch,
                table_state_batch_value,
            )

            # 4. Expansion & Backpropagation
            self._backpropagate_batch(leaves, policy_outputs, value_outputs, expansion_contexts, accumulated_values)

        return accumulated_values

    def _prepare_batch(
        self, leaves: list[MctsSearchState]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[int, list[int]]]:
        """Prepare batched inputs for neural network inference."""
        expansion_contexts = {}  # world_idx -> (actions_list)

        batch_prepared_knowledge = []
        batch_table_state_policy = []
        batch_actions_mask = []
        batch_prepared_player_hands = []
        batch_table_state_value = []

        for search_state in leaves:
            world_idx = search_state.world_idx
            leaf = search_state.leaf

            prepared_knowledge, table_state, actions_mask, actions_list = PolicyStateProcessor.encode(leaf.state)
            prepared_player_hands, table_state_value = ValueStateProcessor.encode(leaf.state)

            batch_prepared_knowledge.append(prepared_knowledge)
            batch_table_state_policy.append(table_state)
            batch_actions_mask.append(actions_mask)
            batch_prepared_player_hands.append(prepared_player_hands)
            batch_table_state_value.append(table_state_value)

            if not leaf.is_terminal():
                expansion_contexts[world_idx] = actions_list

        prepared_knowledge_batch = jnp.stack(batch_prepared_knowledge)
        table_state_batch_policy = jnp.stack(batch_table_state_policy)
        actions_mask_batch = jnp.stack(batch_actions_mask)

        prepared_player_hands_batch = jnp.stack(batch_prepared_player_hands)
        table_state_batch_value = jnp.stack(batch_table_state_value)

        return (
            prepared_knowledge_batch,
            table_state_batch_policy,
            actions_mask_batch,
            prepared_player_hands_batch,
            table_state_batch_value,
            expansion_contexts,
        )

    def _run_inference(
        self,
        prepared_knowledge_batch: jnp.ndarray,
        table_state_batch_policy: jnp.ndarray,
        actions_mask_batch: jnp.ndarray,
        prepared_player_hands_batch: jnp.ndarray,
        table_state_batch_value: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run policy and value networks on batched inputs."""
        policy_outputs = call_policy_network_batched(
            self.networks.policy_network.network,
            self.networks.policy_network.params,
            prepared_knowledge_batch,
            table_state_batch_policy,
            actions_mask_batch,
        )

        value_outputs = call_value_network_batched(
            self.networks.value_network.network,
            self.networks.value_network.params,
            prepared_player_hands_batch,
            table_state_batch_value,
        )
        return policy_outputs, value_outputs

    def _backpropagate_batch(
        self,
        leaves: list[MctsSearchState],
        policy_outputs: jnp.ndarray,
        value_outputs: jnp.ndarray,
        expansion_contexts: dict[int, list[int]],
        accumulated_values: list[np.ndarray],
    ) -> None:
        """Expand leaves and backpropagate values through the tree."""
        for i, search_state in enumerate(leaves):
            world_idx = search_state.world_idx
            leaf = search_state.leaf
            path = search_state.path

            if leaf.is_terminal():
                winner_value = 1.0 / (leaf.state.no_players - 1)
                values = np.ones(leaf.state.no_players) * winner_value
                values[leaf.state.current_player] = LOSER_VALUE
            else:
                # NN evaluation
                priors = policy_outputs[i]
                actions_list = expansion_contexts[world_idx]

                # Expand
                leaf.expand(np.array(priors), actions_list, self._c_puct_value)

                # Value
                shifted_values = value_outputs[i]
                values = ValueStateProcessor.decode(shifted_values, leaf.state.current_player)
                winner_value = 1.0 / (leaf.state.no_players - 1)
                values[leaf.state.is_done_array] = winner_value

            # Accumulate values for this simulation step
            accumulated_values[world_idx] += values

            # Backpropagate
            self.backpropagate(path, leaf, values)

    def run(self, game_state: GameState) -> tuple[np.ndarray, np.ndarray]:
        """
        Run MCTS from the given state.

        Args:
            game_state: Current game state to search from.

        Returns:
            Tuple of (action probabilities, value estimates per player).
        """
        roots = []
        for _ in range(self._num_worlds):
            prepared_game_state = StateProcessor.get_mcts_state(game_state)
            roots.append(McNode(ROOT_PRIOR, prepared_game_state))

        # Run batched simulations
        accumulated_values = self._run_batched_simulations(roots)

        # Aggregate results across worlds
        total_root_values = np.zeros(game_state.no_players)
        total_root_actions = np.zeros(ACTION_COUNT)

        for i, root in enumerate(roots):
            visit_counts = root.compute_visit_counts()

            # Average values for this specific world
            world_values = accumulated_values[i] / self._num_simulations

            total_root_values += world_values
            total_root_actions += visit_counts

        avg_root_values = total_root_values / self._num_worlds
        avg_root_actions = total_root_actions / self._num_worlds
        action_probs = self.compute_action_probs(avg_root_actions, self._policy_temp)
        return action_probs, avg_root_values

    def explore(self, root: McNode) -> tuple[list[tuple[McNode, int]], McNode]:
        """
        Traverse the tree from root to a leaf node.

        Args:
            root: Root node to start traversal from.

        Returns:
            Tuple of (path of (node, action) pairs, leaf node).
        """
        node = root
        path = []
        while node.expanded():
            new_node, action = node.select_child()
            path.append((node, action))
            node = new_node
        return path, node

    def backpropagate(self, path: list[tuple[McNode, int]], leaf: McNode, values: np.ndarray) -> None:
        """
        Update visit counts and value sums along the path from leaf to root.

        Args:
            path: List of (node, action) pairs from root to leaf.
            leaf: Leaf node where evaluation occurred.
            values: Value estimates for each player.
        """
        leaf.visit_count += 1

        for node, action in reversed(path):
            child = node.children[action]
            child.visit_count += 1
            child.value_sum += values[node.state.current_player]
            node.uct_scores[action] = McNode.puct_score(node, child, c_puct_value=self._c_puct_value)
