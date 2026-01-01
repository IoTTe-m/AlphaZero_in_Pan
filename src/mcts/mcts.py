from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from jax import numpy as jnp

from src.game_logic import ACTION_COUNT, GameState
from src.mcts.state_processors import PolicyStateProcessor, StateProcessor, ValueStateProcessor
from src.ml.neural_networks import AlphaZeroNNs
from src.ml.policy_net import call_policy_network_batched
from src.ml.value_net import call_value_network_batched


class McNode:
    def __init__(self, prior: float, state: GameState):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children: dict[int, McNode] = {}
        self.state: GameState = state
        self.uct_scores = -np.ones((ACTION_COUNT,)) * np.inf

    @staticmethod
    def puct_score(parent: 'McNode', child: 'McNode', c_puct_value: float) -> float:
        u_value = c_puct_value * child.prior * np.sqrt(parent.visit_count + 1) / (child.visit_count + 1)  # TODO: verify
        q_value = child.value_sum / child.visit_count if child.visit_count > 0 else 0
        return u_value + q_value

    def expanded(self):
        return len(self.children) > 0

    def is_terminal(self):
        return sum(self.state.is_done_array) >= self.state.no_players - 1

    def expand(self, action_probs: np.ndarray, actions_list: list[int], c_puct_value: float):
        for legal_action in actions_list:
            new_state = deepcopy(self.state)
            new_state.execute_action(legal_action)
            child = McNode(prior=float(action_probs[legal_action]), state=new_state)
            self.children[legal_action] = child
            self.uct_scores[legal_action] = McNode.puct_score(self, child, c_puct_value)

    def select_child(self) -> tuple['McNode', int]:
        action_index = int(np.argmax(self.uct_scores))
        return self.children[action_index], action_index

    def compute_visit_counts(self) -> np.ndarray:
        visits = [(action, child.visit_count) for action, child in self.children.items()]
        visit_counts = np.zeros((ACTION_COUNT,))
        for action, count in visits:
            visit_counts[action] = count
        return visit_counts

    def is_player_finished(self, player_number: int) -> bool:
        return self.state.is_done_array[player_number]


@dataclass
class MctsSearchState:
    world_idx: int
    leaf: McNode
    path: list[tuple[McNode, int]]


class MCTS:
    def __init__(self, networks: AlphaZeroNNs, num_worlds: int, num_simulations: int, c_puct_value: float = 1.0, policy_temp: float = 1.0):
        self.networks = networks
        self.num_worlds = num_worlds
        self.num_simulations = num_simulations
        self.c_puct_value = c_puct_value
        self.policy_temp = policy_temp

    # ROLLOUT:
    # zwiększ ilość odwiedzeń node'a o 1
    # wylistuj akcje
    # jeżeli jesteś w liściu, to odpal policy network
    # zapisz odpowiedź sieci
    # wybierz losową akcję przy pomocy śmiesznego wzorku (odpowiedź sieci + jakieś rzeczy przeróżne)
    # rozszerz drzewo przeszukiwań
    # rób rollout aż do liścia
    # policz value (jeżeli przegrany/wygrany, to znamy wartość, w przeciwnym wypadku użyj value network)
    # propaguj value w górę drzewa i zwiększ nagrodę gracza, który gra aktualnie w danym stanie
    @staticmethod
    def compute_action_probs(visit_counts: np.ndarray, temperature: float) -> np.ndarray:
        if temperature == 0:
            best_action = np.argmax(visit_counts)
            action_probs = np.zeros_like(visit_counts)
            action_probs[best_action] = 1.0
            return action_probs
        else:
            visit_counts_temp = visit_counts ** (1 / temperature)
            total_counts = np.sum(visit_counts_temp)
            if total_counts == 0:
                return np.ones_like(visit_counts) / len(visit_counts)
            return visit_counts_temp / total_counts

    def _run_batched_simulations(self, roots: list[McNode]) -> list[np.ndarray]:
        # Store accumulated values for each world to mimic original behavior
        # world_idx -> accumulated_values (no_players,)
        accumulated_values = [np.zeros(roots[0].state.no_players) for _ in roots]

        for _ in range(self.num_simulations):
            # 1. Selection Phase (CPU)
            leaves = self._select_leaves(roots)

            # 2. Batch Preparation Phase
            (
                prepared_knowledge_batch,
                table_state_batch_policy,
                actions_mask_batch,
                prepared_player_hands_batch,
                table_state_batch_value,
                expansion_contexts,
            ) = self._prepare_batch(leaves)

            # 3. Inference Phase (GPU)
            policy_outputs, value_outputs = self._run_inference(
                prepared_knowledge_batch,
                table_state_batch_policy,
                actions_mask_batch,
                prepared_player_hands_batch,
                table_state_batch_value,
            )

            # 4. Expansion & Backpropagation Phase
            self._backpropagate_batch(leaves, policy_outputs, value_outputs, expansion_contexts, accumulated_values)

        return accumulated_values

    def _select_leaves(self, roots: list[McNode]) -> list[MctsSearchState]:
        leaves: list[MctsSearchState] = []
        for world_idx, root in enumerate(roots):
            root.visit_count += 1
            path, leaf = self.explore(root)
            leaves.append(MctsSearchState(world_idx=world_idx, leaf=leaf, path=path))
        return leaves

    def _prepare_batch(
        self, leaves: list[MctsSearchState]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[int, list[int]]]:
        expansion_contexts = {}  # world_idx -> (actions_list)

        # Batch containers
        batch_prepared_knowledge = []
        batch_table_state_policy = []
        batch_actions_mask = []
        batch_prepared_player_hands = []
        batch_table_state_value = []

        # We strictly enforce batch size = num_worlds by iterating over the leaves list
        # which is guaranteed to be size num_worlds
        for search_state in leaves:
            world_idx = search_state.world_idx
            leaf = search_state.leaf

            # Encode state for batch (even if terminal, to keep shapes constant)
            prepared_knowledge, table_state, actions_mask, actions_list = PolicyStateProcessor.encode(leaf.state)
            prepared_player_hands, table_state_value = ValueStateProcessor.encode(leaf.state)

            batch_prepared_knowledge.append(prepared_knowledge)
            batch_table_state_policy.append(table_state)
            batch_actions_mask.append(actions_mask)
            batch_prepared_player_hands.append(prepared_player_hands)
            batch_table_state_value.append(table_state_value)

            if not leaf.is_terminal():
                expansion_contexts[world_idx] = actions_list

        # Convert to JAX arrays
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
    ):
        for i, search_state in enumerate(leaves):
            world_idx = search_state.world_idx
            leaf = search_state.leaf
            path = search_state.path

            if leaf.is_terminal():
                # Terminal evaluation
                values = np.ones(leaf.state.no_players) * (1 / (leaf.state.no_players - 1))
                values[leaf.state.current_player] = -1
            else:
                # NN evaluation
                priors = policy_outputs[i]
                actions_list = expansion_contexts[world_idx]

                # Expand
                leaf.expand(np.array(priors), actions_list, self.c_puct_value)

                # Value
                shifted_values = value_outputs[i]
                values = ValueStateProcessor.decode(shifted_values, leaf.state.current_player)
                values[leaf.state.is_done_array] = 1 / (leaf.state.no_players - 1)

            # Accumulate values for this simulation step
            accumulated_values[world_idx] += values

            # Backpropagate
            self.backpropagate(path, leaf, values)

    def run(self, game_state: GameState) -> tuple[np.ndarray, np.ndarray]:
        # Initialize roots for all worlds
        roots = []
        for _ in range(self.num_worlds):
            prepared_game_state = StateProcessor.get_mcts_state(game_state)
            roots.append(McNode(1.0, prepared_game_state))

        # Run batched simulations
        accumulated_values = self._run_batched_simulations(roots)

        # Aggregate results across worlds
        total_root_values = np.zeros(game_state.no_players)
        total_root_actions = np.zeros(ACTION_COUNT)

        for i, root in enumerate(roots):
            visit_counts = root.compute_visit_counts()

            # Average values for this specific world
            world_values = accumulated_values[i] / self.num_simulations

            total_root_values += world_values
            total_root_actions += visit_counts

        avg_root_values = total_root_values / self.num_worlds
        avg_root_actions = total_root_actions / self.num_worlds
        action_probs = self.compute_action_probs(avg_root_actions, self.policy_temp)
        return action_probs, avg_root_values

    def explore(self, root: McNode) -> tuple[list[tuple[McNode, int]], McNode]:
        node = root
        path = []
        while node.expanded():
            new_node, action = node.select_child()
            path.append((node, action))
            node = new_node
        return path, node

    def backpropagate(self, path: list[tuple[McNode, int]], leaf: McNode, values: np.ndarray):
        # Update the leaf node (it was expanded but not yet credited)
        leaf.visit_count += 1
        # Note: leaf.value_sum is not updated as it has no parent action leading to it in the path

        # Update all nodes on the path from leaf back to root
        for node, action in reversed(path):
            child = node.children[action]
            child.visit_count += 1
            # Credit child with the value for the player who took the action (parent's player)
            # This way Q(s,a) = expected value of action a for the player who takes it
            child.value_sum += values[node.state.current_player]
            node.uct_scores[action] = McNode.puct_score(node, child, c_puct_value=self.c_puct_value)
