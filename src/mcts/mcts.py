from copy import deepcopy

from src.game_logic import ACTION_COUNT, GameState
from src.ml.neural_networks import call_value_network, call_policy_network, AlphaZeroNNs
import numpy as np

from src.mcts.state_processors import StateProcessor, ValueStateProcessor, PolicyStateProcessor


class McNode:
    def __init__(self, prior: float, state: GameState):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state: GameState = state
        self.uct_scores = -np.ones((ACTION_COUNT,)) * np.inf

    @staticmethod
    def puct_score(parent: 'McNode', child: 'McNode', c_puct_value: float) -> float:
        u_value = c_puct_value * child.prior * np.sqrt(parent.visit_count) / (child.visit_count + 1)
        q_value = child.value_sum / child.visit_count if child.visit_count > 0 else 0
        return u_value + q_value

    def expanded(self):
        return len(self.children) > 0

    def is_terminal(self):
        return sum(self.state.is_done_array) >= self.state.no_players - 1

    def expand(self, az_networks: AlphaZeroNNs, c_puct_value: float):
        *policy_args, actions_list = PolicyStateProcessor.encode(self.state)

        action_probs = call_policy_network(az_networks.policy_network, az_networks.policy_network_params, *policy_args)
        for legal_action in actions_list:
            new_state = deepcopy(self.state)
            is_win_state = new_state.execute_action(legal_action)
            # TODO: verify
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


class MCTS:
    def __init__(self, networks: AlphaZeroNNs, num_worlds: int, num_simulations: int, c_puct_value: int = 1, policy_temp: float = 1.0):
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
    def compute_action_probs(visit_counts: np.ndarray, temp: float) -> np.ndarray:
        if temp == 0:
            best_action = np.argmax(visit_counts)
            action_probs = np.zeros_like(visit_counts)
            action_probs[best_action] = 1.0
            return action_probs
        else:
            visit_counts_temp = visit_counts ** (1 / temp) # TODO: check if correct, maybe softmax?
            total_counts = np.sum(visit_counts_temp)
            if total_counts == 0:
                return np.ones_like(visit_counts) / len(visit_counts)
            return visit_counts_temp / total_counts

    def run(self, game_state: GameState) -> tuple[np.ndarray, np.ndarray]:
        root_values = np.zeros(game_state.no_players)
        root_actions = np.zeros(ACTION_COUNT)

        for _ in range(self.num_worlds):
            prepared_game_state = StateProcessor.get_mcts_state(game_state)
            root = McNode(1.0, prepared_game_state)
            current_values = np.zeros(game_state.no_players)
            for _ in range(self.num_simulations):
                rollout_path, leaf = self.explore(root)

                if not leaf.is_terminal():
                    leaf.expand(self.networks, self.c_puct_value)
                    value_args = ValueStateProcessor.encode(leaf.state)
                    shifted_values = call_value_network(self.networks.value_network, self.networks.value_network_params,
                                                        *value_args)
                    values = ValueStateProcessor.decode(shifted_values, leaf.state.current_player)

                    values[leaf.state.is_done_array] = 1 / (leaf.state.no_players - 1)

                    _, leaf_action = leaf.select_child()
                    rollout_path.append((leaf, leaf_action))
                else:
                    values = np.ones(leaf.state.no_players) * (1 / (leaf.state.no_players - 1))
                    values[leaf.state.current_player] = -1

                self.backpropagate(rollout_path, values)
                current_values += values

            root_values += current_values / self.num_simulations # TODO: check if rotation is needed
            root_actions += root.compute_visit_counts()

        avg_root_values = root_values / self.num_worlds
        avg_root_actions = root_actions / self.num_worlds
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

    def backpropagate(self, path: list[tuple[McNode, int]], values: np.ndarray):
        for node, action in reversed(path):
            child = node.children[action]
            child.visit_count += 1
            child.value_sum += values[node.state.current_player]
        