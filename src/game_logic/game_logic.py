from copy import deepcopy
from dataclasses import dataclass
from functools import partial

import numpy as np
from jax import numpy as jnp
from jax import jit, value_and_grad, vmap
from flax import linen as nn

# hearts, diamonds, clubs, spades
SUITS = ['H', 'D', 'C', 'S']
suits_to_numbers = {suit: i for i, suit in enumerate(SUITS)}
RANKS = ['9', '10', 'J', 'Q', 'K', 'A']
ranks_to_numbers = {rank: i for i, rank in enumerate(RANKS)}

ACTION_COUNT = 51


class GameState:
    def __init__(self, no_players: int = 4):
        assert no_players in [2, 3, 4], 'Number of players should be equal to 2, 3 or 4'

        self.cards_count = len(suits_to_numbers) * len(ranks_to_numbers)
        self.no_players = no_players
        self.player_hands = -np.ones((len(suits_to_numbers), len(ranks_to_numbers)))
        self.table_state = np.zeros((self.cards_count, 10))
        self.current_player = -1
        self.cards_on_table = 0
        self.is_done_array = np.zeros(self.no_players)
        self.knowledge_table = -np.ones((self.no_players, len(suits_to_numbers), len(ranks_to_numbers)))

        self.restart()

    @staticmethod
    def print_hand(ranks: np.ndarray, suits: np.ndarray) -> str:
        suit_symbols = ['♥', '♦', '♣', '♠']
        hand: list[str] = []
        for i in range(len(suits)):
            hand += [f'{suit_symbols[suits[i]]}{RANKS[ranks[i]]}']
        return ' '.join(hand)

    @staticmethod
    def decode_card(card_encoding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # [0-5, 6-9]
        # rank, suit
        try:
            rank, suit = np.where(np.array(card_encoding) == 1)[0]
        except Exception as e:
            raise ValueError(f'Invalid card encoding: {card_encoding}') from e
        suit -= 6
        return rank, suit

    @staticmethod
    def fill_knowledge_table(knowledge_table: np.ndarray, player_hands: np.ndarray, no_players: int):
        for player in range(no_players):
            is_this_player = player_hands == player
            knowledge_table[player] = np.where(is_this_player, player_hands, knowledge_table[player])

    def prepare_deal(self) -> np.ndarray:
        # the deal is a 2d matrix
        # rows: suits
        # columns: ranks
        cards_per_player = self.cards_count // self.no_players
        cards = np.repeat(np.arange(0, self.no_players), cards_per_player)
        np.random.shuffle(cards)
        cards = np.reshape(cards, (len(suits_to_numbers), -1))
        return cards

    def restart(self):
        self.player_hands = self.prepare_deal()
        self.table_state = np.zeros((24, 10))
        self.current_player = self.get_starting_player()
        self.cards_on_table = 0
        self.is_done_array = np.zeros(self.no_players)
        # -2: card is on the table, -1: we don't know where the card is
        self.knowledge_table = -np.ones((self.no_players, len(suits_to_numbers), len(ranks_to_numbers)))
        GameState.fill_knowledge_table(self.knowledge_table, self.player_hands, self.no_players)

    def print_table(self):
        for player in range(self.no_players):
            self.get_player_hand(player)

    def get_player_hand(self, player: int) -> tuple[np.ndarray, np.ndarray]:
        ranks, suits = np.where(np.transpose(self.player_hands) == player)
        return ranks, suits

    def get_starting_player(self) -> int:
        # check which player has 9♥
        return int(self.player_hands[0][0])

    def _play_card(self, rank: int, suit: int):
        self.table_state[self.cards_on_table][rank] = 1
        self.table_state[self.cards_on_table][6 + suit] = 1
        self.cards_on_table += 1
        self.player_hands[suit][rank] = -1
        self.knowledge_table[:, suit, rank] = -2

    def execute_action(self, player: int, action: int) -> bool:
        # returns True if action results in player victory
        # else returns False

        # action meanings:
        # 0-23 - play a single card
        # 24-26 - play three 9s, where 24 means spade goes on the bottom of the stack, 25 means spade goes second from bottom and so on
        # 27-29 - play four 9s, where 27 means spade goes on the bottom of the stack and so on
        # 30-33 - play four 10s
        # 34-37 - play four Js
        # 38-41 - play four Qs
        # 42-45 - play four Ks
        # 46-49 - play four As
        # 50 - take cards from table

        if action < 24:
            rank = action % 6
            suit = action // 6
            self._play_card(rank, suit)

        elif action in range(24, 27):
            # how many cards from top to reach spade
            spade_index = action - 24
            card_order = ['D', 'C']
            card_order.insert(spade_index, 'S')
            rank = 0
            for suit in card_order:
                self._play_card(rank, suits_to_numbers[suit])

        elif action in range(27, 30):
            # when playing four 9s, where to put spade
            spade_index = action - 27 + 1
            card_order = ['H', 'D', 'C']
            card_order.insert(spade_index, 'S')
            rank = 0
            for suit in card_order:
                self._play_card(rank, suits_to_numbers[suit])

        elif action in range(30, 50):
            spade_index = (action - 30) % 4
            rank = (action - 30) // 4
            card_order = ['H', 'D', 'C']
            card_order.insert(spade_index, 'S')
            for suit in card_order:
                self._play_card(rank, suits_to_numbers[suit])

        elif action == 50:
            for _ in range(3):
                if self.cards_on_table <= 1:
                    break
                self.cards_on_table -= 1
                card_encoding = self.table_state[self.cards_on_table]
                rank, suit = GameState.decode_card(card_encoding)
                self.table_state[self.cards_on_table] = np.zeros(10)
                self.player_hands[suit][rank] = player
                self.knowledge_table[:, suit, rank] = player

        else:
            raise ValueError('Invalid action')

        if len(self.get_player_hand(player)[0]) == 0:
            self.is_done_array[self.current_player] = 1
            if np.sum(self.is_done_array) == self.no_players - 1:
                return True

        player_shift = -1 if self.table_state[self.cards_on_table - 1][suits_to_numbers['S']] == 1 else 1

        self.current_player = (self.current_player + player_shift) % 4
        while self.is_done_array[self.current_player] == 1:
            self.current_player = (self.current_player + player_shift) % 4
        return False

    def get_possible_actions(self, player: int) -> list[int]:
        actions: list[int] = []
        ranks, suits = self.get_player_hand(player)

        if len(ranks) == 0:
            return actions

        # 9 hearts
        if self.cards_on_table == 0 and ranks[0] == 0 and suits[0] == 0:
            actions.append(0)
            # four 9s
            if np.count_nonzero(ranks == 0) == 4:
                actions += range(27, 30)

        if self.cards_on_table == 0:
            return actions

        card_encoding = self.table_state[self.cards_on_table - 1]
        rank, _ = GameState.decode_card(card_encoding)

        # single card play
        for i in range(rank, 6):
            for suit in range(4):
                if self.player_hands[suit][i] == player:
                    actions += [suit * 6 + i]

        # three nines
        if rank == 0 and np.count_nonzero(ranks == 0) == 3:
            actions += range(24, 27)

        # four cards play
        for i in range(rank + 1, 6):
            if np.count_nonzero(ranks == i) == 4:
                actions += range(26 + 4 * i, 30 + 4 * i)

        if self.cards_on_table > 1:
            actions += [50]

        return actions

    def get_player_knowledge(self) -> np.ndarray:
        return self.knowledge_table[self.current_player]

    def get_hands_card_counts(self) -> np.ndarray:
        _, cards_counts = np.unique(self.get_player_hand(self.current_player), return_counts=True, sorted=True)
        return cards_counts


class StateProcessor:
    @staticmethod
    def change_perspective(knowledge_array: np.ndarray, player_number: int, no_players: int) -> np.ndarray:
        return np.where(knowledge_array == -1, -1, (knowledge_array - player_number) % no_players)

    @staticmethod
    def get_mcts_state(state: GameState) -> GameState:
        """
        Gets current state and converts it into a starting state for Monte Carlo Tree Search
        :param state: GameState
        :return: new_state: GameState
        """
        current_knowledge = state.get_player_knowledge()
        mask_unknown = np.array(current_knowledge == -1)
        cards_per_player = state.get_hands_card_counts()

        # Randomly shuffle unknown cards
        players_to_fill = np.repeat(np.arange(len(cards_per_player)), cards_per_player)
        np.random.shuffle(players_to_fill)

        # Deal the unknown cards to players
        flat_knowledge = current_knowledge.flatten()
        mask_unknown = mask_unknown.flatten()
        flat_knowledge[mask_unknown] = players_to_fill
        filled_hands = flat_knowledge.reshape(current_knowledge.shape)

        # Build new knowledge table
        full_knowledge = -np.ones((state.no_players, len(SUITS), len(RANKS)))
        GameState.fill_knowledge_table(full_knowledge, filled_hands, state.no_players)

        # Create new state
        new_state = deepcopy(state)
        new_state.player_hands = filled_hands
        new_state.knowledge_table = full_knowledge

        return new_state

    @staticmethod
    def encode_actions(actions_list: list[int]) -> np.ndarray:
        # actions array ([0, 3, 23]) -> encoded array ([1,0,0,1,0...]
        encoded_actions = np.zeros((ACTION_COUNT,))
        encoded_actions[actions_list] = 1
        return encoded_actions

    @staticmethod
    def one_hot_encode_hands(player_hands: np.ndarray, no_players: int) -> np.ndarray:
        encoded_hands = np.zeros(player_hands.shape + (no_players + 1,))
        encoded_hands[..., -1] = player_hands == -1
        for player in range(no_players):
            encoded_hands[..., player] = player_hands == player
        return encoded_hands


class ValueStateProcessor:
    @staticmethod
    def encode(state: GameState) -> tuple[jnp.ndarray, jnp.ndarray]:
        # changes player_hands array so that the current player is encoded as 0
        # and one-hot encodes player hands
        player_hands = state.player_hands
        prepared_player_hands = StateProcessor.change_perspective(player_hands, state.current_player, state.no_players)
        prepared_player_hands = StateProcessor.one_hot_encode_hands(prepared_player_hands, state.no_players)
        table_state = state.table_state
        return jnp.array(prepared_player_hands), jnp.array(table_state)

    @staticmethod
    def decode(state_values: jnp.ndarray, current_player: int) -> np.ndarray:
        # for player 3: from [3, 0, 1, 2] to [0, 1, 2, 3]
        split_index = len(state_values) - current_player
        return np.array(jnp.concatenate(([state_values[split_index:], state_values[:split_index]])))


class PolicyStateProcessor:
    @staticmethod
    def encode(state: GameState) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, list[int]]:
        current_knowledge = state.get_player_knowledge()
        prepared_knowledge = StateProcessor.change_perspective(current_knowledge, state.current_player,
                                                               state.no_players)
        prepared_knowledge = StateProcessor.one_hot_encode_hands(prepared_knowledge, state.no_players)
        table_state = state.table_state

        possible_actions = state.get_possible_actions(state.current_player)
        encoded_actions = StateProcessor.encode_actions(possible_actions)

        return jnp.array(prepared_knowledge), jnp.array(table_state), jnp.array(encoded_actions), possible_actions


class ValueNetwork(nn.Module):
    no_players: int
    suits_count: int
    ranks_count: int

    def setup(self):
        # input_size = (self.no_players + self.suits_count + self.ranks_count) * self.suits_count * self.ranks_count
        output_size = self.no_players
        self.model = nn.Sequential(
            [
                nn.Dense(features=512),
                nn.relu,
                nn.Dense(features=256),
                nn.relu,
                nn.Dense(features=128),
                nn.relu,
                nn.Dense(features=32),
                nn.relu,
                nn.Dense(features=output_size),
            ]
        )

    def __call__(self, prepared_player_hands: jnp.ndarray, table_state: jnp.ndarray) -> jnp.ndarray:
        flattened_hands = prepared_player_hands.flatten()
        flattened_table = table_state.flatten()
        concat_features = jnp.concatenate((flattened_hands, flattened_table))
        return self.model(concat_features)


class PolicyNetwork(nn.Module):
    actions_space_size: int

    def setup(self):
        # input_size = (self.no_players + self.suits_count + self.ranks_count) * self.suits_count * self.ranks_count
        output_size = self.actions_space_size
        self.model = nn.Sequential(
            [
                nn.Dense(features=512),
                nn.relu,
                nn.Dense(features=256),
                nn.relu,
                nn.Dense(features=128),
                nn.relu,
                nn.Dense(features=32),
                nn.relu,
                nn.Dense(features=output_size),
            ]
        )

    def __call__(self, prepared_knowledge: jnp.ndarray, table_state: jnp.ndarray,
                 actions_mask: jnp.ndarray) -> jnp.ndarray:
        # a 1 in action_mask means that we want to include this action
        flattened_knowledge = prepared_knowledge.flatten()
        flattened_table = table_state.flatten()
        concat_features = jnp.concatenate((flattened_knowledge, flattened_table))
        logits = self.model(concat_features)
        return nn.softmax(logits, where=actions_mask)


@partial(jit, static_names=('value_network',))
def call_value_network(value_network: ValueNetwork, value_network_params: dict,
                       prepared_player_hands: jnp.ndarray, table_state: jnp.ndarray) -> jnp.ndarray:
    return value_network.apply(
        value_network_params, prepared_player_hands, table_state
    )


@partial(jit, static_names=('policy_network',))
def call_policy_network(policy_network: PolicyNetwork, policy_network_params: dict,
                        prepared_knowledge: jnp.ndarray, table_state: jnp.ndarray,
                        actions_mask: jnp.ndarray) -> jnp.ndarray:
    return policy_network.apply(
        policy_network_params, prepared_knowledge, table_state, actions_mask
    )


@dataclass
class AlphaZeroNNs:
    value_network: ValueNetwork
    policy_network: PolicyNetwork
    value_network_params: dict
    policy_network_params: dict


class Node:
    def __init__(self, prior: float, state: GameState):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state: GameState = state
        self.uct_scores = -np.ones((ACTION_COUNT,)) * np.inf

    def expanded(self):
        return len(self.children) > 0

    def is_terminal(self):
        return sum(self.state.is_done_array) == self.state.no_players - 1

    def expand(self, az_networks: AlphaZeroNNs, c_puct_value: float):
        current_player = self.state.current_player
        *policy_args, actions_list = PolicyStateProcessor.encode(self.state)
        action_probs = call_policy_network(az_networks.policy_network, az_networks.policy_network_params, *policy_args)
        for legal_action in actions_list:
            new_state = deepcopy(self.state)
            is_win_state = new_state.execute_action(current_player, legal_action)
            # TODO: verify
            child = Node(prior=float(action_probs[legal_action]), state=new_state)
            self.children[legal_action] = child
            self.uct_scores[legal_action] = MCTS.puct_score(self, child, c_puct_value)

    def select_child(self) -> tuple['Node', int]:
        action_index = np.argmax(self.uct_scores)
        return self.children[action_index], self.visit_count


class MCTS:
    def __init__(self, networks: AlphaZeroNNs, num_worlds: int, num_simulations: int, c_puct_value: int = 1):
        self.networks = networks
        self.num_worlds = num_worlds
        self.num_simulations = num_simulations
        self.c_puct_value = c_puct_value

    @staticmethod
    def puct_score(parent: Node, child: Node, c_puct_value: float) -> float:
        u_value = c_puct_value * child.prior * np.sqrt(parent.visit_count) / (child.visit_count + 1)
        q_value = child.value_sum / child.visit_count if child.visit_count > 0 else 0
        return u_value + q_value

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

    def run(self, game_state: GameState):
        root_values = np.zeros(game_state.no_players)
        root_actions = np.zeros(ACTION_COUNT)

        for _ in range(self.num_worlds):
            prepared_game_state = StateProcessor.get_mcts_state(game_state)
            root = Node(1.0, prepared_game_state)
            for _ in range(self.num_simulations):
                rollout_path, leaf = self.explore(root)

                if not leaf.is_terminal():
                    leaf.expand(self.networks, self.c_puct_value)
                    value_args = ValueStateProcessor.encode(leaf.state)
                    shifted_values = call_value_network(self.networks.value_network, self.networks.value_network_params,
                                                *value_args)
                    values = ValueStateProcessor.decode(shifted_values, leaf.state.current_player)
                    _, leaf_action = leaf.select_child()
                    rollout_path.append((leaf, leaf_action))
                else:
                    values = np.ones(leaf.state.no_players) * (1 / (leaf.state.no_players - 1))
                    values[leaf.state.current_player] = -1

                self.backpropagate(rollout_path, values)

    def explore(self, root: Node) -> tuple[list[tuple[Node, int]], Node]:
        node = root
        path = []
        while node.expanded():
            new_node, action = node.select_child()
            path.append((node, action))
            node = new_node
        return path, node

    def backpropagate(self, path: list[tuple[Node, int]], values: np.ndarray):
        for node, action in reversed(path):
            child = node.children[action]
            child.visit_count += 1
            child.value_sum += values[node.state.current_player]


        #idk add 1 to root visit count too

    # def search(self, prepared_game_state: GameState):


table = GameState()

score = [0, 0, 0, 0]
suit_symbols = ['♥', '♦', '♣', '♠']

for _ in range(10):
    table.restart()
    for _ in range(10000):
        print()
        print(f'Current player: {table.current_player}')
        if table.cards_on_table > 0:
            card_encoding = table.table_state[table.cards_on_table - 1]
            rank, suit = GameState.decode_card(card_encoding)
            print(f'Top card: {RANKS[rank]}{suit_symbols[suit]}')
        else:
            print('Top card: none')
        print(
            f'0: {GameState.print_hand(table.get_player_hand(0)[0], table.get_player_hand(0)[1])} {table.get_possible_actions(0)}')
        print(
            f'1: {GameState.print_hand(table.get_player_hand(1)[0], table.get_player_hand(1)[1])} {table.get_possible_actions(1)}')
        print(
            f'2: {GameState.print_hand(table.get_player_hand(2)[0], table.get_player_hand(2)[1])} {table.get_possible_actions(2)}')
        print(
            f'3: {GameState.print_hand(table.get_player_hand(3)[0], table.get_player_hand(3)[1])} {table.get_possible_actions(3)}')
        if table.execute_action(table.current_player, table.get_possible_actions(table.current_player)[0]):
            score[table.current_player] += 1
            break

print(score)
