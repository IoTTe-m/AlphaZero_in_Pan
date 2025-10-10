from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy

# hearts, diamonds, clubs, spades
Suits = ["H", "D", "C", "S"]
Suit_to_number_map = {suit: i for i, suit in enumerate(Suits)}
Ranks = ["9", "10", "J", "Q", "K", "A"]
Rank_to_number_map = {rank: i for i, rank in enumerate(Ranks)}


class GameState:
    def __init__(self, no_players: int = 4):
        assert no_players in [2, 3, 4], "Number of players should be equal to 2, 3 or 4"

        self.cards_count = len(Suit_to_number_map) * len(Rank_to_number_map)
        self.no_players = no_players
        self.player_hands = -np.ones((len(Suit_to_number_map), len(Rank_to_number_map)))
        self.table_state = np.zeros((self.cards_count, 10))
        self.current_player = -1
        self.cards_on_table = 0
        self.is_done_array = np.zeros(self.no_players)
        self.knowledge_table = -np.ones((self.no_players, len(Suit_to_number_map), len(Rank_to_number_map)))

        self.restart()

    @staticmethod
    def print_hand(ranks, suits):
        suit_symbols = ["♥", "♦", "♣", "♠"]
        hand = []
        for i in range(len(suits)):
            hand += [f"{suit_symbols[suits[i]]}{Ranks[ranks[i]]}"]
        return " ".join(hand)

    @staticmethod
    def decode_card(card_encoding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # [0-5, 6-9]
        # rank, suit
        try:
            rank, suit = np.where(np.array(card_encoding) == 1)[0]
        except:
            raise ValueError(f"Invalid card encoding: {card_encoding}")
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
        cards = np.reshape(cards, (len(Suit_to_number_map), -1))
        return cards

    def restart(self):
        self.player_hands = self.prepare_deal()
        self.table_state = np.zeros((24, 10))
        self.current_player = self.get_starting_player()
        self.cards_on_table = 0
        self.is_done_array = np.zeros(self.no_players)
        # -2: card is on the table, -1: we don't know where the card is
        self.knowledge_table = -np.ones((self.no_players, len(Suit_to_number_map), len(Rank_to_number_map)))
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
            card_order = ["D", "C"]
            card_order.insert(spade_index, "S")
            rank = 0
            for suit in card_order:
                self._play_card(rank, Suit_to_number_map[suit])

        elif action in range(27, 30):
            # when playing four 9s, where to put spade
            spade_index = action - 27 + 1
            card_order = ["H", "D", "C"]
            card_order.insert(spade_index, "S")
            rank = 0
            for suit in card_order:
                self._play_card(rank, Suit_to_number_map[suit])

        elif action in range(30, 50):
            spade_index = (action - 30) % 4
            rank = (action - 30) // 4
            card_order = ["H", "D", "C"]
            card_order.insert(spade_index, "S")
            for suit in card_order:
                self._play_card(rank, Suit_to_number_map[suit])

        elif action == 50:
            for i in range(3):
                if self.cards_on_table <= 1:
                    break
                self.cards_on_table -= 1
                card_encoding = self.table_state[self.cards_on_table]
                rank, suit = GameState.decode_card(card_encoding)
                self.table_state[self.cards_on_table] = np.zeros(10)
                self.player_hands[suit][rank] = player
                self.knowledge_table[:, suit, rank] = player

        else:
            raise ValueError("Invalid action")

        if len(self.get_player_hand(player)[0]) == 0:
            self.is_done_array[self.current_player] = 1
            if np.sum(self.is_done_array) == self.no_players - 1:
                return True

        player_shift = -1 if self.table_state[self.cards_on_table - 1][Suit_to_number_map["S"]] == 1 else 1

        self.current_player = (self.current_player + player_shift) % 4
        while self.is_done_array[self.current_player] == 1:
            self.current_player = (self.current_player + player_shift) % 4
        return False

    def get_possible_actions(self, player: int):
        actions = []
        ranks, suits = self.get_player_hand(player)

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
        _, cards_counts = np.unique(
            self.get_player_hand(self.current_player), return_counts=True, sorted=True
        )
        return cards_counts


class StateProcessor(ABC):
    @staticmethod
    def change_perspective(knowledge_array: np.ndarray, player_number: int, no_players: int) -> np.ndarray:
        return np.where(knowledge_array == -1, -1, (knowledge_array - player_number) % no_players)

    @staticmethod
    def get_mcts_state(state: GameState) -> GameState:
        '''
        Gets current state and converts it into a starting state for Monte Carlo Tree Search
        :param state: GameState
        :return: new_state: GameState
        '''
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
        full_knowledge = -np.ones((state.no_players, len(Suits), len(Ranks)))
        GameState.fill_knowledge_table(full_knowledge, filled_hands, state.no_players)

        # Create new state
        new_state = deepcopy(state)
        new_state.player_hands = filled_hands
        new_state.knowledge_table = full_knowledge

        return new_state

    @staticmethod
    @abstractmethod
    def encode(state: GameState) -> tuple[np.ndarray, np.ndarray]:
        pass


class ValueStateProcessor(StateProcessor):
    @staticmethod
    def encode(state: GameState) -> tuple[np.ndarray, np.ndarray]:
        player_hands = state.player_hands
        prepared_player_hands = StateProcessor.change_perspective(player_hands, state.current_player, state.no_players)
        table_state = state.table_state
        return prepared_player_hands, table_state


class PolicyStateProcessor(StateProcessor):
    @staticmethod
    def encode(state: GameState) -> tuple[np.ndarray, np.ndarray]:
        current_knowledge = state.get_player_knowledge()
        prepared_knowledge = StateProcessor.change_perspective(current_knowledge, state.current_player,
                                                               state.no_players)
        table_state = state.table_state
        return prepared_knowledge, table_state


table = GameState()

score = [0, 0, 0, 0]
suit_symbols = ["♥", "♦", "♣", "♠"]

for _ in range(10):
    table.restart()
    for _ in range(10000):
        print()
        print(f"Current player: {table.current_player}")
        if table.cards_on_table > 0:
            card_encoding = table.table_state[table.cards_on_table - 1]
            rank, suit = GameState.decode_card(card_encoding)
            print(f"Top card: {Ranks[rank]}{suit_symbols[suit]}")
        else:
            print("Top card: none")
        print(
            f"0: {GameState.print_hand(table.get_player_hand(0)[0], table.get_player_hand(0)[1])} {table.get_possible_actions(0)}")
        print(
            f"1: {GameState.print_hand(table.get_player_hand(1)[0], table.get_player_hand(1)[1])} {table.get_possible_actions(1)}")
        print(
            f"2: {GameState.print_hand(table.get_player_hand(2)[0], table.get_player_hand(2)[1])} {table.get_possible_actions(2)}")
        print(
            f"3: {GameState.print_hand(table.get_player_hand(3)[0], table.get_player_hand(3)[1])} {table.get_possible_actions(3)}")
        if table.execute_action(table.current_player, table.get_possible_actions(table.current_player)[0]):
            score[table.current_player] += 1
            break

print(score)
