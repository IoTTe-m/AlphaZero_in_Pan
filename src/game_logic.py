import numpy as np

SUITS = ['H', 'D', 'C', 'S']
suits_to_numbers = {suit: i for i, suit in enumerate(SUITS)}
RANKS = ['9', '10', 'J', 'Q', 'K', 'A']
SUIT_SYMBOLS = ['♥', '♦', '♣', '♠']
ranks_to_numbers = {rank: i for i, rank in enumerate(RANKS)}
ACTION_COUNT = 51

NUM_SUITS = len(SUITS)
NUM_RANKS = len(RANKS)
CARD_ENCODING_SIZE = NUM_RANKS + NUM_SUITS

OFFSET_SINGLE_CARD = 0
COUNT_SINGLE_CARD = NUM_SUITS * NUM_RANKS

OFFSET_THREE_NINES = OFFSET_SINGLE_CARD + COUNT_SINGLE_CARD
COUNT_THREE_NINES = 3

OFFSET_FOUR_NINES = OFFSET_THREE_NINES + COUNT_THREE_NINES
COUNT_FOUR_NINES = 3

OFFSET_FOUR_CARDS = OFFSET_FOUR_NINES + COUNT_FOUR_NINES
COUNT_FOUR_CARDS = (NUM_RANKS - 1) * 4

ACTION_TAKE_CARDS = OFFSET_FOUR_CARDS + COUNT_FOUR_CARDS
CARDS_TO_TAKE = 3


class GameState:
    def __init__(self, no_players: int = 4):
        assert no_players in [2, 3, 4], 'Number of players should be equal to 2, 3 or 4'
        # TODO: not all of them need to be public
        self.cards_count = len(suits_to_numbers) * len(ranks_to_numbers)
        self.no_players = no_players
        self.player_hands = -np.ones((len(suits_to_numbers), len(ranks_to_numbers)), dtype=np.int32)
        self.table_state = np.zeros((self.cards_count, CARD_ENCODING_SIZE), dtype=np.int32)
        self.current_player = -1
        self.cards_on_table = 0
        self.is_done_array = np.zeros(self.no_players, dtype=np.bool)
        self.knowledge_table = -np.ones((self.no_players, len(suits_to_numbers), len(ranks_to_numbers)), dtype=np.int32)

        self.restart()

    @staticmethod
    def hand_representation(ranks: np.ndarray, suits: np.ndarray) -> str:
        hand: list[str] = []
        for i in range(len(suits)):
            hand += [f'{SUIT_SYMBOLS[suits[i]]}{RANKS[ranks[i]]}']
        return ' '.join(hand)

    @staticmethod
    def decode_card(card_encoding: np.ndarray) -> tuple[int, int]:
        # [0-5, 6-9]
        # rank, suit
        try:
            rank, suit = np.where(np.array(card_encoding) == 1)[0]
        except Exception as e:
            raise ValueError(f'Invalid card encoding: {card_encoding}') from e
        suit -= NUM_RANKS
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
        cards = np.repeat(np.arange(0, self.no_players, dtype=np.int32), cards_per_player)
        np.random.shuffle(cards)
        cards = np.reshape(cards, (len(suits_to_numbers), -1))
        return cards

    def restart(self):
        self.player_hands = self.prepare_deal()
        self.table_state = np.zeros((self.cards_count, CARD_ENCODING_SIZE), dtype=np.int32)
        self.current_player = self.get_starting_player()
        self.cards_on_table = 0
        self.is_done_array = np.zeros(self.no_players, dtype=np.bool)
        # -2: card is on the table, -1: we don't know where the card is
        self.knowledge_table = -np.ones((self.no_players, len(suits_to_numbers), len(ranks_to_numbers)), dtype=np.int32)
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
        self.table_state[self.cards_on_table][NUM_RANKS + suit] = 1
        self.cards_on_table += 1
        self.player_hands[suit][rank] = -1
        self.knowledge_table[:, suit, rank] = -2

    def _play_single_card(self, action: int):
        rank = (action - OFFSET_SINGLE_CARD) % NUM_RANKS
        suit = (action - OFFSET_SINGLE_CARD) // NUM_RANKS
        self._play_card(rank, suit)

    def _play_three_nines(self, action: int):
        # how many cards from top to reach spade
        spade_index = action - OFFSET_THREE_NINES
        card_order = ['D', 'C']
        card_order.insert(spade_index, 'S')
        for suit in card_order:
            self._play_card(0, suits_to_numbers[suit])

    def _play_four_nines(self, action: int):
        # when playing four 9s, where to put spade
        spade_index = action - OFFSET_FOUR_NINES + 1
        card_order = ['H', 'D', 'C']
        card_order.insert(spade_index, 'S')
        for suit in card_order:
            self._play_card(0, suits_to_numbers[suit])

    def _play_four_cards(self, action: int):
        spade_index = (action - OFFSET_FOUR_CARDS) % len(SUITS)
        rank = (action - OFFSET_FOUR_CARDS) // len(SUITS) + 1  # because we start from tens !!!
        card_order = ['H', 'D', 'C']
        card_order.insert(spade_index, 'S')
        for suit in card_order:
            self._play_card(rank, suits_to_numbers[suit])

    def _take_cards(self, player: int):
        for _ in range(CARDS_TO_TAKE):
            if self.cards_on_table <= 1:
                break
            self.cards_on_table -= 1
            card_encoding = self.table_state[self.cards_on_table]
            rank, suit = GameState.decode_card(card_encoding)
            self.table_state[self.cards_on_table] = np.zeros(CARD_ENCODING_SIZE, dtype=np.int32)
            self.player_hands[suit][rank] = player
            self.knowledge_table[:, suit, rank] = player

    def execute_action(self, action: int) -> bool:
        # returns True if action results in player victory
        # else returns False

        player = self.current_player
        # TODO fix naming
        if OFFSET_SINGLE_CARD <= action < OFFSET_THREE_NINES:
            self._play_single_card(action)

        elif OFFSET_THREE_NINES <= action < OFFSET_FOUR_NINES:
            self._play_three_nines(action)

        elif OFFSET_FOUR_NINES <= action < OFFSET_FOUR_CARDS:
            self._play_four_nines(action)

        elif OFFSET_FOUR_CARDS <= action < ACTION_TAKE_CARDS:
            self._play_four_cards(action)

        elif action == ACTION_TAKE_CARDS:
            self._take_cards(player)

        else:
            raise ValueError('Invalid action')

        if len(self.get_player_hand(player)[0]) == 0:
            self.is_done_array[self.current_player] = True
            if np.sum(self.is_done_array) == self.no_players - 1:
                return True

        player_shift = -1 if self.table_state[self.cards_on_table - 1][NUM_RANKS + suits_to_numbers['S']] == 1 else 1

        self.current_player = (self.current_player + player_shift) % self.no_players
        while self.is_done_array[self.current_player]:
            self.current_player = (self.current_player + player_shift) % self.no_players
        return False

    def _get_single_card_actions(self, player: int, table_rank: int) -> list[int]:
        actions = []
        for i in range(table_rank, NUM_RANKS):
            for suit in range(NUM_SUITS):
                if self.player_hands[suit][i] == player:
                    actions.append(OFFSET_SINGLE_CARD + suit * NUM_RANKS + i)
        return actions

    def _get_three_nines_actions(self, table_rank: int, ranks: np.ndarray) -> list[int]:
        if table_rank == 0 and np.count_nonzero(ranks == 0) == 3:
            return list(range(OFFSET_THREE_NINES, OFFSET_FOUR_NINES))
        return []

    def _get_four_nines_actions(self, ranks: np.ndarray) -> list[int]:
        if np.count_nonzero(ranks == 0) == 4:
            return list(range(OFFSET_FOUR_NINES, OFFSET_FOUR_CARDS))
        return []

    def _get_four_cards_actions(self, table_rank: int, ranks: np.ndarray) -> list[int]:
        actions = []
        for i in range(table_rank + 1, NUM_RANKS):
            if np.count_nonzero(ranks == i) == 4:
                start_action = OFFSET_FOUR_CARDS + (i - 1) * 4
                end_action = start_action + 4
                actions.extend(range(start_action, end_action))
        return actions

    def _get_take_cards_action(self) -> list[int]:
        if self.cards_on_table > 1:
            return [ACTION_TAKE_CARDS]
        return []

    def get_possible_actions(self, player: int) -> list[int]:
        actions: list[int] = []
        ranks, suits = self.get_player_hand(player)

        if len(ranks) == 0:
            return actions

        # 9 hearts
        if self.cards_on_table == 0 and ranks[0] == 0 and suits[0] == 0:
            actions.append(OFFSET_SINGLE_CARD)
            # four 9s
            actions.extend(self._get_four_nines_actions(ranks))

        if self.cards_on_table == 0:
            return actions

        card_encoding = self.table_state[self.cards_on_table - 1]
        table_rank, _ = GameState.decode_card(card_encoding)

        actions.extend(self._get_single_card_actions(player, table_rank))
        actions.extend(self._get_three_nines_actions(table_rank, ranks))
        actions.extend(self._get_four_cards_actions(table_rank, ranks))
        actions.extend(self._get_take_cards_action())

        return actions

    def get_player_knowledge(self) -> np.ndarray:
        return self.knowledge_table[self.current_player]

    def get_hands_card_counts(self) -> np.ndarray:
        """
        Counts how many unknown cards each player has.
        """
        filtered_player_hands = self.player_hands[self.player_hands >= 0]
        player_indices, card_counts = np.unique(filtered_player_hands, return_counts=True, sorted=True)
        current_knowledge = self.get_player_knowledge()
        filtered_knowledge = current_knowledge[current_knowledge >= 0]
        knowledge_player_indices, cards_per_player = np.unique(filtered_knowledge, return_counts=True, sorted=True)
        result = np.zeros(self.no_players, dtype=np.int32)
        result[player_indices] = card_counts
        result[knowledge_player_indices] -= cards_per_player
        return result
