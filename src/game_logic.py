import numpy as np

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
        self.player_hands = -np.ones((len(suits_to_numbers), len(ranks_to_numbers)), dtype=np.int32)
        self.table_state = np.zeros((self.cards_count, 10), dtype=np.int32)
        self.current_player = -1
        self.cards_on_table = 0
        self.is_done_array = np.zeros(self.no_players, dtype=np.int32)
        self.knowledge_table = -np.ones((self.no_players, len(suits_to_numbers), len(ranks_to_numbers)), dtype=np.int32)

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
        cards = np.repeat(np.arange(0, self.no_players, dtype=np.int32), cards_per_player)
        np.random.shuffle(cards)
        cards = np.reshape(cards, (len(suits_to_numbers), -1))
        return cards

    def restart(self):
        self.player_hands = self.prepare_deal()
        self.table_state = np.zeros((24, 10), dtype=np.int32)
        self.current_player = self.get_starting_player()
        self.cards_on_table = 0
        self.is_done_array = np.zeros(self.no_players, dtype=np.int32)
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
        self.table_state[self.cards_on_table][6 + suit] = 1
        self.cards_on_table += 1
        self.player_hands[suit][rank] = -1
        self.knowledge_table[:, suit, rank] = -2

    def execute_action(self, action: int) -> bool:
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

        player = self.current_player

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
            rank = (action - 30) // 4 + 1 # because we start from tens !!!
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
                self.table_state[self.cards_on_table] = np.zeros(10, dtype=np.int32)
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
        '''
        Counts how many unknown cards each player has.
        '''
        filtered_player_hands = self.player_hands[self.player_hands >= 0]
        player_indices, card_counts = np.unique(filtered_player_hands, return_counts=True, sorted=True)
        current_knowledge = self.get_player_knowledge()
        filtered_knowledge = current_knowledge[current_knowledge >= 0]
        knowledge_player_indices, cards_per_player = np.unique(filtered_knowledge, return_counts=True, sorted=True)
        result = np.zeros(self.no_players, dtype=np.int32)
        result[player_indices] = card_counts
        result[knowledge_player_indices] -= cards_per_player
        return result


# table = GameState()
# score = [0, 0, 0, 0]
# suit_symbols = ['♥', '♦', '♣', '♠']

# for _ in range(10):
#     table.restart()
#     for _ in range(10000):
#         card_encoding = table.table_state[table.cards_on_table - 1]
#         print()
#         print(f'Current player: {table.current_player}')
#         if table.cards_on_table > 0:
#             rank, suit = GameState.decode_card(card_encoding)
#             print(f'Top card: {RANKS[rank]}{suit_symbols[suit]}')
#         else:
#             print('Top card: none')
#         print(
#             f'0: {GameState.print_hand(table.get_player_hand(0)[0], table.get_player_hand(0)[1])} {table.get_possible_actions(0)}')
#         print(
#             f'1: {GameState.print_hand(table.get_player_hand(1)[0], table.get_player_hand(1)[1])} {table.get_possible_actions(1)}')
#         print(
#             f'2: {GameState.print_hand(table.get_player_hand(2)[0], table.get_player_hand(2)[1])} {table.get_possible_actions(2)}')
#         print(
#             f'3: {GameState.print_hand(table.get_player_hand(3)[0], table.get_player_hand(3)[1])} {table.get_possible_actions(3)}')
#         if table.execute_action(table.get_possible_actions(table.current_player)[0]):
#             score[table.current_player] += 1
#             break

# print(score)
