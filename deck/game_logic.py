import numpy as np

# hearts, diamonds, clubs, spades
Suits = {"H": 6, "D": 7, "C": 8, "S": 9}
Values = ["9", "10", "J", "Q", "K", "A"]

class GameState:
    def __init__(self):
        self.no_players = 4
        self.player_hands = self.prepare_deal()
        self.table_state = np.zeros((24, 10))
        self.current_player = self.get_starting_player()
        self.cards_on_table = 0
        self.is_done_array = np.zeros(self.no_players)

    @staticmethod
    def print_hand(values, suits):
        suit_symbols = ["♥", "♦", "♣", "♠"]
        hand = []
        for i in range(len(suits)):
            hand += [f"{suit_symbols[suits[i]]}{Values[values[i]]}"]
        return " ".join(hand)

    @staticmethod
    def decode_card(card_encoding):
        # [0-5, 6-9]
        # rank, suit
        try:
            rank, suit = np.where(np.array(card_encoding) == 1)[0]
        except:
            raise ValueError(f"Invalid card encoding: {card_encoding}")
        suit -= 6
        return rank, suit

    def prepare_deal(self) -> np.ndarray:
        # the deal is a 2d matrix
        # rows: suits
        # columns: values
        cards_per_player = 24 // self.no_players
        cards = np.repeat(np.arange(0, self.no_players), cards_per_player)
        np.random.shuffle(cards)
        cards = np.reshape(cards, (self.no_players, -1))
        return cards

    def restart(self):
        self.player_hands = self.prepare_deal()
        self.table_state = np.zeros((24, 10))
        self.current_player = self.get_starting_player()
        self.cards_on_table = 0
        self.is_done_array = np.zeros(self.no_players)

    def print_table(self):
        for player in range(self.no_players):
            self.get_player_hand(player)

    def get_player_hand(self, player: int) -> tuple[np.ndarray, np.ndarray]:
        values, suits = np.where(np.transpose(self.player_hands) == player)
        return values, suits

    def get_starting_player(self) -> int:
        # check which player has 9♥
        return int(self.player_hands[0][0])

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
            self.table_state[self.cards_on_table][rank] = 1
            self.table_state[self.cards_on_table][6 + suit] = 1
            self.cards_on_table += 1
            self.player_hands[suit][rank] = -1
        elif action in range(24, 27):
            # how many cards from top to reach spade
            spade_index = action - 24
            card_order = ["D", "C"]
            card_order.insert(spade_index, "S")
            for suit in card_order:
                self.table_state[self.cards_on_table][0] = 1
                self.table_state[self.cards_on_table][Suits[suit]] = 1
                self.cards_on_table += 1
                self.player_hands[suit][0] = -1
        elif action in range(27, 30):
            # when playing four 9s, where to put spade
            spade_index = action - 27 + 1
            card_order = ["H", "D", "C"]
            card_order.insert(spade_index, "S")
            for suit in card_order:
                self.table_state[self.cards_on_table][0] = 1
                self.table_state[self.cards_on_table][Suits[suit]] = 1
                self.cards_on_table += 1
                self.player_hands[suit][0] = -1
        elif action in range(30, 50):
            spade_index = (action - 30) % 4
            rank = (action - 30) // 4
            card_order = ["H", "D", "C"]
            card_order.insert(spade_index, "S")
            for suit in card_order:
                self.table_state[self.cards_on_table][rank] = 1
                self.table_state[self.cards_on_table][Suits[suit]] = 1
                self.cards_on_table += 1
                self.player_hands[suit][rank] = -1
        elif action == 50:
            for i in range(3):
                if self.cards_on_table <= 1:
                    break
                self.cards_on_table -= 1
                card_encoding = self.table_state[self.cards_on_table]
                rank, suit = GameState.decode_card(card_encoding)
                self.table_state[self.cards_on_table] = np.zeros(10)
                self.player_hands[suit][rank] = player
        else:
            raise ValueError("Invalid action")

        if len(self.get_player_hand(player)[0]) == 0:
            self.is_done_array[self.current_player] = 1
            if np.sum(self.is_done_array) == self.no_players - 1:
                return True

        player_shift = -1 if self.table_state[self.cards_on_table - 1][Suits["S"]] == 1 else 1

        self.current_player = (self.current_player + player_shift) % 4
        while self.is_done_array[self.current_player] == 1:
            self.current_player = (self.current_player + player_shift) % 4
        return False

    def get_possible_actions(self, player: int):
        actions = []
        values, suits = self.get_player_hand(player)

        # 9 hearts
        if self.cards_on_table == 0 and values[0] == 0 and suits[0] == 0:
            actions.append(0)
            # four 9s
            if np.count_nonzero(values == 0) == 4:
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
        if rank == 0 and np.count_nonzero(values == 0) == 3:
            actions += range(24, 27)

        # four cards play
        for i in range(rank + 1, 6):
            if np.count_nonzero(values == i) == 4:
                actions += range(26 + 4 * i, 30 + 4 * i)

        if self.cards_on_table > 1:
            actions += [50]

        return actions


table = GameState()
card = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
GameState.decode_card(card)

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
            print(f"Top card: {Values[rank]}{suit_symbols[suit]}")
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
