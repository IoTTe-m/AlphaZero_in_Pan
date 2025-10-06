import random
import numpy as np

# hearts, diamonds, spades, clubs
Suits = ["♥", "♦", "♠", "♣"]
Values = ["9", "10", "J", "Q", "K", "A"]

class Table:
    def __init__(self):
        self.no_players = 4
        self.current_player = 0
        self.deal = self.prepare_deal()

    def prepare_deal(self) -> np.ndarray:
        # the deal is a 2d matrix
        # rows: suits
        # columns: values
        cards_per_player = 24 // self.no_players
        cards = np.repeat(np.arange(0, self.no_players), cards_per_player)
        np.random.shuffle(cards)
        cards = np.reshape(cards, (self.no_players, -1))
        return cards

    def print_table(self):
        for player in range(self.no_players):
            self.get_player_hand(player)

    def get_player_hand(self, player: int):
        values, suits = np.nonzero(np.array(np.transpose(self.deal) == player))
        for i in range(len(suits)):
            print(f"{Suits[suits[i]]}{Values[values[i]]}", end=" ")
        return np.where(self.deal == player)


table = Table()
table.get_player_hand(1)
