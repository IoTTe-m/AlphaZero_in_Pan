"""Core game logic for the Pan card game.

This module implements the game state, rules, and action encoding for Pan,
a Polish card game where players try to get rid of all their cards.
"""

import numpy as np

SUITS = ['H', 'D', 'C', 'S']
SUITS_TO_NUMBERS = {suit: i for i, suit in enumerate(SUITS)}
RANKS = ['9', '10', 'J', 'Q', 'K', 'A']
SUIT_SYMBOLS = ['♥', '♦', '♣', '♠']
RANKS_TO_NUMBERS = {rank: i for i, rank in enumerate(RANKS)}
ACTION_COUNT = 51

NUM_SUITS = len(SUITS)
NUM_RANKS = len(RANKS)
CARD_ENCODING_SIZE = NUM_RANKS + NUM_SUITS

# Special card values
CARD_UNKNOWN = -1
CARD_ON_TABLE = -2

# Rank indices
RANK_NINE = 0
RANK_TEN = 1

# Suit indices
SUIT_HEARTS = 0
SUIT_SPADES = 3

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
    """Represents the complete state of a Pan card game."""

    def __init__(self, no_players: int = 4) -> None:
        """
        Initialize a new game state with the specified number of players.

        Args:
            no_players: Number of players (2, 3, or 4).
        """
        assert no_players in [2, 3, 4], 'Number of players should be equal to 2, 3 or 4'
        self._cards_count = len(SUITS_TO_NUMBERS) * len(RANKS_TO_NUMBERS)
        self._no_players = no_players
        self.player_hands = CARD_UNKNOWN * np.ones((len(SUITS_TO_NUMBERS), len(RANKS_TO_NUMBERS)), dtype=np.int32)
        self.table_state = np.zeros((self._cards_count, CARD_ENCODING_SIZE), dtype=np.int32)
        self.current_player = -1
        self.cards_on_table = 0
        self.is_done_array = np.zeros(self._no_players, dtype=np.bool)
        self.knowledge_table = CARD_UNKNOWN * np.ones((self._no_players, len(SUITS_TO_NUMBERS), len(RANKS_TO_NUMBERS)), dtype=np.int32)

        self.restart()

    @property
    def no_players(self) -> int:
        """Number of players in the game (read-only)."""
        return self._no_players

    @property
    def cards_count(self) -> int:
        """Total number of cards in the deck (read-only)."""
        return self._cards_count

    @staticmethod
    def hand_representation(ranks: np.ndarray, suits: np.ndarray) -> str:
        """
        Convert card ranks and suits arrays to a human-readable string.

        Args:
            ranks: Array of card rank indices.
            suits: Array of card suit indices.

        Returns:
            Human-readable string like "♥9 ♠K".
        """
        hand: list[str] = []
        for i in range(len(suits)):
            hand += [f'{SUIT_SYMBOLS[suits[i]]}{RANKS[ranks[i]]}']
        return ' '.join(hand)

    @staticmethod
    def decode_card(card_encoding: np.ndarray) -> tuple[int, int]:
        """
        Decode a one-hot encoded card into (rank, suit) indices.

        Args:
            card_encoding: One-hot encoded card array.

        Returns:
            Tuple of (rank_index, suit_index).
        """
        try:
            rank, suit = np.where(np.array(card_encoding) == 1)[0]
        except Exception as e:
            raise ValueError(f'Invalid card encoding: {card_encoding}') from e
        suit -= NUM_RANKS
        return rank, suit

    @staticmethod
    def fill_knowledge_table(knowledge_table: np.ndarray, player_hands: np.ndarray, no_players: int) -> None:
        """
        Update the knowledge table with known card locations from player hands.

        Args:
            knowledge_table: Array to update with card ownership info.
            player_hands: Current card ownership array.
            no_players: Number of players in the game.
        """
        for player in range(no_players):
            is_this_player = player_hands == player
            knowledge_table[player] = np.where(is_this_player, player_hands, knowledge_table[player])

    def prepare_deal(self) -> np.ndarray:
        """
        Create a random deal of cards to players.

        Returns:
            2D array (suits x ranks) with player indices.
        """
        cards_per_player = self._cards_count // self._no_players
        cards = np.repeat(np.arange(0, self._no_players, dtype=np.int32), cards_per_player)
        np.random.shuffle(cards)
        cards = np.reshape(cards, (len(SUITS_TO_NUMBERS), -1))
        return cards

    def restart(self) -> None:
        """Reset the game to initial state with a new random deal."""
        self.player_hands = self.prepare_deal()
        self.table_state = np.zeros((self._cards_count, CARD_ENCODING_SIZE), dtype=np.int32)
        self.current_player = self.get_starting_player()
        self.cards_on_table = 0
        self.is_done_array = np.zeros(self._no_players, dtype=np.bool)
        self.knowledge_table = CARD_UNKNOWN * np.ones((self._no_players, len(SUITS_TO_NUMBERS), len(RANKS_TO_NUMBERS)), dtype=np.int32)
        GameState.fill_knowledge_table(self.knowledge_table, self.player_hands, self._no_players)

    def print_table(self) -> None:
        """Print each player's hand for debugging."""
        for player in range(self._no_players):
            self.get_player_hand(player)

    def get_player_hand(self, player: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the cards held by a player.

        Args:
            player: Player index.

        Returns:
            Tuple of (ranks, suits) arrays.
        """
        ranks, suits = np.where(np.transpose(self.player_hands) == player)
        return ranks, suits

    def get_starting_player(self) -> int:
        """Return the player who holds the 9 of Hearts (starts the game)."""
        return int(self.player_hands[0][0])

    def _play_card(self, rank: int, suit: int) -> None:
        """
        Place a single card on the table and update game state.

        Args:
            rank: Rank index of the card.
            suit: Suit index of the card.
        """
        self.table_state[self.cards_on_table][rank] = 1
        self.table_state[self.cards_on_table][NUM_RANKS + suit] = 1
        self.cards_on_table += 1
        self.player_hands[suit][rank] = CARD_UNKNOWN
        self.knowledge_table[:, suit, rank] = CARD_ON_TABLE

    def _play_single_card(self, action: int) -> None:
        """
        Execute a single card play action.

        Args:
            action: Action index encoding the card to play.
        """
        rank = (action - OFFSET_SINGLE_CARD) % NUM_RANKS
        suit = (action - OFFSET_SINGLE_CARD) // NUM_RANKS
        self._play_card(rank, suit)

    def _play_three_nines(self, action: int) -> None:
        """
        Execute playing three nines (without heart) action.

        Args:
            action: Action index encoding spade position.
        """
        spade_index = action - OFFSET_THREE_NINES
        card_order = ['D', 'C']
        card_order.insert(spade_index, 'S')
        for suit in card_order:
            self._play_card(RANK_NINE, SUITS_TO_NUMBERS[suit])

    def _play_four_nines(self, action: int) -> None:
        """
        Execute playing all four nines action.

        Args:
            action: Action index encoding spade position.
        """
        spade_index = action - OFFSET_FOUR_NINES + 1
        card_order = ['H', 'D', 'C']
        card_order.insert(spade_index, 'S')
        for suit in card_order:
            self._play_card(RANK_NINE, SUITS_TO_NUMBERS[suit])

    def _play_four_cards(self, action: int) -> None:
        """
        Execute playing four cards of the same rank action.

        Args:
            action: Action index encoding rank and spade position.
        """
        spade_index = (action - OFFSET_FOUR_CARDS) % len(SUITS)
        rank = (action - OFFSET_FOUR_CARDS) // len(SUITS) + RANK_TEN
        card_order = ['H', 'D', 'C']
        card_order.insert(spade_index, 'S')
        for suit in card_order:
            self._play_card(rank, SUITS_TO_NUMBERS[suit])

    def _take_cards(self, player: int) -> None:
        """
        Take up to 3 cards from the table back into the player's hand.

        Args:
            player: Player index who takes the cards.
        """
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
        """
        Execute an action and advance the game state.

        Args:
            action: Action index to execute.

        Returns:
            True if the game has ended, False otherwise.
        """
        player = self.current_player

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

        player_shift = -1 if self.table_state[self.cards_on_table - 1][NUM_RANKS + SUIT_SPADES] == 1 else 1

        self.current_player = (self.current_player + player_shift) % self.no_players
        while self.is_done_array[self.current_player]:
            self.current_player = (self.current_player + player_shift) % self.no_players
        return False

    def _get_single_card_actions(self, player: int, table_rank: int) -> list[int]:
        """
        Get all valid single card play actions for the player.

        Args:
            player: Player index.
            table_rank: Minimum rank required to play.

        Returns:
            List of valid action indices.
        """
        actions = []
        for i in range(table_rank, NUM_RANKS):
            for suit in range(NUM_SUITS):
                if self.player_hands[suit][i] == player:
                    actions.append(OFFSET_SINGLE_CARD + suit * NUM_RANKS + i)
        return actions

    def _get_three_nines_actions(self, table_rank: int, ranks: np.ndarray) -> list[int]:
        """
        Get actions for playing three nines if available.

        Args:
            table_rank: Current rank on table.
            ranks: Player's card ranks.

        Returns:
            List of valid action indices.
        """
        if table_rank == RANK_NINE and np.count_nonzero(ranks == RANK_NINE) == 3:
            return list(range(OFFSET_THREE_NINES, OFFSET_FOUR_NINES))
        return []

    def _get_four_nines_actions(self, ranks: np.ndarray) -> list[int]:
        """
        Get actions for playing four nines if available.

        Args:
            ranks: Player's card ranks.

        Returns:
            List of valid action indices.
        """
        if np.count_nonzero(ranks == RANK_NINE) == 4:
            return list(range(OFFSET_FOUR_NINES, OFFSET_FOUR_CARDS))
        return []

    def _get_four_cards_actions(self, table_rank: int, ranks: np.ndarray) -> list[int]:
        """
        Get actions for playing four of a kind (same rank) if available.

        Args:
            table_rank: Current rank on table.
            ranks: Player's card ranks.

        Returns:
            List of valid action indices.
        """
        actions = []
        for i in range(table_rank + 1, NUM_RANKS):
            if np.count_nonzero(ranks == i) == NUM_SUITS:
                start_action = OFFSET_FOUR_CARDS + (i - RANK_TEN) * NUM_SUITS
                end_action = start_action + NUM_SUITS
                actions.extend(range(start_action, end_action))
        return actions

    def _get_take_cards_action(self) -> list[int]:
        """Get the take cards action if there are cards to take."""
        if self.cards_on_table > 1:
            return [ACTION_TAKE_CARDS]
        return []

    def get_possible_actions(self, player: int) -> list[int]:
        """
        Return all legal actions for the specified player.

        Args:
            player: Player index.

        Returns:
            List of valid action indices.
        """
        actions: list[int] = []
        ranks, suits = self.get_player_hand(player)

        if len(ranks) == 0:
            return actions

        # First move must be 9 of hearts
        if self.cards_on_table == 0 and ranks[0] == RANK_NINE and suits[0] == SUIT_HEARTS:
            actions.append(OFFSET_SINGLE_CARD)
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
        """Return the current player's knowledge of card locations."""
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
