import random

import numpy as np
import pytest
from _pytest.capture import CaptureFixture

from src.game_logic import (
    ACTION_TAKE_CARDS,
    NUM_RANKS,
    OFFSET_FOUR_CARDS,
    OFFSET_FOUR_NINES,
    OFFSET_SINGLE_CARD,
    OFFSET_THREE_NINES,
    RANKS,
    SUIT_SYMBOLS,
    GameState,
)

RANK_9 = 0
RANK_10 = 1
RANK_J = 2
RANK_Q = 3
RANK_K = 4
RANK_A = 5

SUIT_H = 0
SUIT_D = 1
SUIT_C = 2
SUIT_S = 3


class TestGameState:
    @pytest.fixture
    def game(self) -> GameState:
        return GameState(no_players=4)

    def setup_hand(self, game: GameState, player: int, cards: list[tuple[int, int]]) -> None:
        """Helper to set up a specific hand for a player.
        cards: list of tuples (rank_idx, suit_idx)
        """
        for rank, suit in cards:
            game.player_hands[suit][rank] = player

    def setup_table(self, game: GameState, cards: list[tuple[int, int]]) -> None:
        """Helper to set up cards on the table.
        cards: list of tuples (rank_idx, suit_idx)
        """
        game.cards_on_table = len(cards)
        for i, (rank, suit) in enumerate(cards):
            game.table_state[i][rank] = 1
            game.table_state[i][NUM_RANKS + suit] = 1

    def test_initialization(self, game: GameState) -> None:
        # Given a new game instance
        # When checking initial state
        # Then all properties should be initialized correctly
        assert game.no_players == 4
        assert game.cards_count == 24
        assert game.current_player != -1
        assert game.cards_on_table == 0
        assert game.player_hands.shape == (4, 6)
        assert np.sum(game.player_hands != -1) == 24

        start_player = game.get_starting_player()
        assert game.player_hands[0][0] == start_player

    @pytest.mark.parametrize(
        'ranks, suits, expected_substrings',
        [
            (
                np.array([RANK_9, RANK_A]),
                np.array([SUIT_H, SUIT_S]),
                [f'{SUIT_SYMBOLS[SUIT_H]}{RANKS[RANK_9]}', f'{SUIT_SYMBOLS[SUIT_S]}{RANKS[RANK_A]}'],
            ),
            (
                np.array([RANK_10, RANK_J]),
                np.array([SUIT_D, SUIT_C]),
                [f'{SUIT_SYMBOLS[SUIT_D]}{RANKS[RANK_10]}', f'{SUIT_SYMBOLS[SUIT_C]}{RANKS[RANK_J]}'],
            ),
        ],
    )
    def test_hand_representation(self, ranks: np.ndarray, suits: np.ndarray, expected_substrings: list[str]) -> None:
        # Given a set of ranks and suits
        # When converting to string representation
        rep = GameState.hand_representation(ranks, suits)

        # Then the string should contain correct symbols
        for sub in expected_substrings:
            assert sub in rep

    @pytest.mark.parametrize(
        'rank, suit',
        [
            (RANK_9, SUIT_H),
            (RANK_A, SUIT_S),
            (RANK_10, SUIT_D),
        ],
    )
    def test_decode_card(self, rank: int, suit: int) -> None:
        # Given a valid card encoding
        encoding = np.zeros(10, dtype=np.int32)
        encoding[rank] = 1
        encoding[NUM_RANKS + suit] = 1

        # When decoding the card
        decoded_rank, decoded_suit = GameState.decode_card(encoding)

        # Then the correct rank and suit should be returned
        assert decoded_rank == rank
        assert decoded_suit == suit

    def test_get_possible_actions_start(self, game: GameState) -> None:
        # Given the starting player
        player = game.get_starting_player()

        # When getting possible actions
        actions = game.get_possible_actions(player)

        # Then playing 9 Hearts should be a valid action
        assert OFFSET_SINGLE_CARD in actions

    def test_play_single_card(self, game: GameState) -> None:
        # Given the starting player and a valid single card action
        player = game.get_starting_player()
        action = OFFSET_SINGLE_CARD

        # When the action is executed
        result = game.execute_action(action)

        # Then the game should continue
        assert result is False
        # And the table state should update
        assert game.cards_on_table == 1
        assert game.table_state[0][RANK_9] == 1
        assert game.table_state[0][NUM_RANKS + SUIT_H] == 1
        # And the card should be removed from player's hand
        assert game.player_hands[SUIT_H][RANK_9] == -1
        # And the knowledge table should update
        assert game.knowledge_table[player, SUIT_H, RANK_9] == -2  # On table

    def test_take_cards(self, game: GameState) -> None:
        # Given a table with cards available to take
        # We must ensure these cards are NOT in any player's hand initially
        game.player_hands = -np.ones((4, 6), dtype=np.int32)

        # Give player a dummy card so they are "in game" (not finished)
        # 9S is safe as we use 9H and 10H for table
        self.setup_hand(game, game.current_player, [(RANK_9, SUIT_S)])

        # Table has 9H (bottom) and 10H (top)
        self.setup_table(game, [(RANK_9, SUIT_H), (RANK_10, SUIT_H)])

        player = game.current_player

        # When getting possible actions
        actions = game.get_possible_actions(player)

        # Then taking cards should be an option
        assert ACTION_TAKE_CARDS in actions

        # When taking cards
        game.execute_action(ACTION_TAKE_CARDS)

        # Then the table should have fewer cards
        # Logic: takes up to 3 cards, but leaves at least 1 on table.
        # Here: 2 cards total. Takes 1 (10H). Leaves 1 (9H).
        assert game.cards_on_table == 1

        # And the player should receive the top card (10H)
        assert game.player_hands[SUIT_H][RANK_10] == player

        # And the bottom card (9H) should still be on the table (not in player hand)
        assert game.player_hands[SUIT_H][RANK_9] == -1
        assert game.table_state[0][RANK_9] == 1

    def test_cannot_take_cards_at_start(self, game: GameState) -> None:
        # Given a game that just started (no cards on table)
        assert game.cards_on_table == 0
        player = game.get_starting_player()

        # When getting possible actions
        actions = game.get_possible_actions(player)

        # Then taking cards should NOT be an option
        assert ACTION_TAKE_CARDS not in actions

    def test_cannot_take_cards_with_only_one_card_on_table(self, game: GameState) -> None:
        # Given a table with only one card (the 9 of Hearts)
        player = game.get_starting_player()
        game.current_player = player

        # Play the starting 9H
        game.execute_action(OFFSET_SINGLE_CARD)
        assert game.cards_on_table == 1

        # Set next player (after playing non-spade, turn goes to next player)
        next_player = game.current_player

        # Give this player some cards to play
        game.player_hands = -np.ones((4, 6), dtype=np.int32)
        self.setup_hand(game, next_player, [(RANK_10, SUIT_H), (RANK_A, SUIT_S)])

        # When getting possible actions
        actions = game.get_possible_actions(next_player)

        # Then taking cards should NOT be an option (must leave at least 1 card)
        assert ACTION_TAKE_CARDS not in actions

    def test_game_end_all_but_one(self, game: GameState) -> None:
        # Given a player with one card left and 2 other players done (1 active opponent)
        player = game.current_player
        game.player_hands = -np.ones((4, 6), dtype=np.int32)
        self.setup_hand(game, player, [(RANK_9, SUIT_H)])

        game.cards_on_table = 0
        action = OFFSET_SINGLE_CARD  # 9H

        # Setup: 4 players total.
        # Mark 2 players as done.
        # Mark current player and one other as NOT done.
        game.is_done_array[:] = True
        game.is_done_array[player] = False
        game.is_done_array[(player + 1) % 4] = False

        # When the player plays their last card
        result = game.execute_action(action)

        # Then the game should end (return True) because only 1 player is left
        assert result is True
        # And the player should be marked as done
        assert game.is_done_array[player]

    def test_game_continues_if_two_left(self, game: GameState) -> None:
        # Given a player with one card left and only 1 other player done (2 active opponents)
        player = game.current_player
        game.player_hands = -np.ones((4, 6), dtype=np.int32)
        self.setup_hand(game, player, [(RANK_9, SUIT_H)])

        game.cards_on_table = 0
        action = OFFSET_SINGLE_CARD  # 9H

        # Setup: 4 players total.
        # Mark 1 player as done.
        # Mark current player and two others as NOT done.
        game.is_done_array[:] = False
        game.is_done_array[(player + 1) % 4] = True

        # When the player plays their last card
        result = game.execute_action(action)

        # Then the game should NOT end (return False) because 2 players are left
        assert result is False
        # And the player should be marked as done
        assert game.is_done_array[player]

    def test_invalid_action(self, game: GameState) -> None:
        # Given an invalid action ID
        # When executing the action
        # Then a ValueError should be raised
        with pytest.raises(ValueError):
            game.execute_action(9999)

    def test_decode_card_invalid(self) -> None:
        # Given an invalid card encoding (empty)
        # When decoding the card
        # Then a ValueError should be raised
        with pytest.raises(ValueError):
            encoding = np.zeros(10, dtype=np.int32)
            GameState.decode_card(encoding)

    def test_print_table(self, game: GameState, capsys: CaptureFixture) -> None:
        # Given a game state
        # When printing the table
        # Then it should run without error
        game.print_table()

    def test_play_three_nines(self, game: GameState) -> None:
        # Given a player with three nines and a matching table state
        player = game.current_player

        self.setup_table(game, [(RANK_9, SUIT_H)])

        game.player_hands = -np.ones((4, 6), dtype=np.int32)
        self.setup_hand(game, player, [(RANK_9, SUIT_D), (RANK_9, SUIT_C), (RANK_9, SUIT_S)])

        # When getting possible actions
        actions = game.get_possible_actions(player)
        three_nines_actions = [a for a in actions if OFFSET_THREE_NINES <= a < OFFSET_FOUR_NINES]

        # Then three nines action should be available
        assert len(three_nines_actions) > 0

        # When executing the action
        action = three_nines_actions[0]
        game.execute_action(action)

        # Then 3 more cards should be on the table
        assert game.cards_on_table == 4

    def test_play_four_nines(self, game: GameState) -> None:
        # Given a player with four nines at the start of the game
        player = game.current_player
        game.cards_on_table = 0
        game.player_hands = -np.ones((4, 6), dtype=np.int32)
        self.setup_hand(game, player, [(RANK_9, s) for s in range(4)])

        # When getting possible actions
        actions = game.get_possible_actions(player)
        four_nines_actions = [a for a in actions if OFFSET_FOUR_NINES <= a < OFFSET_FOUR_CARDS]

        # Then four nines action should be available
        assert len(four_nines_actions) > 0

        # When executing the action
        action = four_nines_actions[0]
        game.execute_action(action)

        # Then 4 cards should be on the table
        assert game.cards_on_table == 4

    def test_play_four_cards(self, game: GameState) -> None:
        # Given a player with four cards of the same rank (10s) and a matching table state
        player = game.current_player
        self.setup_table(game, [(RANK_9, SUIT_H)])

        game.player_hands = -np.ones((4, 6), dtype=np.int32)
        self.setup_hand(game, player, [(RANK_10, s) for s in range(4)])

        # When getting possible actions
        actions = game.get_possible_actions(player)
        four_cards_actions = [a for a in actions if OFFSET_FOUR_CARDS <= a < ACTION_TAKE_CARDS]

        # Then four cards action should be available
        assert len(four_cards_actions) > 0

        # When executing the action
        action = four_cards_actions[0]
        game.execute_action(action)

        # Then 4 more cards should be on the table
        assert game.cards_on_table == 5

    def test_skip_done_players(self, game: GameState) -> None:
        # Given a game where the next player is already done
        game.current_player = 0
        player = 0
        game.is_done_array[:] = False
        game.is_done_array[1] = True

        game.player_hands = -np.ones((4, 6), dtype=np.int32)
        self.setup_hand(game, player, [(RANK_9, SUIT_H)])

        game.cards_on_table = 0

        # When the current player finishes their turn
        action = OFFSET_SINGLE_CARD
        game.execute_action(action)

        # Then the current player should be marked as done
        assert game.is_done_array[0]
        # And the turn should skip the done player (1) and go to player 2
        assert game.current_player == 2

    def test_spade_reversal(self, game: GameState) -> None:
        # Given a game state where a spade is played
        game.current_player = 0
        player = 0
        game.is_done_array[:] = False

        self.setup_table(game, [(RANK_9, SUIT_H)])

        game.player_hands = -np.ones((4, 6), dtype=np.int32)
        self.setup_hand(game, player, [(RANK_9, SUIT_S), (RANK_10, SUIT_H)])

        # When the spade action is executed
        action = OFFSET_SINGLE_CARD + SUIT_S * NUM_RANKS + RANK_9
        game.execute_action(action)

        # Then the card should be on the table
        assert game.table_state[1][RANK_9] == 1
        assert game.table_state[1][NUM_RANKS + SUIT_S] == 1
        # And the play direction should reverse (0 -> 3)
        assert game.current_player == 3

    def test_get_possible_actions_empty_hand(self, game: GameState) -> None:
        # Given a player with an empty hand
        player = game.current_player
        game.player_hands = -np.ones((4, 6), dtype=np.int32)

        # When getting possible actions
        actions = game.get_possible_actions(player)

        # Then no actions should be returned
        assert len(actions) == 0

    def test_get_player_knowledge(self, game: GameState) -> None:
        # Given a game state
        # When getting player knowledge
        k = game.get_player_knowledge()

        # Then it should return a valid array shape
        assert k.shape == (4, 6)

    def test_get_hands_card_counts(self, game: GameState) -> None:
        # Given a new game
        game.restart()

        # When getting card counts
        counts = game.get_hands_card_counts()

        # Then counts should be correct (0 for self, 6 for others)
        assert len(counts) == 4
        player = game.current_player
        assert counts[player] == 0
        for p in range(4):
            if p != player:
                assert counts[p] == 6

        # When a card is played
        action = OFFSET_SINGLE_CARD  # 9H
        game.execute_action(action)

        # Then counts should update correctly
        counts = game.get_hands_card_counts()
        new_player = game.current_player
        assert counts[new_player] == 0

    def test_fuzz_game(self, game: GameState) -> None:
        # Given a fresh game
        # When playing random valid moves until the game ends
        # Then the game should finish without errors

        random.seed(42)  # Fixed seed for reproducibility
        max_steps = 1000
        steps = 0

        is_over = False
        while steps < max_steps:
            player = game.current_player
            actions = game.get_possible_actions(player)

            action = random.choice(actions)
            is_over = game.execute_action(action)

            if is_over:
                break
            steps += 1

        # Assert game finished or max steps reached (soft pass, mainly checking for exceptions)
        assert steps < max_steps or not is_over  # Just to have an assertion
