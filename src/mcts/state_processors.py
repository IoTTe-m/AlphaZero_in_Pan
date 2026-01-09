"""State encoding and transformation utilities for MCTS and neural networks.

Provides processors for converting game states into neural network inputs
and handling perspective shifts between players.
"""

from copy import deepcopy

import numpy as np
from jax import numpy as jnp

from src.game_logic import ACTION_COUNT, CARD_UNKNOWN, RANKS, SUITS, GameState


class StateProcessor:
    """Utility class for encoding and transforming game states."""

    @staticmethod
    def change_perspective(knowledge_array: np.ndarray, player_number: int, no_players: int) -> np.ndarray:
        """
        Shift player indices so that the given player becomes player 0.

        Args:
            knowledge_array: Array with player ownership values.
            player_number: Player to shift to index 0.
            no_players: Total number of players.

        Returns:
            Array with shifted player indices.
        """
        return np.where(knowledge_array <= -1, knowledge_array, (knowledge_array - player_number) % no_players)

    @staticmethod
    def get_mcts_state(state: GameState) -> GameState:
        """
        Convert current state into a starting state for MCTS by sampling unknown cards.

        Args:
            state: Current game state with partial information.

        Returns:
            New state with unknown cards randomly assigned to players.
        """
        current_knowledge = state.get_player_knowledge()
        mask_unknown = np.array(current_knowledge == CARD_UNKNOWN)
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
        full_knowledge = CARD_UNKNOWN * np.ones((state.no_players, len(SUITS), len(RANKS)), dtype=np.int32)
        GameState.fill_knowledge_table(full_knowledge, filled_hands, state.no_players)

        # Create new state
        new_state = deepcopy(state)
        new_state.player_hands = filled_hands
        new_state.knowledge_table = full_knowledge

        return new_state

    @staticmethod
    def encode_actions(actions_list: list[int]) -> np.ndarray:
        """
        Convert a list of action indices to a one-hot encoded boolean array.

        Args:
            actions_list: List of valid action indices.

        Returns:
            Boolean array with True at valid action positions.
        """
        encoded_actions = np.zeros((ACTION_COUNT,), dtype=np.bool_)
        encoded_actions[actions_list] = 1
        return encoded_actions

    @staticmethod
    def one_hot_encode_hands(player_hands: np.ndarray, no_players: int) -> np.ndarray:
        """
        One-hot encode player hands with an extra channel for unknown cards.

        Args:
            player_hands: Array with player indices as values.
            no_players: Total number of players.

        Returns:
            One-hot encoded array with shape (*player_hands.shape, no_players + 1).
        """
        encoded_hands = np.zeros(player_hands.shape + (no_players + 1,))
        encoded_hands[..., -1] = player_hands == CARD_UNKNOWN
        for player in range(no_players):
            encoded_hands[..., player] = player_hands == player
        return encoded_hands


class ValueStateProcessor:
    """State processor for the value network."""

    @staticmethod
    def encode(state: GameState) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode game state for value network input, the current player is encoded at index 0.

        Args:
            state: Current game state.

        Returns:
            Tuple of (one-hot encoded hands, table state).
        """
        player_hands = state.player_hands
        prepared_player_hands = StateProcessor.change_perspective(player_hands, state.current_player, state.no_players)
        prepared_player_hands = StateProcessor.one_hot_encode_hands(prepared_player_hands, state.no_players)
        table_state = state.table_state
        return jnp.array(prepared_player_hands), jnp.array(table_state)

    @staticmethod
    def decode(state_values: jnp.ndarray, current_player: int) -> np.ndarray:
        """
        Decode shifted value predictions back to original player ordering.

        Args:
            state_values: Value predictions with current player at index 0.
            current_player: Original player index.

        Returns:
            Value array in original player order.
        """
        split_index = len(state_values) - current_player
        return np.array(jnp.concatenate([state_values[split_index:], state_values[:split_index]]))


class PolicyStateProcessor:
    """State processor for the policy network."""

    @staticmethod
    def encode(state: GameState) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, list[int]]:
        """
        Encode game state for policy network input.

        Args:
            state: Current game state.

        Returns:
            Tuple of (encoded knowledge, table state, action mask, action list).
        """
        current_knowledge = state.get_player_knowledge()
        prepared_knowledge = StateProcessor.change_perspective(current_knowledge, state.current_player, state.no_players)
        prepared_knowledge = StateProcessor.one_hot_encode_hands(prepared_knowledge, state.no_players)
        table_state = state.table_state

        possible_actions = state.get_possible_actions(state.current_player)
        encoded_actions = StateProcessor.encode_actions(possible_actions)

        return jnp.array(prepared_knowledge), jnp.array(table_state), jnp.array(encoded_actions), possible_actions
