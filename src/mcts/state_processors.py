from copy import deepcopy

import numpy as np
from jax import numpy as jnp

from src.game_logic import ACTION_COUNT, RANKS, SUITS, GameState


class StateProcessor:
    @staticmethod
    def change_perspective(knowledge_array: np.ndarray, player_number: int, no_players: int) -> np.ndarray:
        return np.where(knowledge_array <= -1, knowledge_array, (knowledge_array - player_number) % no_players)

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
        full_knowledge = -np.ones((state.no_players, len(SUITS), len(RANKS)), dtype=np.int32)
        GameState.fill_knowledge_table(full_knowledge, filled_hands, state.no_players)

        # Create new state
        new_state = deepcopy(state)
        new_state.player_hands = filled_hands
        new_state.knowledge_table = full_knowledge

        return new_state

    @staticmethod
    def encode_actions(actions_list: list[int]) -> np.ndarray:
        # actions array ([0, 3, 23]) -> encoded array ([1,0,0,1,0...]
        encoded_actions = np.zeros((ACTION_COUNT,), dtype=np.bool_)
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
        return np.array(jnp.concatenate([state_values[split_index:], state_values[:split_index]]))


class PolicyStateProcessor:
    @staticmethod
    def encode(state: GameState) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, list[int]]:
        current_knowledge = state.get_player_knowledge()
        prepared_knowledge = StateProcessor.change_perspective(current_knowledge, state.current_player, state.no_players)
        prepared_knowledge = StateProcessor.one_hot_encode_hands(prepared_knowledge, state.no_players)
        table_state = state.table_state

        possible_actions = state.get_possible_actions(state.current_player)
        encoded_actions = StateProcessor.encode_actions(possible_actions)

        return jnp.array(prepared_knowledge), jnp.array(table_state), jnp.array(encoded_actions), possible_actions
