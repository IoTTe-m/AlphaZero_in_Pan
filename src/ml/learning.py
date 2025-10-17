from collections import deque
from copy import deepcopy

import numpy as np
from jax import numpy as jnp
from src.game_logic import GameState
from src.mcts.mcts import MCTS
from src.mcts.state_processors import PolicyStateProcessor, ValueStateProcessor
from src.ml.neural_networks import AlphaZeroNNs, PolicyNetwork, ValueNetwork


class Learning:
    def __init__(self, nns: AlphaZeroNNs, no_players: int, batch_size: int = 32,
                 num_simulations: int = 256, num_worlds: int = 16,
                 max_buffer_size: int = 10000, c_puct_value: int = 1,
                 policy_temp: float = 1.0):
        self.no_players = no_players
        self.batch_size = batch_size
        self.buffer: deque[tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], np.ndarray, np.ndarray]] = deque(maxlen=max_buffer_size)  # buffer of (state_value, state_policy, policy, value) tuples
        self.mcts = MCTS(
            networks=nns,
            num_worlds=num_worlds,
            num_simulations=num_simulations
        )


    def play_game(self):
        state = GameState(no_players=self.no_players)
        while True:
            policy_probs, values = self.mcts.run(state)
            current_player = state.current_player

            value_state = ValueStateProcessor.encode(state)
            policy_state = PolicyStateProcessor.encode(state)[:-1]  # exclude actions list
            self.buffer.append((value_state, policy_state, policy_probs, values))
            action = np.random.choice(len(policy_probs), p=policy_probs)
            is_win = state.execute_action(current_player, action)
            if is_win:
                break
        
    def sample_value_batch(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        sampled = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)
        prepared_player_hands = [player_hands for (player_hands, _, _), _, _, _ in sampled]
        table_states = [table_state for (_, table_state, _), _, _ , _ in sampled]
        target_values = [target_value for _, _, _, target_value in sampled]
        return (jnp.array(prepared_player_hands), jnp.array(table_states), jnp.array(target_values))
    
    def sample_policy_batch(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        sampled = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)
        prepared_knowledges = [prepared_knowledge for _, (_, prepared_knowledge, _, _), _, _ in sampled]
        table_states = [table_state for _, (_, table_state, _, _), _, _ in sampled]
        encoded_actions = [encoded_actions for _, (_, _, encoded_actions, _), _, _ in sampled]
        possible_actions = [possible_actions for _, (_, _, _, possible_actions), _, _ in sampled]
        target_policies = [target_policy for _, _, target_policy, _ in sampled]
        return (jnp.array(prepared_knowledges), jnp.array(table_states), jnp.array(encoded_actions),
                jnp.array(possible_actions), jnp.array(target_policies))
