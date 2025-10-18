from collections import deque
import numpy as np
from jax import numpy as jnp
from typing import TypeAlias
import optax
from src.game_logic import GameState
from src.mcts.mcts import MCTS
from src.mcts.state_processors import PolicyStateProcessor, ValueStateProcessor
from src.ml.neural_networks import AlphaZeroNNs, PolicyNetwork, ValueNetwork, call_policy_network, call_value_network, compute_value_loss_and_grad, compute_policy_loss_and_grad

ValueStateRepr: TypeAlias = tuple[jnp.ndarray, jnp.ndarray]
PolicyStateRepr: TypeAlias = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, list[int]]
BufferItem: TypeAlias = tuple[ValueStateRepr, PolicyStateRepr, np.ndarray, np.ndarray]

class LearningProcess:
    def __init__(self, nns: AlphaZeroNNs, no_players: int, batch_size: int = 32,
                 games_per_training: int = 4, num_simulations: int = 256, num_worlds: int = 16,
                 max_buffer_size: int = 1024, c_puct_value: int = 1,
                 policy_temp: float = 1.0, max_game_length: int = 5000):
        self.no_players = no_players
        self.batch_size = batch_size
        self.games_per_training = games_per_training
        self.max_game_length = max_game_length
        self.buffer: deque[BufferItem] = deque(maxlen=max_buffer_size)  # buffer of (state_value, state_policy, policy, value) tuples
        self.mcts = MCTS(
            networks=nns,
            num_worlds=num_worlds,
            num_simulations=num_simulations,
            c_puct_value=c_puct_value,
            policy_temp=policy_temp
        )

    def play_game(self):
        state = GameState(no_players=self.no_players)
        for _ in range(self.max_game_length):
            policy_probs, values = self.mcts.run(state)

            value_state = ValueStateProcessor.encode(state)
            policy_state = PolicyStateProcessor.encode(state)
            self.buffer.append((value_state, policy_state, policy_probs, values))
            action = np.random.choice(len(policy_probs), p=policy_probs)
            is_end = state.execute_action(action)
            if is_end:
                break

    def train_networks(self, batch_count: int = 16):
        """
        Samples a batch from the buffer and trains both neural networks.
        """

        for _ in range(batch_count):
            self._train_value_step()
            self._train_policy_step()

    def self_play(self, epochs: int):
        for _ in range(epochs):
            for _ in range(self.games_per_training):
                self.play_game()
            self.train_networks()

    def _train_value_step(self) -> float:
        prepared_player_hands, table_states, target_values = self._sample_value_batch()
        value_loss, value_grads = compute_value_loss_and_grad(
            self.mcts.networks.value_network,
            self.mcts.networks.value_network_params,
            prepared_player_hands,
            table_states,
            target_values
        )
        updates, self.mcts.networks.value_network_opt_state = self.mcts.networks.value_network_optimizer.update(value_grads, self.mcts.networks.value_network_opt_state)
        self.mcts.networks.value_network_params = optax.apply_updates(self.mcts.networks.value_network_params, updates)
        return value_loss.item()

    def _train_policy_step(self) -> float:
        prepared_knowledge, table_states, encoded_actions, possible_actions, target_policies = self._sample_policy_batch()
        policy_loss, policy_grads = compute_policy_loss_and_grad(
            self.mcts.networks.policy_network,
            self.mcts.networks.policy_network_params,
            prepared_knowledge,
            table_states,
            encoded_actions,
            possible_actions,
            target_policies
        )
        updates, self.mcts.networks.policy_network_opt_state = self.mcts.networks.policy_network_optimizer.update(policy_grads, self.mcts.networks.policy_network_opt_state)
        self.mcts.networks.policy_network_params = optax.apply_updates(self.mcts.networks.policy_network_params, updates)
        return policy_loss.item()

    def _sample_value_batch(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        samples_indices = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)
        sampled = [self.buffer[i] for i in samples_indices]

        data_to_unpack, _, _, target_values = tuple(map(list, zip(*sampled)))
        prepared_player_hands, table_states = tuple(map(list, zip(*data_to_unpack)))

        return (jnp.array(prepared_player_hands), jnp.array(table_states), jnp.array(target_values))

    def _sample_policy_batch(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        samples_indices = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)
        sampled = [self.buffer[i] for i in samples_indices]

        _, tuples, target_policies, _ = tuple(map(list, zip(*sampled)))
        table_states, prepared_knowledge, encoded_actions, possible_actions =  tuple(map(list, zip(*tuples)))

        target_policies = [target_policy for _, _, target_policy, _ in sampled]
        return (jnp.array(prepared_knowledge), jnp.array(table_states), jnp.array(encoded_actions),
                jnp.array(possible_actions), jnp.array(target_policies))
