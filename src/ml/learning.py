from collections import deque
import numpy as np
from tqdm import tqdm
from decimal import Decimal
from jax import numpy as jnp
from typing import TypeAlias
import optax
from src.game_logic import GameState
from src.mcts.mcts import MCTS
from src.mcts.state_processors import PolicyStateProcessor, ValueStateProcessor
from src.ml.neural_networks import AlphaZeroNNs, PolicyNetwork, ValueNetwork, call_policy_network, call_value_network, compute_value_loss_and_grad_vect, compute_policy_loss_and_grad_vect

ValueStateRepr: TypeAlias = tuple[jnp.ndarray, jnp.ndarray]
PolicyStateRepr: TypeAlias = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
BufferItem: TypeAlias = tuple[ValueStateRepr, PolicyStateRepr, np.ndarray, np.ndarray]

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

class LearningProcess:
    def __init__(self, nns: AlphaZeroNNs, no_players: int, batch_size: int = 32,
                 games_per_training: int = 4, num_simulations: int = 2048, num_worlds: int = 16,
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

    def self_play(self, epochs: int, batch_count: int):
        epoch_pbar = tqdm(range(epochs), total=epochs, desc="value loss: inf, policy loss: inf")

        for epoch in epoch_pbar:
            games_pbar = tqdm(range(self.games_per_training), total=self.games_per_training, desc="gameing 😎🎮", leave=False)
            for _ in games_pbar:
                self.play_game()
            avg_value_loss, avg_policy_loss = self.train_networks(batch_count)
            epoch_pbar.set_description(f"value loss: {avg_value_loss:.2e}, policy loss: {avg_policy_loss:.2e}")

    def play_game(self):
        state = GameState(no_players=self.no_players)
        pbar = tqdm(range(self.max_game_length), desc="Max game length:", leave=False)
        for _ in pbar:
            policy_probs, values = self.mcts.run(state)

            value_state = ValueStateProcessor.encode(state) # estimate how the game will end from this state 
            *policy_state, _ = PolicyStateProcessor.encode(state) # estimate best action to take from this state
            self.buffer.append((value_state, policy_state, policy_probs, values))
            action = np.random.choice(len(policy_probs), p=policy_probs)
            is_end = state.execute_action(action)
            if is_end:
                break

    def train_networks(self, batch_count: int) -> tuple[float, float]:
        """
        Samples a batch from the buffer and trains both neural networks.
        """
        avg_value_loss = 0.0
        avg_policy_loss = 0.0
        for batch_num in range(batch_count):
            avg_value_loss += self._train_value_step() / batch_count
            avg_policy_loss += self._train_policy_step() / batch_count
        return avg_value_loss, avg_policy_loss

    def _train_value_step(self) -> float:
        prepared_player_hands, table_states, target_values = self._sample_value_batch()
        value_loss, value_grads = compute_value_loss_and_grad_vect(
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
        prepared_knowledge, table_states, encoded_actions, target_policies = self._sample_policy_batch()
        policy_loss, policy_grads = compute_policy_loss_and_grad_vect(
            self.mcts.networks.policy_network,
            self.mcts.networks.policy_network_params,
            prepared_knowledge,
            table_states,
            encoded_actions,
            target_policies
        )
        updates, self.mcts.networks.policy_network_opt_state = self.mcts.networks.policy_network_optimizer.update(policy_grads, self.mcts.networks.policy_network_opt_state)
        self.mcts.networks.policy_network_params = optax.apply_updates(self.mcts.networks.policy_network_params, updates)
        return policy_loss.item()

    def _sample_value_batch(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        no_samples = min(self.batch_size, len(self.buffer))
        samples_indices = np.random.choice(len(self.buffer), size=no_samples, replace=False)
        sampled = [self.buffer[i] for i in samples_indices]
        data_to_unpack, _, _, target_values = tuple(map(list, zip(*sampled)))
        prepared_player_hands, table_states = tuple(map(list, zip(*data_to_unpack)))

        return jnp.array(prepared_player_hands), jnp.array(table_states), jnp.array(target_values)

    def _sample_policy_batch(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        samples_indices = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)
        sampled = [self.buffer[i] for i in samples_indices]

        _, tuples, target_policies, _ = tuple(map(list, zip(*sampled)))
        table_states, prepared_knowledge, encoded_actions =  tuple(map(list, zip(*tuples)))

        target_policies = [target_policy for _, _, target_policy, _ in sampled]
        return jnp.array(prepared_knowledge), jnp.array(table_states), jnp.array(encoded_actions), jnp.array(target_policies)
