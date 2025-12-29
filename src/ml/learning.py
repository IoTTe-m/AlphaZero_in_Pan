from collections import deque
from pathlib import Path

import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from jax import numpy as jnp
from tqdm import tqdm

from src.game_logic import GameState
from src.mcts.mcts import MCTS
from src.mcts.state_processors import PolicyStateProcessor, ValueStateProcessor
from src.ml.neural_networks import (
    AlphaZeroNNs,
    compute_policy_loss_and_grad_vect,
    compute_value_loss_and_grad_vect,
)

type ValueStateRepr = tuple[jnp.ndarray, jnp.ndarray]
type PolicyStateRepr = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
type BufferItem = tuple[ValueStateRepr, PolicyStateRepr, np.ndarray, np.ndarray]


class LearningProcess:
    def __init__(
        self,
        run: wandb.Run,
        save_dir: str,
        nns: AlphaZeroNNs,
        no_players: int,
        batch_size: int = 32,
        games_per_training: int = 4,
        num_simulations: int = 2048,
        num_worlds: int = 16,
        max_buffer_size: int = 1024,
        c_puct_value: int = 1,
        policy_temp: float = 1.0,
        initial_max_game_length: int = 50,
        capped_max_game_length: int = 500,
        game_length_increment: int = 20,
    ):
        self.run = run
        self.save_dir = save_dir
        self.no_players = no_players
        self.batch_size = batch_size
        self.games_per_training = games_per_training
        self.max_game_length = initial_max_game_length
        self.capped_max_game_length = capped_max_game_length
        self.game_length_increment = game_length_increment

        save_path = Path(f'{self.save_dir}/run_{self.run.name}').absolute()
        checkpointer = ocp.StandardCheckpointer()
        options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
        self.manager = ocp.CheckpointManager(save_path, checkpointer, options)

        self.buffer: deque[BufferItem] = deque(maxlen=max_buffer_size)  # buffer of (state_value, state_policy, policy, value) tuples
        self.mcts = MCTS(networks=nns, num_worlds=num_worlds, num_simulations=num_simulations, c_puct_value=c_puct_value, policy_temp=policy_temp)

    def _increase_max_game_length(self):
        self.max_game_length = min(self.max_game_length + self.game_length_increment, self.capped_max_game_length)

    def self_play(self, epochs: int, batch_count: int):
        epoch_pbar = tqdm(range(epochs), total=epochs, desc='value loss: inf, policy loss: inf')

        try:
            for epoch in epoch_pbar:
                games_pbar = tqdm(range(self.games_per_training), total=self.games_per_training, desc='gameing 😎🎮', leave=False)
                for _ in games_pbar:
                    self.play_game()
                avg_value_loss, avg_policy_loss = self.train_networks(batch_count)
                epoch_pbar.set_description(f'value loss: {avg_value_loss:.2e}, policy loss: {avg_policy_loss:.2e}')

                self.manager.save(step=epoch, items=self.mcts.networks.get_state(epoch))

                self.run.log(data={'value_loss': avg_value_loss, 'policy_loss': avg_policy_loss})
                self._increase_max_game_length()
        except KeyboardInterrupt:
            pass
        finally:
            self.manager.close()

    def play_game(self):
        state = GameState(no_players=self.no_players)
        # Store (value_state, policy_state, policy_probs, current_player) for each step
        game_trajectory: list[tuple[ValueStateRepr, PolicyStateRepr, np.ndarray, int]] = []

        pbar = tqdm(range(self.max_game_length), desc='Max game length', leave=False)
        for _ in pbar:
            policy_probs, _ = self.mcts.run(state)

            value_state = ValueStateProcessor.encode(state)  # estimate how the game will end from this state
            # estimate best action to take from this state
            prepared_knowledge_policy, table_state_policy, encoded_action_policy, _ = PolicyStateProcessor.encode(state)
            # Store current_player so we can shift the outcome to match the encoded perspective
            game_trajectory.append(
                (value_state, (prepared_knowledge_policy, table_state_policy, encoded_action_policy), policy_probs, state.current_player)
            )

            action = np.random.choice(len(policy_probs), p=policy_probs)
            is_end = state.execute_action(action)
            if is_end:
                break

        # Compute actual game outcome in absolute player order: winners get +1/(n-1), loser gets -1
        # In Pan, the last player with cards loses
        game_outcome = np.ones(self.no_players) * (1.0 / (self.no_players - 1))  # Winners share +1
        loser = np.where(~state.is_done_array)[0]
        if len(loser) > 0:
            game_outcome[loser[0]] = -1.0

        # Add all states from this game to the buffer with perspective-shifted game outcome
        for value_state, policy_state, policy_probs, current_player in game_trajectory:
            # Shift outcome so index 0 = current_player (matching the encoded state perspective)
            shifted_outcome = np.roll(game_outcome, -current_player)
            self.buffer.append((value_state, policy_state, policy_probs, shifted_outcome))

    def train_networks(self, batch_count: int) -> tuple[float, float]:
        """
        Samples a batch from the buffer and trains both neural networks.
        """
        avg_value_loss = 0.0
        avg_policy_loss = 0.0
        for _ in range(batch_count):
            avg_value_loss += self._train_value_step() / batch_count
            avg_policy_loss += self._train_policy_step() / batch_count
        return avg_value_loss, avg_policy_loss

    def _train_value_step(self) -> float:
        prepared_player_hands, table_states, target_values = self._sample_value_batch()
        value_loss, value_grads = compute_value_loss_and_grad_vect(
            self.mcts.networks.value_network.network, self.mcts.networks.value_network.params, prepared_player_hands, table_states, target_values
        )
        updates, self.mcts.networks.value_network.opt_state = self.mcts.networks.value_network.optimizer.update(
            value_grads, self.mcts.networks.value_network.opt_state
        )
        self.mcts.networks.value_network.params = optax.apply_updates(self.mcts.networks.value_network.params, updates)
        return value_loss.item()

    def _train_policy_step(self) -> float:
        prepared_knowledge, table_states, encoded_actions, target_policies = self._sample_policy_batch()
        policy_loss, policy_grads = compute_policy_loss_and_grad_vect(
            self.mcts.networks.policy_network.network,
            self.mcts.networks.policy_network.params,
            prepared_knowledge,
            table_states,
            encoded_actions,
            target_policies,
        )
        updates, self.mcts.networks.policy_network.opt_state = self.mcts.networks.policy_network.optimizer.update(
            policy_grads, self.mcts.networks.policy_network.opt_state
        )
        self.mcts.networks.policy_network.params = optax.apply_updates(self.mcts.networks.policy_network.params, updates)
        return policy_loss.item()

    def _sample_value_batch(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        no_samples = min(self.batch_size, len(self.buffer))
        samples_indices = np.random.choice(len(self.buffer), size=no_samples, replace=False)
        sampled = [self.buffer[i] for i in samples_indices]
        data_to_unpack, _, _, target_values = tuple(map(list, zip(*sampled, strict=True)))
        prepared_player_hands, table_states = tuple(map(list, zip(*data_to_unpack, strict=True)))

        return jnp.array(prepared_player_hands), jnp.array(table_states), jnp.array(target_values)

    def _sample_policy_batch(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        no_samples = min(self.batch_size, len(self.buffer))
        samples_indices = np.random.choice(len(self.buffer), size=no_samples, replace=False)
        sampled = [self.buffer[i] for i in samples_indices]

        _, tuples, target_policies, _ = tuple(map(list, zip(*sampled, strict=True)))
        prepared_knowledge, table_states, encoded_actions = tuple(map(list, zip(*tuples, strict=True)))

        return jnp.array(prepared_knowledge), jnp.array(table_states), jnp.array(encoded_actions), jnp.array(target_policies)
