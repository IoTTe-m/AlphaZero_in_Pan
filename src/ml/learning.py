from collections import deque
from pathlib import Path

import numpy as np
import optax
import orbax.checkpoint as ocp
from jax import numpy as jnp
from tqdm import tqdm

import wandb
from src.game_logic import OFFSET_SINGLE_CARD, GameState
from src.mcts.mcts import MCTS
from src.mcts.state_processors import PolicyStateProcessor, ValueStateProcessor
from src.ml.neural_networks import AlphaZeroNNs
from src.ml.policy_net import compute_policy_loss_and_grad_vect
from src.ml.value_net import compute_value_loss_and_grad_vect

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
        c_puct_value: float = 1.0,
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

    def _create_random_game_state(self) -> GameState:
        """Create a game state with random starting conditions for curriculum learning."""
        state = GameState(no_players=self.no_players)

        # 50% chance to start with 9♥ already played (skips the forced first move)
        if np.random.random() < 0.5:
            state.execute_action(OFFSET_SINGLE_CARD)  # Play 9♥

        # 30% chance to remove one player's cards (simulate 3-player game)
        # 10% chance to remove two players' cards (simulate 2-player game)
        if self.no_players == 4:
            roll = np.random.random()
            if roll < 0.25:
                # Remove 2 players (keep current player and one other)
                players_to_remove = [p for p in range(self.no_players) if p != state.current_player]
                np.random.shuffle(players_to_remove)
                for p in players_to_remove[:2]:
                    self._remove_player_cards(state, p)
            elif roll < 0.5:
                # Remove 1 player
                players_to_remove = [p for p in range(self.no_players) if p != state.current_player]
                p = np.random.choice(players_to_remove)
                self._remove_player_cards(state, p)

        return state

    def _remove_player_cards(self, state: GameState, player: int):
        """Remove all cards from a player and mark them as done."""
        # Find all cards owned by this player and redistribute to others
        active_players = [p for p in range(state.no_players) if p != player and not state.is_done_array[p]]
        if not active_players:
            return

        for suit in range(state.player_hands.shape[0]):
            for rank in range(state.player_hands.shape[1]):
                if state.player_hands[suit][rank] == player:
                    # Give card to a random active player
                    new_owner = np.random.choice(active_players)
                    state.player_hands[suit][rank] = new_owner
                    # Update knowledge table
                    state.knowledge_table[:, suit, rank] = new_owner

        state.is_done_array[player] = True

    def play_game(self):
        state = self._create_random_game_state()

        game_trajectory: list[tuple[ValueStateRepr, PolicyStateRepr, np.ndarray, int]] = []

        pbar = tqdm(range(self.max_game_length), desc='Max game length', leave=False)
        for _ in pbar:
            policy_probs, _ = self.mcts.run(state)

            value_state = ValueStateProcessor.encode(state)
            prepared_knowledge_policy, table_state_policy, encoded_action_policy, _ = PolicyStateProcessor.encode(state)
            game_trajectory.append(
                (value_state, (prepared_knowledge_policy, table_state_policy, encoded_action_policy), policy_probs, state.current_player)
            )

            action = np.random.choice(len(policy_probs), p=policy_probs)
            is_end = state.execute_action(action)
            if is_end:
                break

        # Compute actual game outcome: winners get +1/(n-1), losers get -1
        game_outcome = np.ones(self.no_players) * (1.0 / max(1, self.no_players - 1))
        game_outcome[~state.is_done_array] = -1.0

        # Add all states from this game to the buffer
        for value_state, policy_state, policy_probs, current_player in game_trajectory:
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
            value_grads, self.mcts.networks.value_network.opt_state, self.mcts.networks.value_network.params
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
            policy_grads, self.mcts.networks.policy_network.opt_state, self.mcts.networks.policy_network.params
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
