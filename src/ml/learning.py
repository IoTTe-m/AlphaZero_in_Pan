"""AlphaZero self-play training loop.

Implements the training process with self-play game generation,
replay buffer management, and curriculum learning.
"""

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

# Curriculum learning probabilities
PROB_SKIP_FIRST_MOVE = 0.5
PROB_REMOVE_TWO_PLAYERS = 0.25
PROB_REMOVE_ONE_PLAYER = 0.5  # cumulative threshold

# Game outcome values
WINNER_VALUE = 1.0
LOSER_VALUE = -1.0


class LearningProcess:
    """AlphaZero self-play training process.

    Manages the training loop including self-play game generation,
    replay buffer, network training, and checkpointing.
    """

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
        """
        Initialize the AlphaZero self-play training process.

        Args:
            run: Weights & Biases run for logging.
            save_dir: Directory to save checkpoints.
            nns: Neural networks (policy and value) to train.
            no_players: Number of players in the game.
            batch_size: Number of samples per training batch.
            games_per_training: Games to play before each training step.
            num_simulations: MCTS simulations per move.
            num_worlds: Parallel MCTS worlds for variance reduction.
            max_buffer_size: Maximum replay buffer capacity.
            c_puct_value: PUCT exploration constant.
            policy_temp: Temperature for action selection.
            initial_max_game_length: Starting max moves per game (curriculum).
            capped_max_game_length: Maximum allowed game length.
            game_length_increment: How much to increase max length per epoch.
        """
        self._run = run
        self._save_dir = save_dir
        self._no_players = no_players
        self._batch_size = batch_size
        self._games_per_training = games_per_training
        self._max_game_length = initial_max_game_length
        self._capped_max_game_length = capped_max_game_length
        self._game_length_increment = game_length_increment

        save_path = Path(f'{self._save_dir}/run_{self._run.name}').absolute()
        checkpointer = ocp.StandardCheckpointer()
        options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
        self._manager = ocp.CheckpointManager(save_path, checkpointer, options)

        self._buffer: deque[BufferItem] = deque(maxlen=max_buffer_size)
        self._mcts = MCTS(networks=nns, num_worlds=num_worlds, num_simulations=num_simulations, c_puct_value=c_puct_value, policy_temp=policy_temp)

    def _increase_max_game_length(self) -> None:
        """Increment max game length for curriculum learning, up to the cap."""
        self._max_game_length = min(self._max_game_length + self._game_length_increment, self._capped_max_game_length)

    def self_play(self, epochs: int, batch_count: int) -> None:
        """
        Run the main training loop: self-play games, train networks, save checkpoints.

        Args:
            epochs: Number of training epochs.
            batch_count: Number of batches to train per epoch.
        """
        epoch_pbar = tqdm(range(epochs), total=epochs, desc='value loss: inf, policy loss: inf')

        try:
            for epoch in epoch_pbar:
                games_pbar = tqdm(range(self._games_per_training), total=self._games_per_training, desc='gameing 😎🎮', leave=False)
                for _ in games_pbar:
                    self._play_game()
                avg_value_loss, avg_policy_loss = self._train_networks(batch_count)
                epoch_pbar.set_description(f'value loss: {avg_value_loss:.2e}, policy loss: {avg_policy_loss:.2e}')

                self._manager.save(step=epoch, items=self._mcts.networks.get_state(epoch))

                self._run.log(data={'value_loss': avg_value_loss, 'policy_loss': avg_policy_loss})
                self._increase_max_game_length()
        except KeyboardInterrupt:
            pass
        finally:
            self._manager.close()

    def _create_random_game_state(self) -> GameState:
        """
        Create a game state with random starting conditions for curriculum learning.

        Returns:
            New game state, possibly with skipped first move or removed players.
        """
        state = GameState(no_players=self._no_players)

        if np.random.random() < PROB_SKIP_FIRST_MOVE:
            state.execute_action(OFFSET_SINGLE_CARD)

        if self._no_players == 4:
            roll = np.random.random()
            if roll < PROB_REMOVE_TWO_PLAYERS:
                players_to_remove = [p for p in range(self._no_players) if p != state.current_player]
                np.random.shuffle(players_to_remove)
                for p in players_to_remove[:2]:
                    self._remove_player_cards(state, p)
            elif roll < PROB_REMOVE_ONE_PLAYER:
                players_to_remove = [p for p in range(self._no_players) if p != state.current_player]
                p = np.random.choice(players_to_remove)
                self._remove_player_cards(state, p)

        return state

    def _remove_player_cards(self, state: GameState, player: int) -> None:
        """
        Remove a player from the game by redistributing their cards.

        Args:
            state: Game state to modify.
            player: Player index to remove.
        """
        active_players = [p for p in range(state.no_players) if p != player and not state.is_done_array[p]]
        if not active_players:
            return

        for suit in range(state.player_hands.shape[0]):
            for rank in range(state.player_hands.shape[1]):
                if state.player_hands[suit][rank] == player:
                    new_owner = np.random.choice(active_players)
                    state.player_hands[suit][rank] = new_owner
                    state.knowledge_table[:, suit, rank] = new_owner

        state.is_done_array[player] = True

    def _play_game(self) -> None:
        """Play one self-play game and add all transitions to the replay buffer."""
        state = self._create_random_game_state()

        game_trajectory: list[tuple[ValueStateRepr, PolicyStateRepr, np.ndarray, int]] = []

        pbar = tqdm(range(self._max_game_length), desc='Max game length', leave=False)
        for _ in pbar:
            policy_probs, _ = self._mcts.run(state)

            value_state = ValueStateProcessor.encode(state)
            prepared_knowledge_policy, table_state_policy, encoded_action_policy, _ = PolicyStateProcessor.encode(state)
            game_trajectory.append(
                (value_state, (prepared_knowledge_policy, table_state_policy, encoded_action_policy), policy_probs, state.current_player)
            )

            action = np.random.choice(len(policy_probs), p=policy_probs)
            is_end = state.execute_action(action)
            if is_end:
                break

        game_outcome = np.ones(self._no_players) * (WINNER_VALUE / max(1, self._no_players - 1))
        game_outcome[~state.is_done_array] = LOSER_VALUE

        # Add all states from this game to the buffer
        for value_state, policy_state, policy_probs, current_player in game_trajectory:
            shifted_outcome = np.roll(game_outcome, -current_player)
            self._buffer.append((value_state, policy_state, policy_probs, shifted_outcome))

    def _train_networks(self, batch_count: int) -> tuple[float, float]:
        """
        Train both networks on sampled batches from the replay buffer.

        Args:
            batch_count: Number of batches to train.

        Returns:
            Tuple of (average value loss, average policy loss).
        """
        avg_value_loss = 0.0
        avg_policy_loss = 0.0
        for _ in range(batch_count):
            avg_value_loss += self._train_value_step() / batch_count
            avg_policy_loss += self._train_policy_step() / batch_count
        return avg_value_loss, avg_policy_loss

    def _train_value_step(self) -> float:
        """
        Perform one training step on the value network.

        Returns:
            Value loss for this step.
        """
        prepared_player_hands, table_states, target_values = self._sample_value_batch()
        value_loss, value_grads = compute_value_loss_and_grad_vect(
            self._mcts.networks.value_network.network, self._mcts.networks.value_network.params, prepared_player_hands, table_states, target_values
        )
        updates, self._mcts.networks.value_network.opt_state = self._mcts.networks.value_network.optimizer.update(
            value_grads, self._mcts.networks.value_network.opt_state, self._mcts.networks.value_network.params
        )
        self._mcts.networks.value_network.params = optax.apply_updates(self._mcts.networks.value_network.params, updates)
        return value_loss.item()

    def _train_policy_step(self) -> float:
        """
        Perform one training step on the policy network.

        Returns:
            Policy loss for this step.
        """
        prepared_knowledge, table_states, encoded_actions, target_policies = self._sample_policy_batch()
        policy_loss, policy_grads = compute_policy_loss_and_grad_vect(
            self._mcts.networks.policy_network.network,
            self._mcts.networks.policy_network.params,
            prepared_knowledge,
            table_states,
            encoded_actions,
            target_policies,
        )
        updates, self._mcts.networks.policy_network.opt_state = self._mcts.networks.policy_network.optimizer.update(
            policy_grads, self._mcts.networks.policy_network.opt_state, self._mcts.networks.policy_network.params
        )
        self._mcts.networks.policy_network.params = optax.apply_updates(self._mcts.networks.policy_network.params, updates)
        return policy_loss.item()

    def _sample_value_batch(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Sample a batch for value network training from the replay buffer.

        Returns:
            Tuple of (player hands, table states, target values).
        """
        no_samples = min(self._batch_size, len(self._buffer))
        samples_indices = np.random.choice(len(self._buffer), size=no_samples, replace=False)
        sampled = [self._buffer[i] for i in samples_indices]
        data_to_unpack, _, _, target_values = tuple(map(list, zip(*sampled, strict=True)))
        prepared_player_hands, table_states = tuple(map(list, zip(*data_to_unpack, strict=True)))

        return jnp.array(prepared_player_hands), jnp.array(table_states), jnp.array(target_values)

    def _sample_policy_batch(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Sample a batch for policy network training from the replay buffer.

        Returns:
            Tuple of (knowledge, table states, action masks, target policies).
        """
        no_samples = min(self._batch_size, len(self._buffer))
        samples_indices = np.random.choice(len(self._buffer), size=no_samples, replace=False)
        sampled = [self._buffer[i] for i in samples_indices]

        _, tuples, target_policies, _ = tuple(map(list, zip(*sampled, strict=True)))
        prepared_knowledge, table_states, encoded_actions = tuple(map(list, zip(*tuples, strict=True)))

        return jnp.array(prepared_knowledge), jnp.array(table_states), jnp.array(encoded_actions), jnp.array(target_policies)
