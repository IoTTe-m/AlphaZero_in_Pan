"""Game controller managing game state and AI interaction for the GUI."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from src.game_logic import ACTION_COUNT, RANKS, SUITS, GameState
from src.gui.config import PlayConfig
from src.mcts.mcts import MCTS
from src.ml.neural_networks import AlphaZeroNNs, PolicyNetwork, ValueNetwork


class GameController:
    """Manages game logic, AI decisions, and state for the Pan game GUI.

    Coordinates between the game state, MCTS AI, and the GUI application.
    Handles loading trained models and executing both human and AI actions.

    Attributes:
        config: Game configuration settings.
        state: Current game state.
        human_player: Player index controlled by the human.
        mcts: MCTS instance for AI decision making.
    """

    def __init__(self, config: PlayConfig):
        """Initialize the game controller with configuration.

        Args:
            config: Game configuration including model path and MCTS settings.
        """
        self.config = config
        self.state = GameState(no_players=config.player_count)
        self.human_player = config.human_player

        alpha_zero_nns = self._load_trained_model(Path(config.checkpoint_path))
        self.mcts = MCTS(
            networks=alpha_zero_nns,
            num_worlds=config.num_worlds,
            num_simulations=config.num_simulations,
            c_puct_value=config.c_puct_value,
            policy_temp=config.policy_temp,
        )

    def _load_trained_model(self, checkpoint_path: Path) -> AlphaZeroNNs:
        """Load a trained AlphaZero model from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint directory/step.

        Returns:
            Initialized AlphaZeroNNs with loaded weights.
        """
        value_network = ValueNetwork(self.config.player_count, len(SUITS), len(RANKS))
        policy_network = PolicyNetwork(ACTION_COUNT)

        rng = jax.random.PRNGKey(0)

        rng, init_rng = jax.random.split(rng)
        value_network_params = value_network.init(
            init_rng,
            jnp.zeros((1, len(SUITS), len(RANKS), self.config.player_count + 1)),
            jnp.zeros((1, len(SUITS) * len(RANKS), len(SUITS) + len(RANKS))),
        )
        rng, init_rng = jax.random.split(rng)
        policy_network_params = policy_network.init(
            init_rng,
            jnp.zeros((1, len(SUITS), len(RANKS), self.config.player_count + 1)),
            jnp.zeros((1, len(SUITS) * len(RANKS), len(SUITS) + len(RANKS))),
            jnp.zeros((1, ACTION_COUNT), dtype=jnp.bool),
        )

        optimizer_chain_value = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(1e-4),
        )
        opt_state_value = optimizer_chain_value.init(value_network_params)

        optimizer_chain_policy = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(1e-4),
        )
        opt_state_policy = optimizer_chain_policy.init(policy_network_params)

        # Load checkpoint using CheckpointManager (matches how it was saved)
        checkpointer = ocp.StandardCheckpointer()
        options = ocp.CheckpointManagerOptions(create=False)
        manager = ocp.CheckpointManager(checkpoint_path.parent.absolute(), checkpointer, options)

        step = int(checkpoint_path.name)
        restored = manager.restore(
            step,
            args=ocp.args.StandardRestore(
                item={  # type: ignore
                    'step': 0,
                    'value': {'params': value_network_params, 'opt_state': opt_state_value},
                    'policy': {'params': policy_network_params, 'opt_state': opt_state_policy},
                }
            ),
        )

        manager.close()

        return AlphaZeroNNs(
            value_network=value_network,
            policy_network=policy_network,
            value_network_params=restored['value']['params'],
            policy_network_params=restored['policy']['params'],
            value_network_optimizer=optimizer_chain_value,
            policy_network_optimizer=optimizer_chain_policy,
            value_network_opt_state=restored['value']['opt_state'],
            policy_network_opt_state=restored['policy']['opt_state'],
        )

    def restart(self) -> None:
        """Restart the game to initial state."""
        self.state.restart()

    def is_human_turn(self) -> bool:
        """Check if it's the human player's turn.

        Returns:
            True if the current player is the human player.
        """
        return self.state.current_player == self.human_player

    def is_game_over(self) -> bool:
        """Check if the game has ended.

        Returns:
            True if only one player remains (the loser).
        """
        return bool(np.sum(self.state.is_done_array) >= self.state.no_players - 1)

    def get_loser(self) -> int | None:
        """Get the losing player if the game is over.

        Returns:
            Index of the losing player, or None if game is not over.
        """
        if not self.is_game_over():
            return None
        losers = np.where(~self.state.is_done_array)[0]
        return int(losers[0]) if len(losers) > 0 else None

    def get_ai_action(self) -> int:
        """Get the AI's chosen action using MCTS.

        Returns:
            Action ID selected by the AI.
        """
        policy_probs, _ = self.mcts.run(self.state)
        return int(np.argmax(policy_probs))

    def get_human_actions(self) -> list[int]:
        """Get legal actions for the human player.

        Returns:
            List of valid action IDs the human can take.
        """
        return self.state.get_possible_actions(self.human_player)

    def execute_action(self, action: int) -> bool:
        """Execute an action in the game.

        Args:
            action: Action ID to execute.

        Returns:
            True if the game ended after this action.
        """
        return self.state.execute_action(action)

    def get_player_hand(self, player: int) -> list[tuple[int, int]]:
        """Get the cards in a player's hand.

        Args:
            player: Player index.

        Returns:
            List of (rank, suit) tuples for cards in hand.
        """
        ranks, suits = self.state.get_player_hand(player)
        return list(zip(ranks, suits, strict=True))

    def get_table_cards(self) -> list[tuple[int, int]]:
        """Get the cards currently on the table.

        Returns:
            List of (rank, suit) tuples for cards on table.
        """
        cards = []
        for i in range(self.state.cards_on_table):
            card_encoding = self.state.table_state[i]
            rank, suit = GameState.decode_card(card_encoding)
            cards.append((rank, suit))
        return cards

    def get_current_player(self) -> int:
        """Get the index of the current player.

        Returns:
            Current player index.
        """
        return self.state.current_player

    def is_player_done(self, player: int) -> bool:
        """Check if a player has finished (no cards left).

        Args:
            player: Player index to check.

        Returns:
            True if the player has no cards remaining.
        """
        return bool(self.state.is_done_array[player])
