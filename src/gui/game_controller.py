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
    def __init__(self, config: PlayConfig):
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
            args=ocp.args.StandardRestore({
                'step': 0,
                'value': {'params': value_network_params, 'opt_state': opt_state_value},
                'policy': {'params': policy_network_params, 'opt_state': opt_state_policy},
            }),
        )

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

    def restart(self):
        self.state.restart()

    def is_human_turn(self) -> bool:
        return self.state.current_player == self.human_player

    def is_game_over(self) -> bool:
        return np.sum(self.state.is_done_array) >= self.state.no_players - 1

    def get_loser(self) -> int | None:
        if not self.is_game_over():
            return None
        losers = np.where(~self.state.is_done_array)[0]
        return int(losers[0]) if len(losers) > 0 else None

    def get_ai_action(self) -> int:
        policy_probs, _ = self.mcts.run(self.state)
        return int(np.argmax(policy_probs))

    def get_human_actions(self) -> list[int]:
        return self.state.get_possible_actions(self.human_player)

    def execute_action(self, action: int) -> bool:
        return self.state.execute_action(action)

    def get_player_hand(self, player: int) -> list[tuple[int, int]]:
        ranks, suits = self.state.get_player_hand(player)
        return list(zip(ranks, suits, strict=True))

    def get_table_cards(self) -> list[tuple[int, int]]:
        cards = []
        for i in range(self.state.cards_on_table):
            card_encoding = self.state.table_state[i]
            rank, suit = GameState.decode_card(card_encoding)
            cards.append((rank, suit))
        return cards

    def get_current_player(self) -> int:
        return self.state.current_player

    def is_player_done(self, player: int) -> bool:
        return bool(self.state.is_done_array[player])
