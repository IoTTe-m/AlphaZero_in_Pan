import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

import wandb
from src.config import TrainingConfig
from src.game_logic import ACTION_COUNT, RANKS, SUITS
from src.ml.learning import LearningProcess
from src.ml.neural_networks import AlphaZeroNNs, PolicyNetwork, ValueNetwork

DEFAULT_CONFIG = Path(__file__).parent / 'configs' / 'default.yaml'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train AlphaZero for Pan card game')
    parser.add_argument(
        '-c',
        '--config',
        type=Path,
        default=DEFAULT_CONFIG,
        help=f'Path to YAML config file (default: {DEFAULT_CONFIG})',
    )
    return parser.parse_args()


def main(config: TrainingConfig):
    run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        config=config.to_dict(),
    )

    value_network = ValueNetwork(config.player_count, len(SUITS), len(RANKS))
    policy_network = PolicyNetwork(ACTION_COUNT)

    rng = jax.random.PRNGKey(0)

    rng, init_rng = jax.random.split(rng)
    value_network_params = value_network.init(
        init_rng,
        jnp.zeros((1, len(SUITS), len(RANKS), config.player_count + 1)),
        jnp.zeros((1, len(SUITS) * len(RANKS), len(SUITS) + len(RANKS))),
    )
    rng, init_rng = jax.random.split(rng)
    policy_network_params = policy_network.init(
        init_rng,
        jnp.zeros((1, len(SUITS), len(RANKS), config.player_count + 1)),
        jnp.zeros((1, len(SUITS) * len(RANKS), len(SUITS) + len(RANKS))),
        jnp.zeros((1, ACTION_COUNT), dtype=jnp.bool),
    )

    optimizer_chain_value = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
    )
    opt_state_value = optimizer_chain_value.init(value_network_params)

    optimizer_chain_policy = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
    )
    opt_state_policy = optimizer_chain_policy.init(policy_network_params)

    alpha_zero_nns = AlphaZeroNNs(
        value_network=value_network,
        policy_network=policy_network,
        value_network_params=value_network_params,
        policy_network_params=policy_network_params,
        value_network_optimizer=optimizer_chain_value,
        policy_network_optimizer=optimizer_chain_policy,
        value_network_opt_state=opt_state_value,
        policy_network_opt_state=opt_state_policy,
    )

    learning = LearningProcess(
        run=run,
        save_dir=config.save_dir,
        nns=alpha_zero_nns,
        no_players=config.player_count,
        batch_size=config.batch_size,
        games_per_training=config.games_per_training,
        num_simulations=config.num_simulations,
        num_worlds=config.num_worlds,
        max_buffer_size=config.max_buffer_size,
        c_puct_value=config.c_puct_value,
        policy_temp=config.policy_temp,
        initial_max_game_length=config.initial_max_game_length,
        capped_max_game_length=config.capped_max_game_length,
        game_length_increment=config.game_length_increment,
    )

    learning.self_play(config.epochs, config.batch_count)
    print('done 🥰🥰🥰🥰🥰🥰')


if __name__ == '__main__':
    args = parse_args()
    cfg = TrainingConfig.from_yaml(args.config)
    main(cfg)
