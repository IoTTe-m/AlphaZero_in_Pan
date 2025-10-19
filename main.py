import optax
import wandb

from src.ml.learning import LearningProcess
from src.ml.neural_networks import ValueNetwork, PolicyNetwork, AlphaZeroNNs
from src.game_logic import SUITS, RANKS, ACTION_COUNT
import jax
import jax.numpy as jnp

def main():
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    BATCH_COUNT = 8

    GAMES_PER_TRAINING = 1
    NUM_SIMULATIONS = 512
    NUM_WORLDS = 4

    MAX_BUFFER_SIZE = 1024
    C_PUCT_VALUE = 1
    POLICY_TEMP = 0.2

    INITIAL_MAX_GAME_LENGTH = 30
    CAPPED_MAX_GAME_LENGTH = 500
    GAME_LENGTH_INCREMENT = 10

    EPOCHS = 100
    PLAYER_COUNT = 4

    SAVE_DIR = "checkpoints/"

    run = wandb.init(
        entity="reinforced-annealer",
        project="pan-alpha-zero",
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "batch_count": BATCH_COUNT,
            "games_per_training": GAMES_PER_TRAINING,
            "num_simulations": NUM_SIMULATIONS,
            "num_worlds": NUM_WORLDS,
            "max_buffer_size": MAX_BUFFER_SIZE,
            "c_puct_value": C_PUCT_VALUE,
            "policy_temp": POLICY_TEMP,
            "initial_max_game_length": INITIAL_MAX_GAME_LENGTH,
            "capped_max_game_length": CAPPED_MAX_GAME_LENGTH,
            "game_length_increment": GAME_LENGTH_INCREMENT,
            "epochs": EPOCHS,
            "player_count": PLAYER_COUNT
        }
    )
    
    value_network = ValueNetwork(PLAYER_COUNT, len(SUITS), len(RANKS))
    policy_network = PolicyNetwork(ACTION_COUNT)

    rng = jax.random.PRNGKey(0)

    rng, init_rng = jax.random.split(rng)
    # input_size = (self.no_players + self.suits_count + self.ranks_count) * self.suits_count * self.ranks_count
    value_network_params = value_network.init(
        init_rng, jnp.zeros((1, len(SUITS), len(RANKS), PLAYER_COUNT + 1)), jnp.zeros((1, len(SUITS) * len(RANKS), len(SUITS)+len(RANKS)))
    )
    rng, init_rng = jax.random.split(rng)
    # input_size = (self.no_players + self.suits_count + self.ranks_count) * self.suits_count * self.ranks_count
    policy_network_params = policy_network.init(
        init_rng, jnp.zeros((1, len(SUITS), len(RANKS), PLAYER_COUNT + 1)), jnp.zeros((1, len(SUITS) * len(RANKS), len(SUITS)+len(RANKS))), jnp.zeros((1, ACTION_COUNT))
    )

    optimizer_chain_value = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(LEARNING_RATE)
    )
    opt_state_value = optimizer_chain_value.init(value_network_params)

    optimizer_chain_policy = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(LEARNING_RATE)
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
        policy_network_opt_state=opt_state_policy
    )

    learning = LearningProcess(
        run=run,
        save_dir=SAVE_DIR,
        nns=alpha_zero_nns,
        no_players=PLAYER_COUNT,
        batch_size=BATCH_SIZE,
        games_per_training=GAMES_PER_TRAINING,
        num_simulations=NUM_SIMULATIONS,
        num_worlds=NUM_WORLDS,
        max_buffer_size=MAX_BUFFER_SIZE,
        c_puct_value=C_PUCT_VALUE,
        policy_temp=POLICY_TEMP,
        initial_max_game_length=INITIAL_MAX_GAME_LENGTH,
        capped_max_game_length=CAPPED_MAX_GAME_LENGTH,
        game_length_increment=GAME_LENGTH_INCREMENT,
    )

    learning.self_play(EPOCHS, BATCH_COUNT)
    print("done 🥰🥰🥰🥰🥰🥰")

if __name__ == '__main__':
    main()
