from functools import partial

import optax
from flax import linen as nn
from jax import jit, value_and_grad, vmap
from jax import numpy as jnp


class ValueNetwork(nn.Module):
    """
    Gets comprehensive state of the game, returns how good each player stands
    """

    no_players: int
    suits_count: int
    ranks_count: int

    def setup(self):
        # input_size = (self.no_players + self.suits_count + self.ranks_count) * self.suits_count * self.ranks_count
        output_size = self.no_players
        self.model = nn.Sequential(
            [
                nn.Dense(features=512),
                nn.relu,
                nn.Dense(features=256),
                nn.relu,
                nn.Dense(features=128),
                nn.relu,
                nn.Dense(features=32),
                nn.relu,
                nn.Dense(features=output_size),
            ]
        )

    def __call__(self, prepared_player_hands: jnp.ndarray, table_state: jnp.ndarray) -> jnp.ndarray:
        flattened_hands = prepared_player_hands.flatten()
        flattened_table = table_state.flatten()
        concat_features = jnp.concatenate((flattened_hands, flattened_table))
        return self.model(concat_features)


@partial(jit, static_argnames=('value_network',))
def call_value_network(
    value_network: ValueNetwork,
    value_network_params: dict,
    prepared_player_hands: jnp.ndarray,
    table_state: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.array(value_network.apply(value_network_params, prepared_player_hands, table_state))


@partial(jit, static_argnames=('value_network',))
def call_value_network_batched(
    value_network: ValueNetwork,
    value_network_params: dict,
    prepared_player_hands: jnp.ndarray,
    table_state: jnp.ndarray,
) -> jnp.ndarray:
    return vmap(call_value_network, in_axes=(None, None, 0, 0))(value_network, value_network_params, prepared_player_hands, table_state)


def compute_value_loss(
    value_network: ValueNetwork,
    params: optax.Params,
    prepared_player_hands: jnp.ndarray,
    table_states: jnp.ndarray,
    target_values: jnp.ndarray,
) -> jnp.ndarray:
    predicted_values = call_value_network(value_network, params, prepared_player_hands, table_states)
    loss = jnp.mean((predicted_values - target_values) ** 2)
    return loss


compute_value_loss_vect_raw = vmap(compute_value_loss, in_axes=(None, None, 0, 0, 0), out_axes=0)


def compute_value_loss_vect(
    value_network: ValueNetwork,
    params: optax.Params,
    prepared_player_hands: jnp.ndarray,
    table_states: jnp.ndarray,
    target_values: jnp.ndarray,
) -> jnp.ndarray:
    return compute_value_loss_vect_raw(value_network, params, prepared_player_hands, table_states, target_values).mean()


compute_value_loss_and_grad_vect = jit(
    value_and_grad(compute_value_loss_vect, argnums=1),
    static_argnames=('value_network',),
)
