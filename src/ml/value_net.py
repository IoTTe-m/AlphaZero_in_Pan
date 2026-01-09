"""Value network for game state evaluation.

Defines the neural network architecture and loss functions for predicting
player outcomes from game states.
"""

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

    def setup(self) -> None:
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
        """
        Forward pass returning predicted values for each player.

        Args:
            prepared_player_hands: Encoded player hands.
            table_state: Encoded table state.

        Returns:
            Value predictions for each player.
        """
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
    """
    JIT-compiled value network forward pass.

    Args:
        value_network: Value network module.
        value_network_params: Network parameters.
        prepared_player_hands: Encoded player hands.
        table_state: Encoded table state.

    Returns:
        Value predictions for each player.
    """
    return jnp.array(value_network.apply(value_network_params, prepared_player_hands, table_state))


@partial(jit, static_argnames=('value_network',))
def call_value_network_batched(
    value_network: ValueNetwork,
    value_network_params: dict,
    prepared_player_hands: jnp.ndarray,
    table_state: jnp.ndarray,
) -> jnp.ndarray:
    """
    Batched JIT-compiled value network forward pass using vmap.

    Args:
        value_network: Value network module.
        value_network_params: Network parameters.
        prepared_player_hands: Batched encoded player hands.
        table_state: Batched encoded table state.

    Returns:
        Batched value predictions.
    """
    return vmap(call_value_network, in_axes=(None, None, 0, 0))(value_network, value_network_params, prepared_player_hands, table_state)


def compute_value_loss(
    value_network: ValueNetwork,
    params: optax.Params,
    prepared_player_hands: jnp.ndarray,
    table_states: jnp.ndarray,
    target_values: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute MSE loss between predicted and target values.

    Args:
        value_network: Value network module.
        params: Network parameters.
        prepared_player_hands: Encoded player hands.
        table_states: Encoded table state.
        target_values: Ground truth values.

    Returns:
        Scalar MSE loss.
    """
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
    """
    Compute mean value loss over a batch of samples.

    Args:
        value_network: Value network module.
        params: Network parameters.
        prepared_player_hands: Batched encoded hands.
        table_states: Batched table states.
        target_values: Batched target values.

    Returns:
        Mean MSE loss.
    """
    return compute_value_loss_vect_raw(value_network, params, prepared_player_hands, table_states, target_values).mean()


compute_value_loss_and_grad_vect = jit(
    value_and_grad(compute_value_loss_vect, argnums=1),
    static_argnames=('value_network',),
)
