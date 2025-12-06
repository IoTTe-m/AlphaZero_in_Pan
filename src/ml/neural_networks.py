from dataclasses import dataclass
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
    value_network_params: optax.Params,
    prepared_player_hands: jnp.ndarray,
    table_state: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.array(value_network.apply(value_network_params, prepared_player_hands, table_state))


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


class PolicyNetwork(nn.Module):
    """
    Gets partial state of the game (player POV), returns probability of each action
    """

    actions_space_size: int

    def setup(self):
        # input_size = (self.no_players + self.suits_count + self.ranks_count) * self.suits_count * self.ranks_count
        output_size = self.actions_space_size
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

    def __call__(
        self,
        prepared_knowledge: jnp.ndarray,
        table_state: jnp.ndarray,
        actions_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        # a 1 in action_mask means that we want to include this action
        flattened_knowledge = prepared_knowledge.flatten()
        flattened_table = table_state.flatten()
        concat_features = jnp.concatenate((flattened_knowledge, flattened_table))
        logits = self.model(concat_features)
        return nn.softmax(logits, where=actions_mask)


@partial(jit, static_argnames=('policy_network',))
def call_policy_network(
    policy_network: PolicyNetwork,
    policy_network_params: dict,
    prepared_knowledge: jnp.ndarray,
    table_state: jnp.ndarray,
    actions_mask: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.array(policy_network.apply(policy_network_params, prepared_knowledge, table_state, actions_mask))


def compute_policy_loss(
    policy_network: PolicyNetwork,
    params: dict,
    prepared_knowledge: jnp.ndarray,
    table_states: jnp.ndarray,
    encoded_actions: jnp.ndarray,
    target_policies: jnp.ndarray,
) -> jnp.ndarray:
    logits = call_policy_network(policy_network, params, prepared_knowledge, table_states, encoded_actions)

    loss = -jnp.mean(jnp.sum(target_policies * jnp.log(logits + 1e-8), axis=-1))
    return loss


compute_value_loss_vect_raw = vmap(compute_value_loss, in_axes=(None, None, 0, 0, 0), out_axes=0)
compute_policy_loss_vect_raw = vmap(compute_policy_loss, in_axes=(None, None, 0, 0, 0, 0), out_axes=0)


def compute_value_loss_vect(
    value_network: ValueNetwork,
    params: optax.Params,
    prepared_player_hands: jnp.ndarray,
    table_states: jnp.ndarray,
    target_values: jnp.ndarray,
) -> jnp.ndarray:
    return compute_value_loss_vect_raw(value_network, params, prepared_player_hands, table_states, target_values).mean()


def compute_policy_loss_vect(
    policy_network: PolicyNetwork,
    params: dict,
    prepared_knowledge: jnp.ndarray,
    table_states: jnp.ndarray,
    encoded_actions: jnp.ndarray,
    target_policies: jnp.ndarray,
) -> jnp.ndarray:
    return compute_policy_loss_vect_raw(
        policy_network,
        params,
        prepared_knowledge,
        table_states,
        encoded_actions,
        target_policies,
    ).mean()


compute_value_loss_and_grad_vect = jit(
    value_and_grad(compute_value_loss_vect, argnums=1),
    static_argnames=('value_network',),
)
compute_policy_loss_and_grad_vect = jit(
    value_and_grad(compute_policy_loss_vect, argnums=1),
    static_argnames=('policy_network',),
)


@dataclass
class NeuralNetworkManager:
    network: nn.Module
    params: optax.Params
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState


@dataclass
class AlphaZeroNNs:
    value_network: NeuralNetworkManager
    policy_network: NeuralNetworkManager

    def __init__(
        self,
        value_network: ValueNetwork,
        policy_network: PolicyNetwork,
        value_network_params: optax.Params,
        policy_network_params: optax.Params,
        value_network_optimizer: optax.GradientTransformation,
        policy_network_optimizer: optax.GradientTransformation,
        value_network_opt_state: optax.OptState,
        policy_network_opt_state: optax.OptState,
    ):
        self.value_network = NeuralNetworkManager(
            network=value_network,
            params=value_network_params,
            optimizer=value_network_optimizer,
            opt_state=value_network_opt_state,
        )
        self.policy_network = NeuralNetworkManager(
            network=policy_network,
            params=policy_network_params,
            optimizer=policy_network_optimizer,
            opt_state=policy_network_opt_state,
        )

    def get_state(self, step: int) -> dict:
        return {
            'step': int(step),
            'value': {
                'params': self.value_network.params,
                'opt_state': self.value_network.opt_state,
            },
            'policy': {
                'params': self.policy_network.params,
                'opt_state': self.policy_network.opt_state,
            },
        }
