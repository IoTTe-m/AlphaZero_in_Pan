from dataclasses import dataclass
from functools import partial

from flax import linen as nn
from jax import numpy as jnp, jit


class ValueNetwork(nn.Module):
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


class PolicyNetwork(nn.Module):
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

    def __call__(self, prepared_knowledge: jnp.ndarray, table_state: jnp.ndarray,
                 actions_mask: jnp.ndarray) -> jnp.ndarray:
        # a 1 in action_mask means that we want to include this action
        flattened_knowledge = prepared_knowledge.flatten()
        flattened_table = table_state.flatten()
        concat_features = jnp.concatenate((flattened_knowledge, flattened_table))
        logits = self.model(concat_features)
        return nn.softmax(logits, where=actions_mask)


@partial(jit, static_names=('value_network',))
def call_value_network(value_network: ValueNetwork, value_network_params: dict,
                       prepared_player_hands: jnp.ndarray, table_state: jnp.ndarray) -> jnp.ndarray:
    return value_network.apply(
        value_network_params, prepared_player_hands, table_state
    )


@partial(jit, static_names=('policy_network',))
def call_policy_network(policy_network: PolicyNetwork, policy_network_params: dict,
                        prepared_knowledge: jnp.ndarray, table_state: jnp.ndarray,
                        actions_mask: jnp.ndarray) -> jnp.ndarray:
    return policy_network.apply(
        policy_network_params, prepared_knowledge, table_state, actions_mask
    )


@dataclass
class AlphaZeroNNs:
    value_network: ValueNetwork
    policy_network: PolicyNetwork
    value_network_params: dict
    policy_network_params: dict
