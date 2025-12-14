from functools import partial

from flax import linen as nn
from jax import jit, value_and_grad, vmap
from jax import numpy as jnp


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


compute_policy_loss_vect_raw = vmap(compute_policy_loss, in_axes=(None, None, 0, 0, 0, 0), out_axes=0)


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


compute_policy_loss_and_grad_vect = jit(
    value_and_grad(compute_policy_loss_vect, argnums=1),
    static_argnames=('policy_network',),
)
