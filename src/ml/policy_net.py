from functools import partial

from flax import linen as nn
from jax import jit, value_and_grad, vmap
from jax import numpy as jnp

LOG_EPSILON = 1e-8


class PolicyNetwork(nn.Module):
    """
    Gets partial state of the game (player POV), returns probability of each action
    """

    actions_space_size: int

    def setup(self) -> None:
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
        """
        Forward pass returning action probabilities masked by legal actions.

        Args:
            prepared_knowledge: Encoded player knowledge state.
            table_state: Encoded table state.
            actions_mask: Boolean mask of legal actions, 1 for legal, 0 for illegal.

        Returns:
            Softmax probability distribution over actions.
        """
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
    """
    JIT-compiled policy network forward pass.

    Args:
        policy_network: Policy network module.
        policy_network_params: Network parameters.
        prepared_knowledge: Encoded player knowledge.
        table_state: Encoded table state.
        actions_mask: Boolean mask of legal actions.

    Returns:
        Action probability distribution.
    """
    return jnp.array(policy_network.apply(policy_network_params, prepared_knowledge, table_state, actions_mask))


@partial(jit, static_argnames=('policy_network',))
def call_policy_network_batched(
    policy_network: PolicyNetwork,
    policy_network_params: dict,
    prepared_knowledge: jnp.ndarray,
    table_state: jnp.ndarray,
    actions_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Batched JIT-compiled policy network forward pass using vmap.

    Args:
        policy_network: Policy network module.
        policy_network_params: Network parameters.
        prepared_knowledge: Batched encoded player knowledge.
        table_state: Batched encoded table state.
        actions_mask: Batched boolean mask of legal actions.

    Returns:
        Batched action probability distributions.
    """
    return vmap(call_policy_network, in_axes=(None, None, 0, 0, 0))(
        policy_network, policy_network_params, prepared_knowledge, table_state, actions_mask
    )


def compute_policy_loss(
    policy_network: PolicyNetwork,
    params: dict,
    prepared_knowledge: jnp.ndarray,
    table_states: jnp.ndarray,
    encoded_actions: jnp.ndarray,
    target_policies: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute cross-entropy loss between predicted and target policies.

    Args:
        policy_network: Policy network module.
        params: Network parameters.
        prepared_knowledge: Encoded player knowledge.
        table_states: Encoded table state.
        encoded_actions: Action mask.
        target_policies: Target action distribution.

    Returns:
        Scalar cross-entropy loss.
    """
    logits = call_policy_network(policy_network, params, prepared_knowledge, table_states, encoded_actions)
    return -jnp.mean(jnp.sum(target_policies * jnp.log(logits + LOG_EPSILON), axis=-1))


compute_policy_loss_vect_raw = vmap(compute_policy_loss, in_axes=(None, None, 0, 0, 0, 0), out_axes=0)


def compute_policy_loss_vect(
    policy_network: PolicyNetwork,
    params: dict,
    prepared_knowledge: jnp.ndarray,
    table_states: jnp.ndarray,
    encoded_actions: jnp.ndarray,
    target_policies: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute mean policy loss over a batch of samples.

    Args:
        policy_network: Policy network module.
        params: Network parameters.
        prepared_knowledge: Batched encoded knowledge.
        table_states: Batched table states.
        encoded_actions: Batched action masks.
        target_policies: Batched target distributions.

    Returns:
        Mean cross-entropy loss.
    """
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
