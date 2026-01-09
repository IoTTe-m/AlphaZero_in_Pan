"""Neural network management for AlphaZero training.

Contains wrapper classes for managing policy and value networks along
with their parameters, optimizers, and optimizer states.
"""

from dataclasses import dataclass

import optax
from flax import linen as nn

from src.ml.policy_net import (
    PolicyNetwork,
)
from src.ml.value_net import (
    ValueNetwork,
)


@dataclass
class NeuralNetworkManager:
    """Container for a neural network with its parameters and optimizer state.

    Attributes:
        network: Flax neural network module.
        params: Network parameters.
        optimizer: Optax optimizer.
        opt_state: Optimizer state.
    """

    network: nn.Module
    params: optax.Params
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState


@dataclass
class AlphaZeroNNs:
    """Combined policy and value networks for AlphaZero.

    Attributes:
        value_network: Managed value network for state evaluation.
        policy_network: Managed policy network for action probabilities.
    """

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
        """Initialize AlphaZero networks.

        Args:
            value_network: Value network module.
            policy_network: Policy network module.
            value_network_params: Value network parameters.
            policy_network_params: Policy network parameters.
            value_network_optimizer: Value network optimizer.
            policy_network_optimizer: Policy network optimizer.
            value_network_opt_state: Value optimizer state.
            policy_network_opt_state: Policy optimizer state.
        """
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
        """Get serializable state for checkpointing.

        Args:
            step: Current training step.

        Returns:
            Dictionary with step and network states.
        """
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
