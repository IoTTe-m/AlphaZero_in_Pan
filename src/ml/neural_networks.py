from dataclasses import dataclass

import optax
from flax import linen as nn

from src.ml.policy_net import (
    PolicyNetwork,
    call_policy_network,
    compute_policy_loss_and_grad_vect,
)
from src.ml.value_net import (
    ValueNetwork,
    call_value_network,
    compute_value_loss_and_grad_vect,
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
