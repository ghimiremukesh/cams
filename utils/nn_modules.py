"""
This file contains neural network architectures used in this project.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import dataclasses
from typing import Callable
from jax.typing import ArrayLike

# configs for different model
@dataclasses.dataclass
class ModelConfig:
    """Config object"""
    in_features: int = 10
    out_features: int = 1
    num_hidden_layers: int = 3
    hidden_features: int = 256
    activation: Callable = nn.relu

@dataclasses.dataclass
class NoTimeConfig:
    """Config object"""
    in_features: int = 9
    out_features: int = 1
    num_hidden_layers: int = 3
    hidden_features: int = 256
    activation: Callable = nn.relu

@dataclasses.dataclass
class NoTimeDualConfig:
    """Config object"""
    in_features: int = 10
    out_features: int = 1
    num_hidden_layers: int = 3
    hidden_features: int = 256
    activation: Callable = nn.relu


# partially input convex neural network
class PICNN(nn.Module):
    '''
    A Partially Input Convex Neural Network in JAX
    '''
    config: ModelConfig

    def setup(self):
        config = self.config
        in_features = config.in_features
        hidden_features = config.hidden_features
        num_hidden_layers = config.num_hidden_layers
        self.nl = config.activation
        self.nl_u = config.activation
        self.num_hidden_layers = num_hidden_layers

        self.params = self.param('adaptive_act_1', lambda rng: 0.1 * jnp.ones(1))
        self.params_u = self.param('adaptive_act_2', lambda rng: 0.1 * jnp.ones(1))
        self.params_zu_u = self.param('adaptive_act_3', lambda rng: 0.1 * jnp.ones(1))
        weight_init = jax.nn.initializers.kaiming_normal()


        u_sizes = [hidden_features] * (num_hidden_layers)

        self.net_u =[nn.Dense(out_features, use_bias=True)
                    for (out_features) in u_sizes]


        zu_u_sizes = [hidden_features] * num_hidden_layers
        self.net_zu_u =[nn.Dense(out_features, use_bias=True, kernel_init=weight_init)
                        for out_features in zu_u_sizes]

        z_zu_sizes = [hidden_features] * (num_hidden_layers - 1) + [1]

        self.net_z_zu = [nn.Dense(out_features, use_bias=False, name=f'cvx_layer_{i}',
                                  kernel_init=weight_init) for i, out_features in enumerate(z_zu_sizes)]

        yu_u_sizes = [in_features - 1] * 1 + [hidden_features] * (num_hidden_layers)
        # yu_u_sizes = [in_features - 1] * 1 + [hidden_features] * (num_hidden_layers)
        self.net_yu_u =[nn.Dense(1, use_bias=True, kernel_init=weight_init)
                        for _ in yu_u_sizes]

        z_yu_sizes = [hidden_features] * (num_hidden_layers) + [1]
        self.net_z_yu =[nn.Dense(out_features, use_bias=False, kernel_init=weight_init)
             for out_features in z_yu_sizes]


        z_u_sizes = [hidden_features] * (num_hidden_layers) + [1]
        self.net_z_u = [nn.DenseGeneral(out_features, use_bias=True, kernel_init=weight_init)
                        for out_features in z_u_sizes]


    def __call__(self, coords: ArrayLike):
        y_input = coords[..., -1:]
        u_input = coords[..., :-1]
        z_input = self.net_z_u[0](u_input) + self.net_z_yu[0](jnp.multiply(y_input, self.net_yu_u[0](u_input)))

        z_input = self.nl(10 * self.params * z_input)

        u_input = self.net_u[0](u_input)

        u_input = self.nl_u(10 * self.params_u * u_input)

        for i in range(1, self.num_hidden_layers + 1):
            z_input = self.net_z_zu[i - 1](jnp.multiply(z_input,
                                                        self.nl(10 * self.params_zu_u *
                                                            self.net_zu_u[i - 1](u_input)))) + \
                      self.net_z_u[i](u_input) + self.net_z_yu[i](jnp.multiply(y_input, self.net_yu_u[i](u_input)))

            if i == self.num_hidden_layers:
                output = z_input  # no activation needed for final layer
                break
            z_output = self.nl(10 * self.params * z_input)
            u_input = self.net_u[i](u_input)

            u_input = self.nl_u(10 * self.params_u * u_input)

            z_input = z_output

        return output
    

class PICNN_Dual(nn.Module):
    '''
    A Partially Input Convex Neural Network in JAX
    '''
    config: ModelConfig

    def setup(self):
        config = self.config
        in_features = config.in_features
        hidden_features = config.hidden_features
        num_hidden_layers = config.num_hidden_layers
        self.nl = config.activation
        self.nl_u = config.activation
        self.num_hidden_layers = num_hidden_layers

        self.params = self.param('adaptive_act_1', lambda rng: 0.1 * jnp.ones(1))
        self.params_u = self.param('adaptive_act_2', lambda rng: 0.1 * jnp.ones(1))
        self.params_zu_u = self.param('adaptive_act_3', lambda rng: 0.1 * jnp.ones(1))
        weight_init = jax.nn.initializers.kaiming_normal()


        u_sizes = [hidden_features] * (num_hidden_layers)

        self.net_u =[nn.Dense(out_features, use_bias=True)
                    for (out_features) in u_sizes]


        zu_u_sizes = [hidden_features] * num_hidden_layers
        self.net_zu_u =[nn.Dense(out_features, use_bias=True, kernel_init=weight_init)
                        for out_features in zu_u_sizes]

        z_zu_sizes = [hidden_features] * (num_hidden_layers - 1) + [1]

        self.net_z_zu = [nn.Dense(out_features, use_bias=False, name=f'cvx_layer_{i}',
                                  kernel_init=weight_init) for i, out_features in enumerate(z_zu_sizes)]

        yu_u_sizes = [in_features - 1] * 1 + [hidden_features] * (num_hidden_layers)
        # yu_u_sizes = [in_features - 1] * 1 + [hidden_features] * (num_hidden_layers)
        self.net_yu_u =[nn.Dense(1, use_bias=True, kernel_init=weight_init)
                        for _ in yu_u_sizes]

        z_yu_sizes = [hidden_features] * (num_hidden_layers) + [1]
        self.net_z_yu =[nn.Dense(out_features, use_bias=False, kernel_init=weight_init)
             for out_features in z_yu_sizes]


        z_u_sizes = [hidden_features] * (num_hidden_layers) + [1]
        self.net_z_u = [nn.DenseGeneral(out_features, use_bias=True, kernel_init=weight_init)
                        for out_features in z_u_sizes]


    def __call__(self, coords: ArrayLike):
        y_input = coords[..., -2:]
        u_input = coords[..., :-2]
        z_input = self.net_z_u[0](u_input) + self.net_z_yu[0](jnp.multiply(y_input, self.net_yu_u[0](u_input)))

        z_input = self.nl(10 * self.params * z_input)

        u_input = self.net_u[0](u_input)

        u_input = self.nl_u(10 * self.params_u * u_input)

        for i in range(1, self.num_hidden_layers + 1):
            z_input = self.net_z_zu[i - 1](jnp.multiply(z_input,
                                                        self.nl(10 * self.params_zu_u *
                                                            self.net_zu_u[i - 1](u_input)))) + \
                      self.net_z_u[i](u_input) + self.net_z_yu[i](jnp.multiply(y_input, self.net_yu_u[i](u_input)))

            if i == self.num_hidden_layers:
                output = z_input  # no activation needed for final layer
                break
            z_output = self.nl(10 * self.params * z_input)
            u_input = self.net_u[i](u_input)

            u_input = self.nl_u(10 * self.params_u * u_input)

            z_input = z_output

        return output