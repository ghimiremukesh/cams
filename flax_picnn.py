import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import flax
from flax import linen as nn
import dataclasses
from typing import Callable
from jax.typing import ArrayLike

@dataclasses.dataclass
class ModelConfig:
    """Config object"""
    in_features: int = 9
    out_features: int = 1
    num_hidden_layers: int = 3
    hidden_features: int = 256
    activation: Callable = nn.relu
    
@dataclasses.dataclass
class PINNConfig:
    """Config object"""
    in_features: int = 10
    out_features: int = 1
    num_hidden_layers: int = 3
    hidden_features: int = 256
    activation: Callable = nn.relu


class PICNN(nn.Module):
    '''
    A Partially Input Convex Neural Network in JAX
    '''
    config: ModelConfig
    # nl: Optional[Callable] = None
    # nl_u: Optional[Callable] = None
    # params: Optional[ArrayLike] = None
    # params_u: Optional[ArrayLike] = None
    # params_zu_u: Optional[ArrayLike] = None
    # net_u: Optional[Callable] = None
    # net_z_u: Optional[Callable] = None
    # net_zu_u: Optional[Callable] = None
    # net_yu_u: Optional[Callable] = None
    # net_z_yu: Optional[Callable] = None
    # net_z_zu: Optional[Callable] = None

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

        # self.params = jnp.array(0.1)
        # self.params_u = jnp.array(0.1)
        # self.params_zu_u = jnp.array(0.1)

        # non-convex layer
        # u_sizes = zip([in_features - 1] * 1 + [hidden_features] * (num_hidden_layers), [hidden_features] *
        #               num_hidden_layers)

        u_sizes = [hidden_features] * (num_hidden_layers)

        self.net_u =[nn.Dense(out_features, use_bias=True)
                    for (out_features) in u_sizes]

        # zu_u_sizes = zip([hidden_features] * (num_hidden_layers),
        #                  [hidden_features] * num_hidden_layers)

        zu_u_sizes = [hidden_features] * num_hidden_layers
        self.net_zu_u =[nn.Dense(out_features, use_bias=True, kernel_init=weight_init)
                        for out_features in zu_u_sizes]

        # z_zu_sizes = zip([hidden_features] * (num_hidden_layers), [hidden_features] *
        #                  (num_hidden_layers - 1) + [1])
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

        # z_u_sizes = zip([in_features - 1] * 1 + [hidden_features] * (num_hidden_layers), [hidden_features] *
        #                 (num_hidden_layers) + [1])
        z_u_sizes = [hidden_features] * (num_hidden_layers) + [1]
        self.net_z_u = [nn.DenseGeneral(out_features, use_bias=True, kernel_init=weight_init)
                        for out_features in z_u_sizes]

        # self.make_cvx()
        # self.final_layer = nn.Linear(hidden_features, 1, use_bias=False)

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

    # def make_cvx(self):
    #     for layers in self.net_z_zu:
    #         pass


#
# if __name__ == '__main__':
#     cfg = ModelConfig
#     net = PICNN(cfg)
#     key = jrandom.key(0)
#     # net.make_cvx()
#     x = jnp.ones((10, 9))
#
#     params = net.init(key, x)
#     y_prev = net.apply(params, x)
#     # make convex
#     for key in params['params'].keys():
#         if 'cvx_layer' in key:
#             params['params'][key]['kernel'] = params['params'][key]['kernel'].clip(0)
#
#     y = net.apply(params, x)

    # net(x)
