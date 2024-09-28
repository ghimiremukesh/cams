import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import math


# from torchmeta.modules import MetaModule


class BatchLinear(nn.Linear):
    '''A linear layer'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class FCBlock(nn.Module):
    '''A fully connected neural network.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, dropout=0.03, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None
        self.num_hidden_layers = num_hidden_layers
        self.hidden_features = hidden_features
        self.dropout = dropout

        params = torch.tensor(0.1, requires_grad=True)
        params_u = torch.tensor(0.1, requires_grad=True)
        params_zu_u = torch.tensor(0.1, requires_grad=True)
        self.params = nn.Parameter(params)
        self.params_u = nn.Parameter(params_u)
        self.params_zu_u = nn.Parameter(params_zu_u)

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init),
                         'lrelu': (nn.LeakyReLU(inplace=True), init_weights_normal, None),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
                         'tanh': (nn.Tanh(), init_weights_xavier, None),
                         'selu': (nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus': (nn.Softplus(), init_weights_normal, None),
                         'elu': (nn.ELU(inplace=True), init_weights_elu, None)}

        self.nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        self.nl_u, _, _ = nls_and_inits['relu']

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        # non-convex layer
        u_sizes = zip([in_features - 1] * 1 + [hidden_features] * (num_hidden_layers), [hidden_features] *
                      num_hidden_layers)
        # self.net_u = nn.ModuleList([nn.Sequential(
        #     nn.Linear(in_features, out_features, bias=True), nn.Dropout(dropout), self.nl)
        #     for (in_features, out_features) in u_sizes])

        self.net_u = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features, out_features, bias=True), nn.Dropout(dropout))
            for (in_features, out_features) in u_sizes])

        zu_u_sizes = zip([hidden_features] * (num_hidden_layers),
                         [hidden_features] * num_hidden_layers)
        # self.net_zu_u = nn.ModuleList([nn.Sequential(nn.Linear(in_features, out_features, bias=True),
        #                                              nn.Dropout(dropout), self.nl)
        #                                for (in_features, out_features) in zu_u_sizes])
        self.net_zu_u = nn.ModuleList([nn.Sequential(nn.Linear(in_features, out_features, bias=True),
                                                     nn.Dropout(dropout))
                                       for (in_features, out_features) in zu_u_sizes])

        z_zu_sizes = zip([hidden_features] * (num_hidden_layers), [hidden_features] *
                         (num_hidden_layers - 1) + [1])
        self.net_z_zu = nn.ModuleList([nn.Sequential(nn.Linear(in_features, out_features, bias=False),
                                                     nn.Dropout(dropout)) for
                                       (in_features, out_features) in z_zu_sizes])

        yu_u_sizes = [in_features - 1] * 1 + [hidden_features] * (num_hidden_layers)
        self.net_yu_u = nn.ModuleList([nn.Sequential(nn.Linear(in_features, 1, bias=True), nn.Dropout(dropout)) for
                                       in_features in yu_u_sizes])

        z_yu_sizes = [hidden_features] * (num_hidden_layers) + [1]
        self.net_z_yu = nn.ModuleList([nn.Sequential(nn.Linear(1, out_features, bias=False), nn.Dropout(dropout))
                                       for out_features in z_yu_sizes])

        z_u_sizes = zip([in_features - 1] * 1 + [hidden_features] * (num_hidden_layers), [hidden_features] *
                        (num_hidden_layers) + [1])
        self.net_z_u = nn.ModuleList([nn.Sequential(nn.Linear(in_features, out_features, bias=True),
                                                    nn.Dropout(dropout)) for (in_features, out_features) in z_u_sizes])

        # self.final_layer = nn.Linear(hidden_features, 1, bias=False)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        """
        new add
        """

        y_input = coords[..., -1:]
        u_input = coords[..., :-1]
        z_input = self.net_z_u[0](u_input) + self.net_z_yu[0](torch.mul(y_input, self.net_yu_u[0](u_input)))

        z_input = self.nl(10 * self.params * z_input)
        # u_input = self.net_u[0](u_input)
        u_input = self.net_u[0](u_input)
        # u_input = self.nl(10 * self.params_u * u_input)
        u_input = self.nl_u(10 * self.params_u * u_input)

        for i in range(1, self.num_hidden_layers + 1):
            z_input = self.net_z_zu[i - 1](torch.mul(z_input,
                                                     self.nl(10 * self.params_zu_u * self.net_zu_u[i - 1](u_input)))) + \
                      self.net_z_u[i](u_input) + self.net_z_yu[i](torch.mul(y_input, self.net_yu_u[i](u_input)))
            # z_input = self.net_z_zu[i - 1](torch.mul(z_input, self.net_zu_u[i - 1](u_input))) + \
            #           self.net_z_u[i](u_input) + self.net_z_yu[i](torch.mul(y_input, self.net_yu_u[i](u_input)))
            if i == self.num_hidden_layers:
                output = z_input  # no activation needed for final layer
                break
            z_output = self.nl(10 * self.params * z_input)
            u_input = self.net_u[i](u_input)
            # u_input = self.nl(10 * self.params_u * u_input)
            u_input = self.nl_u(10 * self.params_u * u_input)

            z_input = z_output

        # output = self.final_layer(z_input)
        # return z
        return output


class SingleBVPNet(nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, dropout=0.03, **kwargs):
        super().__init__()
        self.mode = mode
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type, dropout=dropout)
        print(self)

        # always convexify after init
        self.make_cvx()

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input.clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords)
        return {'model_in': coords_org, 'model_out': output}

    def convexify(self):
        with torch.no_grad():
            for layer in self.net.net_z_zu:
                for sublayer in layer:
                    if isinstance(sublayer, nn.Linear):
                        sublayer.weight.clamp_(0)

    def make_cvx(self):
        with torch.no_grad():
            for layer in self.net.net_z_zu:
                for sublayer in layer:
                    if isinstance(sublayer, nn.Linear):
                        sublayer.weight.abs_()


########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def sine_init_output(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


