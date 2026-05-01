
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def tie_weights(src, trg):
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(module.weight.data, gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    layers = []
    if hidden_depth == 0:
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)])
        for _ in range(hidden_depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
        layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size = 1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for transform in self.transforms:
            mu = transform(mu)
        return mu


def save_numpy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
