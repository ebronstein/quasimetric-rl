import os
import logging
from typing import *

import torch
import torch.nn as nn
import numpy as np

from quasimetric_rl import utils


class MLP(nn.Module):
    input_size: int
    output_size: int
    zero_init_last_fc: bool
    module: nn.Sequential

    def __init__(self,
                 input_size: int,
                 *,
                 hidden_sizes: Collection[int],
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 zero_init_last_fc: bool = False):
        super().__init__()
        self.input_size = input_size
        self.zero_init_last_fc = zero_init_last_fc

        layer_in_size = input_size
        modules: List[nn.Module] = []
        for sz in hidden_sizes:
            modules.extend([
                nn.Linear(layer_in_size, sz),
                activation_fn(),
            ])
            layer_in_size = sz

        # initialize with glorot_uniform
        with torch.no_grad():
            def init_(m: nn.Module):
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            for m in modules:
                m.apply(init_)
            if zero_init_last_fc:
                last_fc = cast(nn.Linear, modules[-1])
                last_fc.weight.zero_()
                last_fc.bias.zero_()

        self.module = torch.jit.script(nn.Sequential(*modules))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.module(input)

    # for type hints
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return super().__call__(input)


class Policy(nn.Module):
    def __init__(self, state_size, device):
        super().__init__()
        action_dim = 2
        self.mlp = MLP(state_size, hidden_sizes=(256, 256), activation_fn=nn.Tanh)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.device = device
        self.to(device)

    def forward(self, obs) -> torch.Tensor:
        
        if type(obs) == np.ndarray:
            obs = torch.from_numpy(obs).float().to(self.device)
        
        hidden = self.mlp(obs)
        
        ac_mean = self.mean(hidden)
        ac_mean = nn.Tanh()(ac_mean)
        ac_logstd = self.log_std(hidden)
        ac_logstd = nn.Tanh()(ac_logstd)
        ac_std = torch.exp(ac_logstd)
        
        ac_dist = torch.distributions.Normal(ac_mean, ac_std)
        action = ac_dist.sample()
        log_prob = ac_dist.log_prob(action).sum(dim=-1)
        
        # tanh transformation
        action = nn.Tanh()(action)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        
        if type(action) == torch.Tensor:
            action = action.cpu().numpy()
            
        infos = {
            "mean": ac_mean.detach().cpu().numpy(),
            "log_std": ac_logstd.detach().cpu().numpy(),
            "std": ac_std.detach().cpu().numpy(),
        }

        return action, log_prob, infos


def save_policy(policy, output_dir, epoch, iter):
    utils.mkdir(output_dir)
    desc = f"{epoch:05d}_{iter:05d}"
    fullpath = os.path.join(output_dir, f'checkpoint_{desc}.pth')
    torch.save(policy.state_dict(), fullpath)
    relpath = os.path.join('.', os.path.relpath(fullpath, os.path.dirname(__file__)))
    logging.info(f"Checkpointed to {relpath}")


def load_policy(checkpoint, state_size, device):
    policy = Policy(state_size, device)
    policy.load_state_dict(torch.load(checkpoint))
    policy.to(device)
    return policy
