import os
import logging

import torch
import torch.nn as nn
import numpy as np

from quasimetric_rl.modules.utils import MLP
from quasimetric_rl import utils


class Policy(nn.Module):
    def __init__(self, state_size, device):
        super().__init__()
        action_dim = 2
        self.policy = MLP(state_size, 2*action_dim, hidden_sizes=(256, 256))
        self.device = device
        self.to(device)

    def forward(self, obs) -> torch.Tensor:
        
        if type(obs) == np.ndarray:
            obs = torch.from_numpy(obs).float().to(self.device)
        
        ac_pred = self.policy(obs)
        ac_mean, ac_logstd = ac_pred.chunk(2, dim=-1)

        return ac_mean, ac_logstd
    
    def get_action(self, obs) -> np.ndarray:
        ac_mean, ac_logstd = self.forward(obs)
        ac_std = torch.exp(ac_logstd)
        ac_dist = torch.distributions.Normal(loc=ac_mean,scale=ac_std)
        action = ac_dist.sample()
        
        if type(action) == torch.Tensor:
            action = action.cpu().numpy()
        
        return action


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
