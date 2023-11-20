from typing import *

import os

import glob
import attrs
import logging
import time

import hydra
import hydra.types
import hydra.core.config_store
from omegaconf import DictConfig

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.backends.cudnn
import torch.multiprocessing

import quasimetric_rl
from quasimetric_rl import utils, pdb_if_DEBUG, FLAGS

from quasimetric_rl.utils.steps_counter import StepsCounter
from quasimetric_rl.modules import InfoT
from quasimetric_rl.base_conf import BaseConf

from .trainer import Trainer


@utils.singleton
@attrs.define(kw_only=True)
class Conf(BaseConf):
    output_base_dir: str = attrs.field(default=os.path.join(os.path.dirname(__file__), 'results'))

    resume_if_possible: bool = True

    env: quasimetric_rl.data.Dataset.Conf = quasimetric_rl.data.Dataset.Conf()

    batch_size: int = attrs.field(default=4096, validator=attrs.validators.gt(0))
    num_workers: int = attrs.field(default=8, validator=attrs.validators.ge(0))
    total_optim_steps: int = attrs.field(default=int(2e5), validator=attrs.validators.gt(0))

    log_steps: int = attrs.field(default=250, validator=attrs.validators.gt(0))
    save_steps: int = attrs.field(default=50000, validator=attrs.validators.gt(0))



cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='config', node=Conf())


@pdb_if_DEBUG
@hydra.main(version_base=None, config_name="config")
def train(dict_cfg: DictConfig):
    print("Train called")
    cfg: Conf = Conf.from_DictConfig(dict_cfg)
    writer = cfg.setup_for_experiment()  # checking & setup logging
    cfg.output_dir = os.path.join(cfg.output_base_dir, 'policy')
    input_dir = os.path.join(cfg.output_base_dir, 'd4rl_maze2d-umaze-v1/iqe(dim=2048,components=64)_dyn=0.1_2critic_seed=60912')
    print(input_dir)

    ckpts = {}  # (epoch, iter) -> path
    ckpt_paths = glob.glob(os.path.join(glob.escape(input_dir), 'checkpoint_*.pth'))
    print(ckpt_paths)
    for ckpt in sorted(ckpt_paths):
        epoch, it = os.path.basename(ckpt).rsplit('.', 1)[0].split('_')[1:3]
        epoch, it = int(epoch), int(it)
        ckpts[epoch, it] = ckpt
    print("ckpts: ", ckpts)

    dataset = cfg.env.make()

    # trainer
    dataloader_kwargs = dict(shuffle=True, drop_last=True)
    if cfg.num_workers > 0:
        torch.multiprocessing.set_forkserver_preload(["torch", "quasimetric_rl"])
        dataloader_kwargs.update(
            num_workers=cfg.num_workers,
            multiprocessing_context=torch.multiprocessing.get_context('forkserver'),
            persistent_workers=True,
        )

    device = cfg.device.make()

    if cast(torch.device, device).type == 'cuda':
        pin_memory_device = 'cuda'  # DataLoader only allows string... lol
        if device.index is not None:
            pin_memory_device += f':{device.index}'
        dataloader_kwargs.update(
            pin_memory=True,
            pin_memory_device=pin_memory_device,
        )

    trainer = Trainer(
        agent_conf=cfg.agent,
        device=cfg.device.make(),
        dataset=dataset,
        batch_size=cfg.batch_size,
        total_optim_steps=cfg.total_optim_steps,
        dataloader_kwargs=dataloader_kwargs,
    )

    # save, load, and resume
    def save(epoch, it, *, suffix=None, extra=dict()):
        desc = f"{epoch:05d}_{it:05d}"
        if suffix is not None:
            desc += f'_{suffix}'
        utils.mkdir(cfg.output_dir)
        fullpath = os.path.join(cfg.output_dir, f'checkpoint_{desc}.pth')
        state_dicts = dict(
            epoch=epoch,
            it=it,
            agent=trainer.agent.state_dict(),
            losses=trainer.losses.state_dict(),
            **extra,
        )
        torch.save(state_dicts, fullpath)
        relpath = os.path.join('.', os.path.relpath(fullpath, os.path.dirname(__file__)))
        logging.info(f"Checkpointed to {relpath}")

    def load(ckpt):
        state_dicts = torch.load(ckpt, map_location='cpu')
        trainer.agent.load_state_dict(state_dicts['agent'])
        trainer.losses.load_state_dict(state_dicts['losses'])
        relpath = os.path.join('.', os.path.relpath(ckpt, os.path.dirname(__file__)))
        logging.info(f"Loaded from {relpath}")

    if cfg.resume_if_possible and len(ckpts) > 0:
        start_epoch, start_it = max(ckpts.keys())
        logging.info(f'Load from existing checkpoint: {ckpts[start_epoch, start_it]}')
        load(ckpts[start_epoch, start_it])
        logging.info(f'Fast forward to epoch={start_epoch} iter={start_it}')
        start_epoch, start_it = 0, 0 # Start training from scratch.
    else:
        start_epoch, start_it = 0, 0

    ### POLICY EXTRACTION CODE GOES HERE.
    # We have access to trainer.
    state1 = dataset.get_observations(0).to(device)
    state2 = dataset.get_observations(5).to(device)
    state3 = dataset.get_observations(10).to(device)
    print("state1: ", state1)
    print("state2: ", state2)
    print("state3: ", state3)
    v0 = trainer.agent.critics[0](state1, state1)
    v1 = trainer.agent.critics[0](state1, state2)
    v2 = trainer.agent.critics[0](state2, state3)
    v3 = trainer.agent.critics[0](state1, state3)
    print("v0: ", v0)
    print("v1: ", v1)
    print("v2: ", v2)
    print("v3: ", v3)

    indicies = np.arange(10000)

    random_states = dataset.get_observations(np.random.randint(dataset.__len__(), size=10000)).numpy()
    random_xy = random_states[:, :2]
    # plot a scatter plot of the random states
    import matplotlib.pyplot as plt
    plt.scatter(random_xy[:, 0], random_xy[:, 1])
    plt.savefig("random_states.png")
    plt.close()


    for i in range(20):
        goal = dataset.get_observations(np.random.randint(dataset.__len__())).to(device)
        import matplotlib.pyplot as plt
        def plot_value():
            xy_array = np.meshgrid(np.linspace(0.5, 3.2, 30), np.linspace(0.5, 3.2, 30))
            x, y = xy_array

            base_observation = np.copy(dataset.get_observations(0).cpu().detach().numpy())
            base_observations = np.tile(base_observation, (x.shape[0], x.shape[1], 1))
            base_observations[:, :, 0] = x
            base_observations[:, :, 1] = y

            base_observations = torch.from_numpy(base_observations).to(device)
            values = trainer.agent.critics[0](base_observations, goal[None, None]).cpu().detach().numpy()

            values[0:-5, 5:-5] = 0
            # values[goal[0], goal[1]] = 100

            mesh = plt.pcolormesh(x, y, values, cmap='viridis')
            plt.colorbar(mesh)
            plt.savefig(f"value{i}.png")
            plt.close()
        plot_value()
    

    import torch.nn as nn
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            action_dim = 2
            self.policy = quasimetric_rl.modules.utils.MLP(len(state1), 2*action_dim, hidden_sizes=(256, 256))

        def forward(self, obs) -> torch.Tensor:
            ac_pred = self.policy(obs)
            ac_mean, ac_logstd = ac_pred.chunk(2, dim=-1)

            return ac_mean, ac_logstd
        
    policy = Policy().to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
    num_total_epochs = 5
    for epoch in range(num_total_epochs):
        epoch_desc = f"Train epoch {epoch:05d}/{num_total_epochs:05d}"
        for it, (data, data_info) in enumerate(tqdm(trainer.iter_training_data(), total=trainer.num_batches, desc=epoch_desc)):
            observations = data.observations
            actions = data.actions
            next_observations = data.next_observations
            future_observations = data.future_observations # goals

            ac_mean, ac_logstd = policy(observations)
            ac_std = torch.exp(ac_logstd)
            ac_dist = torch.distributions.Normal(loc=ac_mean,scale=ac_std)
            log_prob = ac_dist.log_prob(actions).sum(dim=-1)

            temperature = 3
            with torch.no_grad():
                v_obs = trainer.agent.critics[0](observations, future_observations)
                v_next_obs = trainer.agent.critics[0](next_observations, future_observations)
                adv = v_next_obs - v_obs
                exp_adv = torch.exp(adv * temperature)
                exp_adv = torch.min(exp_adv, torch.ones_like(exp_adv) * 100)

            # print("Computed ac_mean", ac_mean[0])
            # print("Computed ac_logstd", ac_logstd[0])
            # print("Computed ac_std", ac_std[0])
            # print("True action", actions[0])
            # print("Computed log_prob", log_prob[0])
            # print("Computed v_obs", v_obs[0])
            # print("Computed v_next_obs", v_next_obs[0])
            # print("Computed adv", adv[0])

            scaled_log_prob = log_prob * exp_adv
            loss = -scaled_log_prob.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            print("loss: ", loss.item())



if __name__ == '__main__':
    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'

    # set up some hydra flags before parsing
    os.environ['HYDRA_FULL_ERROR'] = str(int(FLAGS.DEBUG))

    train()
