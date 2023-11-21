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
from quasimetric_rl.base_conf import BaseConf

from .trainer import Trainer
from .policy import Policy, save_policy, load_policy
from .eval_utils import evaluate_with_trajectories, load_recorded_video, WandBLogger
from .d4rl import make_d4rl_env

import wandb
wandb.init(
    project="convex-final", 
    sync_tensorboard=False,
)


@utils.singleton
@attrs.define(kw_only=True)
class Conf(BaseConf):
    output_base_dir: str = attrs.field(default=os.path.join(os.path.dirname(__file__), 'results'))

    resume_if_possible: bool = True
    eval_only: bool = False

    env: quasimetric_rl.data.Dataset.Conf = quasimetric_rl.data.Dataset.Conf()

    batch_size: int = attrs.field(default=4096, validator=attrs.validators.gt(0))
    num_workers: int = attrs.field(default=8, validator=attrs.validators.ge(0))
    total_optim_steps: int = attrs.field(default=int(2e5), validator=attrs.validators.gt(0))

    log_steps: int = attrs.field(default=250, validator=attrs.validators.gt(0))
    save_steps: int = attrs.field(default=50000, validator=attrs.validators.gt(0))



cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='config', node=Conf())


def get_checkpoint_dict(dir):
    ckpts = {} # (epoch, iter) -> path
    ckpt_paths = glob.glob(os.path.join(glob.escape(dir), 'checkpoint_*.pth'))
    print(f"loading checkpoints from {ckpt_paths}")
    
    for ckpt in sorted(ckpt_paths):
        epoch, it = os.path.basename(ckpt).rsplit('.', 1)[0].split('_')[1:3]
        epoch, it = int(epoch), int(it)
        ckpts[epoch, it] = ckpt
    print("ckpts: ", ckpts)
    
    return ckpts


@pdb_if_DEBUG
@hydra.main(version_base=None, config_name="config")
def train_and_eval(dict_cfg: DictConfig):
    wandb_logger = WandBLogger()
    
    print("Train called")
    cfg: Conf = Conf.from_DictConfig(dict_cfg)
    writer = cfg.setup_for_experiment()  # checking & setup logging
    cfg.output_dir = os.path.join(cfg.output_base_dir, 'policy')
    
    input_dir = os.path.join(cfg.output_base_dir, 'd4rl_maze2d-umaze-v1/iqe(dim=2048,components=64)_dyn=0.1_2critic_seed=60912')
    print(input_dir)
    ckpts = get_checkpoint_dict(input_dir)

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
    def save_trainer(epoch, it, *, suffix=None, extra=dict()):
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

    def load_trainer(ckpt):
        state_dicts = torch.load(ckpt, map_location='cpu')
        trainer.agent.load_state_dict(state_dicts['agent'])
        trainer.losses.load_state_dict(state_dicts['losses'])
        relpath = os.path.join('.', os.path.relpath(ckpt, os.path.dirname(__file__)))
        logging.info(f"Loaded from {relpath}")

    if cfg.resume_if_possible and len(ckpts) > 0:
        start_epoch, start_it = max(ckpts.keys())
        logging.info(f'Load from existing checkpoint: {ckpts[start_epoch, start_it]}')
        load_trainer(ckpts[start_epoch, start_it])
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
        
    ### learning a policy ###
    if not cfg.eval_only:
        
        policy = Policy(len(state1), device)
        optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
        
        num_total_epochs = 40
        for epoch in tqdm(range(num_total_epochs), total=num_total_epochs):
            for it, (data, data_info) in enumerate(trainer.iter_training_data()):
                observations = data.observations
                actions = data.actions
                next_observations = data.next_observations
                future_observations = data.future_observations # goals

                ac_mean, ac_logstd = policy(observations)
                ac_std = torch.exp(ac_logstd)
                ac_dist = torch.distributions.Normal(loc=ac_mean,scale=ac_std)
                predicted_actions = ac_dist.sample()
                log_prob = ac_dist.log_prob(actions).sum(dim=-1)

                temperature = 1
                with torch.no_grad():
                    v_obs_1 = trainer.agent.critics[0](observations, future_observations)
                    v_obs_2 = trainer.agent.critics[1](observations, future_observations)
                    v_obs = torch.min(v_obs_1, v_obs_2)
                    
                    v_next_obs_1 = trainer.agent.critics[0](next_observations, future_observations)
                    v_next_obs_2 = trainer.agent.critics[1](next_observations, future_observations)
                    v_next_obs = torch.min(v_next_obs_1, v_next_obs_2)
                    
                    adv = v_next_obs - v_obs
                    exp_adv = torch.exp(adv * temperature)
                    exp_adv = torch.min(exp_adv, torch.ones_like(exp_adv) * 100)

                scaled_log_prob = log_prob * exp_adv
                loss = -scaled_log_prob.mean()
                optim.zero_grad()
                loss.backward()
                optim.step()

                training_metrics = {
                    "ac_mean": ac_mean.mean(axis=0),
                    "ac_logstd": ac_logstd.mean(axis=0),
                    "ac_std": ac_std.mean(axis=0),
                    "dataset actions": actions,
                    "predicted actions": predicted_actions,
                    "log_prob": log_prob.mean(),
                    "v_obs": v_obs.mean(),
                    "v_next_obs": v_next_obs.mean(),
                    "adv": adv,
                    "loss": loss.item(),
                }
                wandb.log(training_metrics, step=it)
        
            ### save the learned policy ###
            save_policy(policy, cfg.output_dir, epoch, it)
    
    ### evaluate the learned policy ###
    eval(cfg, state_size=len(state1), device=device, wandb_logger=wandb_logger)

    
def eval(cfg, state_size, device, wandb_logger):
    # make env
    env_name = cfg.env.name
    save_video = True
    env = make_d4rl_env(
        env_name,
        save_video=save_video,
        save_video_dir=os.path.join(cfg.output_dir, 'videos'),
        save_video_prefix='eval',
    )
    
    # load the learned policy
    checkpoint_dict = get_checkpoint_dict(cfg.output_dir)  
    print(f"found policy checkpoints: {checkpoint_dict}")
    checkpoint = checkpoint_dict[max(checkpoint_dict.keys())]  # there is only one checkpoint
    
    for epoch_iter in sorted(checkpoint_dict.keys()):
        
        checkpoint = checkpoint_dict[epoch_iter]
        epoch, iter = epoch_iter
    
        policy = load_policy(checkpoint, state_size, device)
        policy.eval()
        
        # evaluate
        if save_video:
            env.start_recording(
                num_episodes=8,
                num_videos_per_row=4,
            )
        stats, trajs = evaluate_with_trajectories(
            policy_fn=policy.get_action,
            env=env,
            num_episodes=10,
        )
        
        metrics = {}
        metrics['average_return'] = np.mean([np.sum(t['reward']) for t in trajs])
        metrics['average_traj_length'] = np.mean([len(t['reward']) for t in trajs])
        metrics['average_normalizd_return'] = np.mean(
            [env.get_normalized_score(np.sum(t['reward'])) for t in trajs]
        )
        print(epoch, metrics)
        wandb_logger.log({"evaluation": metrics}, step=epoch)
        
        if save_video:
            eval_video = load_recorded_video(video_path=env.current_save_path)
            wandb_logger.log({"evaluation/video": eval_video}, step=epoch)


if __name__ == '__main__':
    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'

    # set up some hydra flags before parsing
    os.environ['HYDRA_FULL_ERROR'] = str(int(FLAGS.DEBUG))

    train_and_eval()
