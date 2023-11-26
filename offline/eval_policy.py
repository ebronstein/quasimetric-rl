from typing import *

import argparse
import os
import torch
from omegaconf import OmegaConf, SCMode
import yaml

from quasimetric_rl.data import Dataset
from quasimetric_rl.modules import QRLAgent, QRLConf

import glob

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.backends.cudnn
import torch.multiprocessing


from .eval_utils import (
    evaluate_with_trajectories,
    load_recorded_video,
    WandBLogger,
)
from .d4rl import make_d4rl_env

import wandb

import argparse
import os
import re
import yaml
from typing import List

from omegaconf import OmegaConf, SCMode

from .main import Conf


def _is_exp_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(
        os.path.join(path, "config.yaml")
    )


def _get_exp_paths(paths: List[str]) -> List[str]:
    # Walk through the paths and find the experiment directories.
    exp_paths = []
    for path in paths:
        if _is_exp_dir(path):
            exp_paths.append(path)
        else:
            for root, dirs, _ in os.walk(path):
                for dir in dirs:
                    subdir = os.path.join(root, dir)
                    if _is_exp_dir(subdir):
                        exp_paths.append(subdir)
    return exp_paths


def _get_checkpoint_dict(dir: str, verbose: bool = True):
    ckpts = {}  # (epoch, iter) -> path
    ckpt_paths = glob.glob(os.path.join(glob.escape(dir), "checkpoint_*.pth"))
    if verbose:
        print(f"loading checkpoints from {ckpt_paths}")

    for ckpt in sorted(ckpt_paths):
        epoch, it = os.path.basename(ckpt).rsplit(".", 1)[0].split("_")[1:3]
        epoch, it = int(epoch), int(it)
        ckpts[epoch, it] = ckpt

    if verbose:
        print("ckpts: ", ckpts)

    return ckpts


def main(
    paths: List[str],
    regex: str,
    num_episodes: int,
    goal_type: str,
    save_video: bool,
    wandb_project: str,
    dry_run: bool,
):
    if goal_type != "fixed":
        raise NotImplementedError()

    exp_paths = _get_exp_paths(paths)
    if not exp_paths:
        raise ValueError(f"No experiment directories found in {paths}")

    # Filter exp_paths by regex.
    if regex:
        regex = re.compile(regex)
        exp_paths = list(filter(regex.search, exp_paths))
        if not exp_paths:
            raise ValueError(
                f"No experiment directories found in {paths} matching {regex}"
            )

    if dry_run:
        print("Dry-run mode. Will not save anything.")
        print("Experiment directories:")
        for exp_dir in exp_paths:
            print(f"  {exp_dir}")
        return

    for exp_dir in exp_paths:
        eval(exp_dir, num_episodes, goal_type, save_video, wandb_project)


def eval(
    expr_dir: str,
    num_episodes: int,
    goal_type: str,
    save_video: bool,
    wandb_project: str,
):
    config_path = os.path.join(expr_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"config file not found: {config_path}. Skipping.")
        return

    with open(config_path) as f:
        conf = OmegaConf.create(yaml.safe_load(f))

    wandb.init(
        project=wandb_project,
        sync_tensorboard=False,
    )

    wandb_logger = WandBLogger()

    # 1. How to create env
    dataset: Dataset = Dataset.Conf(
        kind=conf.env.kind, name=conf.env.name
    ).make(
        dummy=True
    )  # dummy: don't load data
    env = make_d4rl_env(
        conf.env.name,
        save_video=save_video,
        save_video_dir=os.path.join(conf.output_dir, "videos"),
        save_video_prefix="eval",
    )

    # 2. How to re-create QRL agent
    agent_conf: QRLConf = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.structured(QRLConf()), conf.agent
        ),  # overwrite with loaded conf
        structured_config_mode=SCMode.INSTANTIATE,  # create the object
    )
    agent: QRLAgent = agent_conf.make(
        env_spec=dataset.env_spec, total_optim_steps=1
    )[0]
    device_str = f"{conf.device.type}:{conf.device.index}"
    agent = agent.to(device_str)

    ckpts = _get_checkpoint_dict(expr_dir)

    for (epoch, it), checkpoint in tqdm(ckpts.items()):
        print("Evaluating epoch", epoch, "iter", it)

        # 3. Load checkpoint
        agent.load_state_dict(torch.load(checkpoint)["agent"])
        actor = agent.actor
        if actor is None:
            print(f"Actor is None for checkpoint {checkpoint}. Skipping.")
            continue

        actor.eval()

        # evaluate
        if save_video:
            env.start_recording(
                num_episodes=8,
                num_videos_per_row=4,
            )

        @torch.no_grad()
        def policy_fn(
            obs: torch.Tensor,
            goal: torch.Tensor,
            agent=agent,
            device=device_str,
        ):
            obs = torch.from_numpy(obs).type(torch.float32)
            goal = torch.from_numpy(goal).type(torch.float32)
            with agent.mode(False):
                adistn = agent.actor(
                    obs[None].to(device), goal[None].to(device)
                )
            return adistn.mode.cpu().numpy()[0]

        stats, trajs = evaluate_with_trajectories(
            policy_fn=policy_fn,
            env=env,
            num_episodes=num_episodes,
            goal_conditioned=True,
        )

        # Metrics
        metrics = {}
        rewards = [np.sum(t["reward"]) for t in trajs]
        normalized_rewards = [
            env.get_normalized_score(np.sum(t["reward"])) for t in trajs
        ]
        metrics["average_return"] = np.mean(rewards)
        metrics["median_return"] = np.median(rewards)
        metrics["std_return"] = np.std(rewards)
        metrics["average_normalized_return"] = np.mean(normalized_rewards)
        metrics["median_normalized_return"] = np.median(normalized_rewards)
        metrics["std_normalized_return"] = np.std(normalized_rewards)
        metrics["average_traj_length"] = np.mean(
            [len(t["reward"]) for t in trajs]
        )
        metrics["pred actions"] = np.concatenate([t["action"] for t in trajs])
        wandb_logger.log({"evaluation": metrics}, step=epoch)

        if save_video:
            eval_video = load_recorded_video(video_path=env.current_save_path)
            wandb_logger.log({"evaluation/video": eval_video}, step=epoch)

        # TODO
        # Save metrics to file
        # eval_dir = os.path.join(expr_dir, "eval", f"goal_{goal_type}")
        # if not os.path.exists(eval_dir):
        #     os.makedirs(eval_dir)
        # metrics_path = os.path.join(eval_dir, f"metrics_{epoch}_{it}.csv")
        # save_metrics_to_csv(
        #     metrics, metrics_path, exclude_keys=["pred actions"]
        # )


if __name__ == "__main__":
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to the experiment directories. Can be individual experiments or directories containing multiple experiments.",
    )
    parser.add_argument(
        "-r",
        "--regex",
        default="",
        help="Regex pattern for experiment directories.",
    )
    parser.add_argument(
        "-n",
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "-g",
        "--goal",
        choices=["random", "fixed"],
        default="fixed",
        help="Goal type.",
    )
    parser.add_argument(
        "-p",
        "--wandb_project",
        type=str,
        default="ee227b-project-fa23",
        help="wandb project name.",
    )
    parser.add_argument(
        "-v",
        "--save_video",
        action="store_true",
        help="Save video.",
    )
    parser.add_argument(
        "-d",
        "--dry_run",
        action="store_true",
        help="Dry run.",
    )

    args = parser.parse_args()

    paths = args.paths
    regex = args.regex
    num_episodes = args.num_episodes
    goal_type = args.goal
    save_video = args.save_video
    wandb_project = args.wandb_project
    dry_run = args.dry_run

    main(
        paths,
        regex,
        num_episodes,
        goal_type,
        save_video,
        wandb_project,
        dry_run,
    )
