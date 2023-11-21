from typing import cast
import os
import glob
import attrs
import logging

import hydra
import hydra.types
import hydra.core.config_store
from omegaconf import DictConfig

from tqdm.auto import tqdm
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
import torch.multiprocessing
from scipy.ndimage import zoom
import matplotlib.patches as patches

import quasimetric_rl
from quasimetric_rl import utils, pdb_if_DEBUG, FLAGS
from d4rl.pointmaze.maze_model import WALL

from quasimetric_rl.base_conf import BaseConf
from .trainer import Trainer
from .main import Conf


# @attrs.define(kw_only=True)
# class VizConf(Conf):
#     input_dir: str


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="config", node=Conf())


def get_checkpoint_paths(input_dir: str):
    ckpts = {}  # (epoch, iter) -> path
    ckpt_paths = glob.glob(
        os.path.join(glob.escape(input_dir), "checkpoint_*.pth")
    )
    print(ckpt_paths)
    for ckpt in sorted(ckpt_paths):
        epoch, it = os.path.basename(ckpt).rsplit(".", 1)[0].split("_")[1:3]
        epoch, it = int(epoch), int(it)
        ckpts[epoch, it] = ckpt
    return ckpts


def load(ckpt: str, trainer: Trainer):
    state_dicts = torch.load(ckpt, map_location="cpu")
    trainer.agent.load_state_dict(state_dicts["agent"])
    trainer.losses.load_state_dict(state_dicts["losses"])
    relpath = os.path.join(
        ".", os.path.relpath(ckpt, os.path.dirname(__file__))
    )
    logging.info(f"Loaded from {relpath}")


@pdb_if_DEBUG
@hydra.main(version_base=None, config_name="config")
def visualize_value(dict_cfg: DictConfig):
    cfg: Conf = Conf.from_DictConfig(dict_cfg)
    # Set the output folder to get the output visualization directory.
    cfg.set_output_folder()
    input_dir = os.path.join(cfg.output_base_dir, cfg.output_folder)
    output_viz_dir = os.path.join(input_dir, "value_viz")

    dataset = cfg.env.make()

    # trainer
    dataloader_kwargs = dict(shuffle=True, drop_last=True)
    if cfg.num_workers > 0:
        torch.multiprocessing.set_forkserver_preload(
            ["torch", "quasimetric_rl"]
        )
        dataloader_kwargs.update(
            num_workers=cfg.num_workers,
            multiprocessing_context=torch.multiprocessing.get_context(
                "forkserver"
            ),
            persistent_workers=True,
        )

    device = cfg.device.make()

    if cast(torch.device, device).type == "cuda":
        pin_memory_device = "cuda"  # DataLoader only allows string... lol
        if device.index is not None:
            pin_memory_device += f":{device.index}"
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

    ckpts = get_checkpoint_paths(input_dir)

    # Load the most recent checkpoint.
    start_epoch, start_it = max(ckpts.keys())
    logging.info(
        f"Load from existing checkpoint: {ckpts[start_epoch, start_it]}"
    )
    load(ckpts[start_epoch, start_it], trainer)
    logging.info(f"Fast forward to epoch={start_epoch} iter={start_it}")

    env = gym.make(cfg.env.name)

    if not os.path.exists(output_viz_dir):
        os.makedirs(output_viz_dir)

    for goal_xy in env.env.empty_and_goal_locations:
        # Append zero velocity to the goal.
        goal = torch.tensor(list(goal_xy) + [0, 0])
        output_file_path = os.path.join(
            output_viz_dir, f"goal_{goal_xy[0]},{goal_xy[1]}.png"
        )
        plot_value(env, dataset, trainer, goal, output_file_path, device)


def plot_value(
    env, dataset, trainer, goal, output_file_path, device, num_cells=30
):
    goal_x = goal[0].item()
    goal_y = goal[1].item()

    maze_arr = env.env.maze_arr
    maze_width, maze_height = maze_arr.shape
    min_x, min_y = -0.5, -0.5
    max_x = maze_width - 0.5
    max_y = maze_height - 0.5
    # max_x = max(env.env.empty_and_goal_locations, key=lambda xy: xy[0])[0]
    # max_y = max(env.env.empty_and_goal_locations, key=lambda xy: xy[1])[1]
    x, y = np.meshgrid(
        np.linspace(min_x, max_x, num_cells),
        np.linspace(min_y, max_y, num_cells),
    )

    base_observation = np.copy(
        dataset.get_observations(0).cpu().detach().numpy()
    )
    base_observations = np.tile(base_observation, (x.shape[0], x.shape[1], 1))
    base_observations[:, :, 0] = x
    base_observations[:, :, 1] = y

    base_observations = torch.from_numpy(base_observations).to(
        dtype=torch.float32, device=device
    )
    goal = goal.to(dtype=torch.float32, device=device)
    values = (
        trainer.agent.critics[0](base_observations, goal[None, None])
        .cpu()
        .detach()
        .numpy()
    )

    # TODO: do this based on the maze layout.
    # values[0:-5, 5:-5] = 0
    # values[goal[0], goal[1]] = 100
    fig, ax = plt.subplots(figsize=(8, 8))
    mesh = ax.pcolormesh(x, y, values, cmap="viridis")

    for w in range(maze_width):
        for h in range(maze_height):
            if maze_arr[w, h] == WALL:
                rect = patches.Rectangle(
                    (w - 0.5, h - 0.5),
                    1,
                    1,
                    linewidth=1,
                    edgecolor="black",
                    facecolor="black",
                )
                ax.add_patch(rect)

    ax.plot(goal_x, goal_y, "r*", markersize=10)

    plt.colorbar(mesh)
    plt.savefig(output_file_path)
    plt.close()


if __name__ == "__main__":
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    # set up some hydra flags before parsing
    os.environ["HYDRA_FULL_ERROR"] = str(int(FLAGS.DEBUG))

    visualize_value()
