import os
from collections import defaultdict
from typing import Dict, List

import gym
import imageio
import numpy as np
import wandb


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate_with_trajectories(
    policy_fn, env: gym.Env, num_episodes: int
) -> Dict[str, float]:
    trajectories = []
    stats = defaultdict(list)

    for _ in range(num_episodes):
        trajectory = defaultdict(list)
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        while not done:
            action = policy_fn(observation)
            next_observation, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=r,
                done=done,
                info=info,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories


def compose_frames(
    all_frames: List[np.ndarray],
    num_videos_per_row: int,
    margin: int = 4,
):
    num_episodes = len(all_frames)

    if num_videos_per_row is None:
        num_videos_per_row = num_episodes

    t = 0
    end_of_all_epidoes = False
    frames_to_save = []
    while not end_of_all_epidoes:
        frames_t = []

        for i in range(num_episodes):
            # If the episode is shorter, repeat the last frame.
            t_ = min(t, len(all_frames[i]) - 1)
            frame_i_t = all_frames[i][t_]

            # Add the lines.
            frame_i_t = np.pad(
                frame_i_t,
                [[margin, margin], [margin, margin], [0, 0]],
                "constant",
                constant_values=0,
            )

            frames_t.append(frame_i_t)

        # Arrange the videos based on num_videos_per_row.
        frame_t = None
        while len(frames_t) >= num_videos_per_row:
            frames_t_this_row = frames_t[:num_videos_per_row]
            frames_t = frames_t[num_videos_per_row:]

            frame_t_this_row = np.concatenate(frames_t_this_row, axis=1)
            if frame_t is None:
                frame_t = frame_t_this_row
            else:
                frame_t = np.concatenate([frame_t, frame_t_this_row], axis=0)

        frames_to_save.append(frame_t)
        t += 1
        end_of_all_epidoes = all([len(all_frames[i]) <= t for i in range(num_episodes)])

    return frames_to_save


class VideoRecorder(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        save_folder: str = "",
        save_prefix: str = None,
        height: int = 128,
        width: int = 128,
        fps: int = 30,
        camera_id: int = 0,
        goal_conditioned: bool = False,
    ):
        super().__init__(env)

        self.save_folder = save_folder
        self.save_prefix = save_prefix
        self.height = height
        self.width = width
        self.fps = fps
        self.camera_id = camera_id
        self.frames = []
        self.goal_conditioned = goal_conditioned

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        self.num_record_episodes = -1

        self.num_videos = 0

        # self.all_save_paths = None
        self.current_save_path = None

    def start_recording(self, num_episodes: int = None, num_videos_per_row: int = None):
        if num_videos_per_row is not None and num_episodes is not None:
            assert num_episodes >= num_videos_per_row

        self.num_record_episodes = num_episodes
        self.num_videos_per_row = num_videos_per_row

        # self.all_save_paths = []
        self.all_frames = []

    def stop_recording(self):
        self.num_record_episodes = None

    def step(self, action: np.ndarray):  # NOQA

        if self.num_record_episodes is None or self.num_record_episodes == 0:
            observation, reward, terminated, truncated, info = self.env.step(action)

        elif self.num_record_episodes > 0:
            # frame = self.env.render(
            #     height=self.height, width=self.width, camera_id=self.camera_id
            # )
            frame = None
            if frame is None:
                try:
                    frame = self.sim.render(
                        width=self.width, height=self.height, mode="offscreen"
                    )
                    frame = np.flipud(frame)
                except Exception:
                    raise NotImplementedError("Rendering is not implemented.")

            self.frames.append(frame.astype(np.uint8))

            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                if self.goal_conditioned:
                    frames = [
                        np.concatenate([self.env.current_goal["image"], frame], axis=0)
                        for frame in self.frames
                    ]
                else:
                    frames = self.frames

                self.all_frames.append(frames)
                self.num_record_episodes -= 1

                if self.num_record_episodes == 0:
                    # Plot all episodes in one file.
                    frames_to_save = compose_frames(
                        self.all_frames, self.num_videos_per_row
                    )
                else:
                    frames_to_save = frames

                filename = "%08d.mp4" % (self.num_videos)
                if self.save_prefix is not None and self.save_prefix != "":
                    filename = f"{self.save_prefix}_{filename}"
                self.current_save_path = os.path.join(self.save_folder, filename)
                os.makedirs(os.path.dirname(self.current_save_path), exist_ok=True)

                with open(self.current_save_path, "wb") as f:
                    imageio.mimsave(f, frames_to_save, "MP4", fps=self.fps)

                self.num_videos += 1

                self.frames = []

        else:
            raise ValueError("Do not forget to call start_recording.")

        return observation, reward, terminated, truncated, info


def load_recorded_video(
    video_path: str,
):
    with open(video_path, "rb") as f:
        video = np.array(imageio.mimread(f, "MP4")).transpose((0, 3, 1, 2))
        assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"

    return wandb.Video(video, fps=20)


def _recursive_flatten_dict(d: dict):
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values


class WandBLogger(object):
    def log(self, data: dict, step: int = None):
        data_flat = _recursive_flatten_dict(data)
        data = {k: v for k, v in zip(*data_flat)}
        wandb.log(data, step=step)
