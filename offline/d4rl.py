import d4rl
import numpy as np
import gym

from .eval_utils import VideoRecorder


class TruncationWrapper(gym.Wrapper):
    """d4rl only supports the old gym API, where env.step returns a 4-tuple without
    the truncated signal. Here we explicity expose the truncated signal."""

    def __init__(self, env):
        super().__init__(env)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        s = self.env.reset()
        return s, {}

    def step(self, a):
        s, r, done, info = self.env.step(a)
        truncated = info.get("TimeLimit.truncated", False)
        return s, r, done, truncated, info
    

def get_d4rl_dataset(env):
    dataset = d4rl.qlearning_dataset(env)
    return dict(
        observations=dataset["observations"],
        actions=dataset["actions"],
        next_observations=dataset["next_observations"],
        rewards=dataset["rewards"],
        dones=dataset["terminals"].astype(np.float32),
    )
    
def make_d4rl_env(
    env_name: str,
    save_video: bool,
    save_video_dir: str,
    save_video_prefix: str,
    ):
    
    env = gym.make(env_name)
    env = TruncationWrapper(env)
    if save_video:
        env = VideoRecorder(
            env,
            save_folder=save_video_dir,
            save_prefix=save_video_prefix,
            goal_conditioned=False,
        )

    return env
