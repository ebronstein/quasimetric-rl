from __future__ import annotations
from typing import *

import logging
import functools
import os
import collections
from tqdm.auto import tqdm
import argparse

import numpy as np
import torch.utils.data
import gym
import gym.spaces

import abc
import attrs

if TYPE_CHECKING:
    import d4rl.pointmaze


def parser():
    parser = argparse.ArgumentParser(description='Input info for getting maze info')
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v1', 
                        choices=['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1'])
    args = parser.parse_args()
    return args


LOAD_EPISODES_REGISTRY: Mapping[Tuple[str, str], Callable[[], Iterator[EpisodeData]]] = {}
CREATE_ENV_REGISTRY: Mapping[Tuple[str, str], Callable[[], gym.Env]] = {}

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class TensorCollectionAttrsMixin(abc.ABC):
    # All fields must be one of
    #    torch.Tensor
    #    NestedMapping[torch.Tensor]
    #    TensorCollectionAttrsMixin
    #    NestedMapping[TensorCollectionAttrsMixin]

    @classmethod
    def types_dict(cls):
        fields = attrs.fields_dict(cls)
        return {k: t for k, t in get_type_hints(cls).items() if k in fields}

    @staticmethod
    def is_tensor_type(ty):
        try:
            return issubclass(ty, torch.Tensor)
        except TypeError:
            return False

    @staticmethod
    def is_nested_tensor_mapping_type(ty):
        try:
            orig = get_origin(ty)
            args = get_args(ty)
            return (
                (issubclass(orig, NestedMapping) and issubclass(args)[0], torch.Tensor)
                or
                (issubclass(orig, Mapping) and args[0] == str and issubclass(args, torch.Tensor))
            )
        except TypeError:
            return False

    @staticmethod
    def is_tensor_collection_attrs_type(ty):
        try:
            return issubclass(ty, TensorCollectionAttrsMixin)
        except TypeError:
            return False

    @classmethod
    def cat(cls, collections: List[Self], *, dim=0) -> Self:
        assert all(isinstance(c, cls) for c in collections)

        if len(collections) == 1:  # differ from torch.cat: no copy
            return collections[0]

        types = cls.types_dict()

        def cat_key(k: str):
            ty = types[k]
            field_values = [getattr(c, k) for c in collections]
            if cls.is_tensor_type(ty):
                # torch.Tensor
                return torch.cat(field_values, dim=dim)  # differ from torch.cat: no copy if len == 1
            elif cls.is_nested_tensor_mapping_type(ty):
                # NestedMapping[torch.Tensor]

                def cat_map(maps: List[NestedMapping[torch.Tensor]]) -> NestedMapping[torch.Tensor]:
                    if len(maps) == 0:
                        return {}

                    def get_tensor_flags(map: NestedMapping[torch.Tensor]):
                        return {map_k: isinstance(map_v, torch.Tensor) for map_k, map_v in map.items()}

                    tensor_flags = get_tensor_flags(maps[0])

                    return {
                        map_k: (
                            torch.cat([m[map_k] for m in maps], dim=dim) if is_tensor else cat_map([m[map_k] for m in maps])
                        ) for map_k, is_tensor in tensor_flags.items()
                    }

                return cat_map(field_values)
            elif cls.is_tensor_collection_attrs_type(ty):
                # TensorCollectionAttrsMixin
                return cast(Type[TensorCollectionAttrsMixin], ty).cat(field_values, dim=dim)
            else:
                # NestedMapping[TensorCollectionAttrsMixin]

                coll_ty: Type[TensorCollectionAttrsMixin] = get_args(ty)[0]

                def cat_map(maps: List[NestedMapping[TensorCollectionAttrsMixin]]) -> NestedMapping[TensorCollectionAttrsMixin]:
                    if len(maps) == 0:
                        return {}

                    def get_coll_flags(map: NestedMapping[TensorCollectionAttrsMixin]):
                        return {map_k: isinstance(map_v, TensorCollectionAttrsMixin) for map_k, map_v in map.items()}

                    coll_flags = get_coll_flags(maps[0])

                    return {
                        map_k: (
                            coll_ty.cat([m[map_k] for m in maps], dim=dim) if is_coll else cat_map([m[map_k] for m in maps])
                        ) for map_k, is_coll in coll_flags.items()
                    }

                return cat_map(field_values)

        return cls(**{k: cat_key(k) for k in types.keys()})

    @staticmethod
    def _make_cvt_fn(elem_cvt_fn: Callable[[Union[torch.Tensor, TensorCollectionAttrsMixin]], Union[torch.Tensor, TensorCollectionAttrsMixin]]):
        def cvt_fn(x: FieldT) -> FieldT:
            if isinstance(x, (torch.Tensor, TensorCollectionAttrsMixin)):
                return elem_cvt_fn(x)
            else:
                return {k: cvt_fn(v) for k, v in x.items()}
        return cvt_fn

    def to(self, *args, **kwargs) -> Self:
        cvt_fn = self._make_cvt_fn(lambda x: x.to(*args, **kwargs))
        return self.__class__(
            **{k: cvt_fn(v) for k, v in attrs.asdict(self, recurse=False).items()}
        )

    def flatten(self, *args, **kwargs) -> Self:
        cvt_fn = self._make_cvt_fn(lambda x: x.flatten(*args, **kwargs))
        return self.__class__(
            **{k: cvt_fn(v) for k, v in attrs.asdict(self, recurse=False).items()}
        )

    def unflatten(self, *args, **kwargs) -> Self:
        cvt_fn = self._make_cvt_fn(lambda x: x.unflatten(*args, **kwargs))
        return self.__class__(
            **{k: cvt_fn(v) for k, v in attrs.asdict(self, recurse=False).items()}
        )

    def narrow(self, *args, **kwargs) -> Self:
        cvt_fn = self._make_cvt_fn(lambda x: x.narrow(*args, **kwargs))
        return self.__class__(
            **{k: cvt_fn(v) for k, v in attrs.asdict(self, recurse=False).items()}
        )

    def __getitem__(self, *args, **kwargs) -> Self:
        cvt_fn = self._make_cvt_fn(lambda x: x.__getitem__(*args, **kwargs))
        return self.__class__(
            **{k: cvt_fn(v) for k, v in attrs.asdict(self, recurse=False).items()}
        )

    def pin_memory(self, *args, **kwargs) -> Self:
        cvt_fn = self._make_cvt_fn(lambda x: x.pin_memory(*args, **kwargs))
        return self.__class__(
            **{k: cvt_fn(v) for k, v in attrs.asdict(self, recurse=False).items()}
        )


@attrs.define(kw_only=True)
class MultiEpisodeData(TensorCollectionAttrsMixin):
    r"""
    The DATASET of MULTIPLE episodes
    """


    # For each episode, L: number of (s, a, s', r, d, to) pairs, so number of transitions (not observations)
    episode_lengths: torch.Tensor
    # cat all states from all episodes, where the last s' is added. I.e., each episode has L+1 states
    all_observations: torch.Tensor
    # cat all actions from all episodes. Each episode has L actions.
    actions: torch.Tensor
    # cat all rewards from all episodes. Each episode has L rewards.
    rewards: torch.Tensor
    # cat all terminals from all episodes. Each episode has L terminals.
    terminals: torch.Tensor
    # cat all timeouts from all episodes. Each episode has L timeouts.
    timeouts: torch.Tensor
    # cat all observation infos from all episodes. Each episode has L + 1 elements.
    observation_infos: Mapping[str, torch.Tensor] = attrs.Factory(dict)
    # cat all transition infos from all episodes. Each episode has L elements.
    transition_infos: Mapping[str, torch.Tensor] = attrs.Factory(dict)

    @property
    def num_episodes(self) -> int:
        return self.episode_lengths.shape[0]

    @property
    def num_transitions(self) -> int:
        return self.rewards.shape[0]

    def __attrs_post_init__(self):
        assert self.episode_lengths.ndim == 1
        N = self.num_transitions
        assert N > 0
        assert self.all_observations.ndim >= 1 and self.all_observations.shape[0] == (N + self.num_episodes), self.all_observations.shape
        assert self.actions.ndim >= 1 and self.actions.shape[0] == N
        assert self.rewards.ndim == 1 and self.rewards.shape[0] == N
        assert self.terminals.ndim == 1 and self.terminals.shape[0] == N
        assert self.timeouts.ndim == 1 and self.timeouts.shape[0] == N
        for k, v in self.observation_infos.items():
            assert v.shape[0] == N + self.num_episodes, k
        for k, v in self.transition_infos.items():
            assert v.shape[0] == N, k



@attrs.define(kw_only=True)
class EpisodeData(MultiEpisodeData):
    r"""
    A SINGLE episode
    """

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.num_episodes == 1

    @classmethod
    def from_simple_trajectory(cls,
                               observations: Union[np.ndarray, torch.Tensor],
                               actions: Union[np.ndarray, torch.Tensor],
                               next_observations: Union[np.ndarray, torch.Tensor],
                               rewards: Union[np.ndarray, torch.Tensor],
                               terminals: Union[np.ndarray, torch.Tensor],
                               timeouts: Union[np.ndarray, torch.Tensor]):
        observations = torch.tensor(observations)
        next_observations=torch.tensor(next_observations)
        all_observations = torch.cat([observations, next_observations[-1:]], dim=0)
        return cls(
            episode_length=torch.tensor([observations.shape[0]]),
            all_observations=all_observations,
            actions=torch.tensor(actions),
            rewards=torch.tensor(rewards),
            terminals=torch.tensor(terminals),
            timeouts=torch.tensor(timeouts),
        )


def register_offline_env(kind: str, spec: str, *, load_episodes_fn, create_env_fn):
    r"""
    Each specific env (e.g., an offline env from d4rl) just needs to register

        1. how to load the episodes
        (this is optional in online settings. see ReplayBuffer)

        load_episodes_fn() -> Iterator[EpisodeData]

        2. how to create an env

        create_env_fn() -> gym.Env

     See d4rl/maze2d.py for example
    """
    assert (kind, spec) not in LOAD_EPISODES_REGISTRY
    LOAD_EPISODES_REGISTRY[(kind, spec)] = load_episodes_fn
    CREATE_ENV_REGISTRY[(kind, spec)] = create_env_fn


d4rl = None
OfflineEnv = None

def lazy_init_d4rl():
    # d4rl requires mujoco_py, which has a range of installation issues.
    # do not load until needed.

    global d4rl, OfflineEnv

    if d4rl is None:
        import importlib
        with suppress_output():
            ## d4rl prints out a variety of warnings
            d4rl = __import__('d4rl')
        OfflineEnv = d4rl.offline_env.OfflineEnv

def load_environment(name: Union[str, gym.Env]) -> 'OfflineEnv':
    lazy_init_d4rl()
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env: gym.Wrapper = gym.make(name)
    env: 'OfflineEnv' = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    env.reset()
    env.step(env.action_space.sample())  # sometimes stepping is needed to initialize internal
    env.reset()
    return env


def sequence_dataset(env: 'OfflineEnv', dataset: Mapping[str, np.ndarray]) -> Generator[Mapping[str, np.ndarray], None, None]:
    """
    Returns an *ordered* iterator through trajectories.
    Args:
        env: `OfflineEnv`
        dataset: `d4rl` dataset with keys:
            observations
            next_observations
            actions
            rewards
            terminals
            timeouts (optional)
            ...
    Returns:
        An iterator through dictionaries with keys:
            all_observations
            actions
            rewards
            terminals
            timeouts (optional)
            ...
    """

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatibility.
    use_timeouts = 'timeouts' in dataset

    all_episodes_data = []
    episode_step = 0
    for i in tqdm(range(N), desc=f"{env.name} dataset timesteps"):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env.max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep or i == N - 1:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            assert 'all_observations' not in episode_data
            episode_data['all_observations'] = np.concatenate(
                [episode_data['observations'], episode_data['next_observations'][-1:]], axis=0)
            all_episodes_data.append(episode_data['all_observations'])
            # yield episode_data
            data_ = collections.defaultdict(list)

        # all_episodes_data.append(episode_data)
        episode_step += 1
    
    return all_episodes_data


def convert_dict_to_EpisodeData_iter(sequence_dataset_episodes: Iterator[Mapping[str, np.ndarray]]):
    for episode in sequence_dataset_episodes:
        episode_dict = dict(
            episode_lengths=torch.as_tensor([len(episode['all_observations']) - 1], dtype=torch.int64),
            all_observations=torch.as_tensor(episode['all_observations'], dtype=torch.float32),
            actions=torch.as_tensor(episode['actions'], dtype=torch.float32),
            rewards=torch.as_tensor(episode['rewards'], dtype=torch.float32),
            terminals=torch.as_tensor(episode['terminals'], dtype=torch.bool),
            timeouts=(
                torch.as_tensor(episode['timeouts'], dtype=torch.bool) if 'timeouts' in episode else
                torch.zeros(episode['terminals'].shape, dtype=torch.bool)
            ),
            observation_infos={},
            transition_infos={},
        )
        for k, v in episode.items():
            if k.startswith('observation_infos/'):
                episode_dict['observation_infos'][k.split('/', 1)[1]] = v
            elif k.startswith('transition_infos/'):
                episode_dict['transition_infos'][k.split('/', 1)[1]] = v
        yield EpisodeData(**episode_dict)

def preprocess_maze2d_fix(env: 'd4rl.pointmaze.MazeEnv', dataset: Mapping[str, np.ndarray]):
    ## In generation, controller is run until reached goal, which is
    ## continuously set.
    ##
    ## There, terminal is always False, and how timeout is set is unknown (not
    ## in the public script)
    ##
    ## Set timeout at time t                      (*)
    ##   iff reached goal at time t
    ##   iff goal at time t != goal at time t+1
    ##
    ## Remove terminals
    ##
    ## Add next_observations
    ##
    ## Also Maze2d *rewards* is field is off-by-one:
    ##    rewards[t] is not the reward received for performing actions[t] at observation[t].
    ## Rather, it is the reward to be received for transitioning *into* observation[t].
    ##
    ## What a mess... This fixes that too.
    ##
    ## NB that this is different from diffuser code!

    assert not np.any(dataset['terminals'])
    dataset['next_observations'] = dataset['observations'][1:]

    goal_diff = np.abs(dataset['infos/goal'][:-1] - dataset['infos/goal'][1:]).sum(-1)  # diff with next
    timeouts = goal_diff > 1e-5

    timeout_steps = np.where(timeouts)[0]
    path_lengths = timeout_steps[1:] - timeout_steps[:-1]

    logging.info(
        f'[ preprocess_maze2d_fix ] Segmented {env.name} | {len(path_lengths)} paths | '
        f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
    )

    dataset['timeouts'] = timeouts

    logging.info('[ preprocess_maze2d_fix ] Fixed terminals and timeouts')

    # Fix rewards
    assert len(env.goal_locations) == 1
    rewards = cast(
        np.ndarray,
        np.linalg.norm(
            dataset['next_observations'][:, :2] - env.get_target(),
            axis=-1,
        ) <= 0.5
    ).astype(dataset['rewards'].dtype)
    # check that it was wrong :/
    assert (rewards == dataset['rewards'][1:]).all()
    dataset['rewards'] = rewards
    logging.info('[ preprocess_maze2d_fix ] Fixed rewards')

    # put things back into a new dict
    dataset = dict(dataset)
    for k in dataset:
        if dataset[k].shape[0] != dataset['next_observations'].shape[0]:
            dataset[k] = dataset[k][:-1]
    return dataset

def get_d4rl_dataset(env):
    dataset = d4rl.qlearning_dataset(env)
    return dict(
        observations=dataset["observations"],
        actions=dataset["actions"],
        next_observations=dataset["next_observations"],
        rewards=dataset["rewards"],
        terminals=dataset["terminals"].astype(np.float32),
    )

def load_episodes_maze2d(name):
    env = load_environment(name)
    dataset = env.get_dataset()
    print(dataset.keys())
    return sequence_dataset(env, preprocess_maze2d_fix(env, dataset))
    # return sequence_dataset(env, preprocess_maze2d_fix(env, env.get_dataset))
    # yield from convert_dict_to_EpisodeData_iter(
    #     sequence_dataset(
    #         env,
    #         preprocess_maze2d_fix(
    #             env,
    #             env.get_dataset(),
    #         ),
    #     ),
    # )


# for name in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']:
#     register_offline_env(
#         'd4rl', name,
#         create_env_fn=functools.partial(load_environment, name),
#         load_episodes_fn=functools.partial(load_episodes_maze2d, name),
#     )

args = parser()
all_episodes_data = load_episodes_maze2d(args.env_name)
np.save(args.env_name + '_obs_info.npy', all_episodes_data)
# print(len(all_episodes_data))
# length_ep0 = all_episodes_data[0].shape[0]

# for i in range(1,len(all_episodes_data)):
#     print(all_episodes_data[i].shape[0])
#     assert length_ep0 == all_episodes_data[i].shape[0]
# print(all_episodes_data[0])
# print(all_episodes_data[0]['observations'].shape, all_episodes_data[0]['next_observations'].shape, all_episodes_data[0]['all_observations'].shape)

