import os
import os.path as osp
import gym
import numpy as np
import torch
from gym.spaces.box import Box

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_


def make_env(env_id, seed, rank, log_dir, allow_early_resets, object, device_id, **kwargs):
    def _thunk():
        if env_id.startswith("graff"):
            from envs.mj_envs.utils import tmp
            from gym import Wrapper
            env = gym.make(env_id, object=object, device_id=device_id, process_id=rank, **kwargs)
            env = Wrapper(env)
            # from envs.mj_envs.utils.gym_env import GymEnv
            # env = GymEnv(env_id, object=object, device_id=rank,  **kwargs)
        else:
            env = gym.make(env_id)

        env.seed(seed + rank)

        obs_shape = {key: item.shape for key, item in env.observation_space.spaces.items()}

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        # If the input has shape (W,H,3) or (W,H,4), wrap for PyTorch convolutions
        img_keys = [key for key, item in obs_shape.items() if (len(item) == 3 and item[2] in [1, 3, 4, 5])]
        print('img_keys: ', img_keys)
        if img_keys != []:
            env = TransposeImage(env, img_keys, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  device_id,
                  allow_early_resets,
                  dataset,
                  object,
                  num_frame_stack=None,
                  **kwargs):
    if object == 'all':
        curr_dir = osp.dirname(osp.abspath(__file__))
        with open(osp.join(curr_dir, '../envs/mj_envs/dex_manip/assets/%s/objects.txt'%dataset), 'r') as f:
            objects = [obj.strip() for obj in f.readlines()]
        objects = objects * (num_processes//len(objects))
        envs = [
            make_env(env_name, seed, i, log_dir, allow_early_resets, object, device_id, **kwargs)
            for (i,object) in zip(range(len(objects)), objects)
        ]
    else:
        envs = [
            make_env(env_name, seed, i, log_dir, allow_early_resets, object, device_id, **kwargs)
            for i in range(num_processes)
        ]

    if len(envs) > 1:
        #################################################################try#############################################
        envs = ShmemVecEnv(envs, context='spawn')
        # envs = SubprocVecEnv(envs, context='spawn')
    else:
        envs = DummyVecEnv(envs)

    norm_keys = [key for key, item in envs.observation_space.spaces.items() if len(item.shape) == 1]
    if norm_keys != []:
        if gamma is None:
            envs = VecNormalize(envs, norm_keys=norm_keys, ret=False)
        else:
            envs = VecNormalize(envs, norm_keys=norm_keys, gamma=gamma)

    envs = VecPyTorch(envs, device)

    # if num_frame_stack is not None:
    #     envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    # elif len(envs.observation_space.shape) == 3:
    #     envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, img_keys=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3 or dim4"
        self.op = op
        self.img_keys = img_keys
        for key in img_keys:
            obs_shape = env.observation_space[key].shape
            self.observation_space.spaces[key] = Box(
                self.observation_space.spaces[key].low[0, 0, 0],
                self.observation_space.spaces[key].high[0, 0, 0], [
                    obs_shape[self.op[0]], obs_shape[self.op[1]],
                    obs_shape[self.op[2]]
                ],
                dtype=self.observation_space[key].dtype)

    def observation(self, ob):
        for key in self.img_keys:
            ob[key] = ob[key].transpose(self.op[0], self.op[1], self.op[2])
        return ob


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        obs = {key: torch.from_numpy(item).float().to(self.device) for key, item in obs.items()}
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = {key: torch.from_numpy(item).float().to(self.device) for key, item in obs.items()}
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                for key in self.norm_keys:
                    self.ob_rms[key].update(obs[key])
            for key in self.norm_keys:
                obs[key] = np.clip((obs[key] - self.ob_rms[key].mean) /
                              np.sqrt(self.ob_rms[key].var + self.epsilon),
                              - self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
### haven't updated for obs dict ###
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
