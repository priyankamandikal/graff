"""
Wrapper around a gym env that provides convenience functions
"""

import gym
import numpy as np


class EnvSpec(object):
    def __init__(self, obs_dim, act_dim, horizon):
        self.observation_dim = obs_dim
        self.action_dim = act_dim
        self.horizon = horizon


class GymEnv(object):
    def __init__(self, env_name, **kwargs):
        env = gym.make(env_name, **kwargs)
        self.env = env
        self.env_id = env.spec.id

        try:
            self._horizon = env.spec.max_episode_steps
        except AttributeError:
            self._horizon = env.spec._horizon

        # try:
        #     self._action_dim = self.env.env.action_dim
        # except AttributeError:
        #     self._action_dim = self.env.action_space.shape[0]
        #
        # try:
        #     self._observation_dim = self.env.env.obs_dim
        # except AttributeError:
        #     self._observation_dim = self.env.observation_space.shape[0]
        #
        # self.spec = EnvSpec(self._observation_dim, self._action_dim, self._horizon)

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self, seed=None):
        try:
            self.env._elapsed_steps = 0
            return self.env.env.reset_model(seed=seed)
        except:
            if seed is not None:
                self.set_seed(seed)
            return self.env.reset()

    def reset_model(self, seed=None):
        # overloading for legacy code
        return self.reset(seed)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        try:
            self.env.env.mujoco_render_frames = True
            self.env.env.mj_render()
        except:
            self.env.render()

    def seed(self, seed=123):
        try:
            self.env.seed(seed)
        except AttributeError:
            self.env._seed(seed)

    def get_obs(self):
        try:
            return self.env.env._get_obs()
        except:
            return self.env.env.get_obs()

    def get_env_infos(self):
        try:
            return self.env.env.get_env_infos()
        except:
            return {}

    # ===========================================
    # Trajectory optimization related
    # Envs should support these functions in case of trajopt

    def get_env_state(self):
        try:
            return self.env.env.get_env_state()
        except:
            raise NotImplementedError

    def set_env_state(self, state_dict):
        try:
            self.env.env.set_env_state(state_dict)
        except:
            raise NotImplementedError

    def real_env_step(self, bool_val):
        try:
            self.env.env.real_step = bool_val
        except:
            raise NotImplementedError

    # ===========================================

    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration', viz_stability=False):
        self.env.env.visualize_policy(policy, horizon, num_episodes, mode, viz_stability)

    def plot_q_variables(self, policy, policy_name, horizon=1000, num_episodes=1, mode='exploration', save_loc='/tmp'):
        self.env.env.plot_q_variables(policy, policy_name, horizon, num_episodes, mode, save_loc)

    def visualize_policy_offscreen(self, policy, horizon=1000,
                                   num_episodes=1,
                                   frame_size=(640,480),
                                   fps=25,
                                   mode='exploration',
                                   save_loc='/tmp/',
                                   camera_name=None):
        self.env.env.visualize_policy_offscreen(policy, horizon, num_episodes, frame_size, fps, mode, save_loc,
                                                camera_name)

    def metrics_policy(self,
                       policy,
                       env_name,
                       mass,
                       scale,
                       horizon=1000,
                       num_episodes=1,
                       seed=0,
                       save_loc='/tmp/',
                       **kwaargs):
        self.env.env.metrics_policy(policy, env_name, mass, scale, horizon, num_episodes, seed, save_loc, **kwaargs)

    def save_videos(self, policy, env_name, mass, scale, horizon=1000, num_episodes=1, frame_size=(640,480), fps=25,
                    viz_rewards=False, viz_stability=False, eval_cat=None, save_loc='/tmp/', camera_name=None, **kwaargs):
        self.env.env.save_videos(policy, env_name, mass, scale, horizon, num_episodes, frame_size, fps, viz_rewards,
                                 viz_stability, eval_cat, save_loc, camera_name, **kwaargs)

    def evaluate_policy(self, policy,
                        num_episodes=5,
                        horizon=None,
                        gamma=1,
                        visual=False,
                        percentile=[],
                        get_full_dist=False,
                        mean_action=False,
                        init_env_state=None,
                        terminate_at_done=True,
                        seed=123):

        self.set_seed(seed)
        horizon = self._horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)

        for ep in range(num_episodes):
            self.reset()
            if init_env_state is not None:
                self.set_env_state(init_env_state)
            t, done = 0, False
            while t < horizon and (done == False or terminate_at_done == False):
                self.render() if visual is True else None
                o = self.get_obs()
                a = policy.get_action(o)[1]['evaluation'] if mean_action is True else policy.get_action(o)[0]
                o, r, done, _ = self.step(a)
                ep_returns[ep] += (gamma ** t) * r
                t += 1

        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        percentile_stats = []
        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        full_dist = ep_returns if get_full_dist is True else None

        return [base_stats, percentile_stats, full_dist]