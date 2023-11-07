"""
Script to train the PPO policy for dexterous grasping
Check scripts/train.sh for training scripts
"""
import os
from a2c_ppo_acktr.arguments import get_args

args = get_args()

import time
import shutil
import os.path as osp
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    exp_dir = args.exp
    log_dir = osp.join(exp_dir, 'monitor')
    tb_dir = osp.join(exp_dir, 'logs')
    save_dir = osp.join(exp_dir, 'models')
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    utils.cleanup_log_dir(log_dir)
    writer = SummaryWriter(tb_dir)
    
    # copy important codes into exp dir 
    code_dir = osp.join(exp_dir, 'codes')
    os.makedirs(code_dir, exist_ok=True)
    with open(osp.join(code_dir, 'args.txt'), 'w') as f:
        f.write(str(args)+'\n')
    curr_dir = osp.dirname(osp.abspath(__file__))
    shutil.copy(osp.join(curr_dir, 'train.py'), code_dir)
    shutil.copy(osp.join(curr_dir, 'a2c_ppo_acktr/arguments.py'), code_dir)
    shutil.copy(osp.join(curr_dir, 'a2c_ppo_acktr/envs.py'), code_dir)
    shutil.copy(osp.join(curr_dir, 'a2c_ppo_acktr/model.py'), code_dir)
    shutil.copy(osp.join(curr_dir, 'envs/mj_envs/dex_manip/graff.py'), code_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda" if args.cuda else "cpu")
    
    grasp_attrs_dict = {'dataset': args.dataset,
                        'obj': args.obj,
                        'policy': args.policy,
                        'cnn_arch': args.cnn_arch,
                        'noise': args.noise,
                        'inputs': args.inputs,
                        'cameras': args.cameras,
                        'img_res': args.img_res,
                        'rewards': args.rewards,
                        'reward_dst_thr': args.reward_dst_thr,
                        'obj_mass': args.obj_mass,
                        'obj_rot': args.obj_rot,
                        'obj_tr': args.obj_tr,
                        'gravity': args.gravity,
                        'debug': args.debug}

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, log_dir, device, int(args.gpu_env), False, dataset=args.dataset, 
                         object=args.obj, grasp_attrs_dict=grasp_attrs_dict)

    if args.load_model == 0:
        actor_critic = Policy(
            envs.observation_space,
            envs.action_space,
            policy=args.policy,
            cnn_args={'arch': args.cnn_arch,
                      'pretrained': args.cnn_pretrained,
                      'cameras': args.cameras},
            base_kwargs={'recurrent': args.recurrent_policy})
        start_update_num = 0
    else:
        from a2c_ppo_acktr.utils import get_vec_normalize
        actor_critic, ob_rms = \
            torch.load(os.path.join(args.exp, 'models', str(args.load_model) + ".pt"))
        start_update_num = args.load_model + 1
        vec_norm = get_vec_normalize(envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms

    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs.copy_(0, obs)
    rollouts.to(device)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    train_rewards = deque(maxlen=10)
    episode_successes = deque(maxlen=10)
    episode_successes_orig = deque(maxlen=10)
    best_ep_rews = -np.inf

    for j in range(start_update_num, num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    train_rewards.append(info['episode']['r'])
                    episode_successes.append(info['episode']['obj_lift'])
                    episode_successes_orig.append(info['episode']['obj_grab'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % args.log_interval == 0 and len(train_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(train_rewards), np.mean(train_rewards),
                            np.median(train_rewards), np.min(train_rewards),
                            np.max(train_rewards), dist_entropy, value_loss,
                            action_loss))
            writer.add_scalar('Loss/total', value_loss * args.value_loss_coef + action_loss -
                              dist_entropy * args.entropy_coef, total_num_steps)
            writer.add_scalar('Loss/value', value_loss, total_num_steps)
            writer.add_scalar('Loss/action', action_loss, total_num_steps)
            writer.add_scalar('Loss/entropy', dist_entropy, total_num_steps)
            writer.add_scalar('Rewards/mean', np.mean(train_rewards), total_num_steps)
            writer.add_scalar('Rewards/median', np.median(train_rewards), total_num_steps)
            writer.add_scalar('Rewards/max', np.max(train_rewards), total_num_steps)
            writer.add_scalar('Rewards/min', np.min(train_rewards), total_num_steps)
            writer.add_scalar('Success_rate/hand-obj-notable', np.mean(episode_successes), total_num_steps)
            writer.add_scalar('Success_rate/obj-notable', np.mean(episode_successes_orig), total_num_steps)

        # save model for every interval-th episode or for the last epoch
        if ((j + 1) % args.save_interval == 0 or j == num_updates - 1):
            # save model
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_dir, str(j + 1) + ".pt"))

            # save best model if available
            if np.mean(train_rewards) > best_ep_rews:
                print('Best model found. Saving.')
                best_ep_rews = np.mean(train_rewards)
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(save_dir, "best.pt"))


if __name__ == "__main__":
    main()
