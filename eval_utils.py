import os
import os.path as osp
import numpy as np
import cv2
from moviepy.editor import ImageSequenceClip, VideoFileClip, concatenate_videoclips
import shutil
from pickle import dump
from tqdm import tqdm

import torch

from baselines.common.vec_env import VecEnvWrapper
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from a2c_ppo_acktr import tensor_utils


class Evaluation():

    def __init__(self, args, grasp_attrs_dict, mode='test'):

        # super(Evaluation, self).__init__(env)


        device = torch.device("cuda" if args.cuda else "cpu")
        self.env = make_vec_envs(
            args.env_name,
            args.seed + 1000,
            1,
            None,
            None,
            device=device,
            dataset=args.dataset,
            object=args.obj,
            device_id=int(args.gpu),
            allow_early_resets=False,
            grasp_attrs_dict=grasp_attrs_dict)
        self.vec_norm = get_vec_normalize(self.env)
        self.vec_norm.eval()

        # settings
        self.args = args
        self.device_id = int(args.gpu)
        self.fps = 100
        self.horizon = 200
        self.cam = args.cameras[0]
        self.angle_increment = 180. / (args.num_eval_episodes - 1)
        self.img_res = args.viz_res
        self.mode = mode
        self.time_elapsed_in_hrs = 0
        self.mass = grasp_attrs_dict['obj_mass']
        self.scale = grasp_attrs_dict['obj_scale']

        # save
        self.exp_dir = args.exp
        if args.save_videos:
            if args.viz_stability:
                self.video_dir = osp.join(self.exp_dir, 'videos_stability', 'F%d'%args.stability_frc, args.obj)
            else:
                self.video_dir = osp.join(self.exp_dir, 'videos', 'F%d'%args.stability_frc, args.obj)
            os.makedirs(self.video_dir, exist_ok=True)
        if args.save_metrics:
            self.metrics_dir = osp.join(self.exp_dir, 'metrics', 'F%d_N%d' % (args.stability_frc, args.num_eval_episodes), args.obj)
            os.makedirs(self.metrics_dir, exist_ok=True)
            if self.mode == 'train':
                with open(osp.join(self.metrics_dir, 'metrics.txt'), 'w') as f:
                    f.write('model; success; stability; reward\n')


    def get_experience_time(self, iter_num):
        # 2 ms * 5 frame skip * num agent steps * num processes * iter num
        time_elapsed_in_secs = 0.002 * 5 * self.args.num_steps * self.args.num_processes * iter_num
        time_elapsed_in_hrs = time_elapsed_in_secs / 3600.
        return time_elapsed_in_hrs


    def take_action(self, actor_critic, obs, recurrent_hidden_states, masks):
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=True)
        obs, reward, done, infos = self.env.step(action)
        masks.fill_(0.0 if done else 1.0)
        return obs, recurrent_hidden_states, masks, reward, infos


    def render_frame(self, time_step, ep_dir):
        rgbd_frame = self.env.envs[0].sim.render(width=self.img_res, height=self.img_res,
                                                 mode='offscreen', camera_name=self.cam,
                                                 depth=True, device_id=self.device_id)
        img = rgbd_frame[0][::-1]  # rgb : (H,W,3)
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
        if self.mode == 'train':
            img_text = '%.2f hrs' % self.time_elapsed_in_hrs
            cv2.putText(img, img_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))
        img_text = self.args.obj
        cv2.putText(img, img_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
        cv2.imwrite(osp.join(ep_dir, 'rgb', str(time_step).zfill(3) + '.png'), img)


    def merge_videos(self, video_dir):
        for modality in ['rgb']:
            clips = []
            for vname in sorted(os.listdir(video_dir)):
                if modality in vname and 'merged' not in vname and vname[0] != '.':
                    vclip = VideoFileClip(osp.join(video_dir, vname))
                    clips.append(vclip)
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(osp.join(video_dir, 'merged_%s.mp4' % modality))


    def evaluate(self, actor_critic, ob_rms, model):

        # Normalize env
        self.vec_norm.ob_rms = ob_rms

        # Initialize recurrent states and masks
        recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size).cuda()
        masks = torch.zeros(1, 1).cuda()

        # Compute experience time for train model
        if self.mode == 'train':
            self.time_elapsed_in_hrs = self.get_experience_time(iter_num=model)
            with open(osp.join(self.exp_dir, 'iter2time.txt'), 'a') as f:
                f.write('%d; %.2f\n' % (model, self.time_elapsed_in_hrs))

        if self.args.save_videos:
            video_dir = osp.join(self.video_dir, str(model))
        if self.args.save_metrics:
            paths = []

        # Roll out each episode
        for ep, idx in enumerate(range(self.args.num_eval_episodes)):

            # Create save dirs
            if self.args.save_videos:
                if ep%10==0:
                    ep_dir = osp.join(video_dir, 'ep_' + str(ep))
                    os.makedirs(osp.join(ep_dir, 'rgb'), exist_ok=True)

            # Initialize metrics lists
            if self.args.save_metrics:
                rewards = []
                env_infos = []
                perturb_infos = []

            # Set obj initialization angle
            if self.args.obj in ['cell_phone', 'stapler', 'teapot', 'toothpaste']:
                angle = idx * self.angle_increment
            else:
                angle = 180. + (idx * self.angle_increment)
            # # training progression videos
            # obj2ang = {'mug': 180.+18*6,
            #            'pan': 180.+18*7,
            #            'hammer': 180+18*9,
            #            'scissors': 180.+18*3,
            #            'cup': 180.,
            #            'teapot': 0.+18*7,
            #            'knife': 180+18*2}
            # angle = obj2ang[self.args.obj]

            obs = self.env.reset(angle=angle)

            # Run episode
            for t in range(self.horizon):
                obs, recurrent_hidden_states, masks, reward, infos = self.take_action(actor_critic, obs, recurrent_hidden_states, masks)
                if self.args.save_videos:
                    if ep%10==0:
                        self.render_frame(t, ep_dir)
                if self.args.save_metrics:
                    rewards.append(reward)
                    env_infos.append(infos[0])

            # Apply perturbation forces
            if self.args.viz_stability:
                perturb_frc = self.args.stability_frc
                for i in range(300):
                    if i in range(50):
                        self.env.envs[0].sim.data.xfrc_applied[self.env.envs[0].obj_bid] = [perturb_frc, 0, 0, 0, 0, 0]
                    elif i in range(50, 100):
                        self.env.envs[0].sim.data.xfrc_applied[self.env.envs[0].obj_bid] = [-perturb_frc, 0, 0, 0, 0, 0]
                    elif i in range(100, 150):
                        self.env.envs[0].sim.data.xfrc_applied[self.env.envs[0].obj_bid] = [0, perturb_frc, 0, 0, 0, 0]
                    elif i in range(150, 200):
                        self.env.envs[0].sim.data.xfrc_applied[self.env.envs[0].obj_bid] = [0, -perturb_frc, 0, 0, 0, 0]
                    elif i in range(200, 250):
                        self.env.envs[0].sim.data.xfrc_applied[self.env.envs[0].obj_bid] = [0, 0, perturb_frc, 0, 0, 0]
                    elif i in range(250, 300):
                        self.env.envs[0].sim.data.xfrc_applied[self.env.envs[0].obj_bid] = [0, 0, -perturb_frc, 0, 0, 0]

                    obs, recurrent_hidden_states, masks, reward, infos = self.take_action(actor_critic, obs, recurrent_hidden_states, masks)

                    if self.args.save_videos:
                        if ep%10==0:
                            self.render_frame(t+i+1, ep_dir)

                    if self.args.save_metrics:
                        perturb_infos.append(infos[0])

            # Save video
            if self.args.save_videos:
                if ep%10==0:
                    file_name = osp.join(video_dir, "%s_rgb.mp4" % str(ep).zfill(2))
                    clip = ImageSequenceClip(osp.join(ep_dir, 'rgb'), fps=self.fps)
                    clip.write_videofile(file_name)
                    print("saved", file_name)
                    shutil.rmtree(ep_dir)

            # Store metric metadata
            if self.args.save_metrics:
                path = dict(
                    rewards=np.array(rewards),
                    env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
                    perturb_infos=tensor_utils.stack_tensor_dict_list(perturb_infos)
                )
                paths.append(path)

        # Merge episode videos
        if self.args.save_videos:
            self.merge_videos(video_dir)

        # Compute metrics
        if self.args.save_metrics:
            avg_reward = np.mean([np.sum(path['rewards']) for path in paths])
            grasp_success, grasp_stability = self.env.envs[0].evaluate_success(paths)
            if self.mode == 'test':
                with open(osp.join(self.metrics_dir, '{}.txt'.format(model)), 'w') as f:
                    f.write('model; success rate; stability; reward\n')
                    f.write('{}; {:.2f}; {:.2f}; {:.2f}'.format(model, grasp_success, grasp_stability, avg_reward))
            elif self.mode == 'generalization':
                os.makedirs(osp.join(self.metrics_dir, 'generalization'), exist_ok=True)
                with open(osp.join(self.metrics_dir, 'generalization', 'mass_{:.1f}_scale_{:.2f}.txt'.format(self.mass, self.scale)), 'w') as f:
                    f.write('success rate; stability; reward\n')
                    f.write('{:.2f}; {:.2f}; {:.2f}\n'.format(grasp_success, grasp_stability, avg_reward))

            print('expt; model; obj_mass; obj_scale; success; stability; reward')
            print('{}; {}; {:.1f}; {:.1f}; {:.2f}; {:.2f}; {:.2f}'.format(self.exp_dir, model, self.mass, self.scale,
                                                                          grasp_success, grasp_stability,
                                                                          avg_reward))

        if self.mode =='train':
            return [np.sum(path['rewards']) for path in paths], grasp_success, grasp_stability