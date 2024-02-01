'''
Script to evaluate a trained policy. Can save videos and log metrics.
Run as:
    obj=pan; python evaluate.py --exp expts/graff_seed1/ --env-name graff-v0 --obj ${obj} --rewards grasp:1 aff:1 --obj_mass 1 --obj_rot --policy cnn-mlp --cnn_arch custom --camera egocentric --inputs proprio loc rgb depth aff --model best --viz_stability --stability_frc 3 --save_metrics --save_videos --num_eval_episodes 100 --mode test --gpu 0
'''

import argparse
import os
import os.path as osp
import sys
import numpy as np
import torch
from eval_utils import Evaluation

sys.path.append('ppo')


class ParseReward(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split(':')
            getattr(namespace, self.dest)[key] = float(value)

parser = argparse.ArgumentParser(description='RL')
# Expt settings
parser.add_argument('--exp', type=str, required=True, help='Name of the expt')
parser.add_argument('--env-name', default='graff-v0', help='environment to train on')
parser.add_argument('--dataset', type=str, default='contactdb', help='Dataset for grasping task.')
parser.add_argument('--obj', type=str, default='pan', help='Object for grasping task.')
parser.add_argument('--model', type=str, default='best', help='model to load. Choose model name from expts/models')
parser.add_argument('--num_eval_episodes', type=int, default=10, help='Number of episodes to visualize')
parser.add_argument('--gpu', type=str, default='0', help='gpu id')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
# Policy
parser.add_argument('--policy', type=str, default='mlp', help='Choose from [mlp, cnn-mlp]')
parser.add_argument('--load_model', type=int, default=0, help='Load saved model to continue training.')
parser.add_argument('--cnn_arch', type=str, default='custom', help='CNN architecture. Choose from [custom, resnet]')
parser.add_argument('--cnn_pretrained', action='store_true', help='Load ImageNet pretrained resnet weights')
parser.add_argument('--noise', action='store_true', help='Add noise to inputs and outputs')
# Inputs
parser.add_argument('--inputs', type=str, nargs='+', default='proprio', help='List type. Choose from [proprio, loc, rgb, depth, aff]')
parser.add_argument('--cameras', type=str, nargs='*', default='egocentric', help='List type. Choose from [first_person, left, right, egocentric]')
parser.add_argument('--img_res', type=int, default=128, help='Resolution of input img to cnn')
# Rewards
parser.add_argument('--rewards', nargs='*', action=ParseReward, default='grasp:1', help='Custom type, set as reward:weight. Choose reward from [grasp, com, aff]')
parser.add_argument('--reward_dst_thr', type=float, default=0.05, help='hand-obj distance threshold beyond which distance reward will not be applied')
# Object params
parser.add_argument('--obj_mass', type=float, default=1, help='Mass of the object')
parser.add_argument('--obj_rot', action='store_true', help='Randomly rotate object')
parser.add_argument('--obj_tr', action='store_true', help='Randomly translate object')
# Mass - Scale expts
parser.add_argument('--orig_scale', type=float, default=1.0, help='Original scale of object mesh')
parser.add_argument('--delta_scale', type=float, default=0.0, help='Value to add to original scale')
# Environment params
parser.add_argument('--gravity', type=float, default=-9.81, help='Gravity value')
# Stability Metric
parser.add_argument('--viz_stability', action='store_true', help='Visualize grasp stability.')
parser.add_argument('--stability_frc', type=int, default=10, help='Perturbation force in Newtons.')
# Visualization
parser.add_argument('--viz_res', type=int, default=256, help='Img res for visualization')
# Save
parser.add_argument('--mode', type=str, default='test', help='Choose from [test, generalization]. Generalization is for mass-scale generalization.')
parser.add_argument('--save_videos', action='store_true', help='Save videos.')
parser.add_argument('--save_metrics', action='store_true', help='Save metrics.')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.set_num_threads(1)
device = torch.device("cuda:0" if args.cuda else "cpu")

# scale
if args.mode == 'generalization':
    obj_scale = args.orig_scale + args.delta_scale
else:
    obj_scale = 1  # always 1

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
                    'obj_scale': obj_scale,
                    'obj_rot': args.obj_rot,
                    'obj_tr': args.obj_tr,
                    'gravity': args.gravity,
                    'debug': False}

actor_critic, ob_rms = torch.load(os.path.join(args.exp, 'models', args.model + ".pt"))
eval_obj = Evaluation(args, grasp_attrs_dict, args.mode)
eval_obj.evaluate(actor_critic, ob_rms, args.model)
