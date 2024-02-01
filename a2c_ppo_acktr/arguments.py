import os
import os.path as osp
import argparse


class ParseReward(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split(':')
            getattr(namespace, self.dest)[key] = float(value)

def get_args():
    parser = argparse.ArgumentParser(description='robot-crib')
    # Expt settings
    parser.add_argument('--exp', type=str, required=True, help='Name of the expt')
    parser.add_argument('--env-name', default='graff-v0', help='environment to train on')
    parser.add_argument('--dataset', type=str, default='contactdb', help='Dataset for grasping task.')
    parser.add_argument('--obj', type=str, default='pan', help='Object for grasping task.')
    # PPO
    parser.add_argument('--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer alpha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False, help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=16, help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5, help='number of forward steps (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10, help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100, help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None, help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--num-env-steps', type=int, default=10e6, help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument('--recurrent-policy', action='store_true', default=False, help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False, help='use a linear schedule on the learning rate')
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
    # Obj params
    parser.add_argument('--obj_mass', type=float, default=1, help='Mass of the object')
    parser.add_argument('--obj_rot', action='store_true', help='Randomly rotate object')
    parser.add_argument('--obj_tr', action='store_true', help='Randomly translate object')
    # Env params
    parser.add_argument('--gravity', type=float, default=-9.81, help='Gravity value')
    # GPU
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-model', type=str, default='0', help='gpu id to load model on')
    parser.add_argument('--gpu-env', type=str, default='1', help='gpu id to render env on')
    # Debug
    parser.add_argument('--debug', action='store_true', help='Debug mode: Save image observations')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_model

    import torch
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
