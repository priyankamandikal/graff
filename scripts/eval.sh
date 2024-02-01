#!/bin/bash

expdir=./expts   # path to folder with model checkpoints
gpu=0             # gpu to use
seed=1            # seed to use

objs=("apple" "cell_phone" "cup" "door_knob" "flashlight" "hammer" "knife" "light_bulb" "mouse" "mug" "pan" "scissors" "stapler" "teapot" "toothbrush" "toothpaste")
obj="door_knob"       # object to evaluate on. Choose from the above list

# no prior
python evaluate.py --exp $expdir/noprior_seed${seed}/ --env-name graff-v0 --obj ${obj} --rewards grasp:1 --obj_mass 1 --obj_rot --policy cnn-mlp --cnn_arch custom --camera egocentric --inputs proprio loc rgb depth --model best --viz_stability --stability_frc 3 --save_metrics --save_videos --num_eval_episodes 100 --mode test --gpu ${gpu};

# com
python evaluate.py --exp $expdir/com_seed${seed}/ --env-name graff-v0 --obj ${obj} --rewards grasp:1 com:1 --obj_mass 1 --obj_rot --policy cnn-mlp --cnn_arch custom --camera egocentric --inputs proprio loc rgb depth --model best --viz_stability --stability_frc 3 --save_metrics --save_videos --num_eval_episodes 100 --mode test --gpu ${gpu};

# graff
python evaluate.py --exp $expdir/graff_seed${seed}/ --env-name graff-v0 --obj ${obj} --rewards grasp:1 aff:1 --obj_mass 1 --obj_rot --policy cnn-mlp --cnn_arch custom --camera egocentric --inputs proprio loc rgb depth aff --model best --viz_stability --stability_frc 3 --save_metrics --save_videos --num_eval_episodes 100 --mode test --gpu ${gpu}
