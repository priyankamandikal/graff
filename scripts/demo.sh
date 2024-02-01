#!/bin/bash

obj=cup
gpu=0

python evaluate.py --exp expts/graff_trained/ --env-name graff-v0 --obj ${obj} --rewards grasp:1 aff:1 --obj_mass 1 --obj_rot --policy cnn-mlp --cnn_arch custom --camera egocentric --inputs proprio loc rgb depth aff --model best --viz_stability --stability_frc 3 --save_videos --num_eval_episodes 100 --mode test --gpu ${gpu}