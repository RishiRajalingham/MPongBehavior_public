#!/usr/bin/env bash

## within singularity container
export HDF5_USE_FILE_LOCKING='FALSE'

source ~/.bashrc
source activate py36
export PYTHONPATH=/home/rishir/envs/py36/lib/python3.6/site-packages/:/om/user/rishir/lib/src/v1.0.0/:$PYTHONPATH
#export PYTHONPATH=/usr/local/bin/python:/om/user/rishir/envs/tf/lib/python3.6/site-packages/
export CUDA_VISIBLE_DEVICES=0,1

python -B /om/user/rishir/lib/PongRnn/datasets/DatasetEncoder.py