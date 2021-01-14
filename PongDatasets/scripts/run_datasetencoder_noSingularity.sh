#!/usr/bin/env bash

#SBATCH -t 1-12:0:0
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem-per-cpu 30000
#SBATCH --mail-type=NONE
#SBATCH --mail-user=rishir@mit.edu
##SBATCH -p jazayeri

source ~/.bashrc
source activate py36
export PYTHONPATH=/home/rishir/envs/py36/lib/python3.6/site-packages/:/om/user/rishir/lib/src/v1.0.0/
echo "$@"
python  /om/user/rishir/lib/PongRnn/datasets/DatasetEncoder.py "$@"