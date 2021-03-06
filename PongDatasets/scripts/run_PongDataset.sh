#!/usr/bin/env bash


#SBATCH -t 1-0:0:0
#SBATCH -n 1
#SBATCH --mem-per-cpu 12000
#SBATCH --mail-type=NONE
#SBATCH --mail-user=rishir@mit.edu

source ~/.bashrc
source activate py36
export PYTHONPATH=/home/rishir/envs/py36/lib/python3.6/site-packages/:/om/user/rishir/lib/src/v1.0.0/:$PYTHONPATH
echo "$@"
python /om/user/rishir/lib/PongRnn/PongDataset.py "$@"