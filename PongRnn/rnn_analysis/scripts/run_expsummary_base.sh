#!/usr/bin/env bash
## e.g. sbatch scripts/run_analyses.sh --results_path /om/user/rishir/lib/PongRnn/dat/rnn_res/
## run this first before renaming any folders for archiving purposes

#SBATCH -t 2-12:0:0
#SBATCH -n 1
#SBATCH --mem-per-cpu 30000
#SBATCH --mail-type=NONE
#SBATCH --mail-user=rishir@mit.edu
##SBATCH --qos=jazayeri
##SBATCH --partition=jazayeri


source ~/.bashrc
source activate py36
export PYTHONPATH=/home/rishir/envs/py36/lib/python3.6/site-packages/:/om/user/rishir/lib/src/v1.0.0/

pyfn="/om/user/rishir/lib/PongRnn/rnn_analysis/PongRNNExpSummary.py"
indirpath="/om/user/rishir/lib/PongRnn/dat/rnn_res/"
outdirpath="/om/user/rishir/lib/PongRnn/fig/rnn_res/"

python $pyfn --results_path $indirpath --fig_out_path $outdirpath