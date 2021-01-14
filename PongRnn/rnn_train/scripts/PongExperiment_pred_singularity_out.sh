#!/usr/bin/env bash

#SBATCH -t 3-12:0:0
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -p jazayeri
#SBATCH --gres=gpu:tesla-k20:1
##SBATCH --mem-per-cpu 120000


singularity_image="/om/user/rishir/lib/PongRnn/PongRnn_latest.sif"
singularity_script_fn="/om/user/rishir/lib/PongRnn/rnn_train/scripts/PongExperiment_pred_singularity_in.sh"
debug_filename="/om/user/rishir/lib/PongRnn/debug_$RANDOM.txt"
module add openmind/singularity/3.2.0

echo "$@"
singularity exec -B /om:/om $singularity_image bash $singularity_script_fn "$@"
