#!/usr/bin/env bash

#SBATCH -t 12:0:0
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem-per-cpu 30000
#SBATCH --mail-type=NONE
#SBATCH --mail-user=rishir@mit.edu
#SBATCH -p jazayeri
#SBATCH --gres=gpu:tesla-k20:1


singularity_image="/om/user/rishir/lib/PongRnn/PongRnn_latest.sif"
singularity_script_fn="/om/user/rishir/lib/PongRnn/datasets/scripts/run_datasetencoder_2.sh"
module add openmind/singularity/3.2.0
singularity exec -B /om:/om $singularity_image bash $singularity_script_fn