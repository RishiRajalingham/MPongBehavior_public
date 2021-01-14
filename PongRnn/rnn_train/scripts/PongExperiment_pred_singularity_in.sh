#!/usr/bin/env bash

## within singularity container
export HDF5_USE_FILE_LOCKING='FALSE'
export PYTHONPATH=/usr/local/bin/python:/om/user/rishir/envs/tf/lib/python3.6/site-packages/
export CUDA_VISIBLE_DEVICES=0,1

debug_filename="/om/user/rishir/lib/PongRnn/debug_out/debug_$RANDOM.txt"
echo "$@" > $debug_filename
#python -B /om/user/rishir/lib/PongRnn/tests/test_gpu_avail.py >> $debug_filename
python -B /om/user/rishir/lib/PongRnn/rnn_train/PongExperiment_prediction.py "$@" >> $debug_filename
