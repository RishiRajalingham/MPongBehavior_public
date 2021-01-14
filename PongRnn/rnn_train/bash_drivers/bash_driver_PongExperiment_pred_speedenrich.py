import os
import itertools

homepath = '/om/user/rishir/lib/PongRnn/rnn_train/'
use_gpu = True

# pred_lstm20_lwvis-sim-mov_0.50_L10.10__IC_pixel_pca

if use_gpu:
    sbatch_prefix = 'sbatch %s/scripts/PongExperiment_pred_singularity_out.sh' % homepath
else:
    sbatch_prefix = 'sbatch %s/scripts/PongExperiment_pred_no_singularity.sh' % homepath

choices = {
    'file_path': ['/om/user/rishir/data/pong_basic/RF/occluded_pong_bounce1_pad8_10speed/'],
    'loss_weight_type': ['vis-sim-mov'],#['sim-mov', 'vis-sim-mov', 'vis-mov', 'mov'],
    'log_dir': ['./logs_speedenrich/'],
    'save_path': ['/om/user/rishir/lib/PongRnn/dat/rnn_res_speedenrich/'],
    'regularization_weight': [0.1],
    'regularization_loss': ['L1'],
    'input_representation': ['pixel_pca'],
    'n_hidden': [20],
    'max_train_time': [24 * 3600.0 * 3.0],
    'rnn_type': ['lstm'], #'lstm'
    'max_n_epochs': [1000],
    'n_epochs_per_ckpt': [250],
    'train_frac': [1.0],
    'random_seed': [1],
    'do_restart_run': [True],
    'train': [True],
    'test': [True],
    'with_initial_cond': [True]
}

choice_keys = choices.keys()
all_choices = itertools.product(*tuple(choices.values())) # use zipper instead
choice_keys_list = choices.keys()

count = 0
max_jobs = 100

#  check here for max_jobs
#  exception

for model_choice in all_choices:
    arg_string = ''
    for fki, fk in enumerate(choice_keys):
        arg_string += ' --' + fk + ' ' + str(model_choice[fki])
    os_fn = '%s %s' % (sbatch_prefix, arg_string)
    os.system(os_fn)
    print(os_fn)
    count += 1
    if count >= max_jobs:
        break # insert error message
    # time.sleep(5) # seem to be necessary to make sure that singularity bind paths work ??????
