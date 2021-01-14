import os
import itertools

homepath = '/om/user/rishir/lib/PongRnn/rnn_train/'
use_gpu = True

if use_gpu:
    sbatch_prefix = 'sbatch %s/scripts/PongExperiment_pred_singularity_out.sh' % homepath
else:
    sbatch_prefix = 'sbatch %s/scripts/PongExperiment_pred_no_singularity.sh' % homepath

choices = {
    'file_path': ['/om/user/rishir/data/pong_basic/RF/occluded_pong_bounce1_pad8_4speed/'],
    'loss_weight_type': ['mov',],
    'log_dir': ['./logs_regularized/'],
    'save_path': ['/om/user/rishir/lib/PongRnn/dat/rnn_res_regularized/'],
    'regularization_weight_simplicity_DS': [1],
    'regularization_weight_simplicity_DS2': [0.5, 1],
    'regularization_weight_L2_activity': [0.5, 1],
    'regularization_weight': [0.1],
    'regularization_loss': ['L2'],
    'input_representation': ['pixel_pca'],
    'n_hidden': [10, 20],
    'max_train_time': [24 * 3600.0 * 3.0],
    'rnn_type': ['lstm'],
    'max_n_epochs': [1000],
    'n_epochs_per_ckpt': [100],
    'train_frac': [0.5],
    'random_seed': [0],
    'do_restart_run': [True],
    'train': [True],
    'test': [True],
    'with_initial_cond': [True]
}

choice_keys = choices.keys()
all_choices = itertools.product(*tuple(choices.values())) # use zipper instead
choice_keys_list = choices.keys()

count = 0
max_jobs = 20

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
