"""
Simple predictor of ball trajectory from pixels.
Heavily recycled from flipflop.py by @ Matt Golub, August 2018. (mgolub@stanford.edu.)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
# print(tf.__version__)
import numpy as np
import h5py
import sys

# PATH_TO_RW = '/om/user/rishir/lib/src/v1.0.0/'
PATH_TO_RW = '/om/user/rishir/lib/recurrent-whisperer/'
sys.path.insert(0, PATH_TO_RW)

from RecurrentWhisperer import RecurrentWhisperer


class GeneralEstimationRnn(RecurrentWhisperer):

    @staticmethod
    def _default_hash_hyperparameters():
        return {
            'input_dims': [512, 50, 81],
            'output_dims': [512, 50, 2],
            'regularization_weight_L2_activity': 0.0,
            'regularization_weight_simplicity_DS': 0.0,
            'regularization_weight_simplicity_DS2': 0.0,
            'regularization_weight': 0.0,
            'regularization_loss': 'L1',
            'loss_weight_type': 'all',
            'rnn_type': 'lstm',
            'n_hidden': 100,
            'input_representation': 'pxl',
            'output_representation': 'pred',
            'with_initial_cond': False,
            'multisensory': False,
            'train_frac': 1.0,
            'noise_sigma': None,
            'surrogate_label': False,
            'random_seed': 0,
            'dtype': 'float32',  # keep as string (rather than tf.float32)
            # for better argparse handling, yaml writing
            'adam_hps': {'epsilon': 0.01},
            'alr_hps': {},
            'agnc_hps': {}
        }

    @staticmethod
    def _default_non_hash_hyperparameters():
        return {
            'log_dir': './logs/',
            'max_ckpt_to_keep': 15,
            'n_trials_plot': 1,
            'max_n_epochs': 1000,
            'max_train_time': None,
            'do_restart_run': False,
            'min_loss': None,
            'min_learning_rate': 1e-10,
            'n_epochs_per_validation_update': 100,
            'max_n_epochs_without_lvl_improvement': 200,
            'do_generate_visualizations': True,
            'do_save_tensorboard_events': True,
            'do_save_ckpt': True,
            'do_save_lvl_ckpt': True,
            'do_save_lvl_train_predictions': True,
            'do_save_lvl_valid_predictions': True,
            'do_save_lvl_mat_files': False,
            'max_lvl_ckpt_to_keep': 1,
            'n_epochs_per_ckpt': 100,
            'n_epochs_per_visualization_update': 100,
            'disable_gpus': False,
            'allow_gpu_growth': True,
            'per_process_gpu_memory_fraction': None,
            'dataset_name': None,
            'n_folds': None,
            'fold_idx': None,
        }


    def _setup_one_loss(self, output, suffix='',
                        weight_channels_equally=True):
        """ output is a time varying function as well.
        if discrete output desired at some time point,
        fill other timepoints with Nans """

        def diff_along_time_axis(a):
            return a[:, 1:,:] - a[:, :-1,:]

        n_hidden = self.hps.n_hidden
        n_output = int(output.shape[-1])

        np_w_out, np_b_out = self._np_init_weight_matrix(n_hidden, n_output)
        w_out = tf.Variable(np_w_out, dtype=tf.float32, name='w_out' + suffix)
        b_out = tf.Variable(np_b_out, dtype=tf.float32, name='b_out' + suffix)
        pred_output_bxtxo = tf.tensordot(self.hidden_bxtxd, w_out, axes=1) + b_out
        loss_raw = tf.squared_difference(pred_output_bxtxo, output)
        loss_masked = tf.math.multiply(loss_raw, self.mask_bxtxo)
        if weight_channels_equally:
            # here, weight all output channels the same, regardless of how many time-bins there are.
            loss = tf.reduce_mean(tf.reduce_mean(loss_masked, axis=1))
        else:
            loss = tf.reduce_mean(loss_masked)

        # regularize only the output weights.
        if self.hps.regularization_loss == 'L1':
            reguralizer = tf.reduce_sum(tf.abs(np_w_out))
        else:
            reguralizer = tf.nn.l2_loss(np_w_out)
        reg_loss = self.hps.regularization_weight * reguralizer

        # regularize the internal activity and dynamics
        lambda1 = self.hps.regularization_weight_L2_activity
        lambda2 = self.hps.regularization_weight_simplicity_DS
        lambda3 = self.hps.regularization_weight_simplicity_DS2

        l2_norm = tf.norm(self.hidden_bxtxd)
        l2_diff_norm = tf.norm(diff_along_time_axis(self.hidden_bxtxd))

        reg_loss_l2_norm = lambda1 * l2_norm
        reg_loss_simple_dynamics = lambda2 * l2_diff_norm
        reg_loss_simple_dynamics_2 = lambda3 * l2_diff_norm/l2_norm

        loss = loss + tf.cast(reg_loss, tf.float32) +\
               tf.cast(reg_loss_l2_norm, tf.float32) +\
               tf.cast(reg_loss_simple_dynamics, tf.float32) + \
               tf.cast(reg_loss_simple_dynamics_2, tf.float32)

        return pred_output_bxtxo, loss

    def _setup_model(self):
        hps = self.hps
        n_batch, n_time, n_dim_i = hps.input_dims
        n_hidden = hps.n_hidden

        # Data handling
        self.inputs_bxtxd = tf.placeholder(tf.float32, hps.input_dims)
        self.output_bxtxo = tf.placeholder(tf.float32, hps.output_dims)
        self.mask_bxtxo = tf.placeholder(tf.float32, hps.output_dims)

        # RNN
        if hps.rnn_type == 'vanilla':
            self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
        elif hps.rnn_type == 'gru':
            self.rnn_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
        elif hps.rnn_type == 'lstm':
            self.rnn_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of [vanilla, gru, lstm] but was %s' % hps.rnn_type)

        initial_state = self.rnn_cell.zero_state(n_batch, dtype=tf.float32)

        self.hidden_bxtxd, self.final_states_bxs = tf.nn.dynamic_rnn(self.rnn_cell,
                                                                     self.inputs_bxtxd,
                                                                     initial_state=initial_state)

        # tf.keras.layers.RNN(self.rnn_cell, self.inputs_bxtxd, init)

        self.pred_output_bxtxo, self.loss = self._setup_one_loss(self.output_bxtxo)
        self.validation_result = []

    def _setup_saver(self):
        self.saver = tf.train.Saver(tf.global_variables(),
                                    max_to_keep=self.hps.max_ckpt_to_keep)

    def _setup_training(self, train_data, valid_data):
        pass

    def _load_single_batch_from_file(self, batch_data):
        """ Return dictionary with inputs, outputs.
         Specify for each experiment by overriding this function. """

        file_path = batch_data['file_path']
        with h5py.File(file_path, 'r') as f:
            inputs = f['inputs']
            outputs = f['outputs']
            masks = f['masks']

        batch_data = {
            'inputs': inputs,
            'outputs': outputs,
            'masks': masks,
        }

        return batch_data

    def _train_batch(self, batch_data):
        # if batches are too big to load at once, the batch_data file contains an
        # empty dictionary, with a filename (file_path) to load the individual batch data.
        # if not, the inputs and output are also entries in the dictionary:
        # batch_data is actually a filename for a single file of batch data
        # which contains a dictionary of inputs and output

        batch_data = self._load_single_batch_from_file(batch_data)
        return self._train_batch_base(batch_data)

    def _train_batch_base(self, batch_data):
        # batch_data is actually a filename for a single file of batch data
        # which contains a dictionary of inputs and output

        ops_to_eval = [
            self.train_op,
            self.grad_global_norm,
            self.loss,
            self.merged_opt_summary
        ]

        feed_dict = dict()
        feed_dict[self.inputs_bxtxd] = batch_data['inputs']
        feed_dict[self.output_bxtxo] = batch_data['outputs']
        feed_dict[self.mask_bxtxo] = batch_data['masks']
        feed_dict[self.learning_rate] = self.adaptive_learning_rate()
        feed_dict[self.grad_norm_clip_val] = self.adaptive_grad_norm_clip()

        [_,
         ev_grad_global_norm,
         ev_loss,
         ev_merged_opt_summary] = \
            self.session.run(ops_to_eval, feed_dict=feed_dict)

        if self.hps.do_save_tensorboard_events:

            if self._epoch() == 0:
                '''Hack to prevent throwing the vertical axis on the
                Tensorboard figure for grad_norm_clip_val (grad_norm_clip val
                is initialized to an enormous number to prevent clipping
                before we know the scale of the gradients).'''
                feed_dict[self.grad_norm_clip_val] = np.nan
                ev_merged_opt_summary = \
                    self.session.run(self.merged_opt_summary, feed_dict)

            self.writer.add_summary(ev_merged_opt_summary, self._step())

        summary = {'loss': ev_loss, 'grad_global_norm': ev_grad_global_norm}

        return summary

    def restore_from_checkpoint(self, ckpt_tag):
        return self._initialize_or_restore_from_checkpoint(ckpt_tag)

    def _initialize_or_restore_from_checkpoint(self, ckpt_tag):
        # Initializes all Tensorflow objects, either from existing model
        # checkpoint if detected or otherwise as specified in _setup_model. If
        # starting a training run from scratch, writes a yaml file containing
        # all hyperparameter settings.

        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        ckpt_model_path = None
        for ck_ in ckpt.all_model_checkpoint_paths:
            if ckpt_tag in ck_:
                ckpt_model_path = ck_
        self.seso_saver.restore(self.session, ckpt_model_path)
        return

    def predict(self, batch_data, do_predict_full_lstm_state=False):
        data_batches = self._get_data_batches(batch_data)
        predictions = []

        for batch_data_curr in data_batches:
            if 'inputs' not in batch_data_curr.keys():
                batch_data_curr = self._load_single_batch_from_file(batch_data_curr)
            if do_predict_full_lstm_state and (self.hps.rnn_type == 'lstm'):
                predictions_curr = self._predict_with_lstm_cell_states(batch_data_curr)
            else:
                ops_to_eval = [self.hidden_bxtxd, self.pred_output_bxtxo]
                feed_dict = {self.inputs_bxtxd: batch_data_curr['inputs']}
                ev_hidden_bxtxd, ev_pred_output_bxtxo = \
                    self.session.run(ops_to_eval, feed_dict=feed_dict)

                predictions_curr = {
                    'state': ev_hidden_bxtxd,
                    'output': ev_pred_output_bxtxo,
                }
            predictions.append(predictions_curr)

        summary = {'loss': 0}
        return predictions, summary

    def _predict_with_lstm_cell_states(self, batch_data):

        hps = self.hps
        if hps.rnn_type != 'lstm':
            predictions, summary = self.predict(batch_data)
            return predictions

        n_hidden = hps.n_hidden
        [n_batch, n_time, _] = batch_data['inputs'].shape
        initial_state = self.rnn_cell.zero_state(n_batch, dtype=tf.float32)

        ''' Add ops to the graph for getting the complete LSTM state
        (i.e., hidden and cell) at every timestep.'''
        self.full_state_list = []
        for t in range(n_time):
            input_ = self.inputs_bxtxd[:, t, :]
            if t == 0:
                full_state_t_minus_1 = initial_state
            else:
                full_state_t_minus_1 = self.full_state_list[-1]
            _, full_state_bxd = self.rnn_cell(input_, full_state_t_minus_1)
            self.full_state_list.append(full_state_bxd)

        '''Evaluate those ops'''
        ops_to_eval = [self.full_state_list, self.pred_output_bxtxo]
        feed_dict = {self.inputs_bxtxd: batch_data['inputs']}
        ev_full_state_list, ev_pred_output_bxtxo = \
            self.session.run(ops_to_eval, feed_dict=feed_dict)

        '''Package the results'''
        h = np.zeros([n_batch, n_time, n_hidden])  # hidden states: bxtxd
        c = np.zeros([n_batch, n_time, n_hidden])  # cell states: bxtxd
        for t in range(n_time):
            h[:, t, :] = ev_full_state_list[t].h
            c[:, t, :] = ev_full_state_list[t].c

        ev_lstm_cell_state = tf.nn.rnn_cell.LSTMStateTuple(h=h, c=c)

        predictions = {
            'state': ev_lstm_cell_state,
            'output': ev_pred_output_bxtxo,
        }

        return predictions

    def _predict_from_earlier_checkpoint(self, batch_data, do_predict_full_lstm_state=True, checkpoint_idx=0):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        ckpt_model_path = ckpt.all_model_checkpoint_paths[checkpoint_idx]
        self.seso_saver.restore(self.session, ckpt_model_path)
        predictions, summary = self.predict(batch_data, do_predict_full_lstm_state=do_predict_full_lstm_state)
        return predictions, summary, ckpt_model_path

    def _update_validation(self, train_data, valid_data):
        is_lstm = self.hps.rnn_type == 'lstm'
        example_predictions, summary = self.predict(valid_data,
                                                    do_predict_full_lstm_state=is_lstm)
        self.validation_result.append(example_predictions)
        return

    def _get_data_batches(self, train_data):
        # train_data is a dictionary (file_path); list of files, one for each batch
        # returns one filename
        num_batches = len(train_data['file_path'])
        data_batches = []
        for bi in range(num_batches):
            tmp = {'file_path': train_data['file_path'][bi]}
            data_batches.append(tmp)

        return data_batches

    def _get_batch_size(self, batch_data):
        return self.hps.input_dims[0]

    def _setup_visualizations(self):
        pass

    def _update_visualizations(self, train_data=None, valid_data=None):
        """See docstring in RecurrentWhisperer."""
