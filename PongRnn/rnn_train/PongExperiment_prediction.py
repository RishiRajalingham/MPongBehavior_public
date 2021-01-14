from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, argparse
import numpy as np
import tensorflow as tf
import pickle as pk
import h5py

PATH_ = '/om/user/rishir/lib/PongRnn/'
sys.path.insert(0, PATH_)

from rnn_train.GeneralEstimationRNN import GeneralEstimationRnn
from rnn_train.PongExperiment import PongExperiment
from datasets.PongDataset import map_batch_output, map_with_sine

default_file_path = '/om/user/rishir/data/pong_basic/RF/occluded_pong_bounce1_pad8_4speed/'
default_log_path = './logs/'
default_save_path = '/om/user/rishir/lib/PongRnn/dat/rnn_res/'


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default=default_file_path)
parser.add_argument('--log_dir', default=default_log_path)
parser.add_argument('--save_path', default=default_save_path)
parser.add_argument('--input_representation', default='pxl')
parser.add_argument('--output_representation', default='pred')
parser.add_argument('--multisensory', default=0.0, type=float)
parser.add_argument('--noise_sigma', default=0.0, type=float)
parser.add_argument('--rnn_type', default='lstm')
parser.add_argument('--n_hidden', default=25, type=int)
parser.add_argument('--loss_weight_type', default='vis-sim-mov')
parser.add_argument('--regularization_weight_simplicity_DS', default=0.0, type=float)
parser.add_argument('--regularization_weight_simplicity_DS2', default=0.0, type=float)
parser.add_argument('--regularization_weight_L2_activity', default=0.0, type=float)
parser.add_argument('--regularization_weight', default=0.0, type=float)
parser.add_argument('--regularization_loss', default='L1')
parser.add_argument('--with_initial_cond', default=True, type=str2bool)
parser.add_argument('--surrogate_label', default=False, type=str2bool)
parser.add_argument('--max_train_time', default=3600.0, type=float)
parser.add_argument('--max_n_epochs', default=1000, type=int)
parser.add_argument('--n_epochs_per_ckpt', default=100, type=int)
parser.add_argument('--train_frac', default=1.0, type=float)
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--do_restart_run', default=True, type=str2bool)
parser.add_argument('--train', default=True, type=str2bool)
parser.add_argument('--test', default=True, type=str2bool)

""" 
Instantiation of pong experiment where the input data is generated from 
PongDataset, where every trial is of different durations, and not aligned to 
a particular trial event (i.e. asynchronous). 

In addition to asynch, this version of the experiment separates 
vision, simulation, and movement into three different readouts. 

"""


class EstimationRnnAsyncMultiChannel(GeneralEstimationRnn):

    def _setup_one_loss(self, output, suffix='',
                        movement_index=6, weight_channels_equally=True):

        n_hidden = self.hps.n_hidden
        n_output = int(output.shape[-1])

        np_w_out, np_b_out = self._np_init_weight_matrix(n_hidden, n_output)
        w_out = tf.Variable(np_w_out, dtype=tf.float32, name='w_out' + suffix)
        b_out = tf.Variable(np_b_out, dtype=tf.float32, name='b_out' + suffix)
        pred_output_bxtxo_raw = tf.tensordot(self.hidden_bxtxd, w_out, axes=1) + b_out

        initial_position = np.zeros(pred_output_bxtxo_raw.shape)
        initial_position[:, 0, movement_index] = 16.0
        pred_output_bxtxo = pred_output_bxtxo_raw + initial_position

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
        loss = loss + tf.cast(reg_loss, tf.float32)

        return pred_output_bxtxo, loss

    def _load_single_batch_from_file(self, batch_data):
        """ fill in for how to set up inputs and losses. """

        def manage_nans(x):
            sx = x.shape
            x[np.isnan(x)] = 0
            return np.reshape(x, sx)

        def get_pixel_inputs(input_raw_, input_pxl_):
            if self.hps.noise_sigma > 0.0:
                rnd = np.random.RandomState(seed=0)
                noise = np.random.normal(0, self.hps.noise_sigma, input_pxl_.shape)
                input_pxl_ = input_pxl_ + noise

            if self.hps.multisensory > 0.0:
                input_1 = input_raw_[..., 4]
                inputs = np.concatenate((input_pxl_, input_1), axis=2)
            else:
                inputs = input_pxl_
            return inputs

        file_path = batch_data['file_path']
        with h5py.File(file_path, 'r') as f:
            input_raw = f['input'][:]  # {x, y, heading, speed, bounce, slope, offset}(t)
            input_pxl = f['input_%s' % self.hps.input_representation][:]
            # input_pxl = f['input_pxl'][:]
            output = f['output'][:]  # {y_paddle(t)}
            label = f['label'][:]  # {[y_final, trial_end_idx, vis_end_idx]}

        inputs = get_pixel_inputs(input_raw, input_pxl)
        output_all = map_batch_output(input_raw, output, label, self.hps.with_initial_cond)

        if self.hps.loss_weight_type == 'sim-mov':
            w = [1, 1, 0, 0, 0, 0, 1, 0]
        elif self.hps.loss_weight_type == 'vis-sim-mov':
            w = [0, 0, 1, 1, 1, 1, 1, 0]
        elif self.hps.loss_weight_type == 'vis-mov':
            w = [0, 0, 1, 1, 0, 0, 1, 0]
        elif self.hps.loss_weight_type == 'mov':
            w = [0, 0, 0, 0, 0, 0, 1, 0]
        elif self.hps.loss_weight_type == 'step':
            w = [0, 0, 0, 0, 0, 0, 1, 1]
        elif self.hps.loss_weight_type == 'sine':
            output_all = map_with_sine(output_all)
            w = [0, 0, 0, 0, 0, 0, 1, 1]
        w = w / np.sum(w)
        so = output.shape
        w_mat = np.tile(w, (so[0], so[1], 1))
        masks = np.isfinite(output_all).astype(float)
        masks = np.multiply(w_mat, masks)

        batch_data = {
            'inputs': manage_nans(inputs),
            'outputs': manage_nans(output_all),
            'masks': masks,
        }
        return batch_data


class PongExperimentAsynchMultiChannel(PongExperiment):

    def set_data(self):
        train_fn = self.file_path + 'train.pkl'
        valid_fn = self.file_path + 'valid_small.pkl'

        train_data = pk.load(open(train_fn, 'rb'))
        valid_data = pk.load(open(valid_fn, 'rb'))

        train_data_idx = int(len(train_data['file_path']) * self.train_frac)
        train_data['file_path'] = train_data['file_path'][:train_data_idx]

        d_input = list(train_data['dim_input_%s' % self.input_representation])
        if self.multisensory:
            d_input[-1] += 1

        d_output = list(train_data['dim_input'])
        d_output[-1] = 8

        self.data = {
            'train_data': train_data,
            'valid_data': valid_data,
            'input_dims': d_input,
            'output_dims': d_output,
        }
        return

    def init_rnn_model(self):
        rnn_hps = self.hps
        self.rnn = EstimationRnnAsyncMultiChannel(**rnn_hps)
        return


def main(argv):
    print('Pong Experiment prediction')
    flags_, _ = parser.parse_known_args(argv)
    flags = vars(flags_)
    train_flag = flags.pop('train')
    test_flag = flags.pop('test')
    pong_exp = PongExperimentAsynchMultiChannel(**flags)
    pong_exp.run_all(train=train_flag, test=test_flag,
                     save_summary=True, run_all_checkpoints=True)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
