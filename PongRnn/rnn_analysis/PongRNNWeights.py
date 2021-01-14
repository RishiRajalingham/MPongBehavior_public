import sys

PATH_ = '/om/user/rishir/lib/PongRnn/'
sys.path.insert(0, PATH_)
import numpy as np
import pickle as pk
import os

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from rnn_train.GeneralEstimationRNN import GeneralEstimationRnn
import tensorflow as tf


class RnnWeightExtracter(object):
    def __init__(self, model_fn):
        self.model_fn = model_fn
        model_log_hash = self.model_fn.split('/')[-1].split('_')[0]
        self.ckpt_state_fn = 'checkpoint.' + model_fn.split('/')[-1].split('_')[1]
        self.log_dir = '/om/user/rishir/lib/PongRnn/rnn_train/logs/%s' % model_log_hash
        self.name = model_log_hash
        self.is_lstm = 'lstm' in self.model_fn
        self.train_vars = {}
        self.W = {}

        return

    def get_rnn(self):
        log_dir = self.log_dir
        model_fn = self.model_fn

        model_dat = pk.load(open(model_fn, 'rb'))
        hps = model_dat['hps']
        hps['do_restart_run'] = False
        hps['log_dir'] = './tmp_logs'

        self.hps = hps

        rnn = GeneralEstimationRnn(**hps)
        rnn.ckpt_dir = os.path.join(log_dir, 'ckpt/')
        rnn.hps_dir = os.path.join(log_dir, 'hps/')
        rnn.ckpt_path = os.path.join(rnn.ckpt_dir, 'checkpoint.ckpt')
        rnn.hps_path = os.path.join(rnn.hps_dir, 'hyperparameters.pkl')
        rnn.hps_yaml_path = os.path.join(rnn.hps_dir, 'hyperparameters.yml')
        rnn.alr_path = os.path.join(rnn.hps_dir, 'learn_rate.pkl')
        rnn.agnc_path = os.path.join(rnn.hps_dir, 'norm_clip.pkl')

        rnn._initialize_or_restore_from_checkpoint(self.ckpt_state_fn)
        self.rnn = rnn
        return

    def get_weights(self):
        def parse_weights(weight_vals, input_size=100, ngates=4):
            W = {}
            W_splits = np.split(weight_vals, ngates, axis=1)
            for i in range(ngates):
                W['w_x_gate%d' % i] = W_splits[i][:input_size, :]
                W['w_h_gate%d' % i] = W_splits[i][input_size:, :]
            return W

        tvs = tf.trainable_variables()
        self.train_vars = {}
        for tv in tvs:
            self.train_vars[tv.name] = tv.eval(session=self.rnn.session)

        fn_copy = ['w_out:0', 'b_out:0']
        for fnc in fn_copy:
            self.W[fnc] = self.train_vars[fnc]
        if self.is_lstm:
            # since output is only read-out from hidden state, don't parse cell state weights for output feedback metric.
            lstm_weight_vals = self.train_vars['rnn/lstm_cell/kernel:0']
            self.W.update(parse_weights(lstm_weight_vals, input_size=100, ngates=4))
        else:
            gru_weight_vals = self.train_vars['rnn/gru_cell/gates/kernel:0']
            self.W.update(parse_weights(gru_weight_vals, input_size=100, ngates=2))
        return

    def save_weights(self):
        outfn = self.model_fn.replace('.pkl', '_weights.pkl')
        with open(outfn, 'wb') as f:
            f.write(pk.dumps(self.W))
        return

    def run_all(self):
        self.get_rnn()
        self.get_weights()
        self.save_weights()
        print('Done %s ' % self.model_fn)
        return


def extract_and_save_weights(fn):
    # fn = '/om/user/rishir/lib/PongRnn/dat/rnn_res/pred_gru20_lwvis-mov_0.50_L20.01__IC_gabor_pca/7b536f9176_ckpt-124500_rnn_results.pkl'
    tf.reset_default_graph()
    with tf.Session() as sess:
        RRO = RnnWeightExtracter(fn)
        RRO.run_all()
    return

# model_fns = '/om/user/rishir/lib/PongRnn/dat/rnn_res/model_filepaths.pkl'
# fns = pk.load(open(model_fns, 'rb'))
# for fn in fns:
#     extract_and_save_weights(fn)
