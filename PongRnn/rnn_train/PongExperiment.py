
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc

import sys, os
import pickle as pk

PATH_TO_FIXED_POINT_FINDER = '/om/user/rishir/lib/fixed-point-finder/'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
PATH_ = '/om/user/rishir/lib/PongRnn/'
sys.path.insert(0, PATH_)

# from FixedPointFinder import FixedPointFinder
from rnn_train.GeneralEstimationRNN import GeneralEstimationRnn
from rnn_analysis.PongRNNSummarizer import PongRNNSummarizer
default_file_path = '/om/user/rishir/data/pong_basic/RF/occluded_pong_bounce2_pad8_10speed/'
from glob import glob


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


DEFAULT_SAVE_PATH = '/om/user/rishir/lib/PongRnn/dat/rnn_res/'
# DEFAULT_SAVE_PATH = '/om2/user/rishir/model_data/dat/rnn_res/'

def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__



class PongExperiment(object):

    def __init__(self, **kwargs):

        self.exp_fn = kwargs.get('exp_fn', '')
        self.file_path = kwargs.get('file_path', default_file_path)
        self.input_representation = kwargs.get('input_representation', 'pixel')
        self.output_representation = kwargs.get('output_representation', 'pred')

        self.multisensory = kwargs.get('multisensory', 0)
        self.noise_sigma = kwargs.get('noise_sigma', 0.0)
        self.surrogate_label = kwargs.get('surrogate_label', False)

        self.loss_weight_type = kwargs.get('loss_weight_type', 'all')
        self.regularization_weight_simplicity_DS = kwargs.get('regularization_weight_simplicity_DS', 0.0)
        self.regularization_weight_simplicity_DS2 = kwargs.get('regularization_weight_simplicity_DS2', 0.0)
        self.regularization_weight_L2_activity = kwargs.get('regularization_weight_L2_activity', 0.0)
        self.regularization_weight = kwargs.get('regularization_weight', 0.0)
        self.regularization_loss = kwargs.get('regularization_loss', 'L1')
        self.with_initial_cond = kwargs.get('with_initial_cond', True)
        self.train_frac = kwargs.get('train_frac', 1.0)
        self.rnn_type = kwargs.get('rnn_type', 'lstm')
        self.n_hidden = kwargs.get('n_hidden', 25)
        self.min_loss = kwargs.get('min_loss', None)
        self.min_learning_rate = kwargs.get('min_learning_rate', 1e-10)
        self.log_dir = kwargs.get('log_dir', './logs/')
        self.alr_hps = kwargs.get('alr_hps', {'initial_rate': 0.1})
        self.max_n_epochs = kwargs.get('max_n_epochs', 1000)
        self.n_epochs_per_ckpt = kwargs.get('n_epochs_per_ckpt', 100)
        self.max_train_time = kwargs.get('max_train_time', 2.0*24.0*3600.0)
        self.do_restart_run = kwargs.get('do_restart_run', False)
        self.random_seed = kwargs.get('random_seed', 0)

        self.save_path = kwargs.get('save_path', DEFAULT_SAVE_PATH)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # self.save_path = DEFAULT_SAVE_PATH
        self.name = ''
        self.set_name()
        self.data = {}
        self.set_data()
        self.hps = {}
        self.set_hyperparameters()
        self.rnn = None
        self.results = {}
        self.n_checkpoints = None
        return

    def set_name(self):
        self.name = '%s_%s%d_lw%s_%2.2f' % (self.output_representation,
                                            self.rnn_type, self.n_hidden,
                                            self.loss_weight_type, self.train_frac)
        self.name = '%s_%s%2.2f' % (self.name, self.regularization_loss, self.regularization_weight)

        if self.with_initial_cond:
            self.name = '%s_%s' % (self.name, '_IC')
        if self.input_representation is not None:
            self.name = '%s_%s' % (self.name, self.input_representation)
        if self.multisensory > 0:
            self.name = '%s_ms' % self.name
        if self.surrogate_label:
            self.name = '%s_surrogate' % self.name
        if self.noise_sigma > 0:
            self.name = '%s_%2.2fnoise' % (self.name, self.noise_sigma)
        if self.regularization_weight_L2_activity > 0:
            self.name = '%s_L2reg%2.2f' % (self.name, self.regularization_weight_L2_activity)
        if self.regularization_weight_simplicity_DS > 0:
            self.name = '%s_simreg%2.2f' % (self.name, self.regularization_weight_simplicity_DS)
        if self.regularization_weight_simplicity_DS2 > 0:
            self.name = '%s_simreg2%2.2f' % (self.name, self.regularization_weight_simplicity_DS2)
        return

    def set_data(self):
        """ Specify. """
        self.data = {
            'train_data': [],
            'valid_data': [],
            'input_dims': [],
            'output_dims': [],
        }
        return

    def set_hyperparameters(self):
        self.hps = {
            'input_dims': self.data['input_dims'],
            'output_dims': self.data['output_dims'],
            'input_representation': self.input_representation,
            'output_representation': self.output_representation,
            'with_initial_cond': self.with_initial_cond,
            'multisensory': self.multisensory,
            'loss_weight_type': self.loss_weight_type,
            'regularization_weight': self.regularization_weight,
            'regularization_loss': self.regularization_loss,
            'regularization_weight_L2_activity': self.regularization_weight_L2_activity,
            'regularization_weight_simplicity_DS': self.regularization_weight_simplicity_DS,
            'regularization_weight_simplicity_DS2': self.regularization_weight_simplicity_DS2,
            'train_frac': self.train_frac,
            'surrogate_label': self.surrogate_label,
            'rnn_type': self.rnn_type,
            'n_hidden': self.n_hidden,
            'min_loss': self.min_loss,
            'min_learning_rate': self.min_learning_rate,
            'log_dir': self.log_dir,
            'alr_hps': self.alr_hps,
            'max_n_epochs': self.max_n_epochs,
            'max_n_epochs_without_lvl_improvement': self.max_n_epochs,
            'n_epochs_per_ckpt': self.n_epochs_per_ckpt,
            'max_train_time': self.max_train_time,
            'do_restart_run': self.do_restart_run,
            'random_seed': self.random_seed,
            'noise_sigma': self.noise_sigma,
        }
        print(self.hps)
        return

    def init_rnn_model(self):
        rnn_hps = self.hps
        self.rnn = GeneralEstimationRnn(**rnn_hps)
        return

    def train_model(self, save_initial_ckpt=False):
        rnn = self.rnn
        train_data, valid_data = self.data['train_data'], self.data['valid_data']
        if self.do_restart_run and save_initial_ckpt:
            print('Saving initial checkpoint.')
            rnn._save_checkpoint()
        print('Training...')
        # blockPrint()
        rnn.train(train_data=train_data)
        # enablePrint()
        n_epochs = rnn._epoch()
        print('Training complete in %d epochs.' % n_epochs)
        return

    def test_model(self, checkpoint_idx=None):
        # gc.collect()
        print('test_model start')
        sys.stdout.flush()
        rnn = self.rnn
        train_data, valid_data = self.data['train_data'], self.data['valid_data']
        is_lstm = rnn.hps.rnn_type == 'lstm'
        if checkpoint_idx is None:
            example_predictions, summary = rnn.predict(valid_data,
                                                   do_predict_full_lstm_state=is_lstm)
            file_suffix = 'final'
        else:
            example_predictions, summary, ckpt_model_path = rnn._predict_from_earlier_checkpoint(valid_data,
                                                                                do_predict_full_lstm_state=is_lstm,
                                                                              checkpoint_idx=checkpoint_idx)
            file_suffix = ckpt_model_path.split('.')[-1]

        n_epochs = rnn._epoch()
        self.exp_fn = rnn.run_dir.split('/')[-1]

        res = {
            'file_suffix': file_suffix,
            'n_epochs': n_epochs,
            'valid_data': valid_data,
            'example_predictions': [example_predictions],
        }
        print('test_model end')
        sys.stdout.flush()
        return res

    def save_results(self, results, save_summary=True):
        specs = {
            'name': self.name,
            'file_path': self.file_path,
        }
        results_summary = {
            'results': results,
            'hps': self.hps,
            'meta_valid': self.data['valid_data']['meta'],
            'specs': specs
        }
        run_save_path = '%s/%s/' % (self.save_path, self.name)
        run_save_fn = '%s/%s_%s_rnn_results.pkl' % (run_save_path, self.exp_fn, results['file_suffix'])
        if not os.path.exists(run_save_path):
            os.makedirs(run_save_path)
        print(results_summary)
        print('Saving to %s' % run_save_fn)
        sys.stdout.flush()
        with open(run_save_fn, 'wb') as f:
            f.write(pk.dumps(results_summary))

        if save_summary:
            print('Saving summary to %s' % run_save_fn)
            n_epochs_to_save = len(results['example_predictions'])
            sys.stdout.flush()
            for ne in range(n_epochs_to_save):
                print(ne)
                sys.stdout.flush()
                prs = PongRNNSummarizer(filename=run_save_fn, epoch_n=ne,
                                        fig_out_path=run_save_path)
                # prs.run_all()
                prs.run_performance_summary()

        return

    def test_and_save_checkpoints(self, save_summary=True, run_all_checkpoints=True):

        res = self.test_model(checkpoint_idx=None)
        self.save_results(res, save_summary=save_summary)

        if run_all_checkpoints:
            print(self.rnn.ckpt_dir)
            self.n_checkpoints = len(glob(self.rnn.ckpt_dir + '/*.index'))
            print(self.n_checkpoints)
            for ch_idx in range(self.n_checkpoints):
                res = self.test_model(checkpoint_idx=ch_idx)
                print('tested ckpt %d' % ch_idx)
                self.save_results(res, save_summary=save_summary)
                print('saved ckpt %d' % ch_idx)

        return

    def run_all(self, train=True, test=False,
                save_initial_ckpt=False,
                save_summary=True,
                run_all_checkpoints=True):
        """ start here"""
        print('init')
        sys.stdout.flush()
        self.init_rnn_model()

        if train:
            print('train')
            sys.stdout.flush()
            self.train_model(save_initial_ckpt=save_initial_ckpt)
        if test:
            print('test and save')
            sys.stdout.flush()
            self.test_and_save_checkpoints(save_summary=save_summary,
                                           run_all_checkpoints=run_all_checkpoints)

        print('done')
        sys.stdout.flush()
        return
