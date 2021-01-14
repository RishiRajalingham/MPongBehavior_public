import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

sns.set_style('ticks', {"axes.linewidth": "1", 'axes.yaxis.grid': False})
import sys

PATH_TO_FIXED_POINT_FINDER = '/om/user/rishir/lib/fixed-point-finder/'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
PATH_TO_FIXED_POINT_FINDER = '/om/user/rishir/lib/PongRnn/'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)

import h5py
import numpy as np
import pickle as pk
import pandas as pd
import argparse

from datasets.PongDataset import map_batch_output
from sklearn.decomposition import PCA
from rnn_analysis import utils
from rnn_analysis import rnn_analysis_utils as RAU

parser = argparse.ArgumentParser()
parser.add_argument('--filename', default=None)

""" make sure it only looks at real trials, not the zero padding"""


class PongRNNSummarizer(object):

    def __init__(self, **kwargs):
        self.filename = kwargs.get('filename', '')
        self.fig_out_path = kwargs.get('fig_out_path', None)
        self.epoch_n = kwargs.get('epoch_n', -1)
        self.plot_detailed_summaries = kwargs.get('plot_detailed_summaries', False)
        self.group_over_batches = kwargs.get('group_over_batches', False)
        if self.fig_out_path is None:
            self.fig_out_path = "/".join(self.filename.split('/')[:-1])
        self.out_filename = '%s/%s' % (self.fig_out_path, self.filename.split('/')[-1])

        self.output_tv_cols = kwargs.get('output_tv_cols', [0, 1])
        self.output_types_index = {
            'vis-sim': [0, 1],
            'vis': [2, 3],
            'sim': [4, 5],
            'f': [6],
        }

        self.dat = pk.load(open(self.filename, 'rb'))
        self.specs = self.get_specs()

        self.results_output = {}
        self.err_output = {}
        self.model_characteristics = {}
        self.groupbyvar = []
        self.ngroups = 0
        self.error_comparisons = {}
        self.example_trials = {}

        self.out_filename_sample_output_traj = self.out_filename.replace('.pkl', '_sample_output_trajectories.pdf')
        self.out_filename_sample_state_traj = self.out_filename.replace('.pkl', '_sample_state_trajectories.pdf')
        self.out_filename_scatter = self.out_filename.replace('.pkl', '_scatter.pdf')
        self.out_filename_df_summary = self.out_filename.replace('.pkl', '_summary.pkl')
        # print('summarizer_init %s' % self.filename)
        return

    def get_specs(self):
        dat = self.dat

        specs = dat['specs']
        specs.update(dat['hps'])
        specs['n_epochs'] = dat['results']['n_epochs']

        fn = self.filename.replace('//', '/')
        type_tag = fn.split('/')[-2]
        model_tag = '%s_%d' % (dat['hps']['rnn_type'], dat['hps']['n_hidden'])
        fn_ = fn.replace('//', '/').split('/')[-2:]
        fn_[1] = fn_[1].split('_')[0]
        fn2 = "".join(fn_)

        filename_specs = {
            'filename': fn,
            'filename_tag': fn2,
            'model_tag': model_tag,
            'type_tag': type_tag,
            'type_tag_stimulus': type_tag.split('_')[0],
            'type_tag_loss': "".join(type_tag.split('_')[1:]),
        }

        specs.update(filename_specs)

        return specs

    def parse_output_per_batch(self, output, pred_output, pred_state):
        # remove zero padding necessary to match batch_size

        if pred_state.ndim == 4:
            # concatenate c and h for lstms
            c,h = pred_state[0], pred_state[1]
            pred_state_ = np.concatenate((c, h), axis=2)
            output_state_ = h
        else:
            pred_state_ = pred_state
            output_state_ = pred_state

        # remove padding required for fixed batch size (512 instead of 200)
        non_pad_idx = np.nanmean(output[:, :, :2], axis=(1, 2)) != 0
        output_ = output[non_pad_idx, :]
        pred_output_ = pred_output[non_pad_idx, :]
        pred_state_ = pred_state_[non_pad_idx, :]
        output_state_ = output_state_[non_pad_idx,:]

        res = {}
        for rk in self.output_types_index.keys():
            res['output_%s' % rk] = output_[..., self.output_types_index[rk]]
            res['pred_output_%s' % rk] = pred_output_[..., self.output_types_index[rk]]

        # remove initial condition from _f mask
        res['output_f'][:, 0, :] = np.nan
        res['state'] = pred_state_
        res['output_state'] = output_state_
        return res

    def get_results_summary_base(self):
        """ this gets the different output channels """
        dat = self.dat
        batch_fns = dat['results']['valid_data']['file_path']
        with_initial_cond = dat['hps']['with_initial_cond']
        meta_valid = dat['meta_valid']
        n_epochs = dat['results']['n_epochs']

        results = {
            'meta_valid': meta_valid,
            'n_epochs': n_epochs,
            'batch_idx': [],
            'state': [],
        }
        for rk in self.output_types_index.keys():
            results['output_%s' % rk] = []
            results['pred_output_%s' % rk] = []
        res_curr_batch = {}

        for batch_idx, batch_fn in enumerate(batch_fns):
            with h5py.File(batch_fn, 'r') as f:
                batch_input_raw = f['input'][:]
                batch_output = f['output'][:]
                batch_label = f['label'][:]

            output_all = map_batch_output(batch_input_raw, batch_output,
                                          batch_label, with_initial_cond)

            valid_output = dat['results']['example_predictions']
            if isinstance(valid_output[0], list):
                valid_output = valid_output[self.epoch_n]

            pred_output = np.squeeze(valid_output[batch_idx]['output'])
            pred_state = np.squeeze(valid_output[batch_idx]['state'])

            res_curr_batch = self.parse_output_per_batch(output_all, pred_output, pred_state)

            for rcb in res_curr_batch.keys():
                if batch_idx == 0:
                    results[rcb] = res_curr_batch[rcb]
                else:
                    results[rcb] = np.concatenate((results[rcb], res_curr_batch[rcb]), axis=0)
            rcb = list(res_curr_batch.keys())[-1]
            batch_idx_ = np.ones((res_curr_batch[rcb].shape[0],)) * batch_idx
            results['batch_idx'].extend(batch_idx_)

        output_keys = list(res_curr_batch.keys()) + ['batch_idx']
        for rcb in output_keys:
            results[rcb] = np.array(results[rcb])
            self.results_output = results
        return

    def get_model_characterization(self, characteristics_oi=None, niter=2):
        model_data = self.results_output
        model_data['filename'] = self.filename
        dist_fn = '/om/user/rishir/data/pong_basic/RF/occluded_pong_bounce1_pad8_4speed/train_to_valid_dist.pkl'
        dist_data = pk.load(open(dist_fn, 'rb'))

        res_char = {}
        all_characteristic_funcs = {
            # simulation index variations
            'decode_vis_to_sim_index': lambda x: RAU.get_model_simulation_index(x,
                                                                                train_mask='output_vis',
                                                                                test_mask='output_sim'),
            'decode_vis-sim_to_sim_index': lambda x: RAU.get_model_simulation_index(x,
                                                                                    train_mask='output_vis-sim',
                                                                                    test_mask='output_sim'),
            'decode_sim_to_sim_index': lambda x: RAU.get_model_simulation_index(x,
                                                                                train_mask='output_sim',
                                                                                test_mask='output_sim'),
            'decode_f_to_f': lambda x: RAU.get_model_simulation_index(x,
                                                                      train_mask='output_f',
                                                                      test_mask='output_f'),
            # simple geometric characteristics?
            'geom_vis-sim': lambda x: RAU.get_model_geometry(x, epoch_mask='output_vis-sim'),
            'geom_vis': lambda x: RAU.get_model_geometry(x, epoch_mask='output_vis'),
            'geom_sim': lambda x: RAU.get_model_geometry(x, epoch_mask='output_sim'),

            # 'piecewise_encoding': lambda x: RAU.get_piecewise_encoding(x),
            #
            # # encoding score from latent variables
            # 'encode_vis-sim': lambda x: RAU.get_model_encoding_score(x, epoch_mask='output_vis-sim'),
            # 'encode_vis': lambda x: RAU.get_model_encoding_score(x, epoch_mask='output_vis'),
            # 'encode_sim': lambda x: RAU.get_model_encoding_score(x, epoch_mask='output_sim'),
            #
            # # weight matrix characteristics
            'weights': lambda x: RAU.get_model_connectivity_metrics(x['filename']),
            'feedback_control': lambda x: RAU.get_output_feedback(x),
            #
            # # generalization index
            # 'generalization': lambda x: RAU.get_model_generalization_index(x, dist_data=dist_data),
            #
            # # occlusion state encoding
            # 'occlusion_state': lambda x: RAU.get_occlusion_state_characteristics(x),
            # 'cross_time_state_prediction': lambda x: RAU.get_cross_time_state_prediction(x),

            # decoding of dx,dy
            # 'gradient_rep': lambda x: RAU.get_model_decoding_of_gradient_masked(x),
            'gradient_rep_piecewise_sim': lambda x: RAU.get_piecewise_gradient_encoding(x, mask_fk='output_sim', niter=niter)

        }

        if characteristics_oi is None:
            characteristics_oi = list(all_characteristic_funcs.keys())

        for cfk in characteristics_oi:
            res_char[cfk] = all_characteristic_funcs[cfk](model_data)

        self.model_characteristics = res_char
        return

    def get_output_error_comparisons(self, niter=2):

        def get_error_metrics_base(x, y):
            mu = np.nanmean((x - y) ** 2)
            mu1 = np.nanmean(x ** 2)
            mu2 = np.nanmean(y ** 2)
            mu12 = 2 * np.nanmean(x) * np.nanmean(y)
            mae, mae_sig = np.nanmean(np.abs(x - y)), np.nanstd(np.abs(x - y))
            nse = mu / (mu1 + mu2 - mu12)

            return {
                'mae': mae,
                'nse': nse,
                'rmse': mu ** 0.5,
            }

        def get_error_metrics_dist(x,y):
            err = get_error_metrics_base(x,y)
            n = x.shape[0]
            err_dist = {'mae':[], 'nse':[], 'rmse':[]}

            for i in range(niter):
                idx = np.random.choice(n, n, replace=True)
                tmp = get_error_metrics_base(x[idx], y[idx])
                for fk in err_dist.keys():
                    err_dist[fk].append(tmp[fk])

            for fk in err_dist.keys():
                err[fk+'_sig'] = np.nanstd(err_dist[fk])
            return err

        def get_error_metrics(x, y, z):
            """ group by z, and compute errors"""
            uz = np.unique(z)
            res_all = []
            for uzi in uz:
                t = np.nonzero(z == uzi)[0]
                x_t = x[t]
                y_t = y[t]
                res_all.append(get_error_metrics_dist(x_t, y_t))
                # res_all.append(get_error_metrics_base(x_t, y_t))
            return res_all

        res = self.results_output
        self.error_comparisons = {}
        for rk in self.output_types_index.keys():
            self.error_comparisons['error_%s' % rk] = [res['output_%s' % rk], res['pred_output_%s' % rk]]

        self.groupbyvar = res['batch_idx']
        self.ngroups = np.unique(self.groupbyvar).shape[0]
        if self.group_over_batches:
            err = {}
            for ec in self.error_comparisons.keys():
                x_, y_ = self.error_comparisons[ec][0], self.error_comparisons[ec][1]
                err[ec] = get_error_metrics(x_, y_, self.groupbyvar)
        else:
            err = {}
            for ec in self.error_comparisons.keys():
                x_, y_ = self.error_comparisons[ec][0], self.error_comparisons[ec][1]
                err[ec] = get_error_metrics_base(x_, y_)

        self.err_output = err
        return

    def set_example_trials(self):
        res = self.results_output

        if 'pad' in self.dat['meta_valid'].keys():
            pad = np.unique(self.dat['meta_valid']['pad'])[0]
            img_w = np.unique(self.dat['meta_valid']['img_w'])[0]
            img_h = np.unique(self.dat['meta_valid']['img_h'])[0]
        else:
            pad = 8
            img_w = 32
            img_h = 32

        output_ = res['output_vis-sim']
        pred_output_ = res['pred_output_vis-sim']

        err = np.nanmean(np.abs(output_ - pred_output_), axis=(1, 2))
        err_s_idx = np.argsort(err)
        step_size = int(err_s_idx.shape[0] / 36)
        err_s_idx = err_s_idx[::step_size][:36]
        t_idx, t_end_idx = [], []

        for i, idx in enumerate(err_s_idx):
            tmp = np.nonzero(output_[idx, :, 0] >= (img_w - pad))
            t_idx.append(tmp[0][0])

            tmp = np.nonzero(output_[idx, :, 0] >= img_w)
            t_end_idx.append(tmp[0][0])

        self.example_trials = {
            'pad': pad,
            'img_w': img_w,
            'img_h': img_h,
            'trial_idx': err_s_idx,
            'time_idx': t_idx,
            'time_end_idx': t_end_idx,
        }
        return

    def plot_sample_trial_videos(self):

        img_w = self.example_trials['img_w']
        pad = self.example_trials['pad']
        trials_oi = self.example_trials['trial_idx']
        n_trials, n_time, n_var = self.results_output['pred_output_vis-sim'].shape
        n_frames = len(trials_oi) * n_time

        fig, ax = plt.subplots()
        ax.set_xlim((-1, img_w + 1))
        ax.set_ylim((-1, img_w + 1))
        ax.axis('equal')

        ax.fill_between([0, img_w], [0, 0], [img_w, img_w], alpha=0.2)  # frame
        ball_occluded, = ax.plot([], [], 'r.', lw=2, alpha=0.25)
        ax.fill_between([img_w - pad, img_w], [0, 0], [32, 32], alpha=0.2)  # occluder
        trial_caption = ax.text(1, 30, '')
        ball, = ax.plot([], [], 'ro', lw=2, ms=5)
        ball_simulation_vis, = ax.plot([], [], '.', lw=2, color=[0.5, 0, 0])
        ball_simulation_sim, = ax.plot([], [], '.', lw=2, color=[0.5, 0, 0])
        ball_simulation_vissim, = ax.plot([], [], '.', lw=2, color=[0.5, 0, 0])
        paddle, = ax.plot([], [], 'g-', lw=3)
        fig.suptitle(self.specs['loss_weight_type'])

        def init():
            ball.set_data([], [])
            ball_occluded.set_data([], [])
            ball_simulation_vis.set_data([], [])
            ball_simulation_sim.set_data([], [])
            ball_simulation_vissim.set_data([], [])
            paddle.set_data([], [])
            trial_caption.set_text('')
            return (ball, ball_occluded, ball_simulation_vis,
                    ball_simulation_sim, ball_simulation_vissim, paddle)

        def animate(i):
            trial_idx = trials_oi[int(i / n_time)]
            time_idx = np.mod(i, n_time)
            trial_caption.set_text('%d' % trial_idx)
            x, y = self.results_output['output_vis-sim'][trial_idx, time_idx, :]
            if x > (img_w - pad):
                ball.set_data([], [])
                ball_occluded.set_data(x, y)
            else:
                ball_occluded.set_data([], [])
                ball.set_data(x, y)

            x, y = self.results_output['pred_output_vis'][trial_idx, time_idx, :]
            ball_simulation_vis.set_data(x, y)

            x, y = self.results_output['pred_output_sim'][trial_idx, time_idx, :]
            ball_simulation_sim.set_data(x, y)

            x, y = self.results_output['pred_output_vis-sim'][trial_idx, time_idx, :]
            ball_simulation_vissim.set_data(x, y)

            x, y = img_w, self.results_output['pred_output_f'][trial_idx, time_idx, :]
            x = [x, x]
            y = [y - 1, y + 1]
            paddle.set_data(x, y)
            return (ball, ball_occluded, ball_simulation_vis,
                    ball_simulation_sim, ball_simulation_vissim, paddle)

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=n_frames, interval=20, blit=True)
        anim.save(self.out_filename_sample_output_traj.replace('.pdf', '.gif'),
                  writer='imagemagick', fps=30)
        return

    def plot_sample_state_trajectories(self, with_pca=True):

        def flatten_pca(x_, n_comp=3):
            s = x_.shape
            xs = np.reshape(x_, (s[0] * s[1], s[2]))
            pca = PCA(n_components=n_comp)
            xs2 = pca.fit_transform(xs)
            return np.reshape(xs2, (s[0], s[1], n_comp)), pca

        def plot_sample_traj_base(x_, outfn):
            t = np.tile(range(x_.shape[1]), (x_.shape[2], 1))
            f, axes = plt.subplots(6, 6, figsize=(20, 20))
            axes = axes.ravel()
            for i, idx in enumerate(self.example_trials['trial_idx']):
                tx = self.example_trials['time_idx'][i]
                tex = self.example_trials['time_end_idx'][i]
                axes[i].plot(t[:, :tx].T, x_[idx, :tx], lw=2)
                axes[i].plot(t[:, tx:].T, x_[idx, tx:], '-.')
                axes[i].axvline(tx, linestyle='--')
                axes[i].axvline(tex, linestyle='--')
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                plt.tight_layout()
                sns.despine(ax=axes[i], trim=False, offset=2)

            f.suptitle(self.specs['loss_weight_type'])
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            f.savefig(outfn)
            plt.close(fig=f)

        res = self.results_output
        plot_sample_traj_base(res['state'], self.out_filename_sample_state_traj)

        if with_pca:
            x = res['state']
            n_comp_, n_samples = 3, 1000
            # x_idx = np.random.choice(x.shape[0], n_samples, replace=False)
            # x_subsampled = x[x_idx, :, :]
            x_pca, pca_ = flatten_pca(x, n_comp=n_comp_)
            outfn_pca = self.out_filename_sample_state_traj.replace('.pdf', '_pca.pdf')
            plot_sample_traj_base(x_pca, outfn_pca)

        return

    def plot_sample_output_trajectories(self, output_var='vis-sim'):
        res = self.results_output
        img_w, img_h = self.example_trials['img_w'], self.example_trials['img_h']
        pad = self.example_trials['pad']

        output_ = res['output_%s' % output_var]
        pred_output_ = res['pred_output_%s' % output_var]

        trial_idx = self.example_trials['trial_idx']
        time_idx = self.example_trials['time_idx']

        f, axes = plt.subplots(6, 6, figsize=(20, 20))
        axes = axes.ravel()
        for i, idx in enumerate(trial_idx):
            axes[i].fill_between([0, img_w], [0, 0], [img_h, img_h], facecolor='gray', alpha=0.1)
            axes[i].fill_between([img_w - pad, img_w], [0, 0], [img_h, img_h], facecolor='blue', alpha=0.2)
            tx = time_idx[i]
            axes[i].plot(output_[idx, tx:, 0], output_[idx, tx:, 1], 'b--')
            axes[i].plot(pred_output_[idx, tx:, 0], pred_output_[idx, tx:, 1], 'r--')

            axes[i].plot(output_[idx, :tx, 0], output_[idx, :tx, 1], 'b.')
            axes[i].plot(pred_output_[idx, :tx, 0], pred_output_[idx, :tx, 1], 'r.')

            axes[i].axvline(img_w - pad, linestyle='--')

            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_xlim([-1, img_w + 1])
            axes[i].set_ylim([-1, img_h + 1])
            sns.despine(ax=axes[i], trim=False, offset=2)

        f.suptitle(output_var)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        f.savefig(self.out_filename_sample_output_traj)
        plt.close(fig=f)
        return

    def plot_output_error_summary(self):
        def align_to_trial_mark(output, pred_output, i_, j_, pre_time_pad=5, post_time_pad=5):
            ui = np.unique(i_)
            uj = [j_[i_ == ii][0] for ii in ui]
            pred_output_aligned = []
            for i, j in zip(ui, uj):
                tmp_p = pred_output[i, (j - pre_time_pad):(j + post_time_pad)]
                if tmp_p.shape[0] != (pre_time_pad + post_time_pad):
                    tmp_p = np.ones((pre_time_pad + post_time_pad, 1)) * np.nan
                pred_output_aligned.append(tmp_p)

            pred_output_aligned = np.squeeze(pred_output_aligned)
            output_aligned = np.tile(np.nansum(output[ui], axis=1), (1, pre_time_pad + post_time_pad))
            # time_ax1 = list(np.linspace(-pre_time_pad, 0, pre_time_pad))
            # time_ax2 = list(np.linspace(0, post_time_pad, post_time_pad))
            time_ax = list(np.arange(-pre_time_pad, post_time_pad, 1))
            return output_aligned, pred_output_aligned, time_ax

        res = self.results_output
        output_f, pred_output_f = res['output_f'], res['pred_output_f']
        output_xy, pred_output_xy = res['output_vis-sim'], res['pred_output_vis-sim']

        g = sns.jointplot(output_f, pred_output_f, kind="hex",
                          height=4, space=0, color="b").set_axis_labels("True y_f", "Predicted y_f")
        g.fig.suptitle('%s: %d epochs' % (self.specs['loss_weight_type'], self.results_output['n_epochs']))
        g.savefig(self.out_filename_scatter)

        f, axes = plt.subplots(1, 3, figsize=(5, 3))

        pad = np.unique(self.dat['meta_valid']['pad'])[0]
        img_w = np.unique(self.dat['meta_valid']['img_w'])[0]

        i1, j1 = np.nonzero(output_xy[:, :, 0] > img_w)
        o1, p1, t1 = align_to_trial_mark(output_f, pred_output_f, i1, j1)
        i2, j2 = np.nonzero(output_xy[:, :, 0] > (img_w - pad))
        o2, p2, t2 = align_to_trial_mark(output_f, pred_output_f, i2, j2)
        # i3, j3 = np.nonzero(np.diff(res['output_mb'][:, :, 0], axis=1) > 0)
        # o3, p3, t3 = align_to_trial_mark(output_f, pred_output_f, i3, j3)

        err_to_plot = {
            'occlude': np.abs(o2 - p2),
            'end': np.abs(o1 - p1),
            # 'bounce': np.abs(o3 - p3),
        }

        for ei, err_fk in enumerate(err_to_plot.keys()):
            err = err_to_plot[err_fk]
            err_mu = np.nanmean(err, axis=0)
            err_sig = np.nanstd(err, axis=0)
            axes[ei].errorbar(t1, err_mu, yerr=err_sig)
            axes[ei].set_ylabel('MAE')
            axes[ei].set_xlabel('t')
            axes[ei].set_title('Aligned to %s' % err_fk)
        plt.tight_layout()
        f.suptitle(self.specs['loss_weight_type'])
        f.savefig(self.out_filename_scatter.replace('.pdf', '_2.pdf'))
        plt.close(fig=f)
        return

    def plot_output_error_detailed(self):
        mae = np.abs(self.error_comparisons['f'][0] - self.error_comparisons['f'][1])
        err = np.nanmean(mae, axis=1)

        meta_err = self.results_output['meta_valid'].iloc[:err.shape[0]]
        meta_err = meta_err[['x0', 'y0', 'speed0', 'heading0']].reset_index(drop=True)

        for fk in meta_err.keys():
            if np.unique(meta_err[fk]).shape[0] > 10:
                meta_err[fk] = pd.qcut(meta_err[fk], 10)

        meta_err['error'] = err
        f, axes = plt.subplots(1, 2, figsize=(10, 5))
        vars_to_plot = [['y0', 'x0'], ['speed0', 'heading0']]
        for vpi, vp in enumerate(vars_to_plot):
            tmp = meta_err.groupby(vp).mean()['error'].unstack()
            sns.heatmap(tmp, cmap="YlGnBu", ax=axes[vpi]).set_title('Error')
        plt.tight_layout()
        f.suptitle(self.specs['loss_weight_type'])
        f.savefig(self.out_filename_scatter.replace('.pdf', '_3.pdf'))
        plt.close(fig=f)
        return

    def plot_sample_state_videos(self):
        def sample_trials(res_, nsamples=1000):
            x = res_['state']
            meta = res_['meta_valid']
            nsamples = np.min([nsamples, x.shape[0]])
            ind = np.random.choice(range(x.shape[0]), nsamples, replace=False)
            x_ = x[ind]
            meta_curr = meta.iloc[ind]
            ref = res_['output_vis-sim'][ind]
            x_[np.isnan(ref[:, :, 0]), :] = np.nan
            return x_, meta_curr, ref

        def midrange(x):
            mm = np.nanmax(x, axis=(0, 1))
            m = np.nanmin(x, axis=(0, 1))
            return (x - m) / (mm - m)

        res = self.results_output
        states_, meta_, ref_ = sample_trials(res, nsamples=1000)
        states_3d, pca = utils.flatten_pca(states_)

        dims_to_plot = [[0, 1], [0, 2], [1, 2]]
        for dp in dims_to_plot:
            xx, yy = states_3d[:, :, dp[0]], states_3d[:, :, dp[1]]
            ss = np.array(meta_['pad']) * 2
            cc = midrange(np.tile(meta_['y0'], (xx.shape[1], 1)).T)

            scatter_data = {'X': xx, 'Y': yy, 'S': ss, 'C': cc}
            suffix = '_%d%d.gif' % (dp[0], dp[1])
            out_fn = self.out_filename_sample_state_traj.replace('.pdf', suffix)
            utils.scatter_animation(scatter_data, out_fn, title_str=self.specs['loss_weight_type'])
        return

    def save_summary_dataframe(self):
        df = pd.DataFrame()
        spec_all = []
        for gi in range(self.ngroups):
            spec_curr = {}
            spec_curr.update(self.specs)
            for ec in self.error_comparisons.keys():
                if self.group_over_batches:
                    err_batch = self.err_output[ec][gi]
                else:
                    err_batch = self.err_output[ec]
                for err_fn in err_batch.keys():
                    spec_curr['%s_%s' % (ec, err_fn)] = err_batch[err_fn]
            spec_all.append(spec_curr)
        df = df.append(spec_all, ignore_index=True)
        df.to_pickle(self.out_filename_df_summary)

    def run_performance_summary(self):
        self.get_results_summary_base()
        self.get_output_error_comparisons()
        self.save_summary_dataframe()
        return

    def run_all(self):

        self.get_results_summary_base()
        self.get_output_error_comparisons()
        self.get_model_characterization()

        if self.plot_detailed_summaries:
            self.set_example_trials()
            self.plot_sample_output_trajectories()
            self.plot_sample_state_trajectories()
            self.plot_sample_trial_videos()
            self.plot_sample_state_videos()
            self.plot_output_error_summary()
            self.plot_output_error_detailed()

        self.save_summary_dataframe()

        return


def main(argv):
    flags_, _ = parser.parse_known_args(argv)
    flags = vars(flags_)
    summarizer = PongRNNSummarizer(**flags)
    summarizer.run_all()
    return


if __name__ == "__main__":
    main(sys.argv[1:])
