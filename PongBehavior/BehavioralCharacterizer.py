import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import pickle as pk
import sys
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from scipy import linalg
import warnings
from copy import deepcopy

sns.set_style('ticks', {"axes.linewidth": "1", 'axes.yaxis.grid': False})

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

sns.set_style('ticks', {"axes.linewidth": "1", 'axes.yaxis.grid': False})
PATH_ = '/om/user/rishir/lib/MentalPong/'
sys.path.insert(0, PATH_)

"""
This is useful to run the full comparison between hundreds of models and (dozens of?) primates, but can be unwieldy for
one quick comparison between two datasets.
"""


class BehavioralCharacterizer(object):
    def __init__(self):
        self.standard_ntime = 100
        self.niter = 10
        self.py_meta_idx_oi = None

        self.masks = None
        self.model_fks = None
        self.res_model_dat = None  # all metrics for all models/primates
        self.model_characterization = None  # characterization (e.g. simulation index)
        self.model_specs = None
        return

    def set_trials_oi(self, dat):
        """ Note that py_meta_idx_oi is automatically sorted by np.unique.
        numpy_groupby also sorts based on the same np.unique. """
        if isinstance(dat, list):
            tmp = None
            for dat_ in dat:
                tmp_curr = np.unique(dat_.datasets[0]['scalar']['py_meta_index'])
                if tmp is None:
                    tmp = tmp_curr
                else:
                    tmp = np.intersect1d(tmp, tmp_curr)
            self.py_meta_idx_oi = tmp
        elif isinstance(dat, dict):
            tmp = None
            for dat_fk in dat.keys():
                dat_ = dat[dat_fk]
                tmp_curr = np.unique(dat_.datasets[0]['scalar']['py_meta_index'])
                if tmp is None:
                    tmp = tmp_curr
                else:
                    tmp = np.intersect1d(tmp, tmp_curr)
            self.py_meta_idx_oi = tmp
        else:
            self.py_meta_idx_oi = np.unique(dat.datasets[0]['scalar']['py_meta_index'])
        return

    def get_trials_oi(self, meta):
        idx_ = meta['py_meta_index']
        if self.py_meta_idx_oi is not None:
            idx = idx_[idx_.isin(self.py_meta_idx_oi)].index
        else:
            idx = idx_.index
        meta_new = meta.iloc[idx].reset_index(drop=True)
        return meta_new, idx

    def set_dataset_variables(self, model_dat, required_model_fn_substring='_IC'):
        dataset_info = model_dat['dataset_info']
        specs = model_dat['specs']
        meta = dataset_info['meta_valid']
        meta['py_meta_index'] = meta.index
        meta = meta.reset_index(drop=True)

        meta, idx = self.get_trials_oi(meta)
        groupvar = np.array(meta['py_meta_index'])

        mask_vars = {'output_sim', 'output_vis-sim', 'output_vis', 'output_f'}
        masks = {}
        for mfk in mask_vars:
            tmp = np.squeeze(dataset_info[mfk][:, :, 0])
            mask_tmp = self.pad_to_standard_size(tmp[idx, :])
            mask_tmp, _ = self.numpy_groupby(mask_tmp, groupvar)
            mask_tmp[:, 0] = np.nan  # remove initial condition
            masks[mfk] = mask_tmp

        self.model_fks = [m for m in model_dat.keys() if (m != 'dataset_info') and (required_model_fn_substring in m)]
        self.masks = masks
        self.model_specs = specs

        return

    @staticmethod
    def map_position_to_mworks(x0, rnn_display_size=32.0, mwk_screen_size=20):
        hw = rnn_display_size / 2.0
        return (mwk_screen_size / 2.0) * (x0 - hw) / hw

    @staticmethod
    def numpy_groupby(data_matrix, groupby_var):
        mu, sig = [], []
        for ug in np.unique(groupby_var):
            ti = np.nonzero(groupby_var == ug)[0]
            mu.append(np.nanmean(data_matrix[ti, :], axis=0))
            sig.append(np.nanstd(data_matrix[ti, :], axis=0))
        return np.array(mu), np.array(sig)

    @staticmethod
    def get_noise_corrected_nse_base(x1, x2, y1, y2):
        """ eqns from Norman-Haignere and McDermott, 2018 """
        mu_xy = np.nanmean([np.nanmean(x1 * y1), np.nanmean(x1 * y2), np.nanmean(x2 * y1), np.nanmean(x2 * y2)])
        mu_x = np.nanmean([np.nanmean(x1), np.nanmean(x2)])
        mu_y = np.nanmean([np.nanmean(y1), np.nanmean(y2)])

        mu_x2 = np.nanmean([np.nanmean(x1 * x1), np.nanmean(x2 * x2)]) - np.nanmean((x1 - x2) ** 2) / 2
        mu_y2 = np.nanmean([np.nanmean(y1 * y1), np.nanmean(y2 * y2)]) - np.nanmean((y1 - y2) ** 2) / 2

        return (mu_x2 + mu_y2 - 2 * mu_xy) / (mu_x2 + mu_y2 - 2 * mu_x * mu_y)

    @staticmethod
    def get_noise_corrected_corr_base(x1, x2, y1, y2, corrtype='pearson'):
        def nnan_corr(x, y, ctype='pearson'):
            ind = np.isfinite(x) & np.isfinite(y)
            x, y = x[ind], y[ind]
            if ctype == 'pearson':
                return pearsonr(x, y)[0]
            elif ctype == 'spearman':
                return spearmanr(x, y)[0]

        def corr_fn(x, y):
            return nnan_corr(x, y, ctype=corrtype)

        rxx = corr_fn(x1, x2)
        ryy = corr_fn(y1, y2)
        rxy = np.nanmean([corr_fn(x1, y1), corr_fn(x1, y2), corr_fn(x2, y1), corr_fn(x2, y2)])
        try:
            rhon = rxy / ((rxx * ryy) ** 0.5)
        except ValueError:
            rhon = np.nan
        return rhon

    @staticmethod
    def load_data_from_file(dat_fn):
        data_ = {}
        if isinstance(dat_fn, dict):
            for fk_ in dat_fn.keys():
                data_[fk_] = pk.load(open(dat_fn[fk_], 'rb'))
        elif isinstance(dat_fn, list):
            for dfn in dat_fn:
                dat_curr = pk.load(open(dfn, 'rb'))
                data_.update(dat_curr)
        else:
            data_ = pk.load(open(dat_fn, 'rb'))
        return data_

    def compare_behavior(self, xx, yy, mask=None, metrics_oi=None):
        res = {}
        if metrics_oi is None:
            metrics_oi = xx.keys()
        for met in metrics_oi:
            res[met] = {'rhon_p': [], 'rhon_s': [], 'nse': []}

            for i in range(len(xx[met]['mu1'])):
                x1, x2 = xx[met]['mu1'][i], xx[met]['mu2'][i]
                if mask is not None:
                    x1, x2 = x1[np.isfinite(mask)], x2[np.isfinite(mask)]
                    x1, x2 = x1.flatten(), x2.flatten()
                for j in range(len(yy[met]['mu1'])):
                    y1, y2 = yy[met]['mu1'][j], yy[met]['mu2'][j]
                    if mask is not None:
                        y1, y2 = y1[np.isfinite(mask)], y2[np.isfinite(mask)]
                    y1, y2 = y1.flatten(), y2.flatten()

                    rhon_p = self.get_noise_corrected_corr_base(x1, x2, y1, y2, corrtype='pearson')
                    rhon_s = self.get_noise_corrected_corr_base(x1, x2, y1, y2, corrtype='spearman')
                    nse = self.get_noise_corrected_nse_base(x1, x2, y1, y2)
                    res[met]['rhon_p'].append(rhon_p)
                    res[met]['rhon_s'].append(rhon_s)
                    res[met]['nse'].append(nse)
        return res

    def compare_behavior_map(self, epoch, source_models, target_models,
                             cons_metricn='rhon_p', metrics_oi=None):
        """ applies compare_behavior to selected pairs
            of source and target models, with specified metrics.
        """
        results_compare = {}
        for ti, tfn in enumerate(target_models):
            res_t = self.res_model_dat[tfn]
            compare_to_t = {}
            for si, sfn in enumerate(source_models):
                res_s = self.res_model_dat[sfn]
                res_s_t = self.compare_behavior(res_s, res_t,
                                                metrics_oi=metrics_oi,
                                                mask=self.masks[epoch])
                compare_to_t[sfn] = res_s_t
            for metricn in metrics_oi:
                # flip order of dict entries
                tmp = {}
                for sfn in source_models:
                    tmp[sfn] = compare_to_t[sfn][metricn][cons_metricn]
                results_compare['cons_%s_%s' % (tfn, metricn)] = tmp
        return results_compare

    def pad_to_standard_size(self, x):
        return np.pad(x, ((0, 0), (0, self.standard_ntime - x.shape[1])),
                      'constant', constant_values=((0, 0), (0, np.nan)))

    def get_metric_base(self, signal_, groupvar_, label_):
        """
        CAREFUL: the metrics are time-varying. each is NxM (200x100)
        where N is the number of conditions, and M is the number of time-points.
        To allow for characterization of paddle dynamics, no time masks are applied yet.
        """

        def pad_mean(x, g):
            mu, sig = self.numpy_groupby(x, g)
            return self.pad_to_standard_size(mu)

        def get_endpoint_mask(grpvar):
            fitting_mask = []
            for i in grpvar:
                idx = np.nonzero(self.py_meta_idx_oi == i)[0]
                tmp = self.masks['output_f'][idx, :]
                fitting_mask.append(np.isfinite(tmp))
            return np.squeeze(fitting_mask)

        def get_residual(x, l, fitting_mask):
            """ implements partial correlation:
            learn the best linear mapping between behavioral endpoints
            and correct endpoint, and compute residual after applying
            this mapping to all timepoints.
            """

            def fit_map(xx, ll):
                x_, l_ = xx.flatten(), ll.flatten()
                nn = np.isfinite(x_) & np.isfinite(l_)
                x_, l_ = x_[nn], l_[nn]
                x_ = np.reshape(x_, (x_.shape[0], 1))
                l_ = np.reshape(l_, (l_.shape[0], 1))
                l_ = np.concatenate((l_, np.ones(l_.shape)), axis=1)  # add column of ones for intercept
                return linalg.lstsq(l_, x_)[0]

            def apply_map(xx, ll, beta_):
                xs = xx.shape

                x_, l_ = xx.flatten(), ll.flatten()
                xhat = np.ones(x_.shape) * np.nan
                nn = np.isfinite(x_) & np.isfinite(l_)
                x_, l_ = x_[nn], l_[nn]

                x_ = np.reshape(x_, (x_.shape[0], 1))
                l_ = np.reshape(l_, (l_.shape[0], 1))
                l_ = np.concatenate((l_, np.ones(l_.shape)), axis=1)

                xhat[nn] = np.squeeze(x_ - l_.dot(beta_))
                return np.reshape(xhat, xs)

            beta = fit_map(x[fitting_mask], l[fitting_mask])
            return apply_map(x, l, beta)

        def get_difference(x, l, fitting_mask):
            """ raw difference, without regression."""
            return x - l

        def get_error_trial_avg_and_total(data, err_func, prefix=''):
            res = {}
            residual = err_func(data['signal'],
                                data['label'],
                                data['mask'])
            # compute error including trial-by-trial variance (rmse).
            res[prefix + 'err_total'] = pad_mean(residual, data['groupvar'])
            res[prefix + 'abs_err_total'] = pad_mean(np.abs(residual), data['groupvar'])

            residual_mu = err_func(data['signal_mu'],
                                   data['label_mu'],
                                   data['mask_mu'])
            # compute error after averaging out trial-by-trial variance (bias).
            res[prefix + 'err_bias'] = residual_mu
            res[prefix + 'abs_err_bias'] = np.abs(residual_mu)
            return res

        """
         this only really helps for removing the effect of the third variable (ground truth pos)
         for the final timepoint. constrain comparisons to endpoint only.
        """

        groupvar_ = np.array(groupvar_)
        signal_ = np.array(signal_)
        label_ = np.tile(np.array(label_), (signal_.shape[1], 1)).T
        endpoint_mask = get_endpoint_mask(groupvar_)

        data_summary = {
            'groupvar': groupvar_,
            'signal': signal_,
            'label': label_,
            'mask': endpoint_mask,
            'signal_mu': pad_mean(signal_, groupvar_),
            'label_mu': pad_mean(label_, groupvar_),
            'mask_mu': np.isfinite(pad_mean(endpoint_mask, groupvar_)),
        }
        res_all = {
            'label': data_summary['label_mu'],
            'pos': data_summary['signal_mu']
        }
        res_1 = get_error_trial_avg_and_total(data_summary, get_residual, prefix='residual_')
        res_2 = get_error_trial_avg_and_total(data_summary, get_difference, prefix='diff_')
        res_all.update(res_1)
        res_all.update(res_2)
        return res_all

    def get_metric(self, signal, groupvar, label):

        def insert_in_dict(res_dict, res_key, res_curr):
            for mk in res_curr.keys():
                if mk not in res_dict.keys():
                    res_dict[mk] = {}
                if res_key not in res_dict[mk].keys():
                    res_dict[mk][res_key] = []
                res_dict[mk][res_key].append(res_curr[mk])
            return res_dict

        met_all = self.get_metric_base(signal, groupvar, label)
        res = {}
        res = insert_in_dict(res, 'mu', met_all)

        ntrials_per_cond = [np.sum(groupvar == i) for i in np.unique(self.py_meta_idx_oi)]
        for i in range(self.niter):
            if np.nanmin(ntrials_per_cond) > 1:  # enough trials per condition
                x_split1, x_split2, y_split1, y_split2 = train_test_split(signal, groupvar,
                                                                          test_size=0.5,
                                                                          stratify=groupvar)
                label_split1 = label[y_split1.index]
                label_split2 = label[y_split2.index]
                met1 = self.get_metric_base(x_split1, y_split1, label_split1)
                met2 = self.get_metric_base(x_split2, y_split2, label_split2)
            else:
                met1, met2 = met_all, met_all

            res = insert_in_dict(res, 'mu1', met1)
            res = insert_in_dict(res, 'mu2', met2)

        return res

    def get_performance_base(self, model_name):
        # edit this to include both bias (here already) and variance.
        x_ = self.res_model_dat[model_name]['pos']['mu']
        y_ = self.res_model_dat[model_name]['label']['mu']

        mask = np.isfinite(self.masks['output_f'])
        err_all, err_shuf_all = [], []
        for i in range(len(x_)):
            x, y = x_[i], y_[i]
            x, y = x[mask].flatten(), y[mask].flatten()
            err = np.nanmean((x - y) ** 2) ** 0.5
            err_all.append(err)

            y2 = np.random.permutation(y)
            err_shuf = np.nanmean((y2 - y) ** 2) ** 0.5
            err_shuf_all.append(err_shuf)

        err_all = np.array(err_all)
        err_shuf_all = np.array(err_shuf_all)
        return err_all, err_shuf_all

    def get_performance(self, all_models):
        err_shuf_all = []
        results_performance = {}
        for fki, fk in enumerate(all_models):
            err, err_s = self.get_performance_base(fk)
            results_performance[fk] = err
            err_shuf_all.extend(err_s)
        results_performance['null'] = np.nanmean(err_shuf_all)
        return results_performance

    def parse_mwk_dataset(self, dat, occlusion_type='all'):
        def subsample_data(metadata_, signal_, label_, meta_query_str):
            meta_q = metadata_.query(meta_query_str)
            idx_ = meta_q.index
            meta_new = metadata_.iloc[idx_].reset_index(drop=True)
            signal_new = signal_[idx_, :]
            label_new = label_[idx_].reset_index(drop=True)
            return meta_new, signal_new, label_new

        metadata = dat.datasets[0]['scalar']
        signal = dat.datasets[0]['analog_sample']['paddle_pos_y']
        label = dat.datasets[0]['scalar']['yf_mworks']

        query_str = 'ignore == 0'
        if occlusion_type == 'occ':
            query_str = 'ignore == 0 & occ_alpha == 1'
        elif occlusion_type == 'vis':
            query_str = 'ignore == 0 & occ_alpha < 1'

        meta, signal, label = subsample_data(metadata, signal, label, query_str)
        meta, idx = self.get_trials_oi(meta)
        signal = self.pad_to_standard_size(signal[idx, :])
        label = label[idx].reset_index(drop=True)

        groupvar = meta['py_meta_index']

        return {'signal': signal, 'groupvar': groupvar, 'label': label}

    def parse_model_dataset(self, dat, dataset_info):
        meta = dataset_info['meta_valid']
        meta['py_meta_index'] = meta.index
        meta = meta.reset_index(drop=True)

        signal = np.squeeze(dat['pred_output_f'])

        lab = dataset_info['output_f']
        lab[:, 0, :] = np.nan  # remove initial condition
        label = np.squeeze(np.nanmean(lab, axis=1))

        meta, idx = self.get_trials_oi(meta)

        signal = self.pad_to_standard_size(signal[idx, :])
        signal = self.map_position_to_mworks(signal)

        label = label[idx]
        label = self.map_position_to_mworks(label)

        groupvar = meta['py_meta_index']

        res = {'signal': signal, 'groupvar': groupvar,
               'label': label}
        return res

    def get_met(self, dat, occlusion_type='all', dataset_info=None):
        if isinstance(dat, dict):
            dat_ = self.parse_model_dataset(dat, dataset_info)
        else:
            dat_ = self.parse_mwk_dataset(dat, occlusion_type=occlusion_type)
        return self.get_metric(dat_['signal'], dat_['groupvar'], dat_['label'])

    def get_res_model_dat(self, other_dat, model_dat):
        res_model_dat = {}
        for fk in self.model_fks:
            res_model_dat[fk] = self.get_met(model_dat[fk], dataset_info=model_dat['dataset_info'])
        for fk in other_dat.keys():
            for occ_type in ['occ', 'vis']:
                res_model_dat[fk + occ_type] = self.get_met(other_dat[fk], occlusion_type=occ_type)
        self.res_model_dat = res_model_dat
        return

    def run_all_from_mem(self, model_dat, other_dat):
        self.set_trials_oi(other_dat)
        self.set_dataset_variables(model_dat)
        self.get_res_model_dat(other_dat, model_dat)
        return

    def run_all_from_file(self, model_dat_fn, other_data_fns):

        other_dat = self.load_data_from_file(other_data_fns)
        self.set_trials_oi(other_dat)

        model_dat = self.load_data_from_file(model_dat_fn)
        self.set_dataset_variables(model_dat)

        self.get_res_model_dat(other_dat, model_dat)
        return

    def plot_example_traces(self, dat, outfn=None):
        def get_mean_sig(dat_):
            idx_ = dat_.datasets[0]['scalar']['py_meta_index']
            idx = (idx_.isin(self.py_meta_idx_oi) & dat_.datasets[0]['scalar']['occ_alpha'] == 1.0).index
            y = dat_.datasets[0]['analog_sample']['paddle_pos_y'][idx, :]
            m = dat_.datasets[0]['scalar']['py_meta_index'].iloc[idx].reset_index(drop=True)
            return self.numpy_groupby(y, m)

        def mask_trial(mask_fn, ii):
            idx_all = np.nonzero(np.isfinite(self.masks[mask_fn][ii, :]))[0]
            start_i, end_i = idx_all[0], idx_all[-1]
            return start_i, end_i

        t_ax = range(100)

        def plot_one_trial(axes_, s_s_, e_s_, s_v_, e_v_):
            def plot_one_base(ax, s, e, col):
                ax.plot(t_ax[s:e], mu[i, s:e], '-', alpha=0.5, color=col, zorder=2)
                ax.plot(t_ax[s], mu[i, s], 'o', mec='w', color=col, zorder=2)
                ax.plot(t_ax[e], mu[i, e], 'o', mec='w', color=col, zorder=2)
                return

            plot_one_base(axes_, s_s_, e_s_, 'r')
            plot_one_base(axes_, s_v_, e_v_, 'b')
            return

        f, axes = plt.subplots(figsize=(3, 4))
        mu, sig = get_mean_sig(dat)
        for i in range(100):
            s_s, e_s = mask_trial('output_sim', i)
            s_v, e_v = mask_trial('output_vis', i)
            axes.plot(t_ax[s_v:e_s], mu[i, s_v:e_s], '-', alpha=0.1, color='k', zorder=1)
            if np.abs(mu[i, e_s]) < np.abs(mu[i, e_v]):
                plot_one_trial(axes, s_s, e_s, s_v, e_v)

            maxval = np.max(np.abs(mu[i, s_v:e_s]))
            if np.abs(mu[i, e_s]) < maxval * 0.98:
                plot_one_trial(axes, s_s, e_s, s_v, e_v)
        time_tix = [0, 20, 40, 60, 80]
        axes.set_xticks(time_tix)
        axes.set_xticklabels([np.floor(i * 41.666) for i in time_tix]
                             )
        sns.despine(ax=axes, offset=10, trim=False)
        if outfn is None:
            outfn = '/om/user/rishir/lib/MentalPong/fig/hum_ex_paddle_traces.pdf'
        f.savefig(outfn)
        return

    def plot_scatter(self, res_1_fn, res_2_fn, metric='err_bias',
                     plot_trajectory=True, plot_line=True):
        res1, res2 = self.res_model_dat[res_1_fn], self.res_model_dat[res_2_fn]
        ntr = res1[metric]['mu'][0].shape[0]
        f, axes = plt.subplots(1, 1, figsize=(5, 5))
        for i in range(ntr):
            tmp = np.nonzero(np.isfinite(self.masks['output_sim'][i, :]))[0]
            end_idx = tmp[-1]
            start_idx = tmp[0]
            if plot_trajectory:
                axes.plot(res1[metric]['mu'][0][i, start_idx:(end_idx + 1)],
                          res2[metric]['mu'][0][i, start_idx:(end_idx + 1)],
                          '-', alpha=0.25, zorder=1, color=[0.35, 0.65, 1.0])
            axes.plot(res1[metric]['mu'][0][i, end_idx], res2[metric]['mu'][0][i, end_idx],
                      'o', mec='k', mfc=[0.35, 0.65, 1.0], markersize=6)
        if plot_line:
            m = np.isfinite(self.masks['output_f'])
            x = res1[metric]['mu'][0][m].flatten()
            y = res2[metric]['mu'][0][m].flatten()
            axes.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), 'k:')
        axes.set_xlabel('%s behavior' % res_1_fn)
        axes.set_ylabel('%s behavior' % res_2_fn)
        axes.set_title('Paddle %s' % metric)
        sns.despine(ax=axes, offset=10, trim=False)
        plt.tight_layout()
        suffix = '%s_v_%s_%s_tracj%d_lsl%d' % (res_1_fn, res_2_fn, metric, plot_trajectory, plot_line)
        f.savefig('/om/user/rishir/lib/MentalPong/fig/%s.pdf' % suffix)
        return

    default_epochs = ['output_sim', 'output_f']
    default_source_models = ['mov', 'vis-mov', 'vis-sim-mov', 'sim-mov']
    default_target_models = ['Human', 'Monkey']

    def plot_performance(self, all_models, custom_prefix=None):
        f, axes = plt.subplots(1, 1, figsize=(15, 4))
        err_shuf_all = []
        results_performance = {}
        for fki, fk in enumerate(all_models):
            err, err_s = self.get_performance_base(fk)
            results_performance[fk] = err
            err_mu, err_sig = np.nanmean(err), np.nanstd(err)
            axes.bar(fki, err_mu, yerr=err_sig, color='k')
            err_shuf_all.extend(err_s)
        err_chance = np.nanmean(err_shuf_all)
        axes.axhline(err_chance, linestyle='--', color='r', alpha=0.25)
        axes.set_ylim([0, 10])
        axes.set_xlim([-1, len(all_models)])
        axes.set_xticks(range(len(all_models)))
        sns.despine(ax=axes, offset=10, trim=False)

        axes.set_ylabel('Performance (rms error)')
        f.suptitle('perf')
        plt.tight_layout()
        fig_fn = 'performance'
        if custom_prefix is not None:
            fig_fn = '%s_%s' % (custom_prefix, fig_fn)
        f.savefig('/om/user/rishir/lib/MentalPong/fig/%s.pdf' % fig_fn)
        return results_performance

    def plot_comparison(self, epoch, source_models, target_model,
                        metricn='pos', cons_metricn='rhon_p',
                        metrics_oi=None,
                        plot_bar=True, custom_prefix=None,
                        f=None, axes=None):

        if f is None:
            f, axes = plt.subplots(1, 1, figsize=(3, 5))

        if target_model == 'Human':
            barcolor = 'r'
        elif target_model == 'Monkey':
            barcolor = 'b'
        else:
            barcolor = 'k'

        res_t = self.res_model_dat[target_model]
        results_compare = {}
        for si, sfn in enumerate(source_models):
            if sfn == target_model:
                continue
            res_s = self.res_model_dat[sfn]

            cmp_res_ref_model = self.compare_behavior(res_s, res_t,
                                                      metrics_oi=metrics_oi,
                                                      mask=self.masks[epoch])
            tmp = cmp_res_ref_model[metricn][cons_metricn]
            results_compare[sfn] = cmp_res_ref_model
            mu, sig = np.nanmean(tmp), np.nanstd(tmp)
            if plot_bar:
                axes.bar(si, mu, yerr=sig, color=barcolor)
            else:
                axes.errorbar(si, mu, yerr=sig, fmt='o', color=barcolor)

        axes.axhline(1.0, linestyle='--', color='r', alpha=0.25)
        sns.despine(ax=axes, offset=10, trim=False)
        axes.set_xlim([-1, len(source_models)])
        axes.set_xticks(range(len(source_models)))
        # axes.set_xticklabels(source_models, rotation=90)
        axes.set_ylabel('Similarity to %s' % target_model.lower())
        f.suptitle('%s, %s' % (metricn, cons_metricn))
        plt.tight_layout()
        prefix = 'bar' if plot_bar else 'scatter'
        fig_fn = '%s_%s_%s_%s_%s' % (prefix, target_model, epoch, metricn, cons_metricn)
        if custom_prefix is not None:
            fig_fn = '%s_%s' % (custom_prefix, fig_fn)
        f.savefig('/om/user/rishir/lib/MentalPong/fig/%s.pdf' % fig_fn)
        return results_compare

    def get_shuffled_null(self, niter=10):
        def get_null_performance():
            y = self.res_model_dat['Humanocc']['label']['mu'][0]
            y = np.array(y)
            mask = np.isfinite(self.masks['output_f'])
            err_all, err_shuf_all = [], []
            for i in range(niter):
                y2 = np.random.uniform(low=-10, high=10, size=y.shape)
                err_shuf = np.nanmean(np.abs(y2[mask] - y[mask]))
                err_shuf_all.append(err_shuf)
            return np.nanmean(err_shuf_all)

        def get_null_consistency(target='Humanocc', metricn='residual_err_total'):

            self.res_model_dat['Random'] = {
                metricn: {
                    'mu': [], 'mu1': [], 'mu2': []
                }
            }
            # deepcopy(self.res_model_dat[target])
            nt = self.res_model_dat[target][metricn]['mu'][0].shape[0]
            rp = np.random.permutation(nt)
            for mfk in ['mu', 'mu1', 'mu2']:
                tmp = self.res_model_dat[target][metricn][mfk]
                for i in range(len(tmp)):
                    tmp_i = np.array(tmp[i])
                    self.res_model_dat['Random'][metricn][mfk].append(tmp_i[rp, :])

            res_s = self.res_model_dat['Random']
            res_t = self.res_model_dat[target]
            metrics_oi = [metricn]
            epoch = 'output_f'
            res_s_t = self.compare_behavior(res_s, res_t,
                                            metrics_oi=metrics_oi,
                                            mask=self.masks[epoch])
            return np.nanmean(res_s_t['residual_err_total']['rhon_p'])

        shuffled_null = {
            'error_f_mae': get_null_performance(),
            'cons_Humanocc_residual_err_total': get_null_consistency(),

        }
        return shuffled_null


class BehavioralComparer(object):
    def __init__(self, model_dat_fn='', human_dat_fn='', monkey_dat_fn='',
                 save_path='', pred_models_only=True, prefix=''):

        self.model_dat_fn = model_dat_fn
        self.human_dat_fn = human_dat_fn
        self.monkey_dat_fn = monkey_dat_fn
        self.pred_models_only = pred_models_only
        self.prefix = prefix
        self.save_path = save_path
        self.suffix = None

        self.all_models = None
        self.primate_models = None
        self.source_models = None
        self.set_model_names()
        self.pull_out_specs = []
        self.set_model_attributes()
        self.res_perf = {}
        self.res_cons = {}

        self.model_characterization = None
        self.bc = None

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.save_fn_full = '%s/%scomparison_full_%s.pkl' % (self.save_path, self.prefix, self.suffix)
        self.save_fn_cons = '%s/%scomparison_cons_%s.pkl' % (self.save_path, self.prefix, self.suffix)
        self.save_fn_summary = '%s/%scomparison_summary_%s.pkl' % (self.save_path, self.prefix, self.suffix)
        self.figoutpath = '/om/user/rishir/lib/MentalPong/fig/compare/'
        return

    def set_model_names(self):
        model_dat = pk.load(open(self.model_dat_fn, 'rb'))
        all_models = [m for m in model_dat.keys() if m != 'dataset_info']
        if self.pred_models_only:
            self.all_models = [m for m in all_models if ('_IC' in m) and ('pca' in m) and ('pred_' in m)]
            self.suffix = 'pred'
        else:
            self.all_models = [m for m in all_models if ('_IC' in m) and ('pca' in m) and ('pred_' not in m)]
            self.suffix = 'mov'
        self.primate_models = ['Humanocc', 'Monkeyocc', 'Humanvis', 'Monkeyvis']
        self.source_models = self.all_models + self.primate_models
        return

    def set_model_attributes(self):
        model_dat = pk.load(open(self.model_dat_fn, 'rb'))
        all_keys = []
        for fk in self.all_models:
            all_keys.extend(model_dat[fk].keys())

        example_model_fn = self.all_models[0]
        pull_out_specs = []
        all_keys = list(np.unique(all_keys))

        for ak in all_keys:
            tmp_ = model_dat[example_model_fn][ak]
            if isinstance(tmp_, list):
                tmp_ = tmp_[0]
            if isinstance(tmp_, dict):
                for ak2 in tmp_.keys():
                    if not isinstance(tmp_[ak2], dict):
                        spec = [[ak], [ak2]]
                        pull_out_specs.append(spec)

        self.pull_out_specs = pull_out_specs
        return

    def load_model_attributes(self):
        def parse_model_characterization(model_dat_, model_fk):
            charac_keys = [m for m in model_dat_[model_fk].keys() if m != 'output_f']
            m_c = {}
            for ck in charac_keys:
                m_c[ck] = model_dat_[model_fk][ck]
            return m_c

        model_dat = pk.load(open(self.model_dat_fn, 'rb'))
        self.model_characterization = {}
        for fk in self.all_models:
            self.model_characterization[fk] = parse_model_characterization(model_dat, fk)
        return

    def parse_behavior(self):
        other_data_fns = {
            'Human': self.human_dat_fn,
            'Monkey': self.monkey_dat_fn,
        }
        self.bc = BehavioralCharacterizer()
        self.bc.run_all_from_file(self.model_dat_fn, other_data_fns)
        self.res_perf = self.bc.plot_performance(self.source_models, custom_prefix=None)

        cons_metricn = 'rhon_p'
        metricn_list = ['residual_err_bias', 'residual_abs_err_bias',
                        'residual_err_total', 'residual_abs_err_total']
        target_list = self.primate_models
        epoch = 'output_f'
        cons_summary = self.bc.compare_behavior_map(epoch, self.source_models,
                                                    target_list,
                                                    cons_metricn=cons_metricn,
                                                    metrics_oi=metricn_list)

        self.res_cons = cons_summary
        with open(self.save_fn_full, 'wb') as f:
            f.write(pk.dumps(self.bc))

        with open(self.save_fn_cons, 'wb') as f:
            f.write(pk.dumps(self.res_cons))
        return

    def reload_behavior(self):
        self.bc = pk.load(open(self.save_fn_full, 'rb'))
        self.res_cons = pk.load(open(self.save_fn_cons, 'rb'))
        self.res_perf = self.bc.get_performance(self.source_models)
        return

    def save_summary(self):
        def model_name_to_df_row(specs_, m_):
            if 'max' in m_:
                m_n = "_".join(m_.split('_')[:-1])  # remove "max" suffix
                tmp = specs_[specs_['filename_tag'] == m_n]
                return tmp.iloc[0]
            elif 'epoch' in m_:  # concatenated filename_tag with n_epoch
                m_n = m_[:m_.find('_epoch')]
                m_n2 = m_[m_.find('epoch') + 5:]
                idx = [fki for fki in range(specs_.shape[0]) if
                       (m_n2 in specs_.iloc[fki]['filename']) and (m_n in specs_.iloc[fki]['filename_tag'])]
                tmp = specs_.loc[idx[0]]
                return tmp

        m_specs = self.bc.model_specs
        pull_out_specs = self.pull_out_specs
        cons_summary = self.res_cons

        df_all = []
        run_models = self.all_models + self.primate_models

        for m in run_models:
            x = {
                'name': m,
                'error_mu': np.nanmean(self.res_perf[m]),
            }
            for fk in cons_summary.keys():
                if m not in cons_summary[fk]:
                    x[fk] = np.nan
                else:
                    x_dist = cons_summary[fk][m]
                    x[fk] = np.nanmean(x_dist)
                    x[fk + 'sig'] = np.nanstd(x_dist)
                    # add 2 independent samples for following bootstrap SD estimates
                    x[fk + 'iter0'] = x_dist[0]
                    x[fk + 'iter1'] = x_dist[1]

            if m in self.all_models:
                tmp_df_row = model_name_to_df_row(m_specs, m)
                x.update(tmp_df_row)
                sim_m = self.model_characterization[m]
                for p_s in pull_out_specs:
                    for fn_ in p_s[0]:
                        if fn_ not in sim_m.keys():
                            continue
                        s_ = sim_m[fn_]
                        if isinstance(s_, list):
                            s_ = s_[0]
                        for subfn_ in p_s[1]:
                            if subfn_ not in s_.keys():
                                continue
                            x['%s_%s' % (fn_, subfn_)] = np.nanmean(s_[subfn_])
                            x['%s_%s_sig' % (fn_, subfn_)] = np.nanstd(s_[subfn_])
            df_all.append(x)

        df_all = pd.DataFrame(df_all)
        df_all.to_pickle(self.save_fn_summary)
        return

    def run_all(self, recompute_consistency=True):

        if recompute_consistency:
            self.parse_behavior()
        else:
            self.reload_behavior()

        self.load_model_attributes()
        self.save_summary()
        print('Done. Saved to %s' % self.save_fn_summary)
        return

    # PATH_ = '/om/user/rishir/lib/MentalPong/behavior/'
    # sys.path.insert(0, PATH_)
    # import numpy as np
    # import pandas as pd
    # import pickle as pk
    # import BehavioralCharacterizer as BC
    # import warnings

    """ get dataframe of model specs"""


class QuickComparer(object):
    def __init__(self):
        self.bc = None

        # load data from model_dat without any actual model data, only dataset data (meta, masks, etc.)
        default_model_dat_fn = '/om/user/rishir/lib/PongRnn/fig/rnn_res/perc_model_res_random_seed_lt_10_small.pkl'
        model_dat = pk.load(open(default_model_dat_fn, 'rb'))
        for fk in list(model_dat.keys()):
            if (fk != 'specs') and (fk != 'dataset_info'):
                model_dat.pop(fk)
        self.model_dat = model_dat

        return

    @staticmethod
    def subsample_data(data, trials_oi=None):
        data_out = deepcopy(data)
        if trials_oi is not None:
            df = data_out.datasets[0]['scalar']
            data_out.datasets[0]['scalar'] = df.iloc[trials_oi].reset_index(drop=True)

            analog_fns = data_out.datasets[0]['analog_sample'].keys()
            for fk in analog_fns:
                data_out.datasets[0]['analog_sample'][fk] = data_out.datasets[0]['analog_sample'][fk][trials_oi, :]
        return data_out

    def compare(self, data1, data2,
                trials_oi_data1=None,
                trials_oi_data2=None,
                cons_metricn='rhon_p',
                metricn='residual_err_total',
                source_fn='data1occ', target_fn='data2occ'):

        data_to_compare_1 = self.subsample_data(data1, trials_oi_data1)
        data_to_compare_2 = self.subsample_data(data2, trials_oi_data2)

        data = {
            'data1': data_to_compare_1,
            'data2': data_to_compare_2,
        }
        self.bc = BehavioralCharacterizer()
        self.bc.run_all_from_mem(self.model_dat, data)
        epoch = 'output_f'
        return self.bc.compare_behavior_map(epoch, [source_fn], [target_fn],
                                            cons_metricn=cons_metricn,
                                            metrics_oi=[metricn])
