import os
import sys
import numpy as np
import pickle as pk
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import pairwise_distances
from copy import deepcopy

PATH_TO_FIXED_POINT_FINDER = '/om/user/rishir/lib/PongRnn/'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from rnn_analysis import utils, data_utils

PATH_ = '/om/user/rishir/lib/PongRnn/rnn_analysis/run_scripts/'
sys.path.insert(0, PATH_)




def get_model_geometry(model_data, epoch_mask='output_vis-sim'):
    def get_manifold_stats_speed(x, y, suffix=''):
        pos = np.array(x)
        pos_mask = np.transpose(np.tile(y[:, :, 0], (x.shape[2], 1, 1)), (1, 2, 0))

        pos[np.isnan(pos_mask)] = np.nan
        speed = np.diff(pos, axis=1, append=np.nan)
        acc = np.diff(speed, axis=1, append=np.nan)

        pos_mu = np.nanmean(np.nansum(pos ** 2, axis=2) ** 0.5, axis=1)
        speed_mu = np.nanmean(np.nansum(speed ** 2, axis=2) ** 0.5, axis=1)
        acc_mu = np.nanmean(np.nansum(acc ** 2, axis=2) ** 0.5, axis=1)

        speed_var = np.nanstd(np.nansum(speed ** 2, axis=2) ** 0.5, axis=1)
        acc_var = np.nanstd(np.nansum(acc ** 2, axis=2) ** 0.5, axis=1)

        rel_speed = speed_mu / pos_mu
        rel_acc = acc_mu / pos_mu

        rel_speed_var = speed_var / pos_mu
        rel_acc_var = acc_var / pos_mu

        m_stats = {
            # 'curvature' + suffix:  np.nanmean(rel_curve),
            'rel_speed' + suffix: np.nanmean(rel_speed),
            'rel_acc' + suffix: np.nanmean(rel_acc),
            'rel_speed_var' + suffix: np.nanmean(rel_speed_var),
            'rel_acc_var' + suffix: np.nanmean(rel_acc_var),

            'raw_speed' + suffix: np.nanmean(speed_mu),
            'raw_acc' + suffix: np.nanmean(acc_mu),
            'raw_speed_var' + suffix: np.nanmean(speed_var),
            'raw_acc_var' + suffix: np.nanmean(acc_var),
            'raw_pos' + suffix: np.nanmean(pos_mu),
        }
        return m_stats

    def get_task_smoothness(x, y, prefix=''):
        def get_pairwise_distances(d):
            d_ = pairwise_distances(d)
            d_ = np.diagonal(d_, offset=1).flatten()
            return np.squeeze(d_)

        state_matrix = utils.flatten_to_mat(x, y)['X']
        meta_matrix = utils.flatten_to_mat(y, y)['X']

        d_states = get_pairwise_distances(state_matrix)
        d_meta = get_pairwise_distances(meta_matrix)

        rp = pearsonr(d_states, d_meta)[0]
        rs = spearmanr(d_states, d_meta)[0]

        return {'%s_rho_pearson' % prefix: rp,
                '%s_rho_spearman' % prefix: rs,
                }

    def get_geometry_shape_from_pairwise_distances(x, y):
        def dist_of_pairwise_distances(D):
            d_D = pairwise_distances(D)
            d_D = np.diagonal(d_D, offset=1)
            mu = np.nanmean(d_D.flatten())
            sig = np.nanstd(d_D.flatten())
            return mu, sig

        state_matrix = utils.flatten_to_mat(x, y)['X']
        state_matrix_shuffle = deepcopy(state_matrix).flatten()
        np.random.shuffle(state_matrix_shuffle)
        state_matrix_shuffle = np.reshape(state_matrix_shuffle, state_matrix.shape)

        dd_mu, dd_sig = dist_of_pairwise_distances(state_matrix)
        dd_mu_shuf, dd_sig_shuf = dist_of_pairwise_distances(state_matrix_shuffle)

        return {
            'pairwise_dist_mu': dd_mu,
            'pairwise_dist_sig': dd_sig,
            'pairwise_dist_shuf_mu': dd_mu_shuf,
            'pairwise_dist_shuf_sig': dd_sig_shuf,
        }

    def get_all_geometry_metrics(state_):
        states = np.array(state_)#
        epoch_mask_mat = model_data[epoch_mask]
        stats = utils.get_low_dim_manifold_with_stats(states, epoch_mask_mat)
        x_lowdim = stats.pop('X_lowdim')

        stats.update(get_manifold_stats_speed(states, epoch_mask_mat, suffix='_full'))
        stats.update(get_manifold_stats_speed(x_lowdim, epoch_mask_mat, suffix='_lowdim'))
        # stats.update(get_geometry_shape_from_pairwise_distances(states, epoch_mask))
        stats.update(get_task_smoothness(states, epoch_mask_mat, prefix='tasksmooth'))
        stats.update(get_task_smoothness(x_lowdim, epoch_mask_mat, prefix='tasksmooth_lowd'))
        return stats


    stats_all = {}
    stats_1 = get_all_geometry_metrics(model_data['state'])
    for fk in stats_1.keys():
        stats_all['state_' + fk] = stats_1[fk]

    stats_2 = get_all_geometry_metrics(model_data['output_state'])
    for fk in stats_2.keys():
        stats_all['output_state_' + fk] = stats_2[fk]

    return stats_all


def get_piecewise_encoding(model_data):
    def get_prediction_variables(epoch_mask='output_vis'):
        xy = np.array(model_data[epoch_mask])
        full_state = np.array(model_data['state'])
        full_meta = data_utils.get_full_meta_from_xy(xy[:, :, 0], xy[:, :, 1])
        mask = np.array(full_meta['x'])

        state_matrix = utils.flatten_to_mat(full_state, mask)['X']
        meta_matrices = {}
        for fk in full_meta.keys():
            full_y = np.array(full_meta[fk])
            meta_matrices[fk] = utils.flatten_to_mat(full_y, mask)['X']

        return state_matrix, meta_matrices

    def get_piecewise_prediction(M1, M2, mfn='xy'):
        # format as visual_meta, occ_meta, visual_vs_occluded_bias
        M1_a = np.concatenate((M1[mfn], np.zeros(M1[mfn].shape), np.ones((M1[mfn].shape[0], 1))), axis=1)
        M2_a = np.concatenate((np.zeros(M2[mfn].shape), M2[mfn], np.zeros((M2[mfn].shape[0], 1))), axis=1)
        return np.concatenate((M1_a, M2_a), axis=0)

    res_ = {}

    S1_, M1_ = get_prediction_variables(epoch_mask='output_vis')
    S2_, M2_ = get_prediction_variables(epoch_mask='output_sim')
    S = np.concatenate((S1_, S2_), axis=0)
    for mfn_ in ['xy', 'xydspeeddtheta']:
        M = get_piecewise_prediction(M1_, M2_, mfn=mfn_)
        t = np.isfinite(np.mean(M, axis=1))
        res_['R2_%s' % mfn_] = utils.get_R2_highD(S[t, :], M[t, :])

    return res_


def get_model_encoding_score(model_data, epoch_mask='output_vis-sim'):
    def get_prediction_variables():
        xy = np.array(model_data[epoch_mask])
        full_state = np.array(model_data['state'])
        full_meta = data_utils.get_full_meta_from_xy(xy[:, :, 0], xy[:, :, 1])
        mask = np.array(full_meta['x'])

        state_matrix = utils.flatten_to_mat(full_state, mask)['X']
        meta_matrices = {}
        for fk in full_meta.keys():
            full_y = np.array(full_meta[fk])
            meta_matrices[fk] = utils.flatten_to_mat(full_y, mask)['X']

        return state_matrix, meta_matrices

    def get_variance_explained_metrics(state_matrix, meta_matrices):
        res_ = {}
        keys_oi = ['xy', 'xydxdy', 'xydspeeddtheta']
        for yfn in keys_oi:
            meta_matrix = np.array(meta_matrices[yfn])
            state_matrix_curr, meta_matrix_curr = utils.mask2d(state_matrix, meta_matrix)
            res_['prop_var_exp_%s' % yfn] = utils.get_R2_highD(state_matrix_curr, meta_matrix_curr)
        return res_

    X_mat, Y_mat_dict = get_prediction_variables()
    res_r2 = get_variance_explained_metrics(X_mat, Y_mat_dict)

    res_all = {}
    res_all.update(res_r2)

    return res_all


def get_cross_time_state_prediction(model_data):
    def patch_model_data(md):
        md_out = deepcopy(md)
        X = np.array(md['output_f'])
        tmp = np.tile(X, (1, 1, 2))
        tmp[:, 0, :] = np.nan
        md_out['output_f'] = tmp
        return md_out

    model_data_patched = patch_model_data(model_data)
    data_start = data_utils.get_data_aligned_to_epoch_start(model_data_patched,
                                                            use_pca_proj=False,
                                                            epoch_align='output_vis')
    state_start, meta_start = data_start[0]['state'], data_start[0]['meta']
    data_occ = data_utils.get_data_aligned_to_epoch_start(model_data_patched,
                                                          use_pca_proj=False,
                                                          epoch_align='output_sim')
    state_occ, meta_occ = data_occ[0]['state'], data_occ[0]['meta']
    data_end = data_utils.get_data_aligned_to_epoch_start(model_data_patched,
                                                          use_pca_proj=False,
                                                          epoch_align='output_f')
    state_end, meta_end = data_end[0]['state'], data_end[0]['meta']

    states_to_predict = {
        'start': np.nanmean(state_start, axis=1),
        'occ': np.nanmean(state_occ, axis=1),
        'end': np.nanmean(state_end, axis=1),
    }
    R2_cross_epoch = {}
    for i in states_to_predict.keys():
        for j in states_to_predict.keys():
            if j == i:
                continue
            R2_cross_epoch['%s_to_%s' % (j, i)] = utils.get_R2_highD(states_to_predict[i],
                                                                     states_to_predict[j])

    return R2_cross_epoch


def get_occlusion_state_characteristics(model_data, use_pca_proj=True):
    def regress_meta_vars_from_state(state_, meta_):
        keys_oi = ['y', 'dx', 'dy', 'dtheta', 'dspeed']
        reg_r = {}
        for fk in keys_oi:
            y = np.nanmean(meta_[fk], axis=1).flatten()
            X_mat = np.nanmean(state_, axis=1)
            for k in [10, 200]:
                res_lr = utils.linear_regress(X_mat, y, k_folds=k)
                reg_r['decode_%s_rho_k%d' % (fk, k)] = res_lr['r']
                reg_r['decode_%s_mae_k%d' % (fk, k)] = res_lr['mae']
        return reg_r

    def regress_state_from_meta_vars(state_, meta_):
        keys_oi = ['y', 'xydxdy', 'xydspeeddtheta']
        reg_r = {}
        X = np.nanmean(state_, axis=1)
        for fk in keys_oi:
            Y = np.squeeze(np.nanmean(meta_[fk], axis=1))
            if Y.ndim < 2:
                Y = np.expand_dims(Y, axis=1)
            reg_r['encode_R2_%s' % fk] = utils.get_R2_highD(X, Y)

        return reg_r

    state_occ, meta_occ = data_utils.get_data_aligned_to_epoch_start(model_data,
                                                                     use_pca_proj=use_pca_proj)

    res_summary = {}
    res_dec = regress_meta_vars_from_state(state_occ, meta_occ)
    res_summary.update(res_dec)
    res_enc = regress_state_from_meta_vars(state_occ, meta_occ)
    res_summary.update(res_enc)

    return res_summary


def get_model_generalization_index(model_data, dist_data=None):
    if dist_data is None:
        dist_fn = '/om/user/rishir/data/pong_basic/RF/occluded_pong_bounce1_pad8_4speed/train_to_valid_dist.pkl'
        dist_data = pk.load(open(dist_fn, 'rb'))
    dist_to_train_min = np.nanmin(dist_data, axis=0)
    dist_to_train_mean = np.nanmean(dist_data, axis=0)

    mask = model_data['output_f']
    mask[:, 0, :] = np.nan
    y_pred = model_data['pred_output_f'][np.isfinite(mask)]
    y_true = model_data['output_f'][np.isfinite(mask)]
    y_err_abs = np.abs(y_true - y_pred)

    res = {
        'rho_mean_dist': pearsonr(dist_to_train_mean, y_err_abs)[0],
        'rho_min_dist': pearsonr(dist_to_train_min, y_err_abs)[0],
    }
    return res


def get_model_simulation_index(model_data,
                               train_mask='output_vis-sim',
                               test_mask='output_sim'):
    def train_test_base(X, y, X2, y2, k_folds):
        kf = KFold(n_splits=k_folds, random_state=0)
        y_pred = deepcopy(y)
        y2_pred = deepcopy(y2)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X2_train, X2_test = X2[train_index], X2[test_index]
            y2_train, y2_test = y2[train_index], y2[test_index]

            X_train_res = utils.flatten_to_mat(X_train, y_train)
            y_train_res = utils.flatten_to_mat(y_train, y_train)
            reg = LinearRegression()
            reg.fit(X_train_res['X'], y_train_res['X'])

            X_test_res = utils.flatten_to_mat(X_test, y_test)
            y_test_res = utils.flatten_to_mat(y_test, y_test)
            X_tmp = np.array(X_test_res['X'].copy())
            tmp = reg.predict(X_tmp)
            y_pred[test_index] = utils.unflatten_to_3d_mat(tmp, y_test_res)

            X2_test_res = utils.flatten_to_mat(X2_test, y2_test)
            y2_test_res = utils.flatten_to_mat(y2_test, y2_test)
            X_tmp = np.array(X2_test_res['X'].copy())
            tmp = reg.predict(X_tmp)
            y2_pred[test_index] = utils.unflatten_to_3d_mat(tmp, y2_test_res)

        return y_pred, y2_pred

    def train_test_per_k(states, output_train, output_test, k=2):
        y_pred_, y2_pred_ = train_test_base(states, output_train,
                                            states, output_test, k)

        y1, y2 = output_test, y2_pred_
        rho, nse, rmse, mae = [], [], [], []
        for i in range(y1.shape[-1]):
            a, b = y1[..., i].flatten(), y2[..., i].flatten()
            t = np.isfinite(a) & np.isfinite(b)
            rho_ = pearsonr(a[t], b[t])[0]
            rho.append(rho_)
            nse_ = utils.get_noise_corrected_nse_base(a[t], a[t], b[t], b[t])
            nse.append(nse_)
            rmse_ = np.nanmean((a[t] - b[t]) ** 2) ** 0.5
            rmse.append(rmse_)
            mae_ = np.nanmean(np.abs(a[t] - b[t]))
            mae.append(mae_)

        return {
            'rho': np.array(rho),
            'nse': np.array(nse),
            'rmse': np.array(rmse),
            'mae': np.array(mae),
        }

    if 'output_state' not in model_data.keys():
        return {'rho': np.nan, 'nse': np.nan, 'rmse': np.nan, 'mae': np.nan}

    states_ = model_data['output_state']
    output_train_ = model_data[train_mask]
    output_test_ = model_data[test_mask]
    # print(states_.shape, output_train_.shape, output_test_.shape)
    res_full = {}
    for kk in [2, 5, 10]:
        res = train_test_per_k(states_, output_train_, output_test_, k=kk)
        for fk in res.keys():
            res_full['%s_k%d' % (fk, kk)] = res[fk]
    return res_full


def get_piecewise_gradient_encoding(model_data, num_bins=32, bin_steps=2, mask_fk='output_sim', niter=10, k_folds=20,
                                    min_nobs=5, preproc_method='none'):
    def get_decoding_of_gradient(states, full_meta, masks, label_fk, bin_fk, mask_fk):
        base_func = get_model_decoding_of_gradient_base
        label = full_meta[label_fk].copy()
        xy = full_meta[bin_fk].copy()
        t = np.isfinite(masks[mask_fk][:, :, 0])
        state_matrix = states[t, :]
        meta_matrix = label[t, :]
        binvar_matrix = xy[t, :]
        return base_func(state_matrix, meta_matrix, binvar_matrix,
                         num_bins=num_bins, bin_steps=bin_steps,
                         k_folds=k_folds, min_nobs=min_nobs, leave_one_out=False,
                         preproc_method=preproc_method, niter=niter,
                         pool_over_bins=False, return_full_results=True)

    res_, full_meta_, masks_ = data_utils.load_one_base(model_data, mask_early_late=False)

    states_ = res_['state']
    bin_fk = 'xy'
    label_fk = 'dxdy'
    res_full = get_decoding_of_gradient(states_, full_meta_, masks_, label_fk, bin_fk, mask_fk)
    res_summary = {
        'r2_mean': np.nanmean(res_full['R2_over_bins']['r2_over_bins_niter'], axis=0),
    }
    return res_summary


def get_model_decoding_of_gradient_masked(model_data, return_results=False):
    res, full_meta, masks = data_utils.load_one_base(model_data, mask_early_late=True)
    masks_of_interest = [
        'output_sim_roll5_nobounce', 'output_vis_nobounce',
        'output_sim_roll5_nobounce_early', 'output_vis_nobounce_early',
        'output_sim_roll5_nobounce_late', 'output_vis_nobounce_late',
    ]
    results_r2 = {}
    for label in ['xy', 'dxdy']:
        for mask_fn in masks_of_interest:
            summary1 = data_utils.get_matrices_from_res(res, full_meta, masks,
                                                        mask_fn=mask_fn,
                                                        label=label)
            results_r2['grad_r2_%s_%s' % (label, mask_fn)] = utils.get_R2_highD(summary1['y'],
                                                                                summary1['X'],
                                                                                k_folds=20)
            if return_results:
                results_r2['grad_res_%s_%s' % (label, mask_fn)] = \
                    utils.linear_regress(summary1['X'], summary1['y'], k_folds=20)

    return results_r2


def get_model_decoding_of_gradient_base(state_matrix, meta_matrix, binvar_matrix,
                                        num_bins=8, bin_steps=1, k_folds=20,
                                        min_nobs=2, leave_one_out=False, max_k=100,
                                        niter=5, preproc_method='none',
                                        pool_over_bins=False, return_full_results=False):
    def remove_nans(x_, y_):
        t = np.isfinite(np.mean(x_, axis=1)) & np.isfinite(np.mean(y_, axis=1))
        return x_[t, :], y_[t, :]

    def get_edge_bounds(ii):
        i1 = ii - int(bin_steps / 2.0)
        i2 = ii + int(bin_steps / 2.0)
        i1 = int(np.nanmin([np.nanmax([i1, 0]), num_bins]))
        i2 = int(np.nanmin([np.nanmax([i2, 0]), num_bins]))
        return i1, i2

    def get_bin_selections():
        x, y = binvar_matrix[:, 0], binvar_matrix[:, 1]
        xbins = list(np.linspace(0, 32.0, num_bins + 1))
        ybins = list(np.linspace(0, 32.0, num_bins + 1))
        bin_xy_center = []
        bin_selection = []
        for i_ in range(len(xbins) - 1):
            for j_ in range(len(ybins) - 1):
                xi, xj = get_edge_bounds(i_)
                yi, yj = get_edge_bounds(j_)

                tx = (x >= xbins[xi]) & (x < xbins[xj])
                ty = (y >= ybins[yi]) & (y < ybins[yj])

                bin_selection.append(tx & ty)
                bin_xy_center.append([xbins[j_], ybins[i_]])

        bin_res = {'select': bin_selection,
                   'centers': bin_xy_center}
        return bin_res

    def get_gradient_R2_over_each(states, meta_curr, bin_selections=None):
        """ This method uses different regressions for different bins, and measures
        R2 on each bin. the average across bins is reported.
         upside is that is not sensitive to variance across bins,
         and won't be killed by a small number of badly predicted bins
         downside is that if bins are unevenly sampled, the output is not very representative.
         not just in terms of number of samples, but also in total variance per bin.
         """
        X2, Z2 = states, meta_curr

        r2_over_bins, n_over_bins = [], []
        r2_over_bins_niter = []

        if bin_selections is not None:
            for t_select in bin_selections:
                nobs = np.nansum(t_select)
                n_over_bins.append(nobs)

                if leave_one_out:
                    k_fold_val = np.max([nobs, min_nobs])
                    k_fold_val = np.min([k_fold_val, max_k])
                else:
                    k_fold_val = k_folds

                if nobs < k_fold_val:
                    r_2 = np.nan
                    r2_over_splits = [np.nan] * niter
                else:

                    # should not resample over R2, since this will be biased up.
                    # resample (niter) in regression to calculate r
                    r_2, r2_over_splits = utils.get_R2_highD(Z2[t_select, :], X2[t_select, :],
                                                             niter=niter, k_folds=k_fold_val,
                                                             preproc_method=preproc_method, return_splits=True)

                r2_over_bins.append(r_2)
                r2_over_bins_niter.append(r2_over_splits)
            r = np.nanmean(r2_over_bins)
        else:
            r = utils.get_R2_highD(Z2, X2, k_folds=k_folds)

        res_full = {
            'r2_over_bins': np.array(r2_over_bins),
            'r2_over_bins_niter': np.array(r2_over_bins_niter),
            'bin_selections': bin_selections,
            'r': r,
        }
        if return_full_results:
            return res_full
        else:
            return r

    def get_gradient_R2_over_all(states, meta_curr, bin_selections=None):
        """ this method uses different regressions for different bins,
        but then aggregates all into a single prediction that is used to compute R2.
        downside is that poor prediction in a small number of bins can tank the whole thing.
        upside is that if bins are unevenly sampled, this is immune to that.
        """
        X2, Z2 = states, meta_curr
        Z_pred = np.ones(Z2.shape) * np.nan

        if bin_selections is not None:
            for t_select in bin_selections:
                if np.nansum(t_select) > 2:
                    reg_res = utils.linear_regress(X2[t_select, :], Z2[t_select, :])
                    Z_pred[t_select, :] = reg_res['y_pred']
        else:
            reg_res = utils.linear_regress(X2, Z2)
            Z_pred = reg_res['y_pred']

        Z2, Z_pred = remove_nans(Z2, Z_pred)
        return utils.get_R2_highD(Z2, Z_pred, cross_validate_regression=False)

    res = {}
    bin_results = get_bin_selections()
    bin_selects = bin_results['select']
    # res['bin_centers'] = bin_results['centers']

    if pool_over_bins:
        get_gradient_R2 = get_gradient_R2_over_all
    else:
        get_gradient_R2 = get_gradient_R2_over_each

    # res['R2_full'] = get_gradient_R2(state_matrix, meta_matrix, bin_selections=None)
    res['R2_over_bins'] = get_gradient_R2(state_matrix, meta_matrix, bin_selections=bin_selects)
    res['bin_results'] = bin_results

    return res


def get_model_decoding_of_gradient(model_data, k_folds=20, pool_over_bins=False):
    """ how well do the model states predict the gradient of task variables
    (dx,dy)"""

    def get_relevant_variables(res_):
        states = res_['state'].copy()
        sim_x = np.array(res_['output_vis-sim'][:, :, 0])
        sim_y = np.array(res_['output_vis-sim'][:, :, 1])
        full_meta = data_utils.get_full_meta_from_xy(sim_x, sim_y)

        mask_vars = {'output_sim', 'output_vis-sim', 'output_vis', 'output_f'}
        masks = {}
        for mfk in mask_vars:
            mask_tmp = res_[mfk][:, :, [0]]
            if mfk == 'output_f':
                mask_tmp[:, 0, :] = np.nan  # remove initial condition
            masks[mfk] = mask_tmp

        return states, full_meta, masks

    def get_decoding_of_gradient(states, full_meta, masks, label_fk, bin_fk, mask_fk):
        label = full_meta[label_fk].copy()
        xy = full_meta[bin_fk].copy()
        t = np.isfinite(masks[mask_fk][:, :, 0])
        state_matrix = states[t, :]
        meta_matrix = label[t, :]
        binvar_matrix = xy[t, :]

        return get_model_decoding_of_gradient_base(state_matrix,
                                                   meta_matrix,
                                                   binvar_matrix,
                                                   k_folds=k_folds,
                                                   pool_over_bins=pool_over_bins)

    states_, full_meta_, masks_ = get_relevant_variables(model_data)
    res = {}
    bin_fk_ = 'xy'
    for label_fk_ in ['dspeeddtheta', 'dxdy']:
        for mask_fk_ in ['output_vis', 'output_sim']:
            res_curr = get_decoding_of_gradient(states_, full_meta_, masks_,
                                                label_fk_, bin_fk_, mask_fk_)
            for res_fk in res_curr.keys():
                res['decode_%s_%s_%s' % (label_fk_, mask_fk_, res_fk)] = res_curr[res_fk]

    return res


def get_model_connectivity_metrics(filename):
    def parse_weights(w_all):
        """ separate into input (x) and state (h) weights
        and further separate into forward and backwards (transposed)
         """
        w_dict = {
            'x': [],
            'x_t': [],
            'h': [],
            'h_t': []
        }
        for wfk in w_all.keys():
            w = w_all[wfk]
            if 'w_x' in wfk:
                w_dict['x'].append(w)
                w_dict['x_t'].append(w.T)
            elif 'w_h' in wfk:
                w_dict['h'].append(w)
                w_dict['h_t'].append(w.T)
        return w_dict

    def get_weight_sparseness_base(w):
        ncomp = np.min(w.shape)
        pca = PCA(n_components=ncomp, svd_solver='full')
        pca.fit(w)
        eigenvals = pca.explained_variance_
        return np.nansum(eigenvals) ** 2 / np.nansum(eigenvals ** 2)

    def get_weight_metrics(w_all):
        pr = [get_weight_sparseness_base(w) for w in w_all]
        res_ = {
            'W_PR': np.nanmean(pr),
            'W_mu': np.nanmean(w_all),
            'W_abs_mu': np.nanmean(np.abs(w_all)),
        }
        return res_

    res = {}
    weight_filename = filename.replace('.pkl', '_weights.pkl')
    if os.path.isfile(weight_filename) is False:
        return res

    weights = pk.load(open(weight_filename, 'rb'))
    weight_dict = parse_weights(weights)
    for wk in weight_dict:
        w_dict_curr = weight_dict[wk]
        res_w = get_weight_metrics(w_dict_curr)
        for res_w_fk in res_w.keys():
            res['%s_%s' % (res_w_fk, wk)] = res_w[res_w_fk]
    return res


def get_output_feedback(model_data):
    def get_pca_proj_mapping(x_tmp, y_tmp, ncomp=None):
        x_res = utils.flatten_to_mat(x_tmp, y_tmp)
        if ncomp is None:
            ncomp = np.min(x_res['X'].shape)
        pca = PCA(n_components=ncomp, svd_solver='full')
        pca.fit(x_res['X'])
        return pca.components_  # nPC x nD

    def get_weight_vector_projection(W, task_output_idx=6, w_pca=None, suffix=''):
        def get_norm_dot(x, y):
            return np.dot(x, y) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)

        # this is projection of two unit vectors. should also try unit vector only for readout.

        def get_norm_dot_high_d(x, y):
            return [get_norm_dot(x, y[:, yi]) for yi in range(y.shape[1])]
            # y is a subspace

        # projection of output readout axis on recurrent weights
        output_w = W['w_out:0'][:, task_output_idx]
        random_output_w = np.random.random(output_w.shape)

        proj_true, proj_null = [], []
        proj_true_t, proj_null_t = [], []
        for wk in W.keys():
            if 'w_h_gate' not in wk:
                continue
            w_to_gate = W[wk]  # (i,j) node i to node j
            if w_pca is not None:
                w_to_gate = np.matmul(w_to_gate, w_pca)  # format res_pca to nF, nPCA

            proj_true.append(get_norm_dot_high_d(output_w, w_to_gate))
            proj_null.append(get_norm_dot_high_d(random_output_w, w_to_gate))

            proj_true_t.append(get_norm_dot_high_d(output_w, w_to_gate.T))
            proj_null_t.append(get_norm_dot_high_d(random_output_w, w_to_gate.T))

        res_curr = {
            'proj_norm_true%s' % suffix: np.nanmean(np.array(proj_true) ** 2),
            'proj_norm_null%s' % suffix: np.nanmean(np.array(proj_null) ** 2),
            'trans_proj_norm_true%s' % suffix: np.nanmean(np.array(proj_true_t) ** 2),
            'trans_proj_norm_null%s' % suffix: np.nanmean(np.array(proj_null_t) ** 2),
        }

        return res_curr

    res = {}
    filename = model_data['filename']
    weight_fn = filename.replace('.pkl', '_weights.pkl')
    if os.path.isfile(weight_fn) is False:
        return res

    w_ = pk.load(open(weight_fn, 'rb'))
    res.update(get_weight_vector_projection(w_, w_pca=None, suffix='_raw'))

    # x_ = model_data['output_state']
    # y_ = model_data['output_vis-sim']
    # pca_map = get_pca_proj_mapping(x_, y_, ncomp=0.95)
    # res.update(get_weight_vector_projection(w_, w_pca=pca_map.T, suffix='_pca'))

    return res
