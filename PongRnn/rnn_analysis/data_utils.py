import sys

PATH_ = '/om/user/rishir/lib/PongRnn/'
sys.path.insert(0, PATH_)
from rnn_analysis import utils
import numpy as np
from copy import deepcopy

figoutpath = '/om/user/rishir/lib/MentalPong/fig/model_characterization/'
datoutpath = '/om/user/rishir/lib/MentalPong/dat/gradient_axis_example/'
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

""" 
Mostly utils for loading and processing model outputs.
"""


def load__comparison_summary_df(fn='/om/user/rishir/lib/MentalPong/dat/comparison_summary_pred.pkl',
                                output_rep='pred'):
    import pandas as pd
    df_full = pd.read_pickle(fn)
    t = np.isnan(df_full['regularization_weight'])
    df_full['regularization_weight'][t] = 0
    df_full['regularization_loss'][t] = 'none'
    df_full['reg'] = df_full['regularization_loss'] + '_' + df_full['regularization_weight'].astype(str)
    df_full['arch'] = df_full['rnn_type'] + '_' + df_full['n_hidden'].astype(str)

    df = df_full.query('input_representation == "pixel_pca" or input_representation == "gabor_pca"')
    df = df.query('output_representation == "%s"' % output_rep)
    df = df.reset_index(drop=True)

    df_model = df.query('name != "Monkeyocc"')  # .query('error_f_mae < 3')
    df_primate = df_full.query('name == "Monkeyocc" or name == "Humanocc"')

    if 'geom_vis-sim_pairwise_dist_mu' in df_model.keys():
        x = df_model['geom_vis-sim_pairwise_dist_mu']
        y = df_model['geom_vis-sim_PR']
        df_model['geom_vis-sim_pairwise_dist_mu_normalized'] = x / y
    return df, df_model, df_primate


def get_full_meta_from_xy(x_, y_):
    x = np.expand_dims(x_, axis=2)
    y = np.expand_dims(y_, axis=2)
    dx = np.diff(x, axis=1, append=np.nan)
    dy = np.diff(y, axis=1, append=np.nan)

    # fix last entry, from diff operation
    t = np.isfinite(y) & np.isnan(dy)
    ni, nj, nk = np.nonzero(t)
    for i, j in zip(ni, nj):
        dx[i, j, :] = dx[i, j - 1, :]
        dy[i, j, :] = dy[i, j - 1, :]

    dtheta = np.arctan2(dy, dx)
    dspeed = (dy ** 2 + dx ** 2) ** 0.5

    full_meta = {
        'x': x, 'y': y, 'xy': np.concatenate((x, y), axis=2),
        'dx': dx, 'dy': dy, 'dxdy': np.concatenate((dx, dy), axis=2),
        'xydxdy': np.concatenate((x, y, dx, dy), axis=2),
        'dtheta': dtheta, 'dspeed': dspeed,
        'dspeeddtheta': np.concatenate((dspeed, dtheta), axis=2),
        'xydspeeddtheta': np.concatenate((x, y, dspeed, dtheta), axis=2),
    }

    return full_meta


def load_one_base(res_example, mask_early_late=False):
    mask_vars = {'output_sim', 'output_vis-sim', 'output_vis', 'output_f'}
    masks = {}
    for mfk in mask_vars:
        mask_tmp = res_example[mfk][:, :, [0]]
        if mfk == 'output_f':
            mask_tmp[:, 0, :] = np.nan  # remove initial condition
        masks[mfk] = mask_tmp

    res = res_example.copy()
    sim_x = np.array(res['output_vis-sim'][:, :, 0])
    sim_y = np.array(res['output_vis-sim'][:, :, 1])
    full_meta = get_full_meta_from_xy(sim_x, sim_y)

    full_meta = add_subspace_gradient_to_meta(res, full_meta, masks)
    masks = remove_transition(masks, epoch='output_sim')
    all_epochs = []
    for epoch in ['output_vis', 'output_sim', 'output_vis-sim']:
        epochs_unrolled = [fk for fk in masks.keys() if epoch in fk]
        all_epochs.extend(epochs_unrolled)
    for epoch in all_epochs:
        masks = remove_bounces(full_meta, masks, epoch=epoch)

    if mask_early_late:
        masks = split_into_early_late_subepochs(masks)

    return res, full_meta, masks


def get_data_aligned_to_epoch_start(model_data, epoch_align='output_sim', max_offset=0, use_pca_proj=True,
                                    epoch_pca='output_vis-sim'):
    def get_val_at_occlusion(val, mask, offset=0):
        ntr = mask.shape[0]
        val_masked = val.copy()
        n_time = val_masked.shape[1]

        for i in range(ntr):
            j = np.nonzero(np.isfinite(mask[i, :]))[0][0]
            end_i, start_i = j + 1 + offset, j + offset
            if (start_i > 0) and (start_i < n_time) and (end_i > 0) and (end_i < n_time):
                val_masked[i, end_i:] = np.nan
                val_masked[i, :start_i] = np.nan
            else:
                val_masked[i, :] = np.nan

        val_masked = np.expand_dims(val_masked, axis=2)

        return val_masked

    def get_state_at_occlusion(X, res, align_to=epoch_align, offset=0):
        mask_data = np.array(res[align_to][:, :, 0])
        X_occ = np.ones(X.shape) * np.nan
        for i in range(X.shape[2]):
            X_occ[:, :, i] = np.squeeze(get_val_at_occlusion(X[:, :, i],
                                                             mask_data,
                                                             offset=offset))
        return X_occ

    def get_meta_at_occlusion(res, align_to=epoch_align, offset=0):
        mask_data = np.array(res[align_to][:, :, 0])
        sim_x = np.array(res['output_vis-sim'][:, :, 0])
        sim_y = np.array(res['output_vis-sim'][:, :, 1])
        full_meta = get_full_meta_from_xy(sim_x, sim_y)
        occ_meta = {}
        for fk_ in full_meta.keys():
            occ_meta[fk_] = get_val_at_occlusion(full_meta[fk_], mask_data, offset=offset)
        return occ_meta

    X_state = np.array(model_data['state'])
    if use_pca_proj:
        y_mask = np.array(model_data[epoch_pca][:, :, [0]])
        res_pca = utils.get_low_dim_manifold_with_stats(X_state, y_mask)
        X_state = np.array(res_pca['X_lowdim'])
        # npc = res_pca['nPC']
        # X_state = np.array(res_pca['X_lowdim'][:, :, :npc])

    data_aligned = []
    offset_list = list(np.arange(-max_offset, max_offset))
    if max_offset == 0:
        offset_list = [0]

    for offset_curr in offset_list:
        meta_occ = get_meta_at_occlusion(model_data, offset=offset_curr)
        for fk in meta_occ.keys():
            meta_occ[fk] = np.nanmean(meta_occ[fk], axis=1)
        state_occ = get_state_at_occlusion(X_state, model_data.copy(), offset=offset_curr)
        state_occ = np.nanmean(state_occ, axis=1)
        res_current_offset = {
            'meta': meta_occ,
            'state': state_occ,
        }
        data_aligned.append(res_current_offset)

    return data_aligned


def add_subspace_gradient_to_meta(res, full_meta, masks):
    def decode_with_flatten(XX, yy, mask_fn='output_sim', k=2):
        mask = masks[mask_fn]
        X_flat = utils.flatten_to_mat(XX, mask)['X']
        y_flat = utils.flatten_to_mat(yy, mask)['X']
        tmp = utils.linear_regress(X_flat, y_flat, k_folds=k)
        return tmp

    X = res['state'].copy()
    y = np.squeeze(full_meta['xy'])
    mask_epoch = 'output_sim'
    reg_xy = decode_with_flatten(X, y, mask_fn=mask_epoch, k=2)
    coeff_mat = np.nanmean(reg_xy['coeff'], axis=0)
    X_subspace = np.matmul(X, coeff_mat.T)
    X_subspace_diff = np.diff(X_subspace, axis=1, append=np.nan)

    t = np.isfinite(X_subspace) & np.isnan(X_subspace_diff)
    ni, nj, nk = np.nonzero(t)
    for i, j in zip(ni, nj):
        X_subspace_diff[i, j, :] = X_subspace_diff[i, j - 1, :]

    dx, dy = X_subspace_diff[:, :, [0]], X_subspace_diff[:, :, [1]]
    dtheta = np.arctan2(dy, dx)
    dspeed = (dy ** 2 + dx ** 2) ** 0.5
    full_meta['subspace_dx'] = dx
    full_meta['subspace_dy'] = dy
    full_meta['subspace_dtheta'] = dspeed
    full_meta['subspace_dspeed'] = dtheta
    return full_meta


def remove_transition(masks_, epoch='output_sim'):
    for ntimesteps in [5]:
        x_orig = masks_[epoch]
        x = np.roll(x_orig, ntimesteps, axis=1)
        x[np.isnan(x_orig)] = np.nan
        masks_['%s_roll%d' % (epoch, ntimesteps)] = x
    return masks_


def remove_bounces(full_meta, masks, epoch='output_sim'):
    dy_copy = deepcopy(full_meta['dy'])
    mask = deepcopy(masks[epoch])
    mask_2 = deepcopy(masks[epoch])

    t = np.isnan(masks[epoch])
    dy_copy[t] = np.nan
    dy_del = np.diff(dy_copy, append=np.nan, axis=1)
    dy_del_thres = 0.01
    t_select = np.nanmax((np.abs(dy_del) >= dy_del_thres), axis=1)

    trial_to_nan = np.squeeze(t_select) == 1
    mask[trial_to_nan, :, :] = np.nan
    trial_to_nan2 = np.squeeze(t_select) == 0
    mask_2[trial_to_nan2, :, :] = np.nan

    masks['%s_nobounce' % epoch] = mask
    # masks['%s_onlybounce' % epoch] = mask_2
    return masks


def split_into_early_late_subepochs(masks, padding=10):
    for mask_fn in ['output_sim_roll5_nobounce', 'output_vis', 'output_vis_nobounce']:
        x = masks[mask_fn]
        masks['%s_early' % mask_fn] = np.ones((x.shape)) * np.nan
        masks['%s_late' % mask_fn] = np.ones((x.shape)) * np.nan
        for i in range(x.shape[0]):
            j = np.nonzero(np.isfinite(x[i, :, :]))[0]
            if len(j) != 0:
                masks['%s_early' % mask_fn][i, j[0]:(j[0] + padding)] = 1
                masks['%s_late' % mask_fn][i, j[-1] - padding:(j[-1])] = 1
    return masks


def get_bin_selections(binvar_matrix):
    x, y = binvar_matrix[:, 0], binvar_matrix[:, 1]
    xbins = [0, 4, 8, 12, 16, 20, 24, 28, 32.0]
    ybins = [0, 4, 8, 12, 16, 20, 24, 28, 32.0]

    bin_xy_center = []
    bin_selection = []
    for xi in range(len(xbins) - 1):
        for yi in range(len(ybins) - 1):
            tx = (x >= xbins[xi]) & (x < xbins[xi + 1])
            ty = (y >= ybins[yi]) & (y < ybins[yi + 1])
            bin_selection.append(tx & ty)
            bin_xy_center.append([xbins[xi], ybins[yi]])

    bin_res = {'select': bin_selection,
               'centers': bin_xy_center}
    return bin_res


def get_matrices_from_res(res_, full_meta_, masks_, label='dspeed',
                          mask_fn='output_sim', t_select=None):
    states = res_['state'].copy()
    full_label = full_meta_['xydxdy'].copy()
    label = full_meta_[label].copy()
    xy = full_meta_['xy'].copy()
    mask = masks_[mask_fn][:, :, 0]

    if t_select is not None:
        states = states[t_select, :, :]
        label = label[t_select, :, :]
        full_label = full_label[t_select, :, :]
        xy = xy[t_select, :, :]
        mask = mask[t_select, :]

    t = np.isfinite(mask)

    state_matrix = states[t, :]
    meta_matrix = label[t, :]
    binvar_matrix = xy[t, :]
    full_label_matrix = full_label[t, :]
    bin_res = get_bin_selections(binvar_matrix)
    bin_selections = bin_res['select']
    bin_centers = bin_res['centers']
    summary = {
        'X': state_matrix,
        'y': meta_matrix,
        'z': binvar_matrix,
        'all': full_label_matrix,
        'bin_selections': bin_selections,
        'bin_centers': bin_centers,

    }
    return summary
