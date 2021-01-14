import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

sns.set_style('ticks', {"axes.linewidth": "1", 'axes.yaxis.grid': False})
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, ShuffleSplit
from scipy import stats, linalg


def mask2d(data_matrix, mask_matrix):  # keep rows where mask is finite
    idx = np.nonzero(np.isfinite(np.mean(mask_matrix, axis=1)))[0]
    mask_masked = mask_matrix[idx, :]
    data_masked = data_matrix[idx, :]
    return data_masked, mask_masked


""" 
Methods to map a 3D data matrix (trials, time, neurons) into a 
2D data matrix, while applying a mask for specific time bins for specific trials.
"""


def get_mask(mask):
    ms = mask.shape
    mask_2 = np.reshape(mask, (ms[0] * ms[1], ms[2]))
    return np.nonzero(np.isfinite(np.mean(mask_2, axis=1)))[0]


def flatten_to_mat(x, mask):
    xs = x.shape
    x2 = np.reshape(x, (xs[0] * xs[1], xs[2]))
    idx = get_mask(mask)
    x3 = x2[idx, :]
    return {'X': x3, 'idx': idx, 's': xs}


def unflatten_to_3d_mat(X, res):
    x_, idx, xs = res['X'], res['idx'], res['s']
    xs = list(xs)
    if np.prod(X.shape) != np.prod(xs):
        xs[-1] = X.shape[-1]  # third dimension has changed
    x2 = np.ones((xs[0] * xs[1], xs[2])) * np.nan
    x2[idx, :] = X
    x = np.reshape(x2, xs)
    return x


def get_low_dim_manifold_with_stats(x, y):
    if y.ndim < x.ndim:
        y = np.expand_dims(y, axis=2)
    x_res = flatten_to_mat(x, y)
    ncomp = np.nanmin([x_res['X'].shape[0], x_res['X'].shape[1]])
    pca = PCA(n_components=ncomp, svd_solver='full')
    tmp = pca.fit_transform(x_res['X'])
    eigen_vals = pca.explained_variance_
    exp_var = pca.explained_variance_ratio_
    pca_axes = pca.components_

    tnan = np.isfinite(np.nanmean(tmp, axis=1))
    X_mat = tmp[tnan, :]

    stats_ = {
        'exp_var': exp_var,
        'PR': np.nansum(eigen_vals) ** 2 / np.nansum(eigen_vals ** 2),
        'nPC': np.nonzero(np.cumsum(exp_var) >= 0.9999)[0][0],
        'X_lowdim': unflatten_to_3d_mat(tmp, x_res),
        'X_lowdim_mat': X_mat,
        'W_pca': pca_axes,
    }
    return stats_


""" 
Methods for measuring consistency (noise adjusted correlation and normalized square error).
"""


def numpy_groupby(data_matrix, groupby_var):
    mu, sig = [], []
    for ug in np.unique(groupby_var):
        ti = np.nonzero(groupby_var == ug)[0]
        mu.append(np.nanmean(data_matrix[ti], axis=0))
        sig.append(np.nanstd(data_matrix[ti], axis=0))
        # mu.append(np.nanmean(data_matrix[ti, :], axis=0))
        # sig.append(np.nanstd(data_matrix[ti, :], axis=0))
    return np.array(mu), np.array(sig)


def get_noise_corrected_nse_base(x1, x2, y1, y2):
    """ eqns from Norman-Haignere and McDermott, 2018 """
    mu_xy = np.nanmean([np.nanmean(x1 * y1), np.nanmean(x1 * y2), np.nanmean(x2 * y1), np.nanmean(x2 * y2)])
    mu_x = np.nanmean([np.nanmean(x1), np.nanmean(x2)])
    mu_y = np.nanmean([np.nanmean(y1), np.nanmean(y2)])

    mu_x2 = np.nanmean([np.nanmean(x1 * x1), np.nanmean(x2 * x2)]) - np.nanmean((x1 - x2) ** 2) / 2
    mu_y2 = np.nanmean([np.nanmean(y1 * y1), np.nanmean(y2 * y2)]) - np.nanmean((y1 - y2) ** 2) / 2

    return (mu_x2 + mu_y2 - 2 * mu_xy) / (mu_x2 + mu_y2 - 2 * mu_x * mu_y)


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

    def spearman_brown_correction(r):
        return (2 * r) / (1 + r)

    assert(x1.shape[0] == x2.shape[0])
    assert (y1.shape[0] == y2.shape[0])
    assert (x1.shape[0] == y1.shape[0])

    rxx = corr_fn(x1, x2)
    ryy = corr_fn(y1, y2)
    rxy = np.nanmean([corr_fn(x1, y1), corr_fn(x1, y2), corr_fn(x2, y1), corr_fn(x2, y2)])
    try:
        rhon = rxy / ((rxx * ryy) ** 0.5)
    except:
        rhon = np.nan
    res = {
        'rxx': rxx,
        'ryy': ryy,
        'rxy': rxy,
        'rhon': rhon,
        'rxx_sb': spearman_brown_correction(rxx),
        'ryy_sb': spearman_brown_correction(ryy),
    }
    return res


""" 
Methods for plotting animations.
"""


def flatten_pca(x_, n_comp=3):
    s = x_.shape
    xs = np.reshape(x_, (s[0] * s[1], s[2]))
    ys = np.ones((s[0] * s[1], n_comp)) * np.nan
    idx = np.isfinite(np.nanmean(xs, axis=1))
    xxs = xs[idx]
    pca = PCA(n_components=n_comp)
    yys = pca.fit_transform(xxs)
    ys[idx] = yys
    return np.reshape(ys, (s[0], s[1], n_comp)), pca


#
# def scatter_animation(scatter_data, outfn, title_str=None):
#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.add_subplot(1, 1, 1)
#     X, Y = scatter_data['X'], scatter_data['Y']
#     S, C = scatter_data['S'], scatter_data['C']
#
#     time_idx = 0
#     xx = X[:, time_idx]
#     yy = Y[:, time_idx]
#     c = np.array(C[:, time_idx])
#     all_lines = []
#     for i in range(X.shape[0]):
#         line1, = ax.plot(X[i, [time_idx, time_idx + 1]],
#                          Y[i, [time_idx, time_idx + 1]], 'k-', lw=0.5)
#         all_lines.append(line1)
#     state = ax.scatter(xx, yy, s=S, c=c,
#                        vmin=0, vmax=1.0)
#     time_caption = ax.text(0, 0, '%d' % time_idx)
#     axlim = [-1.8, 1.8]
#     ax.set_xlim(axlim)
#     ax.set_ylim(axlim)
#     fig.suptitle(title_str)
#
#     # initialization function: plot the background of each frame
#     def init():
#         return (state, time_caption,) + tuple(all_lines)
#
#     #  animation function. This is called sequentially
#     def animate(i):
#         time_idx = i
#         x = X[:, time_idx]
#         y = Y[:, time_idx]
#         c = np.array(C[:, time_idx])
#         for i in range(X.shape[0]):
#             if time_idx < (X.shape[1] - 1):
#                 all_lines[i].set_xdata(X[i, [time_idx, time_idx + 1]])
#                 all_lines[i].set_ydata(Y[i, [time_idx, time_idx + 1]])
#
#         state.set_offsets(np.array([x, y]).T)
#         state.set_sizes(S)
#         state.set_array(c)
#         time_caption.set_text('%d' % time_idx)
#         return (state, time_caption,) + tuple(all_lines)
#
#     # call the animator. blit=True means only re-draw the parts that have changed.
#     anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                    frames=90, interval=50, blit=True)
#     anim.save(outfn, writer='imagemagick', fps=30)
#     plt.close(fig=fig)
#     # HTML(anim.to_html5_video())
#
#     return


""" other utils """


#
# def regress_out(X, y):
#     """ this is not cross-validated, to ensure maximum performance
#     for regressing out a confounding variable, even with limited data. """
#     # def remove_nans(x_, y_):
#     #     t = np.isfinite(np.mean(x_, axis=1)) & np.isfinite(np.mean(y_, axis=1))
#     #     return x_[t, :], y_[t, :]
#
#     # X, y = remove_nans(X, y)
#     beta = linalg.lstsq(X, y)[0]
#     residual = X - y.dot(beta)
#     return residual

def regress_out_highD(x_mat, y_mat):
    """ returns y, after regressing out x """
    reg = LinearRegression().fit(x_mat, y_mat)
    y_mat_pred = reg.predict(x_mat)
    return (y_mat - y_mat_pred)


#
def preprocess_features(X_train, X_test, method='none'):

    if method == 'pca':
        ncomp = int((min(X_train.shape) - 1))
        pca = PCA(n_components=ncomp, svd_solver='arpack')
        pca = pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
    elif method == 'scale':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    elif method == 'none':
        X_train, X_test = X_train, X_test

    if np.nansum(np.nanvar(X_train, axis=0)) < 10**(-10):
        # Add one column of random noise to avoid all zero-variance features.
        X_train_add = np.random.random(X_train.shape)
        X_test_add = np.random.random(X_test.shape)
        X_train = np.concatenate( (X_train, X_train_add[:,[0]]), axis=1)
        X_test = np.concatenate((X_test, X_test_add[:, [0]]), axis=1)

    return X_train, X_test


def linear_regress(X, y, k_folds=2, random_state=0, train_size=None, preproc_method='none'):
    """
    Note: fitting the intercept can lead to spurious correlations in small datasets.
    """
    def nnan_pearsonr(x, y):
        ind = np.isfinite(x) & np.isfinite(y)
        x, y = x[ind], y[ind]
        return pearsonr(x, y)

    def nnan_spearmanr(x, y):
        ind = np.isfinite(x) & np.isfinite(y)
        x, y = x[ind], y[ind]
        return spearmanr(x, y)

    if k_folds is not None:
        all_splits = KFold(n_splits=k_folds,
                           shuffle=True,
                           random_state=random_state)
    elif train_size is not None:
        n_splits = np.int(y.shape[0] / train_size)
        test_size = y.shape[0] - train_size
        all_splits = ShuffleSplit(n_splits=n_splits,
                                  train_size=train_size,
                                  test_size=test_size,
                                  random_state=random_state)
    else:
        all_splits = KFold(n_splits=10,
                           shuffle=True,
                           random_state=None)

    y_pred = np.ones(y.shape) * np.nan
    coeff = []
    for train_index, test_index in all_splits.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        X_train, X_test = preprocess_features(X_train, X_test, method=preproc_method)


        reg = LinearRegression()
        reg.fit(X_train, y_train)
        coeff.append(reg.coef_)
        y_pred[test_index] = reg.predict(np.array(X_test))
    r, p = nnan_pearsonr(y, y_pred)
    rs, ps = nnan_spearmanr(y, y_pred)
    mae = np.nanmean(np.abs(y - y_pred))

    res_summary = {
        'r': r,
        'p': p,
        'rs': rs,
        'ps': ps,
        'mae': mae,
        'y_pred': y_pred,
        'y_true': y,
        'coeff': np.array(coeff),
    }
    return res_summary


def get_R2_highD(X, Y, k_folds=2, train_size=None,
                 cross_validate_regression=True, niter=10,
                 preproc_method='none', return_splits=False):
    """
    X and Y are both matrices.
    *** Returns variance explained of X by Y. ***
    -- Orthogonalize X into X_pca,
    -- Perform linear regression between Y and each dimension i of X_pca,
        (this step is by default cross-validated, but the flag allows training on all data for
        cases where the Y corresponds to cross-validated predictions of X from earlier modeling.
    -- Weighted sum of R2_i for each dimension is the total proportion of variance explained.

    Note: It's critical to compute estimates of r over different splits before squaring to get R2
    to avoid biasing up by averaging positive values.


    """

    def nnan_pearsonr(x, y):
        ind = np.isfinite(x) & np.isfinite(y)
        x, y = x[ind], y[ind]
        return pearsonr(x, y)

    def quick_pca(x, return_PR=False):
        ncomp = np.min(x.shape)
        pca = PCA(n_components=ncomp, svd_solver='full')
        x_pca = pca.fit_transform(x)
        if return_PR:
            eigenvals = pca.explained_variance_
            PR = np.nansum(eigenvals) ** 2 / np.nansum(eigenvals ** 2)
            return x_pca, PR
        else:
            return x_pca

    def get_cv_linear_map_corr(X, y, random_state=None):
        if cross_validate_regression:
            res_lr = linear_regress(X, y,
                                    k_folds=k_folds,
                                    train_size=train_size,
                                    random_state=random_state,
                                    preproc_method=preproc_method)
            R = res_lr['r']
        else:
            reg = LinearRegression()
            reg.fit(X, y)
            y2 = reg.predict(X)
            R, p = nnan_pearsonr(y, y2)
        return R

    def get_prop_var_exp_cv(X_, Y_, random_state=None):
        # this needs X to be already orthogonalized

        ncomp_x, ncomp_y = X_.shape[1], Y_.shape[1]
        rho = np.zeros((ncomp_x, 1))
        for i1 in range(ncomp_x):
            source = Y_
            target = X_[:, i1]
            rho[i1, 0] = get_cv_linear_map_corr(source, target, random_state=random_state)

        var_per_comp = np.nanvar(X_, axis=0)
        prop_var_exp_per_comp = np.nansum(rho ** 2, axis=1)
        var_exp_per_comp = [var_per_comp[i] * prop_var_exp_per_comp[i] for i in range(ncomp_x)]
        prop_var_exp = np.nansum(var_exp_per_comp) / np.nansum(var_per_comp)
        return prop_var_exp

    def get_prop_var_exp_cv_bootstrap(X_, Y_):
        res_dist = []
        for boot_i in range(niter):
            res_dist.append(get_prop_var_exp_cv(X_, Y_, random_state=boot_i))
        return np.array(res_dist)

    #
    # def get_cv_linear_map_corr(X, y, random_state=None):
    #     if cross_validate_regression:
    #         r_over_iter = []
    #         for i in range(niter):
    #             res_lr = linear_regress(X, y,
    #                                     k_folds=k_folds,
    #                                     train_size=train_size,
    #                                     random_state=i,
    #                                     preproc_method=preproc_method)
    #             r_over_iter.append(res_lr['r'])
    #         R = np.nanmean(r_over_iter)
    #     else:
    #         reg = LinearRegression()
    #         reg.fit(X, y)
    #         y2 = reg.predict(X)
    #         R, p = nnan_pearsonr(y, y2)
    #         r_over_iter = [R]
    #     return R, r_over_iter

    # def get_prop_var_exp_cv(X_, Y_):
    #     # this needs X to be already orthogonalized
    #
    #     ncomp_x, ncomp_y = X_.shape[1], Y_.shape[1]
    #     rho = np.zeros((ncomp_x, 1))
    #     rho_over_niter = np.zeros((ncomp_x, 1))
    #
    #     for i1 in range(ncomp_x):
    #         source = Y_
    #         target = X_[:, i1]
    #         rho[i1, 0] = get_cv_linear_map_corr(source, target)
    #
    #     var_per_comp = np.nanvar(X_, axis=0)
    #     prop_var_exp_per_comp = np.nansum(rho ** 2, axis=1)
    #     var_exp_per_comp = [var_per_comp[i] * prop_var_exp_per_comp[i] for i in range(ncomp_x)]
    #     prop_var_exp = np.nansum(var_exp_per_comp) / np.nansum(var_per_comp)
    #     return prop_var_exp

    X_pca = quick_pca(X)
    Y_pca = quick_pca(Y)
    res_d = get_prop_var_exp_cv_bootstrap(X_pca, Y_pca)
    if return_splits:
        return np.nanmean(res_d), res_d
    else:
        return np.nanmean(res_d)



def partial_corr(C, add_offset=False, return_pval=False):
    C = np.asarray(C)

    # add column of ones for regression
    if add_offset:
        C2 = np.ones((C.shape[0], 1))
        C = np.concatenate((C, C2), axis=1)

    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    P_corr_pval = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr, pval = stats.pearsonr(res_i, res_j)
            P_corr[i, j] = corr
            P_corr[j, i] = corr

            P_corr_pval[i, j] = pval
            P_corr_pval[j, i] = pval

    if add_offset:
        P_corr = P_corr[:-1, :-1]
        P_corr_pval = P_corr_pval[:-1, :-1]

    if return_pval:
        return P_corr, P_corr_pval
    else:
        return P_corr


def plot_rolling_average(df, xval, yval, f, axes, winsize=20, color='k'):
    df = df.sort_values(xval)
    mx = df[xval].rolling(winsize).mean()
    sx = df[xval].rolling(winsize).std()
    nx = df[xval].rolling(winsize).count()
    my = df[yval].rolling(winsize).mean()
    sy = df[yval].rolling(winsize).std()

    semy = sy / (nx ** 0.5)

    axes.plot(mx, my,
              alpha=0.5, color=color)
    axes.fill_between(mx, my - 2 * semy, my + 2 * semy,
                      alpha=0.1, color=color)
    return
