import numpy as np
import h5py
from scipy.signal import fftconvolve

gabor_bank_fn = '/om/user/rishir/lib/motion_encoder_python/tmp.h5'
with h5py.File(gabor_bank_fn, 'r') as f:
    gabor_bank = f['gabor_bank'][:]
    gabor_params = f['gabor_params'][:]

def run_gabor_fast(s_minibatch, l_minibatch, max_time=90):
    """ s_minibatch: nbatch * T * X * Y * 1
        l_minibatch: nbatch * 1
    """
    max_time_curr = int(np.nanmax(l_minibatch))
    if max_time_curr == 0:
        max_time_curr = max_time
    gs = gabor_bank.shape

    def fast_conv(x):
        x = np.squeeze(x)
        out_all = []
        for i in range(gabor_bank.shape[0]):
            tmp_filt = []
            for j in range(gabor_bank.shape[1]):
                gabor_filter_ = gabor_bank[i, j, ...]
                g = np.reshape(gabor_filter_, (gs[4], gs[2], gs[3]))
                tmp = fftconvolve(x, g, 'same')
                tmp = np.pad(tmp, ((0, max_time - max_time_curr),
                                   (0, 0), (0, 0)), 'edge')
                tmp_filt.append(tmp)
            out = np.sqrt(tmp_filt[0] ** 2 + tmp_filt[1] ** 2)
            out_all.append(out)
        out_all = np.array(out_all)
        out_all = np.transpose(out_all, (1, 2, 3, 0))  # transpose to (t,x,y,filt)
        return out_all

    s_minibatch_clipped = s_minibatch[:, :max_time_curr, ...]
    out_minibatch = list(map(fast_conv, s_minibatch_clipped))
    out_minibatch = np.squeeze(out_minibatch)
    return out_minibatch