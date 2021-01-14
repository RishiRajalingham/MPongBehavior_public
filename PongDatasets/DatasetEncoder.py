import sys, argparse, os
import numpy as np
from sklearn.decomposition import IncrementalPCA
from glob import glob
import h5py
from datetime import datetime

import pickle as pk
from joblib import dump, load

from gabor_filter import run_gabor_fast

tmp_path_default = '/om/user/rishir/lib/PongRnn/datasets/tmp/'
path_default = '/om/user/rishir/data/pong_basic/RF/occluded_pong_bounce1_pad8_4speed/'
input_rep_fn_default = 'input_pixel'
job_default = 'valid_full'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default=path_default)
parser.add_argument('--temp_path', default=tmp_path_default)
parser.add_argument('--input_rep_fn', default=input_rep_fn_default)
parser.add_argument('--job', default=job_default)


class FullDatasetEncoder(object):
    def __init__(self, **kwargs):
        self.dataset_path = kwargs.get('dataset_path', path_default)
        self.temp_path = kwargs.get('temp_path', tmp_path_default)
        self.input_rep_fn = kwargs.get('input_rep_fn', input_rep_fn_default)

        self.n_minibatch = 32
        self.transform_batch_size = self.n_minibatch * 90
        self.n_comp = 100  # make 256 for gabors to match downsampled pixels

        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)
        self.pixel_pca_fn = '%s/%s' % (self.temp_path, 'pixel_pca_transform.joblib')
        self.gabor_pca_fn = '%s/%s' % (self.temp_path, 'gabor_pca_transform.joblib')
        self.recompute_pca_pxl = not os.path.isfile(self.pixel_pca_fn)
        self.recompute_pca_gabor = not os.path.isfile(self.gabor_pca_fn)
        self.pixels_pca_transformer = IncrementalPCA(n_components=self.n_comp, batch_size=self.transform_batch_size)
        self.gabors_pca_transformer = IncrementalPCA(n_components=self.n_comp, batch_size=self.transform_batch_size)

        return

    def get_mini_batches(self, filename):
        with h5py.File(filename, 'r') as f_:
            data = f_[self.input_rep_fn][:]
            label = f_['label'][:]
            run_pixel = 'input_pixel_pca' not in f_.keys()
            run_gabor = 'input_gabor_pca' not in f_.keys()
        # shaped as [batch, in_depth, in_height, in_width, in_channels]
        # S = np.expand_dims(S, axis=5)  # add channel dimension
        data_batches = []
        label_batches = []
        nbatches = int(data.shape[0] / self.n_minibatch)
        for i in range(nbatches):
            data_ = data[(i * self.n_minibatch):((i + 1) * self.n_minibatch), ...]
            data_batches.append(data_)
            label_batches.append(label[(i * self.n_minibatch):((i + 1) * self.n_minibatch), 1])
        return data_batches, label_batches, run_pixel, run_gabor

    def get_mini_batches_filelist(self, filenames):
        data_batches_all, label_batches_all = [], []
        r_p, r_g = True, True
        for filename in filenames:
            data_batches, label_batches, run_pixel, run_gabor = self.get_mini_batches(filename)
            data_batches_all.extend(data_batches)
            label_batches_all.extend(label_batches)
            r_p = r_p & run_pixel
            r_g = r_g & run_gabor
        return data_batches_all, label_batches_all, r_p, r_g

    @staticmethod
    def smart_fit_pca(y_batch, l_batch, pca_transform):
        """ static end frames should not be over-counted
        when doing dim reduction, as this will screw up
        variance estimates"""
        s = y_batch.shape
        mask = np.ones(y_batch.shape)
        for i, li in enumerate(l_batch):
            mask[i, int(li):, ...] = np.nan
        mat_shape = (s[0] * s[1], np.prod(s[2:]))
        mask = np.reshape(mask, mat_shape)
        nan_mask = np.isfinite(np.nanmean(mask, axis=1))

        y_mat = np.reshape(y_batch, mat_shape)
        y_mat = y_mat[nan_mask, :]
        pca_transform.partial_fit(y_mat)
        return pca_transform

    def fit_pixel_pca(self, s_batches, l_batches):
        for i, x in enumerate(s_batches):
            self.pixels_pca_transformer = \
                self.smart_fit_pca(x, l_batches[i],
                                   self.pixels_pca_transformer)

        dump(self.pixels_pca_transformer, self.pixel_pca_fn)
        return

    def transform_pixel_pca(self, s_batches):
        y_all = []
        for i, x in enumerate(s_batches):
            xs = x.shape
            y = np.reshape(x, (xs[0] * xs[1], np.prod(xs[2:])))
            y_ = self.pixels_pca_transformer.transform(y)
            y_ = np.reshape(y_, (xs[0], xs[1], self.n_comp))
            y_all.extend(y_)
        return np.array(y_all)

    def fit_gabor_pca(self, s_batches, l_batches):
        for i, x in enumerate(s_batches):
            y = run_gabor_fast(x, l_batches[i])
            self.gabors_pca_transformer = \
                self.smart_fit_pca(y, l_batches[i],
                                   self.gabors_pca_transformer)
        dump(self.gabors_pca_transformer, self.gabor_pca_fn)
        return

    def transform_gabor_pca(self, s_batches, l_batches):
        y_all = []
        for i, x in enumerate(s_batches):
            xs = x.shape
            # Y = run_gabor_one(X, l_batches[i])
            y = run_gabor_fast(x, l_batches[i])
            ys = y.shape

            y = np.reshape(y, (ys[0] * ys[1], np.prod(ys[2:])))
            y_ = self.gabors_pca_transformer.transform(y)
            y_ = np.reshape(y_, (xs[0], xs[1], self.n_comp))
            y_all.extend(y_)
        return np.array(y_all)

    @staticmethod
    def prune_file_list(all_fns, date_month):
        def get_month(fn):
            mtime = os.path.getmtime(fn)
            return float(datetime.fromtimestamp(mtime).strftime('%m'))

        return [ff for ff in all_fns if get_month(ff) < date_month]

    @staticmethod
    def prune_valid_august2020(all_fns):
        import os, time
        return_fns = []
        for ffn in all_fns:
            if 'Aug' in time.ctime(os.path.getmtime(ffn)):
                continue
            else:
                return_fns.append(ffn)
        return return_fns

    @staticmethod
    def overwrite_h5(file_handle, new_data_fn, new_data):
        if new_data_fn not in file_handle.keys():
            file_handle.create_dataset(new_data_fn, data=new_data)
        else:
            del file_handle[new_data_fn]
            file_handle.create_dataset(new_data_fn, data=new_data)
        return

    def run_simple_downsample(self):
        from skimage.transform import rescale
        def run_one_downsample(h5fn):
            with h5py.File(h5fn, 'r') as f_:
                input_pixel = f_['input_pixel'][:]
                if 'input_pixel_downsampled' in f_.keys():
                    downsampled_image = f_['input_pixel_downsampled'][:]
                else:
                    downsampled_image = None

            if downsampled_image is None:
                input_ds = np.array([[rescale(dj, 0.16, anti_aliasing=True) for dj in di] for di in input_pixel])
                with h5py.File(h5fn, 'a') as f_:
                    self.overwrite_h5(f_, 'input_pixel_downsampled', input_ds)
            else:
                x = downsampled_image
                # new_shape = (xs[0] * xs[1], np.prod(xs[2:]))
                new_shape = (512, 90, 256)
                y = np.reshape(x, new_shape)

                with h5py.File(h5fn, 'a') as f_:
                    self.overwrite_h5(f_, 'input_pixel_downsampled', y)
            return

        train_fns = glob('%s/train/*.h5' % self.dataset_path)
        valid_fns = glob('%s/valid/*.h5' % self.dataset_path)
        all_fns = train_fns + valid_fns
        for fn in all_fns:
            run_one_downsample(fn)

        fns = ['%s/train.pkl' % self.dataset_path, '%s/valid.pkl' % self.dataset_path]
        for fn in fns:
            tmp = pk.load(open(fn, 'rb'))
            tmp['dim_input_pixel_downsampled'] = (512, 90, 256)
            with open(fn, 'wb') as f:
                f.write(pk.dumps(tmp))
        return

    def run_gabor_on_downsampled(self):
        return

    def set_dimensions(self):
        fn = '%s/train.pkl' % self.dataset_path
        train_ = pk.load(open(fn, 'rb'))
        train_['dim_input_pixel_pca'] = (512, 90, self.n_comp)
        train_['dim_input_gabor_pca'] = (512, 90, self.n_comp)
        with open(fn, 'wb') as f:
            f.write(pk.dumps(train_))
        return

    def run_all(self, run_pixel=True, run_gabor=True,
                train_on_all_valid_batches=False,
                run_on_train_batches=False,
                run_on_valid_batches=False,
                run_on_valid_sample=True,
                overwrite=True):

        def train_pca():
            if train_on_all_valid_batches:
                # use all training batches to fit pca
                train_fns = glob('%s/train/*.h5' % self.dataset_path)
                train_fns = train_fns[:10]
                s_train, l_train, run_pixel_2, run_gabor_2 = self.get_mini_batches_filelist(train_fns)
                pca_suffix = '_pca_all'
            else:
                # use single training batch to fit pca
                train_fn = '%s/valid/batch_0.h5' % self.dataset_path
                s_train, l_train, run_pixel_2, run_gabor_2 = self.get_mini_batches(train_fn)
                pca_suffix = '_pca'

            if run_pixel:
                if self.recompute_pca_pxl:
                    self.fit_pixel_pca(s_train, l_train)
                else:
                    self.pixels_pca_transformer = load(self.pixel_pca_fn)

            if run_gabor:
                if self.recompute_pca_gabor:
                    self.fit_gabor_pca(s_train, l_train)
                else:
                    self.gabors_pca_transformer = load(self.gabor_pca_fn)
            print('Done fitting ')
            return pca_suffix

        def test_pca(all_fns_to_run, pca_suffix):
            print(len(all_fns_to_run))
            sys.stdout.flush()
            for fn in all_fns_to_run:
                s_test, l_test, run_pixel_2, run_gabor_2 = self.get_mini_batches(fn)
                run_pixel_curr = run_pixel & (run_pixel_2 | overwrite)
                run_gabor_curr = run_gabor & (run_gabor_2 | overwrite)

                if run_pixel_curr:
                    s_test_pxl = self.transform_pixel_pca(s_test)
                    with h5py.File(fn, 'a') as f_:
                        self.overwrite_h5(f_, 'input_pixel' + pca_suffix, s_test_pxl)

                if run_gabor_curr:
                    s_test_gab = self.transform_gabor_pca(s_test, l_test)
                    with h5py.File(fn, 'a') as f_:
                        self.overwrite_h5(f_, 'input_gabor' + pca_suffix, s_test_gab)

        pca_sfx = train_pca()
        all_fns = []
        if run_on_train_batches:
            tmp = glob('%s/train/*.h5' % self.dataset_path)
            tmp = [af for af in tmp if 'sample' not in af]
            all_fns += tmp
        if run_on_valid_batches:
            tmp = glob('%s/valid/*.h5' % self.dataset_path)
            tmp = [af for af in tmp if 'sample' not in af]
            all_fns += self.prune_valid_august2020(tmp)
        if run_on_valid_sample:
            all_fns += ['%s/valid/batch_sample.h5' % self.dataset_path]

        # date_month = 11.0
        # all_fns = self.prune_file_list(all_fns, date_month)
        run_fns = list(set(all_fns))
        test_pca(run_fns, pca_sfx)

        self.set_dimensions()

        # # rerunning for only validation files.
        # all_fns_valid = glob('%s/valid/*.h5' % self.dataset_path)
        # all_fns_valid =
        # run_fns = all_fns_valid

        return




def main(argv):
    flags_, _ = parser.parse_known_args(argv)
    flags = vars(flags_)
    job = flags.pop('job')
    denc = FullDatasetEncoder(**flags)
    if job == 'pixel_downsample':
        denc.run_simple_downsample()
    elif job == 'pca_pixel':
        denc.run_all(run_pixel=True, run_gabor=False,
                     run_on_train_batches=True, )
    elif job == 'pca_pixel_gabor':
        denc.run_all(run_pixel=True, run_gabor=True)
    elif job == 'gabor_on_downsampled':
        denc.run_gabor_on_downsampled()
    elif job == 'valid_full':
        denc.run_all(run_pixel=True, run_gabor=True,
                     run_on_valid_batches=True,
                     run_on_valid_sample=False)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
