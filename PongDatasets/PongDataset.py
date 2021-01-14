import numpy as np
import pandas as pd
import itertools
import h5py
import pickle as pk
import sys, argparse, os
from sklearn.model_selection import train_test_split
from copy import deepcopy
from PIL import Image, ImageDraw

datapath_default = '/om/user/rishir/data/pong_basic/RF/'

parser = argparse.ArgumentParser()
parser.add_argument('--img_h', default=32, type=int)
parser.add_argument('--img_w', default=32, type=int)
parser.add_argument('--pad', default=8, type=int)
parser.add_argument('--grid_size', default=4, type=int)

parser.add_argument('--n_frames', default=100, type=int)
parser.add_argument('--n_frames_vis_min', default=15, type=int)
parser.add_argument('--n_frames_vis_max', default=45, type=int)
parser.add_argument('--n_frames_occ_min', default=15, type=int)
parser.add_argument('--n_frames_occ_max', default=45, type=int)
parser.add_argument('--output_shape_length', default=1, type=int)

parser.add_argument('--n_speed_bins', default=5, type=int)
parser.add_argument('--max_n_bounce', default=1, type=int)
parser.add_argument('--pos_resolution', default=0.01, type=float)
parser.add_argument('--heading_resolution', default=150, type=int)
parser.add_argument('--max_n_trials', default=10**7, type=int)




class PongTrial(object):

    def __init__(self, **kwargs):

        self.img_w = kwargs.get('img_w', 32)
        self.img_h = kwargs.get('img_h', 32)
        self.ball_size = kwargs.get('ball_size', 1)
        self.grid_size = kwargs.get('grid_size', 4)
        self.pad = kwargs.get('pad', 8)
        # self.n_grid_points_x = np.ceil(self.img_w / self.grid_size).astype(int)
        # self.n_grid_points_y = np.ceil(self.img_h / self.grid_size).astype(int)
        self.max_n_bounce = kwargs.get('max_n_bounce', 1)

        self.n_frames = kwargs.get('n_frames', 100)
        self.n_frames_vis_min = kwargs.get('n_frames_vis_min', 15)
        self.n_frames_vis_max = kwargs.get('n_frames_vis_max', 45)
        self.n_frames_occ_min = kwargs.get('n_frames_occ_min', self.n_frames_vis_min)
        self.n_frames_occ_max = kwargs.get('n_frames_occ_max', self.n_frames_vis_max)
        self.n_frames_max = self.n_frames_vis_max + self.n_frames_occ_max

        self.output_shape = kwargs.get('output_shape', 'step')
        self.output_shape_length = kwargs.get('output_shape_length', 1)

        # these are class input parameters
        self.x = None
        self.y = None
        self.r = None
        self.heading = None
        self.speed = None

        # these are class output parameters
        self.trial = None
        self.meta = None
        self.output = None
        self.label = None
        self.t_occ = None
        self.t_end = None
        return

    def generate_trial(self):
        trial = []
        x, y, r, heading, speed = self.x, self.y, self.r, self.heading, self.speed

        bounce = 0
        # m,b =

        dx = speed * np.cos(heading)
        dy = speed * np.sin(heading)
        m = dy/dx
        b = y - m*x
        ball_metadata = [x, y, heading, speed, bounce, m, b]

        for i in range(self.n_frames):
            trial.append(np.array(ball_metadata))

            ball_metadata[0] += dx
            ball_metadata[1] += dy
            ball_metadata[2] = np.arctan(dy / dx)
            ball_metadata[3] = (dy ** 2 + dx ** 2) ** 0.5
            ball_metadata[4] = bounce
            ball_metadata[5] = dy/dx
            ball_metadata[6] = ball_metadata[1] - m*ball_metadata[0]

            overshoot_lower = r - ball_metadata[1]
            overshoot_upper = ball_metadata[1] - (self.img_h - r)
            overshoot_left = r - ball_metadata[0]

            if overshoot_lower > 0:
                dy = -dy
                bounce = 1
                ball_metadata[1] += overshoot_lower * 2
            elif overshoot_upper > 0:
                dy = -dy
                bounce = 1
                ball_metadata[1] -= overshoot_upper * 2
            elif overshoot_left > 0:
                dx = -dx
                bounce = 1
                ball_metadata[0] += overshoot_left * 2
            else:
                bounce = 0
        self.trial = np.squeeze(trial)

        return

    def prune_trial(self):
        x = self.trial[:, 0]
        n_bounce = np.nansum(self.trial[:, 4])

        trial_end_idx = np.nonzero(x >= self.img_w)[0]
        vis_end_idx = np.nonzero(x >= (self.img_w - self.pad))[0]

        if (vis_end_idx.shape[0] == 0) or (trial_end_idx.shape[0] == 0):
            # never reached wall or occluder
            self.trial = None
            return

        if n_bounce > self.max_n_bounce:
            # too many bounces
            self.trial = None
            return

        t_v = vis_end_idx[0]
        t_o = trial_end_idx[0] - vis_end_idx[0]
        t_e = trial_end_idx[0]

        self.trial = self.trial[:(t_e+1)]

        if (t_v > self.n_frames_vis_max) or (t_v < self.n_frames_vis_min):
            self.trial = None

        if (t_o > self.n_frames_occ_max) or (t_o < self.n_frames_occ_min):
            self.trial = None

        if self.trial is not None:
            self.trial = self.trial[:self.n_frames_max, :]
            nan_pad_size = self.n_frames_max - self.trial.shape[0]
            self.trial = np.pad(self.trial, ((0, nan_pad_size), (0, 0)),
                                'constant', constant_values=np.nan)

        return

    def generate_meta(self):
        if self.trial is not None:
            self.meta = {
                'x0': self.trial[0, 0],
                'y0': self.trial[0, 1],
                'heading0': self.trial[0, 2],
                'speed0': self.trial[0, 3],
                'n_bounce': np.nansum(self.trial[:, 4]),
                'm0': self.trial[0, 5],
                'b0': self.trial[0, 6],
                'pad': self.pad,
                'img_w': self.img_w,
                'img_h': self.img_h,

            }
        else:
            self.meta = None
        return

    def generate_output(self):
        if self.trial is None:
            self.output = None
        else:
            x = self.trial[:, 0]
            try:
                trial_end_idx = np.nonzero(x >= self.img_w)[0][0]
                vis_end_idx = np.nonzero(x >= (self.img_w - self.pad))[0][0]
                y_final = self.trial[trial_end_idx, 1]
                self.output = np.ones(x.shape) * np.nan

                self.t_occ = vis_end_idx
                self.t_end = trial_end_idx

                self.label = [y_final, trial_end_idx, vis_end_idx]

                if self.output_shape == 'step':
                    start_idx = trial_end_idx - self.output_shape_length + 1
                    end_idx = trial_end_idx + 1
                    self.output[start_idx:end_idx] = y_final
            except:
                print(np.nonzero(x >= self.img_w))
                self.trial = None
                self.output = None

        return

    def run_forward(self, x=0, y=0, r=0, heading=0, speed=0):
        self.x = x
        self.y = y
        self.r = r
        self.heading = heading
        self.speed = speed
        self.generate_trial()
        self.prune_trial()
        self.generate_meta()
        self.generate_output()
        return


class TrialEncoder(object):
    def __init__(self, **kwargs):
        self.representation = kwargs.get('representation', 'simple_pxl')
        self.grid_size = kwargs.get('grid_size')
        self.ball_size = kwargs.get('ball_size')
        self.pad = kwargs.get('pad')
        self.img_h = kwargs.get('img_h')
        self.img_w = kwargs.get('img_w')

        self.img_w_out = 100
        self.img_rescale = self.img_w_out / self.img_w

        return

    def prep_filters(self):
        return

    def convert_trial(self, trial_data):
        if self.representation == 'simple_pxl':
            return self.convert_trial_simple_pxl(trial_data)
        elif self.representation == 'pixel':
            return self.convert_trial_pxl(trial_data)
        return

    def convert_trial_simple_pxl(self, data):
        def bin_xy(x_, y_):
            gs = self.grid_size
            mv = self.img_w
            ngrid_points = np.ceil(mv / gs).astype(int)
            grid_x = np.zeros((ngrid_points + 1, ngrid_points + 1))

            if np.isnan(x_):
                return grid_x
            x_b = np.floor(x_ / gs).astype(int)
            y_b = np.floor(y_ / gs).astype(int)
            x_m = 1 - (x_ % gs) / gs
            y_m = 1 - (y_ % gs) / gs

            max_idx = ngrid_points - 1
            if (x_b < max_idx) and (y_b < max_idx):
                grid_x[y_b, x_b] = x_m * y_m
                grid_x[y_b + 1, x_b] = (1 - x_m) * y_m
                grid_x[y_b, x_b + 1] = x_m * (1 - y_m)
                grid_x[y_b + 1, x_b + 1] = (1 - x_m) * (1 - y_m)

            return grid_x

        def bin_frame_joint(x_, y_):
            return list(map(bin_xy, x_, y_))

        def bin_all_joint(x_, y_):
            return list(map(bin_frame_joint, x_, y_))

        def add_occluder(pixel_grid_):
            pixel_grid_ = np.squeeze(pixel_grid_)
            pad_start_x = self.img_w - self.pad
            pad_start_bin = np.ceil(pad_start_x / self.grid_size).astype(int)
            # pad_val = (pad_start_x % self.grid_size) / self.grid_size
            # pixel_grid_[:, :, :, pad_start_bin] = pad_val
            pixel_grid_[:, :, :, pad_start_bin:] = 1
            return pixel_grid_

        x = list(data[:, :, 0])
        y = list(data[:, :, 1])
        pixel_grid_raw = bin_all_joint(x, y)
        pixel_grid_occluded = add_occluder(pixel_grid_raw)
        # [nbatch, ntime, nbin, _] = pixel_grid_occluded.shape
        # pixel_grid_occluded = np.reshape(pixel_grid_occluded, (nbatch, ntime, nbin * nbin))
        return pixel_grid_occluded


    def convert_trial_pxl(self, trial_metadata):
        occluder_metadata = [self.img_w - self.pad, 0, self.img_w, self.img_w]
        occluder_metadata = [tmp * self.img_rescale for tmp in occluder_metadata]

        def convert_to_pixel(x_, y_):
            x = x_ * self.img_rescale
            y = y_ * self.img_rescale
            r_ = self.ball_size * self.img_rescale
            im = Image.new('L', (self.img_w_out , self.img_w_out ), color=0)
            draw = ImageDraw.Draw(im)
            ball_metadata = [x - r_, y - r_, x + r_, y + r_]
            draw.ellipse(ball_metadata, fill=(255))
            draw.rectangle(occluder_metadata, fill=(128))
            return np.asarray(im)

        def convert_to_pixel_all_frames(x_, y_):
            return list(map(convert_to_pixel, x_, y_))

        def convert_to_pixel_all_trials(x_, y_):
            return list(map(convert_to_pixel_all_frames, x_, y_))

        x,y = list(trial_metadata[:,:,0]), list(trial_metadata[:,:,1])
        return np.array(convert_to_pixel_all_trials(x,y))

    def convert_trial_mt(self, trial_metadata):
        return


class PongDataset(object):

    def __init__(self, **kwargs):
        self.img_h = kwargs.get('img_h', 32)
        self.img_w = kwargs.get('img_h', 32)
        self.pad = kwargs.get('pad', 8)
        self.grid_size = kwargs.get('grid_size', 4)
        self.n_frames = kwargs.get('n_frames', 100)
        self.n_frames_vis_min = kwargs.get('n_frames_vis_min', 15)
        self.n_frames_vis_max = kwargs.get('n_frames_vis_max', 45)
        self.n_frames_occ_min = kwargs.get('n_frames_occ_min', 15)
        self.n_frames_occ_max = kwargs.get('n_frames_occ_max', 45)
        self.output_shape = kwargs.get('output_shape', 'step')
        self.output_shape_length = kwargs.get('output_shape_length', 1)

        self.pong_trial = PongTrial(**self.__dict__)
        self.pos_resolution = kwargs.get('pos_resolution', 0.01)
        self.x_grid = np.arange(0.05, 0.55, self.pos_resolution)
        self.y_grid = np.arange(0.05, 0.95, self.pos_resolution)
        self.r = kwargs.get('r', 0.02)
        self.heading_span = kwargs.get('heading_span', 0.4)
        self.heading_resolution = kwargs.get('heading_resolution', 144)
        self.n_speed_bins = kwargs.get('n_speed_bins', 5)
        self.max_n_bounce = kwargs.get('max_n_bounce', 1)

        self.n_batch = kwargs.get('n_batch', 512)
        self.max_n_trials = kwargs.get('max_n_trials', 10000000)
        self.test_size = kwargs.get('test_size', 0.1)
        self.seed = kwargs.get('seed', 0)

        self.dataset = {}
        self.dataset_splits = {}
        self.datapath = '/om/user/rishir/data/pong_basic/RF/'
        self.filename = 'occluded_pong_bounce%d_pad%d_%dspeed' % (self.max_n_bounce,
                                                                  self.pad, self.n_speed_bins)
        self.save_path = '%s/%s/' % (self.datapath, self.filename)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        return

    def get_dataset(self):
        def pre_prune_trial(param_x):
            dist_to_travel_1 = (self.img_w - self.pad) - param_x[0]
            travel_speed = param_x[4] * np.cos(param_x[3])
            time_travel_1 = dist_to_travel_1 / travel_speed
            if (time_travel_1 > (self.n_frames_vis_max-1)) or (time_travel_1 < (self.n_frames_vis_min+1)):
                return False
            time_travel_2 = self.pad / travel_speed
            if (time_travel_2 > (self.n_frames_occ_max-1)) or (time_travel_2 < (self.n_frames_occ_min+1)):
                return False
            return True

        print('Get dataset')
        span_in_rad = np.pi*self.heading_span
        heading_grid = np.linspace(-span_in_rad, span_in_rad, self.heading_resolution)

        choices = {
            'x': self.x_grid * self.img_w,
            'y': self.y_grid * self.img_h,
            'r': [self.img_h * self.r],
            'heading': heading_grid,
            'speed': list(np.linspace(0, 1, self.n_speed_bins+1)[1:]),
        }

        all_choices = list(itertools.product(*tuple(choices.values())))
        choice_keys = choices.keys()

        pruned_trials_logical = list(map(pre_prune_trial, tuple(all_choices)))
        pruned_trial_idx = np.nonzero(pruned_trials_logical)[0]
        if len(pruned_trial_idx) > self.max_n_trials:
            pruned_trial_idx = np.random.choice(pruned_trial_idx,
                                                self.max_n_trials, replace=False)
        all_choices = [all_choices[i] for i in pruned_trial_idx]
        dataset = {
            'trial': [],
            'output': [],
            'label': [],
            'meta': [],
        }
        n_trial_count = 0
        for gen_choice in all_choices:
            param = {}
            for fki, fk in enumerate(choice_keys):
                param[fk] = gen_choice[fki]

            self.pong_trial.run_forward(**param)
            if self.pong_trial.trial is not None:
                dataset['trial'].append(self.pong_trial.trial)
                dataset['output'].append(self.pong_trial.output)
                dataset['label'].append(self.pong_trial.label)
                dataset['meta'].append(self.pong_trial.meta)
                n_trial_count += 1

        dataset['meta'] = pd.DataFrame(dataset['meta'])
        self.dataset = dataset

        return

    def get_train_test_split(self):
        print('Get train_test splits')

        def split_dataset(dataset_, index):
            """ split into training and validation set. for each, run split_and_save_batches"""
            dataset_split_ = {}
            for fk_ in dataset_.keys():
                if isinstance(dataset_[fk_], pd.DataFrame):
                    dataset_split_[fk_] = dataset_[fk_].iloc[index, :]
                else:
                    dataset_split_[fk_] = np.array(dataset_[fk_])[index]
            return dataset_split_

        """ split into training and validation set."""
        dataset = self.dataset
        lab = pd.DataFrame(dataset['label'])
        train_l, valid_l = train_test_split(lab, test_size=self.test_size,
                                            random_state=self.seed)
        split_idx = {'train': train_l.index, 'valid': valid_l.index}

        for fk in split_idx.keys():
            self.dataset_splits[fk] = split_dataset(dataset, split_idx[fk])
        return

    def convert_to_pixel_input(self, data):

        def bin_xy(x_, y_):
            gs = self.grid_size
            mv = self.img_w
            ngrid_points = np.ceil(mv / gs).astype(int)
            grid_x = np.zeros((ngrid_points + 1, ngrid_points + 1))

            if np.isnan(x_):
                return grid_x
            x_b = np.floor(x_ / gs).astype(int)
            y_b = np.floor(y_ / gs).astype(int)
            x_m = 1 - (x_ % gs) / gs
            y_m = 1 - (y_ % gs) / gs

            max_idx = ngrid_points - 1
            if (x_b < max_idx) and (y_b < max_idx):
                grid_x[y_b, x_b] = x_m * y_m
                grid_x[y_b + 1, x_b] = (1 - x_m) * y_m
                grid_x[y_b, x_b + 1] = x_m * (1 - y_m)
                grid_x[y_b + 1, x_b + 1] = (1 - x_m) * (1 - y_m)

            return grid_x

        def bin_frame_joint(x_, y_):
            return list(map(bin_xy, x_, y_))

        def bin_all_joint(x_, y_):
            return list(map(bin_frame_joint, x_, y_))

        def add_occluder(pixel_grid_):
            pixel_grid_ = np.squeeze(pixel_grid_)
            pad_start_x = self.img_w - self.pad
            pad_start_bin = np.ceil(pad_start_x / self.grid_size).astype(int)
            # pad_val = (pad_start_x % self.grid_size) / self.grid_size
            # pixel_grid_[:, :, :, pad_start_bin] = pad_val
            pixel_grid_[:, :, :, pad_start_bin:] = 1
            return pixel_grid_

        x = list(data[:, :, 0])
        y = list(data[:, :, 1])
        pixel_grid_raw = bin_all_joint(x, y)
        pixel_grid_occluded = add_occluder(pixel_grid_raw)
        [nbatch, ntime, nbin, _] = pixel_grid_occluded.shape
        pixel_grid_occluded = np.reshape(pixel_grid_occluded, (nbatch, ntime, nbin * nbin))
        pixel_grid_raw = np.reshape(pixel_grid_raw, (nbatch, ntime, nbin * nbin))
        return pixel_grid_raw, pixel_grid_occluded

    def save_batches(self):
        print('Save batches')

        def save_one_batch(dataset_split_, start_idx, end_idx, out_fn):
            tmp = dataset_split_['trial'][start_idx:end_idx]
            # tmp_pxl_raw, tmp_pxl_occ = self.convert_to_pixel_input(tmp)
            data_to_save = {
                'label': dataset_split_['label'][start_idx:end_idx],
                'output': dataset_split_['output'][start_idx:end_idx],
                # 'input_pxl_raw': tmp_pxl_raw,
                # 'input_pxl_occ': tmp_pxl_occ,
                'input': tmp,
            }
            out_data_curr = {}
            for ds_fk in data_to_save:
                out_data_curr['dim_%s' % ds_fk] = data_to_save[ds_fk].shape
            # print('Saving to %s' % out_fn)
            with h5py.File(out_fn, 'w') as f:
                for fk_ in data_to_save.keys():
                    f.create_dataset(fk_, data=data_to_save[fk_])
            # print('Saved %s' % out_fn)
            return out_data_curr

        def split_and_save_batches(dataset_split_, out_dir_):
            """ Note that meta and batches are saved in the same order.
            index of meta is with respect to the full dataset, but
            not relevant for selecting data from batches. Think of it like
            a unique hash id for each trial condition. """
            file_list = []
            inputs_dim = dataset_split_['trial'].shape
            n_samples = inputs_dim[0]
            num_batches = int(n_samples / self.n_batch)

            if not os.path.exists(out_dir_):
                os.makedirs(out_dir_)

            out_data = {}
            for bi in range(num_batches):
                out_fn = out_dir_ + '/batch_' + str(bi) + '.h5'
                start_idx = self.n_batch * bi
                end_idx = self.n_batch * (bi + 1)
                out_data = save_one_batch(dataset_split_, start_idx, end_idx, out_fn)
                file_list.append(out_fn)

            out_fn = out_dir_ + '.pkl'
            out_data.update({
                'file_path': file_list,
                'meta': dataset_split_['meta'],
            })
            with open(out_fn, 'wb') as f:
                f.write(pk.dumps(out_data))
            return out_fn

        for fk in self.dataset_splits:
            out_dir = '%s/%s/%s' % (self.datapath, self.filename, fk)
            split_and_save_batches(self.dataset_splits[fk], out_dir)

    def save_meta_sample(self):
        def save_meta_sample_base(meta, n_samples=100, out_fn_suffix='meta_sample'):
            out_fn = '%s/%s/%s.pkl' % (self.datapath, self.filename, out_fn_suffix)
            meta['meta_index'] = meta.index
            meta = meta.reset_index(drop=True)

            n_samples_max = meta.query('n_bounce > 0').shape[0]
            n_samples = np.min([n_samples, n_samples_max])
            # ensure that there are equally many bounce and no bounce trials in any set [0,2*N]
            x1 = meta.query('n_bounce > 0').sample(n_samples, replace=False, random_state=0).reset_index(drop=True)
            x0 = meta.query('n_bounce == 0').sample(n_samples, replace=False, random_state=0).reset_index(drop=True)

            meta_subsample = pd.concat([x1, x0]).sort_index().reset_index(drop=True)
            meta_subsample = meta_subsample.to_dict()

            with open(out_fn, 'wb') as f:
                f.write(pk.dumps(meta_subsample))
            return

        print('Save validation meta-sample')
        save_meta_sample_base(self.dataset_splits['valid']['meta'], n_samples=100,
                              out_fn_suffix='valid_meta_sample')
        print('Save train meta-sample')
        save_meta_sample_base(self.dataset_splits['train']['meta'], n_samples=100,
                              out_fn_suffix='train_meta_sample')

        return

    def run_all(self):
        self.get_dataset()
        self.get_train_test_split()
        self.save_batches()
        self.save_meta_sample()
        return

def update_with_new_representation(orig_dataset_path_='occluded_pong_bounce1_pad8_4speed/',
                                   representation='pixel'):

    orig_dataset_path = '%s/%s' % (datapath_default, orig_dataset_path_)

    valid_fn = '%s/valid.pkl' % orig_dataset_path
    valid_meta = pk.load(open(valid_fn, 'rb'))
    train_fn = '%s/train.pkl' % orig_dataset_path
    train_meta = pk.load(open(train_fn, 'rb'))

    pad = np.unique(valid_meta['meta']['pad'])
    img_w = np.unique(valid_meta['meta']['img_w'])
    img_h = np.unique(valid_meta['meta']['img_h'])
    ball_size = 1.0

    enc = TrialEncoder(representation=representation,
                       grid_size=None,
                       ball_size=ball_size,
                       pad=pad,
                       img_w=img_w,
                       img_h=img_h)

    all_files = valid_meta['file_path'] + train_meta['file_path']
    for fn in all_files:
        with h5py.File(fn, 'a') as f:
            if 'input_%s' % representation not in f.keys():
                inputs = f['input'][:]
                f['input_%s' % representation] = enc.convert_trial(inputs)

            if 'input_pxl_raw' in f.keys():
                del f['input_pxl_raw']
            print(fn)
    return

def map_with_sine(output_all):
    """ output_all = {xy, xy_vis, xy_sim, y_mov, y_simstep}.
    replace the simstep channel with a sine output.
    """
    return output_all

def map_batch_output(input_raw_, output_, label_, with_initial_cond):
    xy = input_raw_[..., [0, 1]]
    trial_end_idx = label_[:, 1].astype(int)
    vis_end_idx = label_[:, 2].astype(int)

    xy_vis = deepcopy(xy)
    xy_sim = deepcopy(xy)
    y_simstep = deepcopy(xy[:,:,[1]])
    y_final = deepcopy(output_)
    if with_initial_cond:
        zero_point = 32/2.0
    else:
        zero_point = np.nan
    for vi, v_idx in enumerate(vis_end_idx):
        t_idx = trial_end_idx[vi]
        xy_vis[vi, v_idx:, :] = np.nan
        xy_sim[vi, :v_idx, :] = np.nan

        y_final[vi,0] = zero_point

        y_simstep[vi, :v_idx, :] = np.nan
        y_simstep[vi,v_idx:t_idx,:] = xy_sim[vi,t_idx,1]
        y_simstep[vi, t_idx:, :] = np.nan

    so = y_final.shape
    y_mov = np.reshape(y_final, (so[0], so[1], 1))
    y_simstep = np.reshape(y_simstep, (so[0], so[1], 1))
    output_all = np.concatenate((xy, xy_vis, xy_sim, y_mov, y_simstep), axis=2)
    return output_all


def main(argv):
    if argv[0] == 'update':
        update_with_new_representation()
    else:
        flags_, _ = parser.parse_known_args(argv)
        flags = vars(flags_)
        pong_dataset = PongDataset(**flags)
        pong_dataset.run_all()
        fn_tmp = pong_dataset.filename
        update_with_new_representation(orig_dataset_path_=fn_tmp,
                                       representation='pixel')


    return


if __name__ == "__main__":
    main(sys.argv[1:])
