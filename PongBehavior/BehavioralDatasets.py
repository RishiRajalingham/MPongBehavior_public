import sys
import pickle as pk
from glob import glob
import numpy as np
import MentalPongBehavior as MB


class BehavioralDataset(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.suffix = kwargs.get('suffix', '')
        self.data_dir_str = kwargs.get('data_dir_str')

        self.date_range = kwargs.get('date_range')
        self.out_dir = kwargs.get('out_dir')
        self.if_joystick = kwargs.get('if_joystick')
        self.if_eye = kwargs.get('if_eye')
        self.clip_early = kwargs.get('clip_early', False)

        self.data_fns = None
        return

    @staticmethod
    def get_date_from_filename(fn):
        fn_split = fn.split('/')
        if '-' in fn_split[-1]:
            fn_file = fn_split[-1]
            sess_date = float(fn_file.split('-')[-2])
        else:
            sess_date = float(fn_split[-2])  # folder has date, for human data
        return sess_date

    def set_data_filenames(self):
        def set_data_fns_base(data_dir_str, date_range):
            fns = sorted(glob(data_dir_str))
            fns_oi = []
            for fn in fns:
                sess_date = self.get_date_from_filename(fn)
                if self.date_range is None:
                    fns_oi.append(fn)
                elif sess_date in date_range:
                    fns_oi.append(fn)
            return fns_oi

        if isinstance(self.data_dir_str, list):
            self.data_fns = []
            for dds, dr in zip(self.data_dir_str, self.date_range):
                self.data_fns.extend(set_data_fns_base(dds, dr))
        else:
            self.data_fns = set_data_fns_base(self.data_dir_str, self.date_range)
        return

    def get_data(self, verbose=False):
        self.set_data_filenames()
        data = MB.MentalPongBehavior(if_joystick=self.if_joystick,
                                     if_eye=self.if_eye,
                                     clip_early=self.clip_early)
        data.update_with_data(self.data_fns)
        run_save_fn = '%s/%s%s.pkl' % (self.out_dir, self.name, self.suffix)
        with open(run_save_fn, 'wb') as f:
            f.write(pk.dumps(data))
        print('Done %s.' % self.name)
        sys.stdout.flush()

        if verbose:
            df = data.datasets[0]['scalar']
            print('dataset size', df.shape[0])
            print('dataset_idx:', np.unique(df['dataset_idx']))
        return


hum_data_dir_default = '/om/group/jazayeri/PongPsychophysics_AP_RR/data/'
perle_data_dir_default = '/om/user/rishir/data/monk_inlab/perle/mat/'
carmen_data_dir_default = '/om/user/rishir/data/monk_inlab/carmen/mat/'
out_dir_default = '/om/user/rishir/data/behavior'

# dates used previously, missing a lot of carmen data.
# PERLE_START_DATE_pb = 20191107
# PERLE_END_DATE_pb = 20191130
# CARMEN_START_DATE_pb = 20191207
# CARMEN_END_DATE_pb = 20200130

# dates used for getting all possible behavior
# PERLE_START_DATE_pb = 20191107
# PERLE_END_DATE_pb = 20191130
# CARMEN_START_DATE_pb = 20191209
# CARMEN_END_DATE_pb = 20200220


# new dates, including latest carmen data
# and trying to match #trials across monkeys
PERLE_START_DATE_pb = 20191110
PERLE_END_DATE_pb = 20191130
CARMEN_START_DATE_pb = 20200119 #20191215
CARMEN_END_DATE_pb = 20200220

dfns = [
    # *** Human data ***
    {
        'name': 'human_pong_basic',
        'data_dir_str': '%s/pong_basic/*/*.mat' % hum_data_dir_default,
        'date_range': np.arange(20190930, 20200130 + 1),
        'out_dir': out_dir_default,
        'if_joystick': False,
        'if_eye': True,
    },
    {
        'name': 'human_pong_basic_heldout',
        'data_dir_str': '%s/pong_basic_heldout/*/*.mat' % hum_data_dir_default,
        'date_range': np.arange(20190900, 20200100 + 1),
        'out_dir': out_dir_default,
        'if_joystick': False,
        'if_eye': False,
    },
    {
        'name': 'human_pong_basic_small_paddle',
        'data_dir_str': '%s/pong_small_paddle/*/*.mat' % hum_data_dir_default,
        'date_range': np.arange(20191029, 20191030 + 1),
        'out_dir': out_dir_default,
        'if_joystick': False,
        'if_eye': True,
    },
    {
        'name': 'Human_TelePong_H',
        'data_dir_str': '%s/telepong/*/*.mat' % hum_data_dir_default,
        'date_range': np.arange(20191204, 20191230 + 1),
        'out_dir': out_dir_default,
        'if_joystick': False,
        'if_eye': True,
    },

    # *** pooled monkey data
    {
        'name': 'monkey_CP_pong_basic',
        'data_dir_str': ['%s/*.mat' % perle_data_dir_default, '%s/*.mat' % carmen_data_dir_default, ],
        'date_range': [np.arange(PERLE_START_DATE_pb, PERLE_END_DATE_pb + 1),
                       np.arange(CARMEN_START_DATE_pb, CARMEN_END_DATE_pb + 1)],
        'out_dir': out_dir_default,
        'if_joystick': True,
        'if_eye': True,
    },
    {
        'name': 'monkey_CP_pong_basic_clipped',
        'data_dir_str': ['%s/*.mat' % perle_data_dir_default, '%s/*.mat' % carmen_data_dir_default, ],
        'date_range': [np.arange(PERLE_START_DATE_pb, PERLE_END_DATE_pb + 1),
                       np.arange(CARMEN_START_DATE_pb, CARMEN_END_DATE_pb + 1)],
        'out_dir': out_dir_default,
        'if_joystick': True,
        'if_eye': True,
        'clip_early': True,
    },

    # *** Perle data ***
    {
        'name': 'perle_pong_basic_learning',
        'data_dir_str': '%s/*.mat' % perle_data_dir_default,
        'date_range': np.arange(20190802, 20190806 + 1),
        'out_dir': out_dir_default,
        'if_joystick': True,
        'if_eye': False,
    },
    {
        'name': 'perle_pong_basic_generalization',
        'data_dir_str': '%s/*.mat' % perle_data_dir_default,
        'date_range': np.array([20190809, 20191101]),
        # two sessions because generalization was tested on new 50 first, and then another new 150 later.
        'out_dir': out_dir_default,
        'if_joystick': True,
        'if_eye': False,
    },
    {
        'name': 'perle_pong_basic',
        'data_dir_str': '%s/*.mat' % perle_data_dir_default,
        'date_range': np.arange(PERLE_START_DATE_pb, PERLE_END_DATE_pb + 1),
        'out_dir': out_dir_default,
        'if_joystick': True,
        'if_eye': True,
    },
    {
        'name': 'perle_pong_basic_clipped',
        'data_dir_str': '%s/*.mat' % perle_data_dir_default,
        'date_range': np.arange(PERLE_START_DATE_pb, PERLE_END_DATE_pb + 1),
        'out_dir': out_dir_default,
        'if_joystick': True,
        'if_eye': True,
        'clip_early': True,
    },
    {
        'name': 'perle_TelePong_H_visible',
        'data_dir_str': '%s/*.mat' % perle_data_dir_default,
        'date_range': np.arange(20191201, 20191205 + 1),
        'out_dir': out_dir_default,
        'if_joystick': True,
        'if_eye': True,
        # Need to retroactively fix 20191201 to show dataset_idx=1.
    },
    {
        'name': 'perle_TelePong_H_learning',
        'data_dir_str': '%s/*.mat' % perle_data_dir_default,
        'date_range': np.arange(20191206, 20191211 + 1),
        'out_dir': out_dir_default,
        'if_joystick': True,
        'if_eye': True,
    },
    {
        'name': 'perle_TelePong_H',
        'data_dir_str': '%s/*.mat' % perle_data_dir_default,
        'date_range': np.arange(20191212, 20191218 + 1),
        'out_dir': out_dir_default,
        'if_joystick': True,
        'if_eye': True,
    },
    # *** Carmen data ***
    {
        'name': 'carmen_pong_basic_learning',
        'data_dir_str': '%s/*.mat' % carmen_data_dir_default,
        'date_range': np.arange(20191129, 20191203 + 1),
        'out_dir': out_dir_default,
        'if_joystick': True,
        'if_eye': True,
    },
    {
        'name': 'carmen_pong_basic_generalization',
        'data_dir_str': '%s/*.mat' % carmen_data_dir_default,
        'date_range': np.arange(20191206, 20191206 + 1),
        'out_dir': out_dir_default,
        'if_joystick': True,
        'if_eye': True,
    },
    {
        'name': 'carmen_pong_basic',
        'data_dir_str': '%s/*.mat' % carmen_data_dir_default,
        'date_range': np.arange(CARMEN_START_DATE_pb, CARMEN_END_DATE_pb + 1),
        'out_dir': out_dir_default,
        'if_joystick': True,
        'if_eye': True,
    },
    {
        'name': 'carmen_pong_basic_clipped',
        'data_dir_str': '%s/*.mat' % carmen_data_dir_default,
        'date_range': np.arange(CARMEN_START_DATE_pb, CARMEN_END_DATE_pb + 1),
        'out_dir': out_dir_default,
        'if_joystick': True,
        'if_eye': True,
        'clip_early': True,
    },
]
#
# recompute_vals = ['monkey_CP_pong_basic', #'monkey_CP_pong_basic_clipped',
#                   'carmen_pong_basic', #'carmen_pong_basic_clipped',
#                   'perle_pong_basic', #'perle_pong_basic_clipped'
#                   'human_pong_basic',
#                   ]  # ['human_pong_basic', 'monkey_CP_pong_basic', 'carmen_pong_basic']
# recompute_vals = ['perle_pong_basic_generalization']

recompute_vals = ['human_pong_basic_heldout']
for dfn in dfns:
    # dfn['suffix'] = '_full'
    if recompute_vals is None:
        bd = BehavioralDataset(**dfn)
        bd.get_data()
    elif dfn['name'] in recompute_vals:
        bd = BehavioralDataset(**dfn)
        bd.get_data(verbose=True)
