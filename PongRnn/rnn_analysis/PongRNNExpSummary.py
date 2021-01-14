import sys

PATH_TO_FIXED_POINT_FINDER = '/om/user/rishir/lib/PongRnn/'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from rnn_analysis.PongRNNSummarizer import PongRNNSummarizer
import numpy as np
import pickle as pk
import pandas as pd
import argparse
import os
from glob import glob

default_path = '/om/user/rishir/lib/PongRnn/dat/rnn_res/'
default_fig_out_path = '/om/user/rishir/lib/PongRnn/fig/rnn_res/'
epoch_n = 0


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--update_lstm_only', default=False, type=str2bool)
parser.add_argument('--update_only', default=True, type=str2bool)
parser.add_argument('--results_path', default=default_path)
parser.add_argument('--fig_out_path', default=default_fig_out_path)
parser.add_argument('--pred_models_only', default=True, type=str2bool)

# default_func_update = ['occlusion_state', 'encode_vis-sim', 'encode_vis', 'encode_sim']
# default_func_update = ['cross_time_state_prediction']

default_func_update = ['geom_vis-sim', 'geom_vis', 'geom_sim', 'gradient_rep_piecewise_sim']


# default_func_update = ['geom_vis-sim']#, 'geom_vis', 'geom_sim']

class ExpResultsSummary(object):
    """
    -- Load all PongSummarizer result_summary into a dataframe (summary_df_all).
    -- Select best performing model per group (e.g. over epochs)
    after applying a query (random_seed<x).
    -- Measure detailed model characterizations on this reduced set of models.
    -- Save detailed model_dat.pkl

    """

    def __init__(self, **kwargs):
        self.results_path = kwargs.get('results_path', default_path)
        self.fig_out_path = kwargs.get('fig_out_path', default_fig_out_path)
        self.pred_models_only = kwargs.get('pred_models_only', True)
        self.update_lstm_only = kwargs.get('update_lstm_only', False)
        if self.fig_out_path is None:
            self.fig_out_path = self.results_path.replace('dat', 'fig')
        if not os.path.exists(self.fig_out_path):
            os.makedirs(self.fig_out_path)

        self.summary_df_all = None
        self.best_model_res = {}
        return

    def prune_model_list(self, all_models):
        if self.update_lstm_only:
            all_models = [m for m in all_models if 'lstm' in m]
        if self.pred_models_only:
            return [m for m in all_models if ('_IC' in m) and ('pca' in m) and ('pred_' in m)]
        else:
            return [m for m in all_models if ('_IC' in m) and ('pca' in m) and ('pred_' not in m)]

    def recompute_results_summary(self, force_recompute=True,
                                  search_suffix='rnn_results.pkl'):
        # rishir -- make this 'final_rnn_results.pkl'?

        def run_summarizer(fn_):
            print('rerun %s' % fn_)
            prs = PongRNNSummarizer(filename=fn_, epoch_n=epoch_n)
            # prs.run_all()
            prs.run_performance_summary()
            return

        print('Recomputing results summary')
        search_str = '%s/*/*%s' % (self.results_path, search_suffix)
        fns = glob(search_str)
        fns = self.prune_model_list(fns)
        for fn in fns:
            fn2 = fn.replace('.pkl', '_summary.pkl')
            if (os.path.isfile(fn2) is False) or force_recompute:
                run_summarizer(fn)
            else:
                df_summary = pd.read_pickle(fn2)
                if 'error_f_rmse' not in df_summary.keys():
                    run_summarizer(fn)

        return

    def load_all_results_summary(self):
        print('Loading results summary')
        fns = glob(self.results_path + '*/*_summary.pkl')
        # rishir -- make this final_rnn_results_summary.pkl ?
        x = []
        for fn in fns:
            # print(fn)
            df = pd.read_pickle(fn)
            x.append(df)
        self.summary_df_all = pd.concat(x).reset_index(drop=True)
        return

    def prep_figout_dir(self, query_str_):
        query_str_format = query_str_.replace(' ', '') \
            .replace('&', '').replace('==', '').replace('"', '')
        query_str_format = query_str_format.replace('<', '_lt_') \
            .replace('>', '_gt_')
        figoutpath_ = '%s/%s' % (self.fig_out_path, query_str_format)
        if not os.path.exists(figoutpath_):
            os.makedirs(figoutpath_)
        return figoutpath_, query_str_format

    # compare best performing models of each type
    def get_best_model_prediction(self, query_str='random_seed==0',
                                  query_col_name='loss_weight_type',
                                  err_min='error_f_rmse',
                                  plot_detailed_summaries=True):

        df_all = self.summary_df_all.query(query_str)
        figoutpath, query_str_formatted = self.prep_figout_dir(query_str)

        def keep_only_important_entries(prs_result_dict, prs_model_characteristics, prs_error):
            dataset_specific_info_keys = ['meta_valid', 'output_vis', 'output_vis-sim', 'output_sim', 'output_f']
            model_specific_info_keys = ['pred_output_f']

            dataset_info, model_info = {}, {}
            for fk in dataset_specific_info_keys:
                dataset_info[fk] = prs_result_dict[fk]
            for fk in model_specific_info_keys:
                model_info[fk] = prs_result_dict[fk]
            model_info.update(prs_model_characteristics)
            model_info.update(prs_error)

            return dataset_info, model_info

        def get_best_model_per_type_base(df_curr, model_rank='min'):
            groupvars = ['filename_tag', 'n_epochs']
            err_mu = df_curr.groupby(groupvars).mean()[err_min]
            if model_rank == 'min':
                idx = err_mu.idxmin()
            elif model_rank == 'max':
                idx = err_mu.idxmax()
            else:
                idx = err_mu.idxmax()

            tmp = df_curr.query('filename_tag == "%s"' % idx[0]) \
                .query('n_epochs == %d' % idx[1])
            fn = np.unique(tmp['filename'])
            prs = PongRNNSummarizer(filename=fn[0],
                                    plot_detailed_summaries=plot_detailed_summaries,
                                    fig_out_path=figoutpath)

            prs.get_results_summary_base()
            prs.get_output_error_comparisons()
            prs.get_model_characterization()
            specs = prs.specs
            dataset_info, model_info = keep_only_important_entries(prs.results_output,
                                                                   prs.model_characteristics,
                                                                   prs.err_output)
            return dataset_info, model_info, specs

        def get_best_model_per_type(df_all_):
            specs_full = []
            for lw in np.unique(df_all_[query_col_name]):
                query_within_str = '%s == "%s"' % (query_col_name, lw)
                df_curr = df_all_.query(query_within_str)
                for model_rank in ['max']:  # ['min', 'max']
                    dataset_info, model_info, specs = get_best_model_per_type_base(df_curr, model_rank=model_rank)
                    specs_full.append(specs)
                    self.best_model_res['%s_%s' % (lw, model_rank)] = model_info
                    if 'dataset_info' not in self.best_model_res.keys():
                        self.best_model_res['dataset_info'] = dataset_info
            self.best_model_res['specs'] = pd.DataFrame(specs_full)

            return

        get_best_model_per_type(df_all)
        best_model_res_fn = '%s/perc_model_res_%s.pkl' \
                            % (self.fig_out_path, query_str_formatted)
        with open(best_model_res_fn, 'wb') as f:
            f.write(pk.dumps(self.best_model_res))
        print('dat: best model res')
        return

    def update_best_model_prediction(self, query_str='random_seed==0', funcs_to_update=None):
        """ RISHI: fill this code.
        load precomputed best performing model summary, and update it with
        extra model characterizations. """
        figoutpath, query_str_formatted = self.prep_figout_dir(query_str)

        def run_one_model_characterizer(fn_, func_to_run=None):
            if func_to_run is None:
                return {}
            else:
                prs = PongRNNSummarizer(filename=fn_,
                                        plot_detailed_summaries=False,
                                        fig_out_path=figoutpath)

                prs.get_results_summary_base()
                prs.get_model_characterization(characteristics_oi=func_to_run)
                return prs.model_characteristics

        best_model_res_fn = '%s/perc_model_res_%s.pkl' \
                            % (self.fig_out_path, query_str_formatted)
        best_model_res = pk.load(open(best_model_res_fn, 'rb'))
        m_specs = best_model_res['specs']
        all_models = [fk for fk in best_model_res.keys() if (fk != 'specs') and (fk != 'dataset_info')]
        all_models = self.prune_model_list(all_models)

        for m in all_models:
            m_n = "_".join(m.split('_')[:-1])  # remove "max" suffix
            tmp = m_specs[m_specs['filename_tag'] == m_n]['filename'].iloc[0]
            model_char = run_one_model_characterizer(tmp, func_to_run=funcs_to_update)
            best_model_res[m].update(model_char)

        # save to best_model_res_fn
        with open(best_model_res_fn, 'wb') as f:
            f.write(pk.dumps(best_model_res))
        print('dat updated: best model res')
        return

    def run_all(self, recompute=False, query_str='random_seed<50',
                query_col_name='filename_tag'):

        self.recompute_results_summary(force_recompute=recompute)
        self.load_all_results_summary()
        self.get_best_model_prediction(query_str=query_str,
                                       query_col_name=query_col_name,
                                       plot_detailed_summaries=False)
        return

    def update_with_results(self, query_str='random_seed==0', funcs_to_update=None):
        if funcs_to_update is not None:
            self.update_best_model_prediction(query_str=query_str, funcs_to_update=funcs_to_update)
        return


def main(argv):
    flags_, _ = parser.parse_known_args(argv)
    flags = vars(flags_)
    update_only = flags.pop('update_only')

    summarizer = ExpResultsSummary(**flags)
    for query_str in ['random_seed<50']:
        # model_characteristics are automatically recomputed for the
        # performance-based selected models. Don't need to recompute unless
        # performance values or models are changed. thus recompute=False
        if update_only:
            summarizer.update_with_results(query_str=query_str,
                                           funcs_to_update=default_func_update)
        else:
            summarizer.run_all(recompute=False,
                               query_col_name='filename_tag',
                               query_str=query_str)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
