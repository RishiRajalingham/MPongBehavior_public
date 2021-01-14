
import sys
PATH_ = '/om/user/rishir/lib/MentalPong/behavior/'
sys.path.insert(0, PATH_)
import BehavioralCharacterizer as BC

model_dat_fn = '/om/user/rishir/lib/PongRnn/fig/rnn_res_speedenrich/perc_model_res_random_seed_lt_50.pkl'
hum_dat_fn = '/om/user/rishir/data/behavior/human_pong_basic.pkl'
monkey_dat_fn = '/om/user/rishir/data/behavior/monkey_CP_pong_basic.pkl'
save_path = '/om/user/rishir/lib/MentalPong/dat_speedenrich/'
comparer = BC.BehavioralComparer( model_dat_fn=model_dat_fn,
                                  human_dat_fn=hum_dat_fn,
                                  monkey_dat_fn=monkey_dat_fn,
                                  save_path=save_path,
                                  pred_models_only=True,
                                  prefix='')
comparer.run_all(recompute_consistency=True)
