from traj_rnns.traj_trainer import TrajTrainer
from tools import config
import os

if __name__ == '__main__':
    print 'os.environ["CUDA_VISIBLE_DEVICES"]= {}'.format(os.environ["CUDA_VISIBLE_DEVICES"])
    print config.config_to_str()
    trajrnn = TrajTrainer(tagset_size=config.d, batch_size=config.batch_size,
                             sampling_num=config.sampling_num)
    trajrnn.data_prepare(griddatapath=config.gridxypath, coordatapath=config.corrdatapath,
                         distancepath=config.distancepath, train_radio=config.seeds_radio)
    # load_model_name = 'model/dtw/porto_dtw_LSTM_training_10000_940_incellTrue_config_True_0.005_20_10_0.2_porto_True_128_train.h5'
    if config.loadModel == False:
        load_model_name = None
    else:
        load_model_name = './model/' + config.distance_type + '/matchingFT/{}_{}_{}_training'\
                                          .format(config.data_type, config.distance_type, config.method_name) +\
                                          '_config_{}_{}_False.pt'.format(config.stard_unit, config.learning_rate)
    if config.method_name == "matching":
        print('Method name: ' + config.method_name)
        trajrnn.matching_train(save_model=config.saveModel, load_model=load_model_name, in_cell_update=config.incell,
                              stard_LSTM=config.stard_unit)
    elif config.method_name == "neutraj" or config.method_name == "t3s" or config.method_name == "srn":
        print('Method name: ' + config.method_name)
        trajrnn.neutraj_train(save_model=False, load_model=load_model_name, in_cell_update=config.incell,
                               stard_LSTM=config.stard_unit)
    elif config.method_name == "t2s":
        print('Method name: ' + config.method_name)
        trajrnn.t2s_train(save_model=False, load_model=load_model_name, in_cell_update=config.incell,
                              stard_LSTM=config.stard_unit)
