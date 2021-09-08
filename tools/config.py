# Data path
# corrdatapath = './features/geolife_traj_coord'
# gridxypath = './features/geolife_traj_grid'
# distancepath = './features/geolife_discret_frechet_distance_all_9200'
# distancepath = './features/geolife_erp_39_115_distance_all_9000'
corrdatapath = './features/porto_traj_coord'
gridxypath = './features/porto_traj_grid'
distancepath = './features/porto_dtw_distance_all_10000'

# t2s = False
trmModel = False
qerror = False
saveModel = False
loadModel = False
tripleLoss = False
tripleWeight = 0.99
triEpoch = -1
no_matching = False

kdSampling = False
evaEpoch = 0

# Training Parameters
method_name = "matching"  # "neutraj" or "matching" or "t2s" or "t3s" or "srn"
GPU = "0"  # "0"
if method_name == "t2s":
    learning_rate = 0.001
else:
    learning_rate = 0.005 #0.005
seeds_radio = 0.2  # default:0.2
epochs = 10000
batch_size = 20  # 20
if method_name == 'matching' or method_name == 't2s':
    sampling_num = 20
if method_name == 'neutraj' or method_name == 't3s' or method_name == 'srn':
    sampling_num = 10
# sampling_num = 20  # neutraj:10 match:20

distance_type = distancepath.split('/')[2].split('_')[1]
data_type = distancepath.split('/')[2].split('_')[0]

# if distance_type == 'dtw' or distance_type == 'erp':
if distance_type == 'dtw' or distance_type == 'erp':
    mail_pre_degree = 16
else:
    if distance_type == 'lcss' or distance_type == 'edr':
        mail_pre_degree = 8
    else:
        mail_pre_degree = 8

# Test Config
# datalength = 1800
if data_type == 'porto':
    datalength = 10000  # geolife:9000 porto:10000
    em_batch = 1000
if data_type == 'geolife':
    datalength = 9000
    em_batch = 900
# em_batch = 1000  # geolife:900 porto:1000
test_num = 1000 #int(datalength - seeds_radio * datalength)   # geolife:7200 porto:8000

# Model Parameters
d = 128
if method_name == 'neutraj':
    stard_unit = False  # It controls the type of recurrent unit (standard cells or SAM argumented cells)
else:
    stard_unit = True
incell = True
recurrent_unit = 'LSTM'  # GRU, LSTM or SimpleRNN
spatial_width = 2

gird_size = [1100, 1100]


def config_to_str():
    configs = 'learning_rate = {} '.format(learning_rate) + '\n' + \
              'mail_pre_degree = {} '.format(mail_pre_degree) + '\n' + \
              'training_ratio = {} '.format(seeds_radio) + '\n' + \
              'embedding_size = {}'.format(d) + '\n' + \
              'epochs = {} '.format(epochs) + '\n' + \
              'datapath = {} '.format(corrdatapath) + '\n' + \
              'datatype = {} '.format(data_type) + '\n' + \
              'corrdatapath = {} '.format(corrdatapath) + '\n' + \
              'distancepath = {} '.format(distancepath) + '\n' + \
              'distance_type = {}'.format(distance_type) + '\n' + \
              'recurrent_unit = {}'.format(recurrent_unit) + '\n' + \
              'batch_size = {} '.format(batch_size) + '\n' + \
              'sampling_num = {} '.format(sampling_num) + '\n' + \
              'incell = {}'.format(incell) + '\n' + \
              'stard_unit = {}'.format(stard_unit) + '\n' + \
              'qerror = {}'.format(qerror) + '\n' + \
              'tripleLoss = {}'.format(tripleLoss) + '\n' + \
              'tripleWeight = {}'.format(tripleWeight) + '\n' + \
              'noMatching = {}'.format(no_matching) + '\n' + \
              'kdSampling = {}'.format(kdSampling)
    return configs


if __name__ == '__main__':
    print '../model/model_training_600_{}_acc_{}'.format((0),1)
    print config_to_str()
