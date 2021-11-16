import tools.test_methods as tm
import time, os, cPickle
import numpy as np
import torch
import random

from tools import config
from tools import sampling_methods as sm
from traj_model import Traj_Network
from wrloss import WeightedRankingLoss
from tqdm import tqdm
from sklearn.neighbors import KDTree

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU


def pad_sequence(traj_grids, maxlen=100, pad_value=0.0):
    paddec_seqs = []
    for traj in traj_grids:
        pad_r = np.zeros_like(traj[0])*pad_value
        while (len(traj) < maxlen):
            traj.append(pad_r)
        paddec_seqs.append(traj)
    return paddec_seqs


def shuffle_sequence(origin):
    ori = range(origin)
    random.shuffle(ori)
    return ori


class TrajTrainer(object):
    def __init__(self, tagset_size,
                 batch_size, sampling_num, learning_rate = config.learning_rate):

        self.target_size = tagset_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.learning_rate = learning_rate

    def build_kdtree(self, train_seqs):
        k = 5
        T = 5
        self.sim_trajs = []
        for idx, seq in enumerate(train_seqs):
            seq_len = len(seq)
            subseq_len = seq_len / T
            sub_counter = seq_len - subseq_len * T
            sim_traj = []
            x_set = []
            y_set = []
            for coord in seq:
                x_set.append(coord[0])
                y_set.append(coord[1])
            assert seq_len == len(x_set)
            l_slice = 0
            if sub_counter > 0:
                u_slice = subseq_len + 1
                sub_counter -= 1
            else:
                u_slice = subseq_len
            for i in range(T):
                x_mean = np.mean(x_set[l_slice: min(seq_len, u_slice)])
                sim_traj.append(x_mean)
                y_mean = np.mean(y_set[l_slice: min(seq_len, u_slice)])
                sim_traj.append(y_mean)
                l_slice = u_slice
                if sub_counter > 0:
                    u_slice += subseq_len + 1
                    sub_counter -= 1
                else:
                    u_slice += subseq_len
                if (u_slice-l_slice) not in [subseq_len, subseq_len+1]:
                    print(idx, seq_len)
            self.sim_trajs.append(sim_traj)
        self.sim_trajs = np.array(self.sim_trajs)
        print("sim_trajs shape:")
        print(self.sim_trajs.shape)
        self.kdtree = KDTree(self.sim_trajs, leaf_size=40)
        print("kdtree built!")

    def data_prepare(self, griddatapath=config.gridxypath,
                     coordatapath=config.corrdatapath,
                     distancepath=config.distancepath,
                     train_radio=config.seeds_radio):
        dataset_length = config.datalength
        traj_grids, useful_grids, max_len = cPickle.load(open(griddatapath, 'r'))
        self.trajs_length = [len(j) for j in traj_grids][:dataset_length]
        self.grid_size = config.gird_size
        self.max_length = max_len

        grid_trajs = [[[i[0]+config.spatial_width, i[1]+config.spatial_width] for i in tg]
                      for tg in traj_grids[:dataset_length]]

        traj_grids, useful_grids, max_len = cPickle.load(open(coordatapath, 'r'))
        x, y = [], []
        for traj in traj_grids:
            for r in traj:
                x.append(r[0])
                y.append(r[1])
        meanx, meany, stdx, stdy = np.mean(x), np.mean(y), np.std(x), np.std(y)
        traj_grids = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy] for r in t] for t in traj_grids]

        coor_trajs = traj_grids[:dataset_length]
        train_size = int(len(grid_trajs)*train_radio/self.batch_size)*self.batch_size
        self.train_size = train_size
        print train_size

        grid_train_seqs, grid_test_seqs = grid_trajs[:train_size], grid_trajs[train_size:]
        coor_train_seqs, coor_test_seqs = coor_trajs[:train_size], coor_trajs[train_size:]

        self.grid_trajs = grid_trajs
        self.grid_train_seqs = grid_train_seqs
        self.coor_trajs = coor_trajs
        self.coor_train_seqs = coor_train_seqs
        pad_trjs = []
        for i, t in enumerate(grid_trajs):
            traj = []
            for j, p in enumerate(t):
                traj.append([coor_trajs[i][j][0], coor_trajs[i][j][1], p[0], p[1]])
            pad_trjs.append(traj)

        print "Padded Trajs shape"
        print len(pad_trjs)
        self.train_seqs = pad_trjs[:train_size]
        if config.method_name == "t2s" or config.method_name == "matching":  # config.t2s:
            self.build_kdtree(self.train_seqs)
        self.padded_trajs = np.array(pad_sequence(pad_trjs, maxlen=150)) #maxlen=max_len
        distance = cPickle.load(open(distancepath, 'r'))
        max_dis = distance.max()
        # if config.distance_type == 'erp':
        #     max_dis = distance.mean()
        self.max_dist = max_dis
        print 'max value in distance matrix :{}'.format(max_dis)
        print config.distance_type
        # if (config.distance_type == 'dtw' or config.distance_type == 'edr' or config.distance_type == 'erp' or config.distance_type == 'lcss') and config.method_name != "t2s":
        if config.method_name != "t2s":
            if config.method_name == 'matching':
                distance = distance / max_dis
                print("Overall distance is divided by max_dis!!!")
            else:
                if config.distance_type == 'dtw' or config.distance_type == 'edr' or config.distance_type == 'erp' or config.distance_type == 'lcss':
                    distance = distance / max_dis
                    print("Overall distance is divided by max_dis!!!")
        print "Train data shape"
        print(np.array(self.train_seqs).shape)
        print "Distance shape"
        print distance[:train_size].shape
        train_distance = distance[:train_size, :train_size]

        print "Train Distance shape"
        print train_distance.shape
        self.distance = distance
        self.train_distance = train_distance
        if config.distance_type == 'dtw':
            self.t_alpha = np.mean(self.train_distance) + 3.0 * np.var(self.train_distance)
        else:
            self.t_alpha = 1.0 / np.max(self.train_distance)

    def t2s_batch_generator(self, train_seqs, train_distance):
        j = 0
        subtraj_dir = '/home/peyang/Data/TrajSimilarity/NeuTraj/features/subtraj_distance/'
        if config.data_type == 'geolife':
            subtraj_dir = '/home/peyang/Data/TrajSimilarity/NeuTraj/features/geolife_subtraj_distance/'
        # train_sequence = shuffle_sequence(self.train_size)
        # print(train_sequence[:10])
        pred_c = -1
        while j < len(train_seqs):
            batch_counter = j / 200
            if batch_counter != pred_c:
                # subtraj_file_name = 'train_' + str(batch_counter) + '_' + config.distance_type + '_distance.npy'
                # # subtraj_file_name = 'training_' + str(batch_counter) + '_' + config.distance_type + '_distance'
                # # subtraj_distance_tmp = np.load(subtraj_dir + subtraj_file_name, 'r')
                # subtraj_distance = np.load(subtraj_dir + subtraj_file_name, 'r')
                # # subtraj_distance = subtraj_distance_tmp / self.max_dist
                if config.distance_type == 'dtw':
                    subtraj_file_name = 'train_' + str(batch_counter) + '_' + config.distance_type + '_distance.npy'
                elif config.distance_type == 'erp':
                    subtraj_file_name = 'training_' + str(batch_counter) + '_' + config.distance_type + '_39_115_distance'
                else:
                    subtraj_file_name = 'training_' + str(batch_counter) + '_' + config.distance_type + '_distance'
                subtraj_distance_tmp = np.load(subtraj_dir + subtraj_file_name, 'r')
                subtraj_distance = subtraj_distance_tmp
                pred_c = batch_counter
                # print(subtraj_file_name)
            anchor_input, trajs_input, negative_input,distance,negative_distance = [],[],[],[],[]
            anchor_input_len, trajs_input_len, negative_input_len = [], [], []
            batch_trajs_keys = {}
            batch_trajs_input, batch_trajs_len = [], []
            subtraj_trajs_distance, subtraj_negative_distance = [], []
            #print("run")
            for i in range(self.batch_size):
                #sampling_index_list = sm.random_sampling(len(self.train_seqs),j+i)
                #negative_sampling_index_list = sm.random_sampling(len(self.train_seqs), j + i)
                all_samples_index_list = sm.top_n_sampling(self.distance, len(self.train_seqs), j + i, p=False)
                #negative_sampling_index_list = sm.top_n_sampling(self.distance, len(self.train_seqs), j + i, p=True)
                #sampling_index_list = sm.distance_sampling(self.distance, len(self.train_seqs), j + i)
                #negative_sampling_index_list = sm.negative_distance_sampling(self.distance, len(self.train_seqs), j + i)
                #all_samples_index_list = sm.distance_sampling(self.distance, len(self.train_seqs), j + i)
                sampling_index_list = all_samples_index_list[0:10]
                negative_sampling_index_list = all_samples_index_list[10:20]

                if config.kdSampling:
                    all_samples_index_list = sm.kdtree_sampling(self.kdtree, self.sim_trajs, j + i)
                    sampling_index_list = all_samples_index_list[0:10]
                    negative_sampling_index_list = all_samples_index_list[10:20]

                # trajs_input.append(train_seqs[j+i])
                # anchor_input.append(train_seqs[j + i])
                # negative_input.append(train_seqs[j + i])
                if not batch_trajs_keys.has_key(j+i):
                    batch_trajs_keys[j+i] = 0
                    batch_trajs_input.append(train_seqs[j + i])
                    batch_trajs_len.append(self.trajs_length[j + i])

                # anchor_input_len.append(self.trajs_length[j + i])
                # trajs_input_len.append(self.trajs_length[j + i])
                # negative_input_len.append(self.trajs_length[j + i])

                # distance.append(1)
                # negative_distance.append(1)
                # subtraj_trajs_distance.append(subtraj_distance[(j+i)%200][(j+i)%200])
                # subtraj_negative_distance.append(subtraj_distance[(j+i)%200][(j+i)%200])

                for traj_index in sampling_index_list:
                    anchor_input.append(train_seqs[j+i])
                    trajs_input.append(train_seqs[traj_index])

                    anchor_input_len.append(self.trajs_length[j + i])
                    trajs_input_len.append(self.trajs_length[traj_index])

                    if not batch_trajs_keys.has_key(traj_index):
                        batch_trajs_keys[j + i] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])

                    distance.append(np.exp(-float(train_distance[j+i][traj_index])*self.t_alpha))
                    subtraj_trajs_distance.append(np.exp(-(subtraj_distance[(j+i)%200][traj_index])*self.t_alpha))
                    # if j==0 and i==0:
                    #     print(np.exp(-float(train_distance[j+i][traj_index])*config.mail_pre_degree))
                    #     print(np.exp(-(subtraj_distance[(j+i)%200][traj_index])*config.mail_pre_degree))

                for traj_index in negative_sampling_index_list:
                    negative_input.append(train_seqs[traj_index])
                    negative_input_len.append(self.trajs_length[traj_index])
                    negative_distance.append(np.exp(-float(train_distance[j+i][traj_index])*self.t_alpha))
                    subtraj_negative_distance.append(np.exp(-(subtraj_distance[(j+i)%200][traj_index])*self.t_alpha))

                    if not batch_trajs_keys.has_key(traj_index):
                        batch_trajs_keys[j + i] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])
            #normlize distance
            # distance = np.array(distance)
            # distance = (distance-np.mean(distance))/np.std(distance)
            max_anchor_length = max(anchor_input_len)
            max_sample_lenght = max(trajs_input_len)
            max_neg_lenght = max(negative_input_len)
            anchor_input = pad_sequence(anchor_input, maxlen=max_anchor_length)
            trajs_input = pad_sequence(trajs_input, maxlen=max_sample_lenght)
            negative_input = pad_sequence(negative_input, maxlen=max_neg_lenght)
            batch_trajs_input = pad_sequence(batch_trajs_input, maxlen=max(max_anchor_length, max_sample_lenght,
                                                                           max_neg_lenght))

            # print(np.array(subtraj_trajs_distance).shape)
            # print(np.array(subtraj_negative_distance).shape)
            yield ([np.array(anchor_input),np.array(trajs_input),np.array(negative_input)],  #, np.array(batch_trajs_input)],
                   [anchor_input_len, trajs_input_len, negative_input_len],  #, batch_trajs_len],
                   [np.array(distance), np.array(negative_distance)],
                   [np.array(subtraj_trajs_distance), np.array(subtraj_negative_distance)])
            j = j + self.batch_size

    def batch_generator(self, train_seqs, train_distance):
        j = 0
        subtraj_dir = '/home/peyang/Data/TrajSimilarity/NeuTraj/features/subtraj_distance/'
        if config.data_type == 'geolife':
            subtraj_dir = '/home/peyang/Data/TrajSimilarity/NeuTraj/features/geolife_subtraj_distance/'
        # train_sequence = shuffle_sequence(self.train_size)
        # print(train_sequence[:10])
        pred_c = -1
        while j < len(train_seqs):
            batch_counter = j / 200
            if batch_counter != pred_c:
                if config.distance_type == 'dtw':
                    subtraj_file_name = 'train_' + str(batch_counter) + '_' + config.distance_type + '_distance.npy'
                elif config.distance_type == 'erp' and config.data_type == 'geolife':
                    subtraj_file_name = 'training_' + str(batch_counter) + '_' + config.distance_type + '_39_115_distance'
                else:
                    subtraj_file_name = 'training_' + str(batch_counter) + '_' + config.distance_type + '_distance'
                subtraj_distance_tmp = np.load(subtraj_dir + subtraj_file_name, 'r')
                if config.distance_type == 'dtw' or config.distance_type == 'edr' or config.distance_type == 'erp' or config.distance_type == 'lcss':
                    subtraj_distance = subtraj_distance_tmp / self.max_dist
                else:
                    # subtraj_distance = subtraj_distance_tmp
                    subtraj_distance = subtraj_distance_tmp / self.max_dist
                pred_c = batch_counter
                # print(subtraj_file_name)
            anchor_input, trajs_input, negative_input,distance,negative_distance = [],[],[],[],[]
            anchor_input_len, trajs_input_len, negative_input_len = [], [], []
            batch_trajs_keys = {}
            batch_trajs_input, batch_trajs_len = [], []
            subtraj_trajs_distance, subtraj_negative_distance = [], []
            #print("run")
            for i in range(self.batch_size):
                #sampling_index_list = sm.random_sampling(len(self.train_seqs),j+i)
                #negative_sampling_index_list = sm.random_sampling(len(self.train_seqs), j + i)
                # sampling_index_list = sm.top_n_sampling(self.distance, len(self.train_seqs), j + i, p=False)
                # negative_sampling_index_list = sm.top_n_sampling(self.distance, len(self.train_seqs), j + i, p=True)
                #sampling_index_list = sm.distance_sampling(self.distance, len(self.train_seqs), j + i)
                #negative_sampling_index_list = sm.negative_distance_sampling(self.distance, len(self.train_seqs), j + i)
                
                # all_samples_index_list = sm.kdtree_sampling(self.kdtree, self.sim_trajs, j + i)
                # sampling_index_list = all_samples_index_list[0:1]
                # negative_sampling_index_list = all_samples_index_list[1:2]

                if config.kdSampling:
                    all_samples_index_list = sm.kdtree_sampling(self.kdtree, self.sim_trajs, j + i)
                    sampling_index_list = all_samples_index_list[0:10]
                    negative_sampling_index_list = all_samples_index_list[10:20]
                else:
                    all_samples_index_list = sm.distance_sampling(self.distance, len(self.train_seqs), j + i)
                    sampling_index_list = all_samples_index_list[0:config.sampling_num/2]
                    negative_sampling_index_list = all_samples_index_list[config.sampling_num/2:config.sampling_num]

                # if i == 0:
                #     print(self.trajs_length[j+i])
                #     print(self.trajs_length[sampling_index_list[0]])
                #     print(subtraj_distance[(j+i)%200][sampling_index_list[0]][self.trajs_length[j+i]/10][self.trajs_length[sampling_index_list[0]]/10])
                #     print(train_distance[j+i][sampling_index_list[0]])

                trajs_input.append(train_seqs[j+i])
                anchor_input.append(train_seqs[j + i])
                negative_input.append(train_seqs[j + i])
                if not batch_trajs_keys.has_key(j+i):
                    batch_trajs_keys[j+i] = 0
                    batch_trajs_input.append(train_seqs[j + i])
                    batch_trajs_len.append(self.trajs_length[j + i])

                anchor_input_len.append(self.trajs_length[j + i])
                trajs_input_len.append(self.trajs_length[j + i])
                negative_input_len.append(self.trajs_length[j + i])

                distance.append(1)
                negative_distance.append(1)
                subtraj_trajs_distance.append(subtraj_distance[(j+i)%200][(j+i)%200])
                subtraj_negative_distance.append(subtraj_distance[(j+i)%200][(j+i)%200])

                for traj_index in sampling_index_list:
                    anchor_input.append(train_seqs[j+i])
                    trajs_input.append(train_seqs[traj_index])

                    anchor_input_len.append(self.trajs_length[j + i])
                    trajs_input_len.append(self.trajs_length[traj_index])

                    if not batch_trajs_keys.has_key(traj_index):
                        batch_trajs_keys[j + i] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])

                    distance.append(np.exp(-float(train_distance[j+i][traj_index])*config.mail_pre_degree))
                    subtraj_trajs_distance.append(np.exp(-(subtraj_distance[(j+i)%200][traj_index])*config.mail_pre_degree))
                    # if j==0 and i==0:
                    #     print(np.exp(-float(train_distance[j+i][traj_index])*config.mail_pre_degree))
                    #     print(np.exp(-(subtraj_distance[(j+i)%200][traj_index])*config.mail_pre_degree))

                for traj_index in negative_sampling_index_list:
                    negative_input.append(train_seqs[traj_index])
                    negative_input_len.append(self.trajs_length[traj_index])
                    negative_distance.append(np.exp(-float(train_distance[j+i][traj_index])*config.mail_pre_degree))
                    subtraj_negative_distance.append(np.exp(-(subtraj_distance[(j+i)%200][traj_index])*config.mail_pre_degree))

                    if not batch_trajs_keys.has_key(traj_index):
                        batch_trajs_keys[j + i] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])
            #normlize distance
            # distance = np.array(distance)
            # distance = (distance-np.mean(distance))/np.std(distance)
            max_anchor_length = max(anchor_input_len)
            max_sample_lenght = max(trajs_input_len)
            max_neg_lenght = max(negative_input_len)
            anchor_input = pad_sequence(anchor_input, maxlen=max_anchor_length)
            trajs_input = pad_sequence(trajs_input, maxlen=max_sample_lenght)
            negative_input = pad_sequence(negative_input, maxlen=max_neg_lenght)
            batch_trajs_input = pad_sequence(batch_trajs_input, maxlen=max(max_anchor_length, max_sample_lenght,
                                                                           max_neg_lenght))

            # print(np.array(subtraj_trajs_distance).shape)
            # print(np.array(subtraj_negative_distance).shape)
            yield ([np.array(anchor_input),np.array(trajs_input),np.array(negative_input)],  #, np.array(batch_trajs_input)],
                   [anchor_input_len, trajs_input_len, negative_input_len],  #, batch_trajs_len],
                   [np.array(distance), np.array(negative_distance)],
                   [np.array(subtraj_trajs_distance), np.array(subtraj_negative_distance)])
            j = j + self.batch_size

    def neutraj_batch_generator(self, train_seqs, train_distance):
        j = 0
        while j < len(train_seqs):
            anchor_input, trajs_input, negative_input, distance, negative_distance = [], [], [], [], []
            anchor_input_len, trajs_input_len, negative_input_len = [], [], []
            batch_trajs_keys = {}
            batch_trajs_input, batch_trajs_len = [], []
            for i in range(self.batch_size):
                # sampling_index_list = sm.random_sampling(len(self.train_seqs),j+i)
                if config.method_name == "srn":
                    sampling_index_list = sm.random_sampling(len(self.train_seqs),j+i)
                    negative_sampling_index_list = sm.random_sampling(len(self.train_seqs), j + i)
                if config.method_name == "neutraj" or config.method_name == "t3s":
                    sampling_index_list = sm.distance_sampling(self.distance, len(self.train_seqs), j + i)
                    negative_sampling_index_list = sm.negative_distance_sampling(self.distance, len(self.train_seqs), j + i)

                trajs_input.append(train_seqs[j + i])
                anchor_input.append(train_seqs[j + i])
                negative_input.append(train_seqs[j + i])
                if not batch_trajs_keys.has_key(j + i):
                    batch_trajs_keys[j + i] = 0
                    batch_trajs_input.append(train_seqs[j + i])
                    batch_trajs_len.append(self.trajs_length[j + i])

                anchor_input_len.append(self.trajs_length[j + i])
                trajs_input_len.append(self.trajs_length[j + i])
                negative_input_len.append(self.trajs_length[j + i])

                distance.append(1)
                negative_distance.append(1)

                for traj_index in sampling_index_list:
                    anchor_input.append(train_seqs[j + i])
                    trajs_input.append(train_seqs[traj_index])

                    anchor_input_len.append(self.trajs_length[j + i])
                    trajs_input_len.append(self.trajs_length[traj_index])

                    if not batch_trajs_keys.has_key(traj_index):
                        batch_trajs_keys[j + i] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])

                    distance.append(np.exp(-float(train_distance[j + i][traj_index]) * config.mail_pre_degree))

                for traj_index in negative_sampling_index_list:
                    negative_input.append(train_seqs[traj_index])
                    negative_input_len.append(self.trajs_length[traj_index])
                    negative_distance.append(np.exp(-float(train_distance[j + i][traj_index]) * config.mail_pre_degree))

                    if not batch_trajs_keys.has_key(traj_index):
                        batch_trajs_keys[j + i] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])
            # normlize distance
            # distance = np.array(distance)
            # distance = (distance-np.mean(distance))/np.std(distance)
            max_anchor_length = max(anchor_input_len)
            max_sample_lenght = max(trajs_input_len)
            max_neg_lenght = max(negative_input_len)
            anchor_input = pad_sequence(anchor_input, maxlen=max_anchor_length)
            trajs_input = pad_sequence(trajs_input, maxlen=max_sample_lenght)
            negative_input = pad_sequence(negative_input, maxlen=max_neg_lenght)
            batch_trajs_input = pad_sequence(batch_trajs_input, maxlen=max(max_anchor_length, max_sample_lenght,
                                                                           max_neg_lenght))

            yield (
            [np.array(anchor_input), np.array(trajs_input), np.array(negative_input), np.array(batch_trajs_input)],
            [anchor_input_len, trajs_input_len, negative_input_len, batch_trajs_len],
            [np.array(distance), np.array(negative_distance)])
            j = j + self.batch_size

    def trained_model_eval(self, print_batch=10, print_test=100, save_model=True, load_model=None,
                           in_cell_update=True, stard_LSTM=False):

        spatial_net = Traj_Network(4, self.target_size, self.grid_size,
                                      self.batch_size, self.sampling_num,
                                      stard_LSTM= stard_LSTM, incell= in_cell_update)

        if load_model != None:
            m = torch.load(open(load_model))
            spatial_net.load_state_dict(m)

            embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch= config.em_batch)
            print 'len(embeddings): {}'.format(len(embeddings))
            print embeddings.shape
            print embeddings[0].shape

            acc1 = tm.test_model(self, embeddings, test_range=range(len(self.train_seqs), len(self.train_seqs)+config.test_num),
                                 similarity=True, print_batch=print_test, r10in50=True)
            return acc1

    def neutraj_train(self, print_batch=10, print_test=3600, save_model=False, load_model=None,
                      in_cell_update=True, stard_LSTM=False):

        spatial_net = Traj_Network(4, self.target_size, self.grid_size, self.batch_size, self.sampling_num,
                                      stard_LSTM=stard_LSTM, incell=in_cell_update)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, spatial_net.parameters()), lr=config.learning_rate)
        # optimizer = torch.optim.Adam(spatial_net.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)

        mse_loss_m = WeightedRankingLoss(batch_size=self.batch_size, sampling_num=self.sampling_num)

        spatial_net.cuda()
        mse_loss_m.cuda()

        if load_model != None:
            m = torch.load(open(load_model))
            spatial_net.load_state_dict(m)
            # embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch= config.em_batch)
            # print 'len(embeddings): {}'.format(len(embeddings))
            # print embeddings.shape
            # print embeddings[0].shape
            #
            # tm.test_model(self,embeddings, test_range=range(len(self.train_seqs), len(self.train_seqs)+config.test_num),
            #                      similarity=True, print_batch=print_test, r10in50=True)

        subtraj_dir = '/home/peyang/Data/TrajSimilarity/NeuTraj/features/subtraj_distance/'
        prev10_loss = 1000.0
        best_per = [0.0, 0.0, 0.0, 0.0]
        for epoch in range(config.epochs):
            spatial_net.train()
            print "Start training Epochs : {}".format(epoch)
            total_loss = 0.0
            total_pos_loss = 0.0
            total_neg_loss = 0.0
            total_whole_loss = 0.0
            # print len(torch.nonzero(spatial_net.rnn.cell.spatial_embedding))
            start = time.time()
            for i, batch in enumerate(self.neutraj_batch_generator(self.train_seqs, self.train_distance)):
                inputs_arrays, inputs_len_arrays, target_arrays = batch[0], batch[1], batch[2]

                # trajs_loss, negative_loss, outputs_ap, outputs_p, outputs_an, outputs_n = spatial_net(inputs_arrays, inputs_len_arrays)
                trajs_loss, negative_loss = spatial_net(inputs_arrays, inputs_len_arrays)

                positive_distance_target = torch.Tensor(target_arrays[0]).view((-1, 1))  # (220, 1)
                negative_distance_target = torch.Tensor(target_arrays[1]).view((-1, 1))

                # whole_loss = mse_loss_m.f(trajs_loss, positive_distance_target, negative_loss, negative_distance_target)
                # sub_loss = mse_loss_m(outputs_ap, outputs_p, outputs_an, outputs_n, inputs_len_arrays, subtraj_target)
                # # loss = autograd.Variable(loss, requires_grad=True)
                # # loss.requires_grad = True
                # # loss += attn_loss
                # loss = sub_loss + whole_loss
                loss = mse_loss_m.f(trajs_loss, positive_distance_target, negative_loss, negative_distance_target, epoch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                optim_time = time.time()
                if not in_cell_update:
                   spatial_net.spatial_memory_update(inputs_arrays, inputs_len_arrays)
                batch_end = time.time()
                total_loss += loss.item()
                total_pos_loss += mse_loss_m.trajs_mse_loss.item()
                total_neg_loss += mse_loss_m.negative_mse_loss.item()

            end = time.time()
            print 'Epoch [{}/{}], Step [{}/{}], Epoch_Positive_Loss: {}, Epoch_Negative_Loss: {}, ' \
                  'Epoch_Total_Loss: {}, Time_cost: {}'. \
                format(epoch + 1, config.epochs, i + 1, len(self.train_seqs) // self.batch_size,
                       total_pos_loss, total_neg_loss,
                       total_loss, end - start)

            if epoch % 10 == 0 and (epoch == 0 or epoch > 0):  # and prev10_loss > total_loss:
                spatial_net.eval()
                embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch=config.em_batch)
                print 'len(embeddings): {}'.format(len(embeddings))
                print embeddings.shape
                print embeddings[0].shape

                acc1 = tm.test_model(self, embeddings, test_range=range(int(len(self.padded_trajs)*0.2),
                                                                  int(len(self.padded_trajs)*0.2)+config.test_num),
                                 similarity=True, print_batch=print_test)

                # acc1 = tm.test_matching_model(self, spatial_net, test_range=range(int(len(self.padded_trajs)*0.2),
                #                                                                   int(len(self.padded_trajs)*0.2)+config.test_num),
                #                               similarity=True, print_batch=print_test)
                # if acc1[0]+acc1[1] > best_per[0]+best_per[1]:
                #     best_per = acc1
                # print(best_per)

                prev10_loss = total_loss

            if save_model and epoch % 5 == 0:
                save_model_name = './model/' + config.distance_type + '/test/{}_{}_{}_training_{}'\
                                      .format(config.data_type, config.distance_type, config.recurrent_unit,
                                              str(epoch)) +\
                                '_config_{}_{}_{}_{}_{}_{}_{}_{}'.format(config.stard_unit, config.learning_rate,
                                                                         config.batch_size, config.sampling_num,
                                                                         config.seeds_radio, config.data_type,
                                                                         str(stard_LSTM), config.d) +\
                                '_train.h5'#'_train_{}_test_{}_{}_{}_{}_{}.h5'.format(acc1[0], acc1[1], acc1[2], acc1[3],
                                                                          #acc1[4], acc1[5])
                print save_model_name
                torch.save(spatial_net.state_dict(), save_model_name)

    def matching_train(self, print_batch=10, print_test=3600, save_model=False, load_model=None,
                      in_cell_update=True, stard_LSTM=False):

        spatial_net = Traj_Network(4, self.target_size, self.grid_size, self.batch_size, self.sampling_num,
                                      stard_LSTM=stard_LSTM, incell=in_cell_update)
        print(stard_LSTM)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, spatial_net.parameters()), lr=config.learning_rate)
        # optimizer = torch.optim.Adam(spatial_net.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)

        mse_loss_m = WeightedRankingLoss(batch_size=self.batch_size, sampling_num=self.sampling_num)

        spatial_net.cuda()
        mse_loss_m.cuda()

        if load_model != None:
            m = torch.load(open(load_model))
            spatial_net.load_state_dict(m)
            print(load_model + ' has been loaded!')
            # embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch= config.em_batch)
            # print 'len(embeddings): {}'.format(len(embeddings))
            # print embeddings.shape
            # print embeddings[0].shape
            #
            # tm.test_model(self,embeddings, test_range=range(len(self.train_seqs), len(self.train_seqs)+config.test_num),
            #                      similarity=True, print_batch=print_test, r10in50=True)

        subtraj_dir = '/home/peyang/Data/TrajSimilarity/NeuTraj/features/subtraj_distance/'
        prev10_loss = 1000.0
        best_per = [0.0, 0.0, 0.0, 0.0]
        for epoch in range(config.epochs):
            spatial_net.train()
            print "Start training Epochs : {}".format(epoch)
            total_loss = 0.0
            total_pos_loss = 0.0
            total_neg_loss = 0.0
            total_whole_loss = 0.0
            # print len(torch.nonzero(spatial_net.rnn.cell.spatial_embedding))
            start = time.time()
            for i, batch in enumerate(self.batch_generator(self.train_seqs, self.train_distance)):
                inputs_arrays, inputs_len_arrays, target_arrays, subtraj_target = batch[0], batch[1], batch[2], batch[3]

                trajs_loss, negative_loss, outputs_ap, outputs_p, outputs_an, outputs_n = spatial_net.matching_forward(inputs_arrays, inputs_len_arrays)

                positive_distance_target = torch.Tensor(target_arrays[0]).view((-1, 1))  # (220, 1)
                # print(positive_distance_target)
                negative_distance_target = torch.Tensor(target_arrays[1]).view((-1, 1))

                if not config.qerror:
                    whole_loss = mse_loss_m.f(trajs_loss, positive_distance_target, negative_loss, negative_distance_target, epoch)
                    sub_loss = mse_loss_m(outputs_ap, outputs_p, outputs_an, outputs_n, inputs_len_arrays, subtraj_target)
                else:
                    whole_loss = mse_loss_m.f(trajs_loss, positive_distance_target, negative_loss, negative_distance_target, epoch)
                    sub_loss = mse_loss_m.qerror_forward(outputs_ap, outputs_p, outputs_an, outputs_n, inputs_len_arrays, subtraj_target)
                # loss = autograd.Variable(loss, requires_grad=True)
                # loss.requires_grad = True
                # loss += attn_loss
                loss = sub_loss + whole_loss

                optimizer.zero_grad()
                if config.no_subloss:
                    whole_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                # optimizer.zero_grad()

                optim_time = time.time()
                #if not in_cell_update:
                #    spatial_net.spatial_memory_update(inputs_arrays, inputs_len_arrays)
                batch_end = time.time()
                total_loss += loss.item()
                total_pos_loss += mse_loss_m.trajs_mse_loss.item()
                total_neg_loss += mse_loss_m.negative_mse_loss.item()
                total_whole_loss += whole_loss.item()
                # if (i + 1) % print_batch == 0:
                #     print 'Epoch [{}/{}], Step [{}/{}], Positive_Loss: {}, Negative_Loss: {}, Whole_Loss: {}, ' \
                #           'Total_Loss: {}, ''Update_Time_cost: {}, All_Time_cost: {}'.\
                #            format(epoch + 1, config.epochs, i + 1, len(self.train_seqs) // self.batch_size,
                #               mse_loss_m.trajs_mse_loss.item(), mse_loss_m.negative_mse_loss.item(),
                #               whole_loss.item(), loss.item(), batch_end-optim_time, batch_end-start)

            end = time.time()
            print 'Epoch [{}/{}], Step [{}/{}], Epoch_Positive_Loss: {}, Epoch_Negative_Loss: {}, ' \
                  'Epoch_Whole_Loss: {}, Epoch_Total_Loss: {}, Time_cost: {}'. \
                format(epoch + 1, config.epochs, i + 1, len(self.train_seqs) // self.batch_size,
                       total_pos_loss, total_neg_loss, total_whole_loss,
                       total_loss, end - start)

            if epoch % 10 == 0 and (epoch == 0 or (epoch > config.evaEpoch and total_loss < 1000.0)):  # and prev10_loss > total_loss:
                spatial_net.eval()
                '''embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch= config.em_batch)
                print 'len(embeddings): {}'.format(len(embeddings))
                print embeddings.shape
                print embeddings[0].shape


                acc1 = tm.test_model(self,embeddings, test_range=range(int(len(self.padded_trajs)*0.2),
                                                                   int(len(self.padded_trajs)*0.2)+config.test_num),
                                 similarity=True, print_batch=print_test)'''
                # with torch.no_grad():
                # tm.test_matching_time(self, spatial_net, test_range=range(0,10000), #range(int(len(self.padded_trajs)*config.seeds_radio),
                #                                                                 #int(len(self.padded_trajs)*config.seeds_radio)+config.test_num),
                #                               similarity=True, print_batch=print_test)
                acc1 = tm.test_matching_model(self, spatial_net, test_range=range(int(len(self.padded_trajs)*config.seeds_radio),
                                                                                  int(len(self.padded_trajs)*config.seeds_radio)+config.test_num),
                                              similarity=True, print_batch=print_test, epochs=epoch)

                if acc1[0]+acc1[1] > best_per[0]+best_per[1]:
                    best_per = acc1
                    if save_model:
                        save_model_name = './model/' + config.distance_type + '/matchingFT/{}_{}_{}_training'\
                                          .format(config.data_type, config.distance_type, config.method_name) +\
                                          '_config_{}_{}_{}.pt'.format(config.stard_unit, config.learning_rate, config.qerror)
                        print save_model_name
                        torch.save(spatial_net.state_dict(), save_model_name)
                print(best_per)

                prev10_loss = total_loss

    def t2s_train(self, print_batch=10, print_test=3600, save_model=False, load_model=None,
                      in_cell_update=True, stard_LSTM=False):

        spatial_net = Traj_Network(4, self.target_size, self.grid_size, self.batch_size, self.sampling_num,
                                      stard_LSTM=stard_LSTM, incell=in_cell_update)
        print(stard_LSTM)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, spatial_net.parameters()), lr=config.learning_rate)
        # optimizer = torch.optim.Adam(spatial_net.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)

        mse_loss_m = WeightedRankingLoss(batch_size=self.batch_size, sampling_num=self.sampling_num)

        spatial_net.cuda()
        mse_loss_m.cuda()

        if load_model != None:
            m = torch.load(open(load_model))
            spatial_net.load_state_dict(m)
            # embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch= config.em_batch)
            # print 'len(embeddings): {}'.format(len(embeddings))
            # print embeddings.shape
            # print embeddings[0].shape
            #
            # tm.test_model(self,embeddings, test_range=range(len(self.train_seqs), len(self.train_seqs)+config.test_num),
            #                      similarity=True, print_batch=print_test, r10in50=True)

        subtraj_dir = '/home/peyang/Data/TrajSimilarity/NeuTraj/features/subtraj_distance/'
        prev10_loss = 1000.0
        best_per = [0.0, 0.0, 0.0, 0.0]
        for epoch in range(config.epochs):
            spatial_net.train()
            print "Start training Epochs : {}".format(epoch)
            total_loss = 0.0
            total_pos_loss = 0.0
            total_neg_loss = 0.0
            total_whole_loss = 0.0
            # print len(torch.nonzero(spatial_net.rnn.cell.spatial_embedding))
            start = time.time()
            for i, batch in enumerate(self.t2s_batch_generator(self.train_seqs, self.train_distance)):
                inputs_arrays, inputs_len_arrays, target_arrays, subtraj_target = batch[0], batch[1], batch[2], batch[3]

                trajs_loss, negative_loss, outputs_ap, outputs_p, outputs_an, outputs_n = spatial_net.t2s_forward(inputs_arrays, inputs_len_arrays)

                positive_distance_target = torch.Tensor(target_arrays[0]).view((-1, 1))  # (220, 1)
                # print(positive_distance_target)
                negative_distance_target = torch.Tensor(target_arrays[1]).view((-1, 1))

                whole_loss = mse_loss_m.f(trajs_loss, positive_distance_target, negative_loss, negative_distance_target, epoch)
                sub_loss = mse_loss_m.t2s_forward(outputs_ap, outputs_p, outputs_an, outputs_n, inputs_len_arrays, subtraj_target)
                # loss = autograd.Variable(loss, requires_grad=True)
                # loss.requires_grad = True
                # loss += attn_loss
                loss = sub_loss + whole_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # optimizer.zero_grad()

                optim_time = time.time()
                #if not in_cell_update:
                #    spatial_net.spatial_memory_update(inputs_arrays, inputs_len_arrays)
                batch_end = time.time()
                total_loss += loss.item()
                total_pos_loss += mse_loss_m.trajs_mse_loss.item()
                total_neg_loss += mse_loss_m.negative_mse_loss.item()
                total_whole_loss += whole_loss.item()

            end = time.time()
            print 'Epoch [{}/{}], Step [{}/{}], Epoch_Positive_Loss: {}, Epoch_Negative_Loss: {}, ' \
                  'Epoch_Whole_Loss: {}, Epoch_Total_Loss: {}, Time_cost: {}'. \
                format(epoch + 1, config.epochs, i + 1, len(self.train_seqs) // self.batch_size,
                       total_pos_loss, total_neg_loss, total_whole_loss,
                       total_loss, end - start)

            if epoch % 10 == 0 and (epoch == 0 or epoch > 250):  # and prev10_loss > total_loss:
                spatial_net.eval()
                '''embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch= config.em_batch)
                print 'len(embeddings): {}'.format(len(embeddings))
                print embeddings.shape
                print embeddings[0].shape


                acc1 = tm.test_model(self,embeddings, test_range=range(int(len(self.padded_trajs)*0.2),
                                                                   int(len(self.padded_trajs)*0.2)+config.test_num),
                                 similarity=True, print_batch=print_test)'''

                tm.test_matching_time(self, spatial_net, test_range=range(int(len(self.padded_trajs)*config.seeds_radio),
                                                                                  int(len(self.padded_trajs)*config.seeds_radio)+config.test_num),
                                              similarity=True, print_batch=print_test)

                prev10_loss = total_loss

            if save_model and (epoch % 5 == 0):
                save_model_name = './model/' + config.distance_type + '/test/{}_{}_{}_training_{}'\
                                      .format(config.data_type, config.distance_type, config.recurrent_unit,
                                              str(epoch)) +\
                                '_config_{}_{}_{}_{}_{}_{}_{}_{}'.format(config.stard_unit, config.learning_rate,
                                                                         config.batch_size, config.sampling_num,
                                                                         config.seeds_radio, config.data_type,
                                                                         str(stard_LSTM), config.d) +\
                                '_train.h5'#'_train_{}_test_{}_{}_{}_{}_{}.h5'.format(acc1[0], acc1[1], acc1[2], acc1[3],
                                                                          #acc1[4], acc1[5])
                print save_model_name
                torch.save(spatial_net.state_dict(), save_model_name)

    def trm_train(self, print_batch=10, print_test=3600, save_model=False, load_model=None,
                      in_cell_update=True, stard_LSTM=False):

        spatial_net = Traj_Network(4, self.target_size, self.grid_size, self.batch_size, self.sampling_num,
                                      stard_LSTM=stard_LSTM, incell=in_cell_update)
        print(stard_LSTM)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, spatial_net.parameters()), lr=config.learning_rate)
        # optimizer = torch.optim.Adam(spatial_net.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)

        mse_loss_m = WeightedRankingLoss(batch_size=self.batch_size, sampling_num=self.sampling_num)

        spatial_net.cuda()
        mse_loss_m.cuda()

        if load_model != None:
            m = torch.load(open(load_model))
            spatial_net.load_state_dict(m)
            # embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch= config.em_batch)
            # print 'len(embeddings): {}'.format(len(embeddings))
            # print embeddings.shape
            # print embeddings[0].shape
            #
            # tm.test_model(self,embeddings, test_range=range(len(self.train_seqs), len(self.train_seqs)+config.test_num),
            #                      similarity=True, print_batch=print_test, r10in50=True)

        subtraj_dir = '/home/peyang/Data/TrajSimilarity/NeuTraj/features/subtraj_distance/'
        prev10_loss = 1000.0
        for epoch in range(config.epochs):
            spatial_net.train()
            print "Start training Epochs : {}".format(epoch)
            total_loss = 0.0
            total_pos_loss = 0.0
            total_neg_loss = 0.0
            total_whole_loss = 0.0
            # print len(torch.nonzero(spatial_net.rnn.cell.spatial_embedding))
            start = time.time()
            for i, batch in enumerate(self.batch_generator(self.train_seqs, self.train_distance)):
                inputs_arrays, inputs_len_arrays, target_arrays, subtraj_target = batch[0], batch[1], batch[2], batch[3]

                trajs_loss, negative_loss, outputs_ap, outputs_p, outputs_an, outputs_n = spatial_net.trm_forward(inputs_arrays, inputs_len_arrays)

                positive_distance_target = torch.Tensor(target_arrays[0]).view((-1, 1))  # (220, 1)
                # print(positive_distance_target)
                negative_distance_target = torch.Tensor(target_arrays[1]).view((-1, 1))

                whole_loss = mse_loss_m.f(trajs_loss, positive_distance_target, negative_loss, negative_distance_target)
                sub_loss = mse_loss_m(outputs_ap, outputs_p, outputs_an, outputs_n, inputs_len_arrays, subtraj_target)
                # loss = autograd.Variable(loss, requires_grad=True)
                # loss.requires_grad = True
                # loss += attn_loss
                loss = sub_loss + whole_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # optimizer.zero_grad()

                optim_time = time.time()
                #if not in_cell_update:
                #    spatial_net.spatial_memory_update(inputs_arrays, inputs_len_arrays)
                batch_end = time.time()
                total_loss += loss.item()
                total_pos_loss += mse_loss_m.trajs_mse_loss.item()
                total_neg_loss += mse_loss_m.negative_mse_loss.item()
                total_whole_loss += whole_loss.item()
                if (i + 1) % print_batch == 0:
                    print 'Epoch [{}/{}], Step [{}/{}], Positive_Loss: {}, Negative_Loss: {}, Whole_Loss: {}, ' \
                          'Total_Loss: {}, ''Update_Time_cost: {}, All_Time_cost: {}'.\
                           format(epoch + 1, config.epochs, i + 1, len(self.train_seqs) // self.batch_size,
                              mse_loss_m.trajs_mse_loss.item(), mse_loss_m.negative_mse_loss.item(),
                              whole_loss.item(), loss.item(), batch_end-optim_time, batch_end-start)

            end = time.time()
            print 'Epoch [{}/{}], Step [{}/{}], Epoch_Positive_Loss: {}, Epoch_Negative_Loss: {}, ' \
                  'Epoch_Whole_Loss: {}, Epoch_Total_Loss: {}, Time_cost: {}'. \
                format(epoch + 1, config.epochs, i + 1, len(self.train_seqs) // self.batch_size,
                       total_pos_loss, total_neg_loss, total_whole_loss,
                       total_loss, end - start)

            if epoch % 10 == 0 and epoch > 220:  # and prev10_loss > total_loss:
                spatial_net.eval()
                '''embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch= config.em_batch)
                print 'len(embeddings): {}'.format(len(embeddings))
                print embeddings.shape
                print embeddings[0].shape


                acc1 = tm.test_model(self,embeddings, test_range=range(int(len(self.padded_trajs)*0.2),
                                                                   int(len(self.padded_trajs)*0.2)+config.test_num),
                                 similarity=True, print_batch=print_test)'''
                # with torch.no_grad():
                acc1 = tm.test_matching_model(self, spatial_net, test_range=range(int(len(self.padded_trajs)*0.2),
                                                                                  int(len(self.padded_trajs)*0.2)+config.test_num),
                                              similarity=True, print_batch=print_test)

                print acc1

                prev10_loss = total_loss

            if save_model and (epoch % 5 == 0):
                save_model_name = './model/' + config.distance_type + '/test/{}_{}_{}_training_{}'\
                                      .format(config.data_type, config.distance_type, config.recurrent_unit,
                                              str(epoch)) +\
                                '_config_{}_{}_{}_{}_{}_{}_{}_{}'.format(config.stard_unit, config.learning_rate,
                                                                         config.batch_size, config.sampling_num,
                                                                         config.seeds_radio, config.data_type,
                                                                         str(stard_LSTM), config.d) +\
                                '_train.h5'#'_train_{}_test_{}_{}_{}_{}_{}.h5'.format(acc1[0], acc1[1], acc1[2], acc1[3],
                                                                          #acc1[4], acc1[5])
                print save_model_name
                torch.save(spatial_net.state_dict(), save_model_name)
