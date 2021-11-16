from tools import preprocess
from tools.distance_compution import trajectory_distance_combine, trajecotry_distance_list
import cPickle
import numpy as np
import time


def distance_comp(coor_path):
    traj_coord = cPickle.load(open(coor_path, 'r'))[0]
    np_traj_coord = []
    for t in traj_coord:
        np_traj_coord.append(np.array(t))
    print np_traj_coord[0]
    print np_traj_coord[1]
    print len(np_traj_coord)

    distance_type = 'dtw'

    start_t = time.time()
    trajecotry_distance_list(np_traj_coord[:1000], batch_size=1000, processors=1, distance_type=distance_type,
                             data_name=data_name)
    end_t = time.time()
    total = end_t - start_t
    print('Computation time is {}'.format(total))

    # trajectory_distance_combine(9000, batch_size=200, metric_type=distance_type, data_name=data_name)
    # trajectory_distance_combain(200000, batch_size=200, metric_type=distance_type, data_name=data_name)


if __name__ == '__main__':
    # coor_path, data_name = preprocess.trajectory_feature_generation(path= './data/porto_trajs_all')
    # coor_path, data_name = preprocess.trajectory_feature_generation(path= './data/geolife_trajs')
    # distance_comp('./features/geolife_traj_coord')
    data_name = 'porto'
    distance_comp('./features/porto_traj_coord')
