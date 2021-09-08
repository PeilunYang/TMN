import cPickle
import traj_dist.distance as tdist
import numpy as np
import multiprocessing
import time


def trajectory_distance(traj_feature_map, traj_keys,  distance_type="hausdorff", batch_size=50, processors=30):
    # traj_keys= traj_feature_map.keys()
    trajs = []
    for k in traj_keys:
        traj = []
        for record in traj_feature_map[k]:
            traj.append([record[1],record[2]])
        trajs.append(np.array(traj))

    pool = multiprocessing.Pool(processes=processors)
    # print np.shape(distance)
    batch_number = 0
    for i in range(len(trajs)):
        if (i!=0) & (i%batch_size == 0):
            print (batch_size*batch_number, i)
            pool.apply_async(trajectory_distance_batch, (i, trajs[batch_size*batch_number:i], trajs, distance_type,
                                                         'geolife'))
            batch_number+=1
    pool.close()
    pool.join()


def trajecotry_distance_list(trajs, distance_type="hausdorff", batch_size=50, processors=30, data_name='porto'):
    pool = multiprocessing.Pool(processes=processors)
    print(data_name)
    # print np.shape(distance)
    start_t = time.time()
    batch_number = 0
    for i in range(len(trajs)):
        if (i != 0) & (i % batch_size == 0):
            print (batch_size*batch_number, i)
            pool.apply_async(trajectory_distance_batch, (i, trajs[batch_size*batch_number:i], trajs, distance_type,
                                                         data_name))
            batch_number += 1
    pool.close()
    pool.join()
    end_t = time.time()


def trajectory_distance_batch(i, batch_trjs, trjs, metric_type="hausdorff", data_name='porto'):
    if metric_type == 'lcss':
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps=0.005)  # eps=0.003
        tmp_matrix = 1.0 - trs_matrix
        len_a = len(batch_trjs)
        len_b = len(trjs)
        min_len_matrix = np.ones((len_a, len_b))
        sum_len_matrix = np.ones((len_a, len_b))
        for ii in range(len_a):
            for jj in range(len_b):
                min_len_matrix[ii][jj] = min(len(batch_trjs[ii]), len(trjs[jj]))
                sum_len_matrix[ii][jj] = len(batch_trjs[ii]) + len(trjs[jj])
        tmp_trs_matrix = tmp_matrix * min_len_matrix
        trs_matrix = sum_len_matrix - 2.0 * tmp_trs_matrix
    elif metric_type == 'edr':
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps=0.005)  # eps=0.003
        len_a = len(batch_trjs)
        len_b = len(trjs)
        max_len_matrix = np.ones((len_a, len_b))
        for ii in range(len_a):
            for jj in range(len_b):
                max_len_matrix[ii][jj] = max(len(batch_trjs[ii]), len(trjs[jj]))
        trs_matrix = trs_matrix * max_len_matrix
    elif metric_type == 'erp':
        aa = np.zeros(2, dtype=float)
        aa[0] = 39.0  # geolife:39.6  porto:40.7
        aa[1] = 115.0  # geolife:115.9 porto:-9.0
        print(aa)
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, g=aa)
    else:
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type)
    # cPickle.dump(trs_matrix, open('./features/BruteForce/'+data_name+'_'+metric_type+'_distance_' + str(i), 'w'))
    # cPickle.dump(trs_matrix, open('./features/'+data_name+'_'+metric_type+'_39_115_distance_' + str(i), 'wb'))
    print 'complete: ' + str(i)


def trajectory_distance_combine(trajs_len, batch_size=100, metric_type="hausdorff", data_name='porto'):
    distance_list = []
    a = 0
    for i in range(1,trajs_len+1):
        if (i != 0) & (i % batch_size == 0):
            distance_list.append(cPickle.load(open('./features/'+data_name+'_'+metric_type+'_39_115_distance_' + str(i))))
            print distance_list[-1].shape
    a = distance_list[-1].shape[1]
    distances = np.array(distance_list)
    print distances.shape
    all_dis = distances.reshape((trajs_len,a))
    print all_dis.shape
    cPickle.dump(all_dis,open('./features/'+data_name+'_'+metric_type+'_39_115_distance_all_'+str(trajs_len), 'w'))
    return all_dis
