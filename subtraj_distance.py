import cPickle
import numpy as np
import traj_dist.distance as tdist


coord_path='./features/porto_traj_coord'
traj_coord=cPickle.load(open(coord_path,'r'))[0]
np_traj_coord=[]
for t in traj_coord:
    np_traj_coord.append(np.array(t))


all_training_subtraj_similarity = []
for i in range(0, 2000):
    training_subtraj_similarity = []
    traj_len = len(np_traj_coord[i])
    for j in range(0, 2000):
        j_traj_len = len(np_traj_coord[j])
        subtraj_similarity = np.zeros((15, 15))
        for p in range(15):
            if p < ((traj_len/10) + 1) :
                for q in range(15):
                    if q < ((j_traj_len/10) + 1):
                        tmp = tdist.dtw(np_traj_coord[i][0: min((p+1)*10, traj_len)], np_traj_coord[j][0: min((q+1)*10, j_traj_len)])
                        # tmp = tdist.edr(np_traj_coord[i][0: min((p+1)*10, traj_len)], np_traj_coord[j][0: min((q+1)*10, j_traj_len)], eps=0.005) * max([min((p+1)*10, traj_len), min((q+1)*10, j_traj_len)])
                        # tmp = tdist.erp(np_traj_coord[i][0: min((p+1)*10, traj_len)], np_traj_coord[j][0: min((q+1)*10, j_traj_len)], g=np.array([39.0,115.0]))
                        # tmp = (1.0 - tdist.lcss(np_traj_coord[i][0: min((p+1)*10, traj_len)], np_traj_coord[j][0: min((q+1)*10, j_traj_len)], eps=0.005)) * min([min((p+1)*10, traj_len), min((q+1)*10, j_traj_len)])
                        # tmp = min((p+1)*10, traj_len) + min((q+1)*10, j_traj_len) - 2.0 * tmp
                        subtraj_similarity[p][q] = tmp
        training_subtraj_similarity.append(subtraj_similarity)
    all_training_subtraj_similarity.append(training_subtraj_similarity)
    print('trajectory ' + str(i) + ' finished subtraj similarity computation.')
    if (i+1) % 200 == 0:
        #cPickle.dump(np.array(all_training_subtraj_similarity[((i+1)/200 - 1)*200: ((i+1)/200)*200]), open('./features/subtraj_distance/training_' + str((i+1)/200 - 1) +'_dtw_distance','w'))
        #cPickle.dump(np.array(all_training_subtraj_similarity[0: 200]), open('./features/subtraj_distance/training_' + str((i+1)/200 - 1) +'_dtw_distance','w'))
        # np.save(open('./features/geolife_subtraj_distance/training_' + str((i+1)/200 - 1) + '_erp_39_115_distance','w'), np.array(all_training_subtraj_similarity[0: 200]))
        np.save(open('./features/subtraj_distance/train_' + str((i+1)/200 - 1) + '_dtw_distance.npy', 'w'), np.array(all_training_subtraj_similarity[0: 200]))
        all_training_subtraj_similarity = []
