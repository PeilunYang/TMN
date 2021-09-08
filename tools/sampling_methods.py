import random
import numpy as np
import config
import cPickle

from tools import config

np.random.seed(2021)


def random_sampling(train_seq_len, index):
    sampling_index_list = random.sample(range(train_seq_len), config.sampling_num)
    return sampling_index_list


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def kdtree_sampling(kdtree, sim_trajs, index):
    kk = 10
    top_kk = 11
    dist, ind = kdtree.query(sim_trajs[index:index+1], k=top_kk)
    #ind = ind.tolist()
    #print(ind)
    # i_near = random.choice(ind)
    i_near_ind = []
    while len(set(i_near_ind)) != kk:
        i_near_ind = list(np.random.randint(1, top_kk, kk))
    i_near = []
    for ii in range(len(i_near_ind)):
        i_near.append(ind[0][i_near_ind[ii]])
    # i_near = ind[0][i_near_ind]

    training_samples_num = config.datalength - config.test_num
    # i_far_ind = []
    # while ((len(set(i_far_ind)) != kk) and (index not in i_far_ind)):
    #     i_far_ind = list(np.random.randint(0, training_samples_num, kk))
    # neg_dist, neg_ind = kdtree.query(sim_trajs[index:index+1], k=training_samples_num)
    flag = True
    while flag:
        # i_far = random.choice(ind)
        i_far_ind = []
        # while len(set(i_far_ind)) != kk:
        #     i_far_ind = list(np.random.randint(1, 51, kk))
        while ((len(set(i_far_ind)) != kk) and (index not in i_far_ind)):
            i_far_ind = list(np.random.randint(0, training_samples_num, kk))
        i_far = []
        for ii in range(len(i_far_ind)):
            #i_far.append(ind[0][i_far_ind[ii]])
            i_far.append(i_far_ind[ii])
        # print(index, i_near, i_far)
        if len(list(set(i_near) & set(i_far))):
            continue
        else:
            flag = False
    return i_near + i_far


def distance_sampling(distance, train_seq_len, index):
    index_dis = distance[index]
    pre_sort = [np.exp(-i*config.mail_pre_degree) for i in index_dis[:train_seq_len]]
    sample_index = []
    t = 0
    importance = []
    for i in pre_sort/np.sum(pre_sort):
        importance.append(t)
        t+=i
    importance = np.array(importance)
    '''count = 0
    for i,j in enumerate(importance):
        if i<1799:
            if((importance[i+1] - importance[i]) > 0.0003):
                count += 1
    print(count)'''
    while len(sample_index)<config.sampling_num:
        a = np.random.uniform()
        idx = np.where(importance>a)[0]
        if len(idx)==0:
            # if((1.0 - importance[train_seq_len-1]) > 0.0003):
            sample_index.append(train_seq_len-1)
        elif ((idx[0]-1) not in sample_index) & (not ((idx[0]-1) == index)):
            # if((importance[idx[0]] - importance[idx[0]-1]) > 0.0003):
            sample_index.append(idx[0]-1)
    sorted_sample_index = []
    for i in sample_index:
        sorted_sample_index.append((i, pre_sort[i]))
    sorted_sample_index = sorted(sorted_sample_index, key=lambda a: a[1], reverse=True)
    return [i[0] for i in sorted_sample_index]

def negative_distance_sampling(distance, train_seq_len, index):
    index_dis = distance[index]
    pre_sort = [np.exp(-i * config.mail_pre_degree) for i in index_dis[:train_seq_len]]
    pre_sort = np.ones_like(np.array(pre_sort)) - pre_sort
    # print [(i,j) for i,j in enumerate(pre_sort)]
    sample_index = []
    t = 0
    importance = []
    for i in pre_sort / np.sum(pre_sort):
        importance.append(t)
        t += i
    importance = np.array(importance)
    # print importance
    while len(sample_index) < config.sampling_num:
        a = np.random.uniform()
        idx = np.where(importance > a)[0]
        if len(idx) == 0:
            if (1.0 - importance[train_seq_len - 1]) > 0.0005:
                sample_index.append(train_seq_len - 1)
        elif ((idx[0] - 1) not in sample_index) & (not ((idx[0] - 1) == index)):
            if (importance[idx[0]] - importance[idx[0] - 1]) > 0.0005:
                sample_index.append(idx[0] - 1)
    sorted_sample_index = []
    for i in sample_index:
        sorted_sample_index.append((i, pre_sort[i]))
    sorted_sample_index = sorted(sorted_sample_index, key=lambda a: a[1], reverse=True)
    return [i[0] for i in sorted_sample_index]


def top_n_sampling(distance, train_seq_len, index, p=False):
    index_dis = distance[index]
    topK = 50

    pre_sort = [(i,j) for i,j in enumerate(index_dis[:train_seq_len])]
    # print(pre_sort)
    post_sort = sorted(pre_sort, key=lambda k: k[1], reverse=p)
    if not p:
        # sample_index = [e[0] for e in post_sort[1:(config.sampling_num+1)]]
        # sample_index = [e[0] for e in post_sort[1:(topK + 1)]]
        sample_index = [e for e in post_sort[1:(topK + 1)]]
    else:
        # sample_index = [e[0] for e in post_sort[:config.sampling_num]]
        sample_index = [e[0] for e in post_sort[:topK]]

    selected_sample_index = random.sample(sample_index, config.sampling_num)
    # random.shuffle(selected_sample_index)
    sel_sample = sorted(selected_sample_index, key=lambda k: k[1], reverse=p)
    sel_sample_index = [e[0] for e in sel_sample]
    return sel_sample_index


if __name__ == '__main__':
    # distance = cPickle.load(open('../features/toy_discret_frechet_distance_all_1800', 'r'))
    distance = cPickle.load(open('../features/porto_hausdorff_distance_all_10000', 'r'))
    # print distance_sampling(distance, 100, 10)
    # print distance_sampling(distance, 100, 10)
    #
    # for i in range(2000):
    #     print negative_distance_sampling(distance, 2000, i)
    #     print negative_distance_sampling(distance, 2000, i)

    for i in range(10):
        print top_n_sampling(distance, 100, 10, False)
        print top_n_sampling(distance, 100, 10, True)
