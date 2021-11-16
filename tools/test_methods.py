import config
import numpy as np
import torch.autograd as autograd
import torch
# from traj_rnns.spatial_memory_lstm_pytorch import SpatialCoordinateRNNPytorch
import time


def test_comput_embeddings(self, spatial_net, test_batch=1025):
    if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
        hidden = autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda()
    else:
        hidden = (autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda(),
                  autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda())
    embeddings_list = []
    j = 0
    s = time.time()
    while j < 1000: #self.padded_trajs.shape[0]:
        #for i in range(self.batch_size):
        out = spatial_net.rnn([autograd.Variable(torch.Tensor(self.padded_trajs[j:j+test_batch]),
                                                     requires_grad=False).cuda(),
                                   self.trajs_length[j:j+test_batch]],hidden)
        # print(out.shape)
        # embeddings = out.data.cpu().numpy()
        embeddings = out.data
        j += test_batch
        embeddings_list.append(embeddings)

    ss = time.time()
    sq_a = out**2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)
    sum_sq_b = torch.sum(sq_a, dim=1).unsqueeze(0)
    bt = out.t()
    dp_res = sum_sq_a + sum_sq_b - 2*out.mm(bt)
    print('similairty computation time: {}'.format(time.time()-ss))
    print(dp_res.shape)
    print 'embedding time of {} trajectories: {}'.format(self.padded_trajs.shape[0], time.time()-s)
    embeddings_list = torch.cat(embeddings_list, dim=0)
    print embeddings_list.size()
    return embeddings_list.cpu().numpy()


def test_model(self, traj_embeddings, test_range, print_batch=10, similarity=False, r10in50=False):
    top_10_count, l_top_10_count = 0, 0
    top_50_count, l_top_50_count = 0, 0
    top10_in_top50_count = 0
    top10_in_top100_count = 0
    test_traj_num = 0
    range_num = test_range[-1]
    all_true_distance, all_test_distance = [], []
    error_true, error_test, errorr1050 = 0., 0., 0.
    
    start_time = time.time()
    for i in test_range:

        if similarity:
            # This is for the exp similarity
            test_distance = [(j, float(np.exp(-np.sum(np.square(traj_embeddings[i] - e)))))
                             for j, e in enumerate(traj_embeddings)]
            t_similarity = np.exp(-self.distance[i][:len(traj_embeddings)]*config.mail_pre_degree)
            true_distance = list(enumerate(t_similarity))
            learned_distance = list(enumerate(self.distance[i][:len(self.train_seqs)]))

            s_test_distance = sorted(test_distance, key=lambda a: a[1], reverse=True)
            s_true_distance = sorted(true_distance, key=lambda a: a[1], reverse=True)
            s_learned_distance = sorted(learned_distance, key=lambda a: a[1])
        else:
            # This is for computing the distance
            test_distance = [(j, float(np.sum(np.square(traj_embeddings[i] - e))))
                             for j, e in enumerate(traj_embeddings)]
            true_distance = list(enumerate(self.distance[i][:len(traj_embeddings)]))
            learned_distance = list(enumerate(self.distance[i][:len(self.train_seqs)]))

            s_test_distance = sorted(test_distance, key=lambda a: a[1])
            s_true_distance = sorted(true_distance, key=lambda a: a[1])
            s_learned_distance = sorted(learned_distance, key=lambda a: a[1])

        #for i in range(len(self.train_seqs)):
        # trained_distance = [(j, float(np.exp(-np.sum(np.square(traj_embeddings[0] - e)))))
        #                     for j, e in enumerate(traj_embeddings)]
        # tt_similarity = np.exp(-self.distance[0][:len(traj_embeddings)]*config.mail_pre_degree)
        # true_train_distance = list(enumerate(tt_similarity))
        # st_train_distance = sorted(trained_distance, key=lambda a: a[1], reverse=True)
        # st_true_distance = sorted(true_train_distance, key=lambda a:a[1], reverse=True)

        top10_recall = [l[0] for l in s_test_distance[:11] if l[0] in [j[0] for j in s_true_distance[:11]]]
        # # top10_train_recall = [l[0] for l in st_train_distance[:11] if l[0] in [j[0] for j in st_true_distance[:11]]]
        top50_recall = [l[0] for l in s_test_distance[:51] if l[0] in [j[0] for j in s_true_distance[:51]]]
        top10_in_top50 = [l[0] for l in s_test_distance[:11] if l[0] in [j[0] for j in s_true_distance[:51]]]
        # top10_in_top100 = [l[0] for l in s_test_distance[:11] if l[0] in [j[0] for j in s_true_distance[:101]]]

        top_10_count += len(top10_recall) - 1
        top_50_count += len(top50_recall) - 1
        top10_in_top50_count += len(top10_in_top50) -1
        # top10_in_top100_count += len(top10_in_top100) -1


        l_top10_recall = [l[0] for l in s_learned_distance[:11] if l[0] in [j[0] for j in s_true_distance[:11]]]
        l_top50_recall = [l[0] for l in s_learned_distance[:51] if l[0] in [j[0] for j in s_true_distance[:51]]]

        l_top_10_count += len(l_top10_recall) - 1
        l_top_50_count += len(l_top50_recall) - 1

        all_true_distance.append(s_true_distance[:50])
        all_test_distance.append(s_test_distance[:50])

        true_top_10_distance = 0.
        for ij in s_true_distance[:11]:
            true_top_10_distance += self.distance[i][ij[0]]
        test_top_10_distance = 0.
        for ij in s_test_distance[:11]:
            # print (i, ij)
            test_top_10_distance += self.distance[i][ij[0]]
        test_top_10_distance_r10in50 = 0.
        temp_distance_in_test50 = []
        for ij in s_test_distance[:51]:
            temp_distance_in_test50.append([ij,self.distance[i][ij[0]]])
        sort_dis_10in50 = sorted(temp_distance_in_test50, key= lambda x: x[1])
        test_top_10_distance_r10in50 = sum([iaj[1] for iaj in sort_dis_10in50[:11]])

        error_true += true_top_10_distance
        error_test += test_top_10_distance
        errorr1050 += test_top_10_distance_r10in50

        test_traj_num += 1
        if (i % print_batch) == 0:
            # print test_distance
            print '**----------------------------------**'
            print s_test_distance[:20]
            print s_true_distance[:20]
            print top10_recall
            print top50_recall
            #print top10_train_recall
            #print(st_train_distance[:20])
            #print(st_true_distance[:20])
            # print(traj_embeddings[1500])
            # print(traj_embeddings[98])
            # print(traj_embeddings[1666])
            # print(traj_embeddings[1734])

    epoch_test_time = time.time() - start_time
    if r10in50:
        error_test = errorr1050

    best_res_recall = 0.0
    best_res_recallin = 0.0
    best_top10, best_top50, best_top10in50, best_top10in100, best_test_time = 0.0, 0.0, 0.0, 0.0, 0.0
    print 'Test on {} trajs'.format(test_traj_num)
    print 'Search Top 50 recall {}'.format(float(top_50_count) / (test_traj_num * 50))
    print 'Search Top 10 recall {}'.format(float(top_10_count) / (test_traj_num * 10))
    print 'Search Top 10 in Top 50 recall {}'.format(float(top10_in_top50_count) / (test_traj_num * 10))
    #print 'Search Top 10 in Top 100 recall {}'.format(float(top10_in_top100_count) / (test_traj_num * 10))
    print 'Test time of {} trajectories: {}'.format(test_traj_num, epoch_test_time)

    res_list = []
    res_list.append(float(top_50_count) / (test_traj_num * 50))
    res_list.append(float(top_10_count) / (test_traj_num * 10))
    res_list.append(float(top10_in_top50_count) / (test_traj_num * 10))
    res_list.append(float(top10_in_top100_count) / (test_traj_num * 10))

    return res_list

    epoch_res_recall = float(top_50_count) / (test_traj_num * 50) + float(top_10_count) / (test_traj_num * 10)
    epoch_res_recallin = float(top10_in_top50_count) / (test_traj_num * 10) + float(top10_in_top100_count) / (
                test_traj_num * 10)
    if epoch_res_recall > best_res_recall:
        best_res_recall = epoch_res_recall
        best_top10 = float(top_10_count) / (test_traj_num * 10)
        best_top50 = float(top_50_count) / (test_traj_num * 50)
        best_top10in50 = float(top10_in_top50_count) / (test_traj_num * 10)
        best_top10in100 = float(top10_in_top100_count) / (test_traj_num * 10)
        best_test_time = epoch_test_time

    print 'Current best performance: '
    print 'Test on {} trajs'.format(test_traj_num)
    print 'Search Top 50 recall {}'.format(best_top50)
    print 'Search Top 10 recall {}'.format(best_top10)
    print 'Search Top 10 in Top 50 recall {}'.format(best_top10in50)
    print 'Search Top 10 in Top 100 recall {}'.format(best_top10in100)
    print 'Test time of {} trajectories: {}'.format(test_traj_num, best_test_time)

    '''print 'Error true:{}'.format((float(error_true) / (test_traj_num * 10))*84000)
    print 'Error test:{}'.format((float(error_test) / (test_traj_num * 10))*84000)
    print 'Error div :{}'.format((float(abs(error_test-error_true)) / (test_traj_num * 10))*84000)
    return (float(top_10_count) / (test_traj_num * 10), \
           float(top_50_count) / (test_traj_num * 50),\
           float(top10_in_top50_count) / (test_traj_num * 10), \
           (float(error_true) / (test_traj_num * 10)) * 84000, \
           (float(error_test) / (test_traj_num * 10)) * 84000, \
           (float(abs(error_test - error_true)) / (test_traj_num * 10)) * 84000)'''


def test_matching_model(self, spatial_net, test_range, print_batch=10, similarity=False, r10in50=False, epochs=100):
    top_10_count, l_top_10_count = 0, 0
    top_50_count, l_top_50_count = 0, 0
    top10_in_top50_count = 0
    top10_in_top100_count = 0
    test_traj_num = 0
    all_true_distance, all_test_distance = [], []
    error_true, error_test, errorr1050 = 0., 0., 0.
    all_attn_ptn = []
    
    start_time = time.time()
    for i in test_range:

        if similarity:
            # This is for the exp similarity
            '''hidden = (autograd.Variable(torch.zeros(self.padded_trajs.shape[0], self.target_size), requires_grad=False),
                    autograd.Variable(torch.zeros(self.padded_trajs.shape[0], self.target_size), requires_grad=False))
            a_input_length = []
            for j in range(self.padded_trajs.shape[0]):
                a_input_length.append(self.trajs_length[i])
            #print(self.padded_trajs.shape)
            a_embedding, p_embedding = spatial_net.smn([autograd.Variable(torch.Tensor(self.padded_trajs[i]).unsqueeze(0).repeat(self.padded_trajs.shape[0], 1, 1), requires_grad=False), a_input_length], 
                                                       [autograd.Variable(torch.Tensor(self.padded_trajs), requires_grad=False), self.trajs_length],
                                                       hidden, hidden)
            a_embeddings, p_embeddings = a_embedding.data.numpy(), p_embedding.data.numpy()'''
            a_embeddings_list, p_embeddings_list = [], []
            # p_attns_list = []
            # hidden1 = (autograd.Variable(torch.zeros(500, self.target_size), requires_grad=False).cuda(),
            #         autograd.Variable(torch.zeros(500, self.target_size), requires_grad=False).cuda())
            # hidden2 = (autograd.Variable(torch.zeros(500, self.target_size), requires_grad=False).cuda(),
            #         autograd.Variable(torch.zeros(500, self.target_size), requires_grad=False).cuda())
            j = 0
            test_batch_size = 500
            while j < self.padded_trajs.shape[0]:
                a_input_length = []
                for z in range(test_batch_size):
                    a_input_length.append(self.trajs_length[i])
                if config.method_name == "matching":
                    if j == 0 and i == test_range[0]:
                        print("Test method: " + config.method_name)
                    a_out, p_out, _, _ = spatial_net.smn.f([autograd.Variable(torch.Tensor(self.padded_trajs[i]).unsqueeze(0).repeat(test_batch_size, 1, 1), requires_grad=False).cuda(), a_input_length],
                                                    [autograd.Variable(torch.Tensor(self.padded_trajs[j:j+test_batch_size]), requires_grad=False).cuda(), self.trajs_length[j:j+test_batch_size]])
                                                    # hidden1, hidden2, None, None)
                elif config.method_name == "t2s":
                    a_out, p_out, _, _ = spatial_net.smn.t2s_forward([autograd.Variable(
                        torch.Tensor(self.padded_trajs[i]).unsqueeze(0).repeat(test_batch_size, 1, 1),
                        requires_grad=False).cuda(), a_input_length],
                                                           [autograd.Variable(
                                                               torch.Tensor(self.padded_trajs[j:j + test_batch_size]),
                                                               requires_grad=False).cuda(),
                                                            self.trajs_length[j:j + test_batch_size]])
                # embeddings = out.data.cpu().numpy()
                a_embedding, p_embedding = a_out.data, p_out.data
                # p_attn_aa = p_attn_a.data
                j += test_batch_size
                a_embeddings_list.append(a_embedding)
                p_embeddings_list.append(p_embedding)
                # p_attns_list.append(p_attn_aa)
            a_embeddings_list = torch.cat(a_embeddings_list, dim=0)
            p_embeddings_list = torch.cat(p_embeddings_list, dim=0)
            # p_attns_list = torch.cat(p_attns_list, dim=0)
            # p_attns = p_attns_list.cpu().numpy()
            # np.save('p_attn_a_matrix.npy', p_attns)
            a_embeddings, p_embeddings = a_embeddings_list.cpu().numpy(), p_embeddings_list.cpu().numpy()
            #return embeddings_list.cpu().numpy()
            #test_distance = [(j, float(np.exp(-np.sum(np.square(traj_embeddings[i] - e)))))
            #                 for j, e in enumerate(traj_embeddings)]
            test_distance = [(j, float(np.exp(-np.sum(np.square(a_embeddings[j] - p_embeddings[j])))))
                             for j, e in enumerate(a_embeddings)]
            #t_similarity = np.exp(-self.distance[i][:len(traj_embeddings)]*config.mail_pre_degree)
            t_similarity = np.exp(-self.distance[i][:len(a_embeddings)]*config.mail_pre_degree)
            true_distance = list(enumerate(t_similarity))
            # learned_distance = list(enumerate(self.distance[i][:len(self.train_seqs)]))

            s_test_distance = sorted(test_distance, key=lambda a: a[1], reverse=True)
            s_true_distance = sorted(true_distance, key=lambda a: a[1], reverse=True)
            # s_learned_distance = sorted(learned_distance, key=lambda a: a[1])
        else:
            # This is for computing the distance
            test_distance = [(j, float(np.sum(np.square(traj_embeddings[i] - e))))
                             for j, e in enumerate(traj_embeddings)]
            true_distance = list(enumerate(self.distance[i][:len(traj_embeddings)]))
            # learned_distance = list(enumerate(self.distance[i][:len(self.train_seqs)]))

            s_test_distance = sorted(test_distance, key=lambda a: a[1])
            s_true_distance = sorted(true_distance, key=lambda a: a[1])
            # s_learned_distance = sorted(learned_distance, key=lambda a: a[1])

        #for i in range(len(self.train_seqs)):
        # trained_distance = [(j, float(np.exp(-np.sum(np.square(traj_embeddings[0] - e)))))
        #                     for j, e in enumerate(traj_embeddings)]
        # tt_similarity = np.exp(-self.distance[0][:len(traj_embeddings)]*config.mail_pre_degree)
        # true_train_distance = list(enumerate(tt_similarity))
        # st_train_distance = sorted(trained_distance, key=lambda a: a[1], reverse=True)
        # st_true_distance = sorted(true_train_distance, key=lambda a:a[1], reverse=True)
        # all_attn_ptn.append(p_attns)
        # np.save('p_attn_a_matrix_{}.npy'.format(epochs), np.array(all_attn_ptn))

        top10_recall = [l[0] for l in s_test_distance[:11] if l[0] in [j[0] for j in s_true_distance[:11]]]
        # top10_train_recall = [l[0] for l in st_train_distance[:11] if l[0] in [j[0] for j in st_true_distance[:11]]]
        top50_recall = [l[0] for l in s_test_distance[:51] if l[0] in [j[0] for j in s_true_distance[:51]]]
        top10_in_top50 = [l[0] for l in s_test_distance[:11] if l[0] in [j[0] for j in s_true_distance[:51]]]
        top10_in_top100 = [l[0] for l in s_test_distance[:11] if l[0] in [j[0] for j in s_true_distance[:101]]]

        top_10_count += len(top10_recall) - 1
        top_50_count += len(top50_recall) - 1
        top10_in_top50_count += len(top10_in_top50) -1
        top10_in_top100_count += len(top10_in_top100) -1

        test_traj_num += 1
        if (i % print_batch) == 0:
            # print test_distance
            print '**----------------------------------**'
            print s_test_distance[:20]
            print s_true_distance[:20]
            print top10_recall
            print top50_recall
            #print top10_train_recall
            #print(st_train_distance[:20])
            #print(st_true_distance[:20])
            #print(traj_embeddings[1500])
            #print(traj_embeddings[98])
            #print(traj_embeddings[1666])
            #print(traj_embeddings[1734])

    epoch_test_time = time.time()-start_time
    if r10in50:
        error_test = errorr1050

    best_res_recall = 0.0
    best_res_recallin = 0.0
    best_top10, best_top50, best_top10in50, best_top10in100, best_test_time = 0.0, 0.0, 0.0, 0.0, 0.0
    print 'Test on {} trajs'.format(test_traj_num)
    print 'Search Top 50 recall {}'.format(float(top_50_count) / (test_traj_num * 50))
    print 'Search Top 10 recall {}'.format(float(top_10_count) / (test_traj_num * 10))
    print 'Search Top 10 in Top 50 recall {}'.format(float(top10_in_top50_count) / (test_traj_num * 10))
    print 'Search Top 10 in Top 100 recall {}'.format(float(top10_in_top100_count) / (test_traj_num * 10))
    print 'Test time of {} trajectories: {}'.format(test_traj_num, epoch_test_time)

    res_list = []
    res_list.append(float(top_50_count) / (test_traj_num * 50))
    res_list.append(float(top_10_count) / (test_traj_num * 10))
    res_list.append(float(top10_in_top50_count) / (test_traj_num * 10))
    res_list.append(float(top10_in_top100_count) / (test_traj_num * 10))

    return res_list

    epoch_res_recall = float(top_50_count) / (test_traj_num * 50) + float(top_10_count) / (test_traj_num * 10)
    epoch_res_recallin = float(top10_in_top50_count) / (test_traj_num * 10) + float(top10_in_top100_count) / (test_traj_num * 10)
    if epoch_res_recall > best_res_recall:
        best_res_recall = epoch_res_recall
        best_top10 = float(top_10_count) / (test_traj_num * 10)
        best_top50 = float(top_50_count) / (test_traj_num * 50)
        best_top10in50 = float(top10_in_top50_count) / (test_traj_num * 10)
        best_top10in100 = float(top10_in_top100_count) / (test_traj_num * 10)
        best_test_time = epoch_test_time

    print 'Current best performance: '
    print 'Test on {} trajs'.format(test_traj_num)
    print 'Search Top 50 recall {}'.format(best_top50)
    print 'Search Top 10 recall {}'.format(best_top10)
    print 'Search Top 10 in Top 50 recall {}'.format(best_top10in50)
    print 'Search Top 10 in Top 100 recall {}'.format(best_top10in100)
    print 'Test time of {} trajectories: {}'.format(test_traj_num, best_test_time)


def test_matching_time(self, spatial_net, test_range, print_batch=10, similarity=False, r10in50=False):
    top_10_count, l_top_10_count = 0, 0
    top_50_count, l_top_50_count = 0, 0
    top10_in_top50_count = 0
    top10_in_top100_count = 0
    test_traj_num = 0
    all_true_distance, all_test_distance = [], []
    error_true, error_test, errorr1050 = 0., 0., 0.

    start_time = time.time()
    for i in test_range:
        #if similarity:
            # This is for the exp similarity
        '''hidden = (autograd.Variable(torch.zeros(self.padded_trajs.shape[0], self.target_size), requires_grad=False),
                    autograd.Variable(torch.zeros(self.padded_trajs.shape[0], self.target_size), requires_grad=False))
            a_input_length = []
            for j in range(self.padded_trajs.shape[0]):
                a_input_length.append(self.trajs_length[i])
            #print(self.padded_trajs.shape)
            a_embedding, p_embedding = spatial_net.smn([autograd.Variable(torch.Tensor(self.padded_trajs[i]).unsqueeze(0).repeat(self.padded_trajs.shape[0], 1, 1), requires_grad=False), a_input_length], 
                                                       [autograd.Variable(torch.Tensor(self.padded_trajs), requires_grad=False), self.trajs_length],
                                                       hidden, hidden)
            a_embeddings, p_embeddings = a_embedding.data.numpy(), p_embedding.data.numpy()'''
        a_embeddings_list, p_embeddings_list = [], []
        # hidden1 = (autograd.Variable(torch.zeros(500, self.target_size), requires_grad=False).cuda(),
        #         autograd.Variable(torch.zeros(500, self.target_size), requires_grad=False).cuda())
        # hidden2 = (autograd.Variable(torch.zeros(500, self.target_size), requires_grad=False).cuda(),
        #         autograd.Variable(torch.zeros(500, self.target_size), requires_grad=False).cuda())
        j = 0
        test_batch_size = 1000
        while j < self.padded_trajs.shape[0]:
            a_input_length = []
            for z in range(test_batch_size):
                a_input_length.append(self.trajs_length[i])
            if config.method_name == "matching":
                a_out, p_out, _, _ = spatial_net.smn.f([autograd.Variable(torch.Tensor(self.padded_trajs[i]).unsqueeze(0).repeat(test_batch_size, 1, 1), requires_grad=False).cuda(), a_input_length],
                                                [autograd.Variable(torch.Tensor(self.padded_trajs[j:j+test_batch_size]), requires_grad=False).cuda(), self.trajs_length[j:j+test_batch_size]])
                                                # hidden1, hidden2, None, None)
            elif config.method_name == "t2s":
                a_out, p_out, _, _ = spatial_net.smn.t2s_forward([autograd.Variable(
                    torch.Tensor(self.padded_trajs[i]).unsqueeze(0).repeat(test_batch_size, 1, 1),
                    requires_grad=False).cuda(), a_input_length],
                                                        [autograd.Variable(
                                                            torch.Tensor(self.padded_trajs[j:j + test_batch_size]),
                                                            requires_grad=False).cuda(),
                                                        self.trajs_length[j:j + test_batch_size]])
        # while j < self.padded_trajs.shape[0]:
        #     a_input_length = []
        #     for z in range(test_batch_size):
        #         a_input_length.append(self.trajs_length[i])
        #     a_out, p_out, _, _ = spatial_net.smn.f([autograd.Variable(
        #         torch.Tensor(self.padded_trajs[i]).unsqueeze(0).repeat(test_batch_size, 1, 1),
        #         requires_grad=False).cuda(), a_input_length],
        #                                             [autograd.Variable(
        #                                                 torch.Tensor(self.padded_trajs[j:j + test_batch_size]),
        #                                                 requires_grad=False).cuda(),
        #                                             self.trajs_length[j:j + test_batch_size]])
            # hidden1, hidden2, None, None)
            # embeddings = out.data.cpu().numpy()
            #a_embedding, p_embedding = a_out.data, p_out.data
            j += test_batch_size
            #a_embeddings_list.append(a_embedding)
            #p_embeddings_list.append(p_embedding)
        #a_embeddings_list = torch.cat(a_embeddings_list, dim=0)
        #p_embeddings_list = torch.cat(p_embeddings_list, dim=0)
    # a_embeddings, p_embeddings = a_embeddings_list.cpu().numpy(), p_embeddings_list.cpu().numpy()
    '''
            return embeddings_list.cpu().numpy()
            # test_distance = [(j, float(np.exp(-np.sum(np.square(traj_embeddings[i] - e)))))
            #                 for j, e in enumerate(traj_embeddings)]
            test_distance = [(j, float(np.exp(-np.sum(np.square(a_embeddings[j] - p_embeddings[j])))))
                             for j, e in enumerate(a_embeddings)]
            # t_similarity = np.exp(-self.distance[i][:len(traj_embeddings)]*config.mail_pre_degree)
            t_similarity = np.exp(-self.distance[i][:len(a_embeddings)] * config.mail_pre_degree)
            true_distance = list(enumerate(t_similarity))
            # learned_distance = list(enumerate(self.distance[i][:len(self.train_seqs)]))

            s_test_distance = sorted(test_distance, key=lambda a: a[1], reverse=True)
            s_true_distance = sorted(true_distance, key=lambda a: a[1], reverse=True)
            # s_learned_distance = sorted(learned_distance, key=lambda a: a[1])
        else:
            # This is for computing the distance
            test_distance = [(j, float(np.sum(np.square(traj_embeddings[i] - e))))
                             for j, e in enumerate(traj_embeddings)]
            true_distance = list(enumerate(self.distance[i][:len(traj_embeddings)]))
            # learned_distance = list(enumerate(self.distance[i][:len(self.train_seqs)]))

            s_test_distance = sorted(test_distance, key=lambda a: a[1])
            s_true_distance = sorted(true_distance, key=lambda a: a[1])
            # s_learned_distance = sorted(learned_distance, key=lambda a: a[1])

        # for i in range(len(self.train_seqs)):
        # trained_distance = [(j, float(np.exp(-np.sum(np.square(traj_embeddings[0] - e)))))
        #                     for j, e in enumerate(traj_embeddings)]
        # tt_similarity = np.exp(-self.distance[0][:len(traj_embeddings)]*config.mail_pre_degree)
        # true_train_distance = list(enumerate(tt_similarity))
        # st_train_distance = sorted(trained_distance, key=lambda a: a[1], reverse=True)
        # st_true_distance = sorted(true_train_distance, key=lambda a:a[1], reverse=True)

        # top10_recall = [l[0] for l in s_test_distance[:11] if l[0] in [j[0] for j in s_true_distance[:11]]]
        # #top10_train_recall = [l[0] for l in st_train_distance[:11] if l[0] in [j[0] for j in st_true_distance[:11]]]
        # top50_recall = [l[0] for l in s_test_distance[:51] if l[0] in [j[0] for j in s_true_distance[:51]]]
        # top10_in_top50 = [l[0] for l in s_test_distance[:11] if l[0] in [j[0] for j in s_true_distance[:51]]]
        # top10_in_top100 = [l[0] for l in s_test_distance[:11] if l[0] in [j[0] for j in s_true_distance[:101]]]

        # top_10_count += len(top10_recall) - 1
        top_10_count += len(set(s_test_distance[:11]) & set(s_true_distance[:11])) - 1
        # top_50_count += len(top50_recall) - 1
        top_10_count += len(set(s_test_distance[:51]) & set(s_true_distance[:51])) - 1
        # top10_in_top50_count += len(top10_in_top50) - 1
        # top10_in_top100_count += len(top10_in_top100) - 1

        test_traj_num += 1
        if (i % print_batch) == 0:
            # print test_distance
            print '**----------------------------------**'
            print s_test_distance[:20]
            print s_true_distance[:20]
            # print top10_recall
            # print top50_recall
            # print top10_train_recall
            # print(st_train_distance[:20])
            # print(st_true_distance[:20])
            # print(traj_embeddings[1500])
            # print(traj_embeddings[98])
            # print(traj_embeddings[1666])
            # print(traj_embeddings[1734])
        # if (i % 100) == 0:
        #     print 'Test time of {} trajectories: {}'.format(100, time.time() - start_time)'''

    '''if r10in50:
        error_test = errorr1050

    print 'Test on {} trajs'.format(test_traj_num)
    print 'Search Top 50 recall {}'.format(float(top_50_count) / (test_traj_num * 50))
    print 'Search Top 10 recall {}'.format(float(top_10_count) / (test_traj_num * 10))
    # print 'Search Top 10 in Top 50 recall {}'.format(float(top10_in_top50_count) / (test_traj_num * 10))
    # print 'Search Top 10 in Top 100 recall {}'.format(float(top10_in_top100_count) / (test_traj_num * 10))
    print 'Test time of {} trajectories: {}'.format(test_traj_num, time.time() - start_time)'''
    print('Embedding time for 10000 trajectories: {}'.format(time.time() - start_time))


if __name__ == '__main__':
    print config.config_to_str()
