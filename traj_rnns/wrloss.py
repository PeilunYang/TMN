import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn import Module, Parameter
import torch
from tools import config
import numpy as np


class WeightMSELoss(Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightMSELoss, self).__init__()
        self.weight = []
        for i in range(batch_size):
            self.weight.append(0.)
            for traj_index in range(10):#sampling_num):
                if config.method_name == "srn":
                    self.weight.append(np.array([1]))
                else:
                    self.weight.append(np.array([config.sampling_num - traj_index]))
                

        self.weight = np.array(self.weight)
        sum = np.sum(self.weight)
        self.weight = self.weight / sum
        self.weight = self.weight.astype(float)
        self.weight = Parameter(torch.Tensor(self.weight).cuda(), requires_grad=False)
        #self.weight = Parameter(torch.Tensor(self.weight), requires_grad = False)
        self.batch_size = batch_size
        self.sampling_num = sampling_num

    def forward(self, inputs, targets, isReLU=False, isSub=False):
        if not isSub:
            div = targets - inputs.view(-1, 1)
            if isReLU:
                div = F.relu(div.view(-1, 1))
            square = torch.mul(div.view(-1, 1), div.view(-1, 1))
            weight_square = torch.mul(square.view(-1, 1), self.weight.view(-1, 1))

            loss = torch.sum(weight_square)
            #loss = torch.sum(square)
            return loss

        else:
            div = targets - inputs.view

    def triple_forward(self, pos_inputs, neg_inputs, pos_targets, neg_targets):
        wweight = []
        for i in range(20):
            wweight.append(0.)
            for traj_index in range(10):
                wweight.append(1.)
        wweight = np.array(wweight)
        sum = np.sum(wweight)
        wweight = wweight / sum
        wweight = wweight.astype(float)
        wweight = Parameter(torch.Tensor(wweight).cuda(), requires_grad=False)
        inputs_div = pos_inputs.view(-1, 1) - neg_inputs.view(-1, 1)
        targets_div = pos_targets - neg_targets
        # zero_div = torch.zeros_like(targets_div)
        # div = torch.max(zero_div.cuda(), targets_div.cuda() - inputs_div.cuda())
        div = F.relu(targets_div.cuda() - inputs_div.cuda())
        weighted_div = torch.mul(div.view(-1, 1), wweight.view(-1, 1))

        loss = torch.sum(weighted_div)
        return loss

    def qerror_forward(self, inputs, targets, isNeg):
        if isNeg:
            qerror = []
            for i in range(len(inputs)):
                if i % 11 == 0: # (i == 0 or (i+1) % 11 == 0) and i != 219 :
                    continue
                else:
                    if inputs[i] > targets[i]:
                        tar_tmp = max(targets[i], 1e-4)
                        qerror.append(torch.Tensor([inputs[i] / tar_tmp]))
                    else:
                        # qerror.append(torch.ones(1).cuda())
                        qerror.append(torch.Tensor([inputs[i] / inputs[i]]))
            # print(len(qerror))
            result = torch.sum(torch.cat(qerror)) / 200.0 # 1e-6 20000.0
        else:
            qerror = []
            for i in range(len(inputs)):
                if i % 11 == 0: # i == 0 or (i+1) % 11 == 0:
                    continue
                else:
                    if inputs[i] > targets[i]:
                        tar_tmp = max(targets[i], 1e-4)
                        qerror.append(inputs[i] / tar_tmp)
                    else:
                        inp_tmp = max(inputs[i], 1e-4)
                        qerror.append(targets[i] / inp_tmp)
            result = torch.sum(torch.cat(qerror)) / 200.0 # 1e-6 20000.0
        return result

    def t2s_forward(self, inputs, targets, isReLU=False, isSub=False):
        div = targets - inputs.view(-1, 1)
        square = torch.mul(div.view(-1, 1), div.view(-1, 1))
        weight_square = torch.mul(square.view(-1, 1), targets)

        loss = torch.sum(weight_square) / targets.size(0)
        # loss = torch.sum(square)
        return loss


class WeightedRankingLoss(Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightedRankingLoss, self).__init__()
        self.positive_loss = WeightMSELoss(batch_size, sampling_num)
        self.negative_loss = WeightMSELoss(batch_size, sampling_num)
        self.weight = []
        for i in range(batch_size):
            self.weight.append(0.)
            for traj_index in range(10):  # sampling_num):
                if config.qerror:
                    self.weight.append(np.array([1]))
                else:
                    self.weight.append(np.array([config.sampling_num - traj_index]))
                # self.weight.append(np.array([1]))

        self.weight = np.array(self.weight)
        sum = np.sum(self.weight)
        self.weight = self.weight / sum
        self.weight = self.weight.astype(float)
        self.weight = Parameter(torch.Tensor(self.weight).cuda(), requires_grad=False)

    def f(self, p_input, p_target, n_input, n_target, epoch):
        if config.method_name == 't2s':
            trajs_mse_loss = self.positive_loss.t2s_forward(p_input, autograd.Variable(p_target).cuda(), False, False)
        else:
            if config.qerror:
                trajs_mse_loss = self.positive_loss.qerror_forward(p_input, autograd.Variable(p_target).cuda(), False)
            else:
                trajs_mse_loss = self.positive_loss(p_input, autograd.Variable(p_target).cuda(), False, False)
        #trajs_mse_loss = self.positive_loss(p_input, autograd.Variable(p_target), False)

        if config.method_name == 't2s':
            negative_mse_loss = self.negative_loss.t2s_forward(n_input, autograd.Variable(n_target).cuda(), True, False)
        else:
            if config.qerror:
                negative_mse_loss = self.negative_loss.qerror_forward(n_input, autograd.Variable(n_target).cuda(), True)
            else:
                negative_mse_loss = self.negative_loss(n_input, autograd.Variable(n_target).cuda(), True, False)
        #negative_mse_loss = self.negative_loss(n_input, autograd.Variable(n_target), True)

        self.trajs_mse_loss = trajs_mse_loss
        self.negative_mse_loss = negative_mse_loss
        loss = sum([trajs_mse_loss, negative_mse_loss])
        if config.tripleLoss:
            if epoch > config.triEpoch: 
                triLoss = self.positive_loss.triple_forward(p_input, n_input, p_target, n_target)
                loss = config.tripleWeight * triLoss + (1.0-config.tripleWeight) * loss
        return loss

    def forward(self, a_emb_p, p_emb, a_emb_n, n_emb, inputs_len_arrays, subtraj_distance):
        # a_emb_p: 149x220x128 inputs_len_arrays
        # anchor_input = torch.LongTensor(inputs_len_arrays[0]).view((-1, 1))  # (220, )
        # trajs_input = torch.LongTensor(inputs_len_arrays[1]).view((-1, 1)) # (220, )
        # negative_input = torch.LongTensor(inputs_len_arrays[2]).view((-1, 1))  # (220, )
        anchor_input = inputs_len_arrays[0]  # (220, )
        trajs_input = inputs_len_arrays[1]  # (220, )
        negative_input = inputs_len_arrays[2]  # (220, )
        subtraj_trajs_distance = torch.Tensor(subtraj_distance[0])  # (220, 15, 15)
        subtraj_neg_distance = torch.Tensor(subtraj_distance[1])  # (220, 15, 15)
        batch_loss = 0.
        batch_num_counter = 0
        for i in range(a_emb_p.size()[1]):
            a_input_len = anchor_input[i]
            p_input_len = trajs_input[i]
            n_input_len = negative_input[i]
            num_counter = 0
            tmp_batch_loss = 0.
            '''
            for j in range(0, (a_input_len/10)+1):  # (a_input_len/10)
                ap_subtraj = a_emb_p.permute(1, 0, 2)[i][min((j+1)*10 - 1, a_input_len-1)].unsqueeze(0)
                an_subtraj = a_emb_n.permute(1, 0, 2)[i][min((j+1)*10 - 1, a_input_len-1)].unsqueeze(0)
                p_subtraj = p_emb.permute(1, 0, 2)[i][p_input_len-1].unsqueeze(0)
                n_subtraj = n_emb.permute(1, 0, 2)[i][n_input_len-1].unsqueeze(0)
                ap_pred = torch.exp(-F.pairwise_distance(ap_subtraj, p_subtraj, p=2))
                an_pred = torch.exp(-F.pairwise_distance(an_subtraj, n_subtraj, p=2))
                #ap_target = autograd.Variable(subtraj_trajs_distance[i][j][(p_input_len/10)]).cuda()
                ap_target = subtraj_trajs_distance[i][j][p_input_len/10]
                #an_target = autograd.Variable(subtraj_neg_distance[i][j][n_input_len/10]).cuda()
                an_target = subtraj_neg_distance[i][j][n_input_len/10]
                ap_loss = torch.mul((ap_target - ap_pred), (ap_target - ap_pred))
                ap_loss = ap_loss * self.weight[i]
                an_loss = torch.mul(F.relu(an_target - an_pred), F.relu(an_target - an_pred))
                # an_loss = torch.mul((an_target - an_pred), (an_target - an_pred))
                an_loss = an_loss * self.weight[i]
                # loss = torch.sum(torch.Tensor([ap_loss, an_loss]))
                loss = ap_loss + an_loss
                # loss = sum([ap_loss, an_loss])
                tmp_batch_loss += loss
                num_counter += 1
                batch_num_counter += 1
            batch_loss += tmp_batch_loss / num_counter
            '''
            
            # for j in range(0, (a_input_len/10)+1):  # (a_input_len/10)
            #     loss = 0.
            #     for p in range(0, (p_input_len/10)+1):
            #         ap_subtraj = a_emb_p.permute(1, 0, 2)[i][min((j+1)*10 - 1, a_input_len-1)].unsqueeze(0)
            #         p_subtraj = p_emb.permute(1, 0, 2)[i][min((p+1)*10 - 1, p_input_len-1)].unsqueeze(0)
            #         ap_pred = torch.exp(-F.pairwise_distance(ap_subtraj, p_subtraj, p=2))
            #         ap_target = subtraj_trajs_distance[i][j][p]
            #         ap_loss = torch.mul((ap_target - ap_pred), (ap_target - ap_pred))
            #         ap_loss = ap_loss * self.weight[i]
            #         loss += ap_loss
            #         num_counter += 1
            #     for q in range(0, (n_input_len/10)+1):
            #         an_subtraj = a_emb_n.permute(1, 0, 2)[i][min((j+1)*10 - 1, a_input_len-1)].unsqueeze(0)
            #         n_subtraj = n_emb.permute(1, 0, 2)[i][min((q+1)*10 - 1, n_input_len-1)].unsqueeze(0)
            #         an_pred = torch.exp(-F.pairwise_distance(an_subtraj, n_subtraj, p=2))
            #         an_target = subtraj_neg_distance[i][j][q]
            #         an_loss = torch.mul(F.relu(an_target - an_pred), F.relu(an_target - an_pred))
            #         an_loss = an_loss * self.weight[i]
            #         loss += an_loss
            #         num_counter += 1
            #     # loss = ap_loss + an_loss
            #     tmp_batch_loss += loss
            #     # num_counter += 1
            # batch_num_counter += 1
            # batch_loss += tmp_batch_loss / num_counter

            ap_min_length = min(a_input_len, p_input_len)
            loss = 0.
            for lp in range(0, ap_min_length/10):
                ap_subtraj = a_emb_p.permute(1, 0, 2)[i][min((lp + 1) * 10 - 1, a_input_len - 1)].unsqueeze(0)
                p_subtraj = p_emb.permute(1, 0, 2)[i][min((lp + 1) * 10 - 1, p_input_len - 1)].unsqueeze(0)
                ap_pred = torch.exp(-F.pairwise_distance(ap_subtraj, p_subtraj, p=2))
                ap_target = subtraj_trajs_distance[i][lp][lp]
                ap_loss = torch.mul((ap_target - ap_pred), (ap_target - ap_pred))
                ap_loss = ap_loss * self.weight[i]
                loss += ap_loss
                num_counter += 1

            an_min_length = min(a_input_len, n_input_len)
            for ln in range(0, an_min_length/10):
                an_subtraj = a_emb_n.permute(1, 0, 2)[i][min((ln + 1) * 10 - 1, a_input_len - 1)].unsqueeze(0)
                n_subtraj = n_emb.permute(1, 0, 2)[i][min((ln + 1) * 10 - 1, n_input_len - 1)].unsqueeze(0)
                an_pred = torch.exp(-F.pairwise_distance(an_subtraj, n_subtraj, p=2))
                an_target = subtraj_neg_distance[i][ln][ln]
                an_loss = torch.mul(F.relu(an_target - an_pred), F.relu(an_target - an_pred))
                an_loss = an_loss * self.weight[i]
                loss += an_loss
                num_counter += 1

            batch_num_counter += 1
            batch_loss += loss / num_counter

        return batch_loss

    def qerror_forward(self, a_emb_p, p_emb, a_emb_n, n_emb, inputs_len_arrays, subtraj_distance):
        # a_emb_p: 149x220x128 inputs_len_arrays
        # anchor_input = torch.LongTensor(inputs_len_arrays[0]).view((-1, 1))  # (220, )
        # trajs_input = torch.LongTensor(inputs_len_arrays[1]).view((-1, 1)) # (220, )
        # negative_input = torch.LongTensor(inputs_len_arrays[2]).view((-1, 1))  # (220, )
        anchor_input = inputs_len_arrays[0]  # (220, )
        trajs_input = inputs_len_arrays[1]  # (220, )
        negative_input = inputs_len_arrays[2]  # (220, )
        subtraj_trajs_distance = torch.Tensor(subtraj_distance[0])  # (220, 15, 15)
        subtraj_neg_distance = torch.Tensor(subtraj_distance[1])  # (220, 15, 15)
        batch_loss = 0.
        batch_num_counter = 0
        total_num_counter = 0
        for i in range(a_emb_p.size()[1]):
            if i % 11 == 0:
                continue
            a_input_len = anchor_input[i]
            p_input_len = trajs_input[i]
            n_input_len = negative_input[i]
            num_counter = 0
            tmp_batch_loss = 0.
            ap_min_length = min(a_input_len, p_input_len)
            loss = 0.
            for lp in range(0, ap_min_length / 10):
                ap_subtraj = a_emb_p.permute(1, 0, 2)[i][min((lp + 1) * 10 - 1, a_input_len - 1)].unsqueeze(0)
                p_subtraj = p_emb.permute(1, 0, 2)[i][min((lp + 1) * 10 - 1, p_input_len - 1)].unsqueeze(0)
                ap_pred = torch.exp(-F.pairwise_distance(ap_subtraj, p_subtraj, p=2))
                ap_target = subtraj_trajs_distance[i][lp][lp]
                if ap_pred > ap_target:
                    ap_tar_tmp = max(ap_target, 1e-4)
                    ap_loss = ap_pred / ap_tar_tmp
                else:
                    ap_pred_tmp = max(ap_pred, 1e-4)
                    ap_loss = ap_target / ap_pred_tmp
                # ap_loss = torch.mul((ap_target - ap_pred), (ap_target - ap_pred))
                # ap_loss = ap_loss * self.weight[i]
                loss += ap_loss
                num_counter += 1
                total_num_counter += 1

            an_min_length = min(a_input_len, n_input_len)
            for ln in range(0, an_min_length / 10):
                an_subtraj = a_emb_n.permute(1, 0, 2)[i][min((ln + 1) * 10 - 1, a_input_len - 1)].unsqueeze(0)
                n_subtraj = n_emb.permute(1, 0, 2)[i][min((ln + 1) * 10 - 1, n_input_len - 1)].unsqueeze(0)
                an_pred = torch.exp(-F.pairwise_distance(an_subtraj, n_subtraj, p=2))
                an_target = subtraj_neg_distance[i][ln][ln]
                if an_pred > an_target:
                    an_tar_tmp = max(an_target, 1e-4)
                    an_loss = an_pred / an_tar_tmp
                else:
                    # an_pred_tmp = max(an_pred, 1e-4)
                    # an_loss = an_target / an_pred_tmp
                    an_loss = an_pred / an_pred # torch.ones(1).cuda()
                # an_loss = torch.mul(F.relu(an_target - an_pred), F.relu(an_target - an_pred))
                # an_loss = an_loss * self.weight[i]
                loss += an_loss
                num_counter += 1
                total_num_counter += 1

            batch_num_counter += 1
            # batch_loss += loss / num_counter
            batch_loss += loss # / 100.0

        # return batch_loss / batch_num_counter
        return batch_loss / total_num_counter

    def t2s_forward(self, a_emb_p, p_emb, a_emb_n, n_emb, inputs_len_arrays, subtraj_distance):
        # a_emb_p: 149x220x128 inputs_len_arrays
        # anchor_input = torch.LongTensor(inputs_len_arrays[0]).view((-1, 1))  # (220, )
        # trajs_input = torch.LongTensor(inputs_len_arrays[1]).view((-1, 1)) # (220, )
        # negative_input = torch.LongTensor(inputs_len_arrays[2]).view((-1, 1))  # (220, )
        anchor_input = inputs_len_arrays[0]  # (220, )
        trajs_input = inputs_len_arrays[1]  # (220, )
        negative_input = inputs_len_arrays[2]  # (220, )
        subtraj_trajs_distance = torch.Tensor(subtraj_distance[0])  # (220, 15, 15)
        subtraj_neg_distance = torch.Tensor(subtraj_distance[1])  # (220, 15, 15)
        batch_loss = 0.
        batch_num_counter = 0
        r = 10
        for i in range(a_emb_p.size()[1]):
            a_input_len = anchor_input[i]
            p_input_len = trajs_input[i]
            n_input_len = negative_input[i]
            num_counter = 0
            tmp_batch_loss = 0.
            loss = 0.
            for lp in range(r):
                ap_up = a_input_len / 10 + 1
                p_up = p_input_len / 10 + 1
                ap_rand = int(np.random.randint(0, ap_up, 1))
                p_rand = int(np.random.randint(0, p_up, 1))
                ap_subtraj = a_emb_p.permute(1, 0, 2)[i][min((ap_rand + 1) * 10 - 1, a_input_len - 1)].unsqueeze(0)
                p_subtraj = p_emb.permute(1, 0, 2)[i][min((p_rand + 1) * 10 - 1, p_input_len - 1)].unsqueeze(0)
                ap_pred = torch.exp(-F.pairwise_distance(ap_subtraj, p_subtraj, p=2))
                ap_target = subtraj_trajs_distance[i][ap_rand][p_rand]
                # ap_loss = 1.0/(config.batch_size*10.0) * ap_target * torch.mul((ap_target - ap_pred), (ap_target - ap_pred))
                ap_loss = 1.0 / r * ap_target * torch.mul((ap_target - ap_pred), (ap_target - ap_pred))
                loss += ap_loss
                num_counter += 1

            for ln in range(r):
                an_up = a_input_len / 10 + 1
                n_up = n_input_len / 10 + 1
                an_rand = int(np.random.randint(0, an_up, 1))
                n_rand = int(np.random.randint(0, n_up, 1))
                an_subtraj = a_emb_n.permute(1, 0, 2)[i][min((an_rand + 1) * 10 - 1, a_input_len - 1)].unsqueeze(0)
                n_subtraj = n_emb.permute(1, 0, 2)[i][min((n_rand + 1) * 10 - 1, n_input_len - 1)].unsqueeze(0)
                an_pred = torch.exp(-F.pairwise_distance(an_subtraj, n_subtraj, p=2))
                an_target = subtraj_neg_distance[i][an_rand][n_rand]
                # an_loss = 1.0/(config.batch_size*10.0) * an_target * torch.mul((an_target - an_pred), (an_target - an_pred))
                an_loss = 1.0 / r * an_target * torch.mul((an_target - an_pred), (an_target - an_pred))
                loss += an_loss
                num_counter += 1

            batch_num_counter += 1
            # batch_loss += loss / num_counter
            batch_loss += loss

        # return batch_loss
        return batch_loss / batch_num_counter
