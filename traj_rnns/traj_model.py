from sam_cells import SAM_LSTMCell, SAM_GRUCell, TRM_LSTMCell
from torch.nn import Module, Parameter
from tools import config

import torch.autograd as autograd
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import math, copy, time
from torch.autograd import Variable

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    #def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
    #    return self.decode(self.encode(src, src_mask), src_mask,
    #                        tgt, tgt_mask)
    def forward(self, src, src_mask, inputs_len):
        output_trm = self.encode(src, src_mask)
        results = []
        for b, v in enumerate(inputs_len):
            tmp = output_trm[b][:v]
            #print(tmp.size()) #(len,128)
            tmp_result = torch.mean(tmp, dim=0)
            #print(tmp_result.size()) #(128,)
            results.append(tmp_result.view(1, -1))
        f_outputs = torch.cat(results, dim=0)
        #a = self.encode_s(src, src_mask)	
        return f_outputs
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
        #return self.encoder(src, src_mask)
        
    def encode_s(self, src, src_mask):
        pool_a = []
        pool_a_len = []
        
        vocab_grid = self.src_embed(src)
        print(vocab_grid.size())
        #tmp = self.encoder(self.src_embed(src[:, 0:10]), src_mask[:,:, 0:10])
        tmp = self.encode(vocab_grid[:, 0:10], src_mask[:,:, 0:10])
        #print(tmp.size())
        tmp_result = torch.mean(tmp, dim=1)
        #print(tmp_result.size())
        pool_a.append(tmp_result.unsqueeze(1))
        '''tmp = self.encode(vocab_grid[:, 10:20], src_mask[:,:, 10:20])
        tmp_result = torch.mean(tmp, dim=1)
        pool_a.append(tmp_result.unsqueeze(1))
        tmp = self.encoder(self.src_embed(src[:, 20:30]), src_mask[:,:, 20:30])
        tmp_result = torch.mean(tmp, dim=1)
        pool_a.append(tmp_result.unsqueeze(1))
        tmp = self.encoder(self.src_embed(src[:, 30:40]), src_mask[:,:, 30:40])
        tmp_result = torch.mean(tmp, dim=1)
        pool_a.append(tmp_result.unsqueeze(1))'''
        
        return tmp_result
        
        
        '''for i in range(15):
            upper = min(src.size(1), i*10+10)
            print(i)		
            tmp = self.encoder(self.src_embed(src[:, i*10:upper]), src_mask[:,:, i*10:upper]) #220x10x128
            print("haha")
            tmp_result = torch.mean(tmp, dim=1) #220x128
            pool_a.append(tmp_result.unsqueeze(1)) # 220x1x128 list_len:15
            pool_a_len.append(src_mask[:,:,upper])
            print(len(pool_a_len))
            
        pool_b = []
        for j in range(3):
            src_b = torch.cat(pool_a[j*5:j*5+5], dim=1)
            print(src_b.size())
            src_mask_b = torch.cat(pool_a_len[j*5:j*5+5], dim=2)
            tmp_b = self.encoder(src_b, src_mask_b)
            tmp_result_b = torch.mean(tmp_b, dim=1)
            pool_b.append(tmp_result.view(1, -1))
        
        #pool_c = []
        src_c = torch.cat(pool_b, dim=0)
        tmp_c = self.encoder(src_c)
        tmp_result_c = torch.mean(tmp_c, dim=1)
                   
        return tmp_result_c	'''		
	    
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        # nn.init.xavier_uniform_(self.w_1.weight)
        # nn.init.xavier_uniform_(self.w_2.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, x):
        # jud = (x<=300)|((x>300)&(x%300==0))|((x>300)&((x-1)%300==0))|(x>89700)
        # xlu = (x-301).clamp(0,90000)
        '''x_coor = x[0]
        y_coor = x[1]
        xl_coor = (x_coor-1).clamp(0, 1099)
        xr_coor = (x_coor+1).clamp(0, 1099)
        yu_coor = (y_coor+1).clamp(0, 1099)
        yb_coor = (y_coor-1).clamp(0, 1099)
        xlu = (yu_coor) * 1100 + xl_coor + 1
        # xu = (x-300).clamp(0,90000)
        xu = (yu_coor) * 1100 + x_coor + 1
        xru = (yu_coor) * 1100 + xr_coor + 1
        #xl = (x-1).clamp(0,90000)
        xl = (y_coor) * 1100 + xl_coor + 1
        xx = (y_coor) * 1100 + x_coor + 1
        #xr = (x+1).clamp(0,90000)
        xr = (y_coor) * 1100 + xr_coor + 1
        xlb = (yb_coor) * 1100 + xl_coor + 1
        #xb = (x+300).clamp(0,90000)
        xb = (yb_coor) * 1100 + x_coor + 1
        xrb = (yb_coor) * 1100 + xr_coor + 1
        neighbor = (self.lut(xlu) + self.lut(xu) + self.lut(xru) + self.lut(xl) + self.lut(xr) +
                    self.lut(xlb) + self.lut(xb) + self.lut(xrb)) / 8.0
        # neighbor = (torch.add(self.lut(xu), torch.add(self.lut(xl), torch.add(self.lut(xr), self.lut(xb))))) / 4
        # f_results = (self.lut(x) * 0.9 + neighbor * 0.1) * math.sqrt(self.d_model)
        f_results = torch.add(self.lut(xx) * 0.9, neighbor * 0.1) * math.sqrt(self.d_model)'''

        return self.lut(x) * math.sqrt(self.d_model)
        #return self.lut(x) * math.sqrt(self.d_model)
        # return f_results

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

def make_model(src_vocab, tgt_vocab, N=2,
               d_model=128, d_ff=512, h=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    # grid = Embeddings(d_model, src_vocab)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        # nn.Sequential(c(grid), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        # nn.Sequential(c(grid), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class RNNEncoder(Module):
    def __init__(self, input_size, hidden_size, grid_size, stard_LSTM=False, incell=True):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stard_LSTM = stard_LSTM
        model = make_model(1100*1100, 1, N=2)
        self.bitrm = model.cuda()
        # self.final_l = torch.nn.Linear(hidden_size * 2, hidden_size)
        #self.zz = torch.Tensor([0.5])
        #self.wweight = autograd.Variable(self.zz, requires_grad=False).cuda()
        self.wweight = Parameter(torch.ones(1) / 2.0)
        self.wweight.requires_grad = False
        self.sig = torch.nn.Sigmoid()

        if self.stard_LSTM:
            if config.recurrent_unit=='GRU':
                self.cell = torch.nn.GRUCell(input_size - 2, hidden_size).cuda()
            elif config.recurrent_unit=='SimpleRNN':
                self.cell = torch.nn.RNNCell(input_size - 2, hidden_size).cuda()
            else:
                self.cell = torch.nn.LSTMCell(input_size - 2, hidden_size).cuda()
                # self.cell = torch.nn.LSTMCell(hidden_size, hidden_size).cuda()
        else:
            if config.recurrent_unit=='GRU':
                self.cell = SAM_GRUCell(input_size, hidden_size, grid_size, incell=incell).cuda()
            elif config.recurrent_unit=='SimpleRNN':
                self.cell = SpatialRNNCell(input_size, hidden_size, grid_size, incell=incell).cuda()
            else:
                self.cell = SAM_LSTMCell(input_size, hidden_size, grid_size, incell=incell).cuda()
                # self.cell = TRM_LSTMCell(input_size, hidden_size, grid_size, incell=False).cuda()				

        print self.cell
        print 'in cell update: {}'.format(incell)
        # self.cell = torch.nn.LSTMCell(input_size-2, hidden_size).cuda()

    def forward(self, inputs_a, initial_state=None):
        inputs, inputs_len = inputs_a  # porto inputs:220x149x4 inputs_len:list
        time_steps = inputs.size(1)  # porto:149
        out = None
        # print(inputs_len)
        
        # input_grid = ((inputs[:, :, 3]*1100 + inputs[:, :, 2] - 2202 + 1).clamp(0, 1100*1100)).type(torch.LongTensor).cuda() #porto:220x149
        # #print(input_grid.size())
        #
        # inputs_grid = autograd.Variable(input_grid, requires_grad=False).cuda() #porto:220x149
        # #print(inputs_grid.size())
        # # src_mask = Variable(((input_grid != 0).unsqueeze(-2)), requires_grad=False).cuda()
        # src_mask = (input_grid != 0).unsqueeze(-2) #porto:220x1x149
        # #print(src_mask[0,:,50:59])
        # # tgt_mask = (inputs_grid != 0).unsqueeze(-2)
        # # tgt_mask = Variable(tgt_mask & autograd.Variable(subsequent_mask(inputs_grid.size(-1)).type_as(tgt_mask.data)))
        # #output_trm = self.bitrm.encode(inputs_grid, src_mask)
        # #output_trm = self.bitrm(inputs_grid, src_mask)
        # f_outputs = self.bitrm(inputs_grid, src_mask, inputs_len)
        # #print(output_trm.size()) # 220*149*128
        # # output_trm = self.bitrm(inputs_grid, inputs_grid, src_mask, tgt_mask)

        '''results = []
        for b, v in enumerate(inputs_len):
            tmp = output_trm[b][:v]
            #print(tmp.size()) #(len,128)
            tmp_result = torch.mean(tmp, dim=0)
            #print(tmp_result.size()) #(128,)
            results.append(tmp_result.view(1, -1))
        f_outputs = torch.cat(results, dim=0)'''

        if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
            out = initial_state
        else:
            out, state = initial_state

        outputs = []
        # lstm_result = []
        for t in range(time_steps):
            if self.stard_LSTM:
                cell_input = inputs[:, t, :][:, :-2]
                # cell_input = output_trm[:, t, :]
            else:
                cell_input = inputs[:, t, :]
            if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
                out = self.cell(cell_input, out)
            else:
                out, state = self.cell(cell_input, (out, state))
            outputs.append(out)
            # lstm_result.append(out.unsqueeze(1))
        mask_out = []
        for b, v in enumerate(inputs_len):
            mask_out.append(outputs[v-1][b, :].view(1, -1))
        fi_outputs = torch.cat(mask_out, dim=0)
        # for t in range(time_steps):
        #     lstm_result.append(fi_outputs.unsqueeze(1))
        # lstm_results = torch.cat(lstm_result, dim=1)

        '''input_grid = torch.LongTensor(inputs.size(0), inputs.size(1))
                for i in range(inputs.size(0)):
                    for j in range(inputs.size(1)):
                        if inputs[i][j][2] != 0:
                            input_grid[i][j] = inputs[i][j][3]*300 + inputs[i][j][2] - 602 + 1
                        else:
                            input_grid[i][j] = 0'''

        # x = ((inputs[:, :, 2] - 2).clamp(0, 1099)).type(torch.LongTensor).cuda()
        # y = ((inputs[:, :, 3] - 2).clamp(0, 1099)).type(torch.LongTensor).cuda()
        
        '''results = []
        for b, v in enumerate(inputs_len):
            tmp = output_trm[b][:42]
            tmp_results = []
            for i in range(tmp.size(0)):
                tmp_results.append(tmp[i].view(1, -1))
            tmp_res = torch.cat(tmp_results, dim=1)
            tmp_res = torch.tanh(self.l_fc1(tmp_res))
            results.append(tmp_res.view(1, -1))
        f_outputs = torch.cat(results, dim=0)'''

        #w_clamped = self.sig(self.wweight)
        # print(w_clamped)
        # final_outputs = fi_outputs * (1.0-self.wweight) + f_outputs * self.wweight
        #print(self.wweight.data)
        #final_outputs = fi_outputs #* 0.5 + f_outputs * 0.5
        # final_outputs = self.final_l(torch.cat((f_outputs, fi_outputs), 1))

        # return f_outputs
        #return final_outputs
        return fi_outputs
        # return torch.cat(mask_out, dim = 0)

    def batch_grid_state_gates(self, inputs_a, initial_state = None):
        inputs, inputs_len = inputs_a
        time_steps = inputs.size(1)
        out, state = initial_state
        outputs = []
        gates_out_all = []
        batch_weight_ih = autograd.Variable(self.cell.weight_ih.data, requires_grad=False).cuda()
        batch_weight_hh = autograd.Variable(self.cell.weight_hh.data, requires_grad=False).cuda()
        batch_bias_ih = autograd.Variable(self.cell.bias_ih.data, requires_grad=False).cuda()
        batch_bias_hh = autograd.Variable(self.cell.bias_hh.data, requires_grad=False).cuda()
        for t in range(time_steps):
            # cell_input = inputs[:, t, :][:,:-2]
            cell_input = inputs[:, t, :]
            self.cell.update_memory(cell_input, (out, state),
                                    batch_weight_ih, batch_weight_hh,
                                    batch_bias_ih, batch_bias_hh)

def fast_cdist(x1, x2):
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x2 = x2 - adjustment

    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    res.clamp_min_(1e-30).sqrt_()
    return res


def topn_point_sampling(anchor_traj, cmp_traj):
    #points_dist = []
    #for i in range(cmp_traj.size(0)):
    #    point_dist = torch.exp(-F.pairwise_distance(anchor_traj[i], cmp_traj[i], p=2)).view(1, -1)
    #    points_dist.append(point_dist)
    #ppd = torch.cat(points_dist, dim=0) #size: 220x149

    pwd = torch.cdist(anchor_traj, cmp_traj) #pwd:220x149x149
    sorted, indices = torch.sort(pwd, descending=False)
    topn_points = indices[:, :, :5]
    neg_points = indices[:, :, -5:]
    return topn_points, neg_points


class SMNEncoder(Module):
    def __init__(self, input_size, hidden_size, grid_size, stard_LSTM=False, incell=True):
        super(SMNEncoder, self).__init__()
        self.input_size = input_size
        if config.no_matching:
            self.hidden_size = hidden_size/2
        else:
            self.hidden_size = hidden_size
        self.stard_LSTM = stard_LSTM
        self.mlp_ele = torch.nn.Linear(2, hidden_size/2).cuda()
        self.nonLeaky = torch.nn.LeakyReLU(0.1)
        self.nonTanh = torch.nn.Tanh()
        self.point_pooling = torch.nn.AvgPool1d(config.pooling_size)
        #model = make_model(1100*1100, 1, N=4)
        #self.bitrm = model
        #self.zz = torch.Tensor([0.5])
        #self.wweight = autograd.Variable(self.zz, requires_grad=False).cuda()
        #self.wweight = Parameter(torch.ones(1) / 2.0)
        #self.wweight.requires_grad = False
        self.seq_model_layer = 1
        if config.no_matching:
            self.seq_model = torch.nn.LSTM(hidden_size/2, hidden_size/2, num_layers=self.seq_model_layer)
        else:
            self.seq_model = torch.nn.LSTM(hidden_size, hidden_size, num_layers=self.seq_model_layer)
        self.t2s_model = torch.nn.LSTM(2, hidden_size, num_layers=self.seq_model_layer)
        self.pos_encoder = PositionalEncoding(d_model=hidden_size, dropout=0.1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        # self.trm = nn.Transformer(d_model=hidden_size, nhead=4, num_encoder_layers=1, num_decoder_layers=1,
        #                           dim_feedforward=128)
        if config.no_matching:
            self.res_linear1 = torch.nn.Linear(hidden_size/2, hidden_size/2).cuda()
            self.res_linear2 = torch.nn.Linear(hidden_size/2, hidden_size/2).cuda()
            self.res_linear3 = torch.nn.Linear(hidden_size/2, hidden_size/2).cuda()
        else:
            self.res_linear1 = torch.nn.Linear(hidden_size, hidden_size).cuda()
            self.res_linear2 = torch.nn.Linear(hidden_size, hidden_size).cuda()
            self.res_linear3 = torch.nn.Linear(hidden_size, hidden_size).cuda()
        # self.batch_norm1 = torch.nn.BatchNorm1d(hidden_size)
        # self.batch_norm2 = torch.nn.BatchNorm1d(hidden_size)

        if self.stard_LSTM:
            if config.recurrent_unit == 'GRU':
                self.cell = torch.nn.GRUCell(input_size - 2, hidden_size).cuda()
            elif config.recurrent_unit == 'SimpleRNN':
                self.cell = torch.nn.RNNCell(input_size - 2, hidden_size).cuda()
            else:
                #self.cell = torch.nn.LSTMCell(input_size - 2, hidden_size)
                self.cell = torch.nn.LSTMCell(hidden_size, hidden_size).cuda()
        else:
            if config.recurrent_unit == 'GRU':
                self.cell = SAM_GRUCell(input_size, hidden_size, grid_size, incell=incell).cuda()
            elif config.recurrent_unit == 'SimpleRNN':
                self.cell = SpatialRNNCell(input_size, hidden_size, grid_size, incell=incell).cuda()
            else:
                self.cell = SAM_LSTMCell(input_size, hidden_size, grid_size, incell=incell).cuda()
                # self.cell = TRM_LSTMCell(input_size, hidden_size, grid_size, incell=False).cuda()				

        print self.cell
        print 'in cell update: {}'.format(incell)

    def trm_forward(self, inputs_a, inputs_b):
        input_a, input_len_a = inputs_a
        input_b, input_len_b = inputs_b
        time_steps_a = input_a.size(1)  # porto:149
        time_steps_b = input_b.size(1)
        mlp_input_a = self.nonLeaky(self.mlp_ele(input_a[:, :, :-2]))  # 220x149x64
        mlp_input_b = self.nonLeaky(self.mlp_ele(input_b[:, :, :-2]))

        input_grid_a = (input_a[:, :, 3] * 1100 + input_a[:, :, 2] - 2202 + 1).clamp(0, 1100 * 1100).long()  # porto:220x149
        trm_mask_a = (input_grid_a == 0).cuda()
        mask_a = (input_grid_a != 0).unsqueeze(-2).cuda()
        # trm_output_a = self.transformer_encoder(mlp_input_a.permute(1, 0, 2), src_key_padding_mask=mask_a).permute(1, 0, 2)  # 220x149x64
        input_grid_b = (input_b[:, :, 3] * 1100 + input_b[:, :, 2] - 2202 + 1).clamp(0, 1100 * 1100).long()  # porto:220x149
        trm_mask_b = (input_grid_b == 0).cuda()
        mask_b = (input_grid_b != 0).unsqueeze(-2).cuda()
        scores_a_o = torch.matmul(mlp_input_a, mlp_input_b.transpose(-2, -1))  # porto:220x149x149
        scores_a_o = scores_a_o.masked_fill(mask_b == 0, -1e9).transpose(-2, -1)
        scores_a_o = scores_a_o.masked_fill(mask_a == 0, -1e9).transpose(-2, -1)
        scores_a = scores_a_o  # porto:220x149x149
        p_attn_a = F.softmax(scores_a, dim=-1)
        p_attn_a = p_attn_a.masked_fill(mask_b == 0, 0.0).transpose(-2, -1)
        p_attn_a = p_attn_a.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
        attn_ab = p_attn_a.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size / 2)
        sum_traj_b = mlp_input_b.unsqueeze(-3).expand(-1, time_steps_a, -1, -1).mul(attn_ab).sum(dim=-2)
        cell_input_a = torch.cat((mlp_input_a, (mlp_input_a - sum_traj_b)), dim=-1)  # porto:220x149x128
        cell_input_a = self.pos_encoder(cell_input_a)
        trm_output_a = self.transformer_encoder(src=cell_input_a.permute(1, 0, 2), src_key_padding_mask=trm_mask_a)
        # trm_output_a = self.trm(src=cell_input_a.permute(1, 0, 2), tgt=mlp_input_b.permute(1, 0, 2),
        #                       src_key_padding_mask=mask_a, tgt_key_padding_mask=mask_b,
        #                       memory_key_padding_mask=mask_a)
        trm_output_ca = F.sigmoid(self.res_linear1(trm_output_a)) * self.nonLeaky(self.res_linear2(trm_output_a))
        trm_output_hata = F.sigmoid(self.res_linear3(trm_output_a)) * self.nonLeaky(trm_output_ca)
        trm_output_fa = trm_output_a + trm_output_hata
        mask_out_a = []
        for b, v in enumerate(input_len_a):
            mask_out_a.append(trm_output_fa[v - 1][b, :].view(1, -1))
        fa_outputs = torch.cat(mask_out_a, dim=0)

        scores_b = scores_a_o.permute(0, 2, 1)
        ## scores_b = scores_b.masked_fill(mask_b == 0, -1e9)
        p_attn_b = F.softmax(scores_b, dim=-1)
        p_attn_b = p_attn_b.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
        p_attn_b = p_attn_b.masked_fill(mask_b == 0, 0.0).transpose(-2, -1)
        attn_ba = p_attn_b.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size / 2)
        sum_traj_a = mlp_input_a.unsqueeze(-3).expand(-1, time_steps_b, -1, -1).mul(attn_ba).sum(dim=-2)
        cell_input_b = torch.cat((mlp_input_b, (mlp_input_b - sum_traj_a)), dim=-1)  # porto:220x149x128
        cell_input_b = self.pos_encoder(cell_input_b)
        trm_output_b = self.transformer_encoder(src=cell_input_b.permute(1, 0, 2), src_key_padding_mask=trm_mask_b)
        # trm_output_b = self.trm(src=mlp_input_b.permute(1, 0, 2), tgt=mlp_input_a.permute(1, 0, 2),
        #                       src_key_padding_mask=mask_b, tgt_key_padding_mask=mask_a,
        #                       memory_key_padding_mask=mask_b)
        trm_output_cb = F.sigmoid(self.res_linear1(trm_output_b)) * self.nonLeaky(self.res_linear2(trm_output_b))
        trm_output_hatb = F.sigmoid(self.res_linear3(trm_output_b)) * self.nonLeaky(trm_output_cb)
        trm_output_fb = trm_output_b + trm_output_hatb
        mask_out_b = []
        for b, v in enumerate(input_len_b):
            mask_out_b.append(trm_output_fb[v - 1][b, :].view(1, -1))
        fb_outputs = torch.cat(mask_out_b, dim=0)

        return fa_outputs, fb_outputs, trm_output_fa, trm_output_fb

    def init_hidden(self, hidden_dim, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, mini_batch_size, hidden_dim)
        return (torch.zeros(self.seq_model_layer, batch_size, hidden_dim).cuda(),
                torch.zeros(self.seq_model_layer, batch_size, hidden_dim).cuda())

    def f(self, inputs_a, inputs_b):
        input_a, input_len_a = inputs_a  # porto inputs:220x149x4 inputs_len:list
        input_b, input_len_b = inputs_b
        if config.pooling_points:
            time_steps_a = input_a.size(1)/config.pooling_size  # porto:149
            time_steps_b = input_b.size(1)/config.pooling_size
        else:
            time_steps_a = input_a.size(1)
            time_steps_b = input_b.size(1)
        mlp_input_a = self.nonLeaky(self.mlp_ele(input_a[:, :, :-2]))
        pooled_mlp_input_a = self.point_pooling(mlp_input_a.permute(0,2,1)).permute(0,2,1)
        mlp_input_b = self.nonLeaky(self.mlp_ele(input_b[:, :, :-2]))
        pooled_mlp_input_b = self.point_pooling(mlp_input_b.permute(0,2,1)).permute(0,2,1)
        input_grid_a = (input_a[:, :, 3] * 1100 + input_a[:, :, 2] - 2202 + 1).clamp(0, 1100 * 1100).long()  # porto:220x149
        mask_a = (input_grid_a != 0).unsqueeze(-2).cuda()  # porto:220x1x149
        pooled_mask_a = self.point_pooling(mask_a.float())
        input_grid_b = (input_b[:, :, 3] * 1100 + input_b[:, :, 2] - 2202 + 1).clamp(0, 1100 * 1100).long()  # porto:220x149
        mask_b = (input_grid_b != 0).unsqueeze(-2).cuda()  # porto:220x1x149
        pooled_mask_b = self.point_pooling(mask_b.float())

        out_a, state_a = self.init_hidden(self.hidden_size, input_a.size(0))
        if config.no_matching:
            cell_input_a = mlp_input_a
        else:
            if config.pooling_points:
                scores_a_o = torch.matmul(mlp_input_a, pooled_mlp_input_b.transpose(-2, -1))  # porto:220x149x149
                scores_a_o = scores_a_o.masked_fill(pooled_mask_b == 0, -1e9).transpose(-2, -1)
                scores_a_o = scores_a_o.masked_fill(mask_a == 0, -1e9).transpose(-2, -1)
                scores_a = scores_a_o  # porto:220x149x149
                p_attn_a = F.softmax(scores_a, dim=-1)
                p_attn_a = p_attn_a.masked_fill(pooled_mask_b == 0, 0.0).transpose(-2, -1)
                p_attn_a = p_attn_a.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
                attn_ab = p_attn_a.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size / 2)
                # sum_traj_b = mlp_input_b.unsqueeze(-3).expand(-1, time_steps_a, -1, -1).mul(attn_ab).sum(dim=-2)
                sum_traj_b = mlp_input_b.unsqueeze(-2).expand(-1, -1, time_steps_a, -1).mul(attn_ab).sum(dim=-2)
                cell_input_a = torch.cat((mlp_input_a, (mlp_input_a-sum_traj_b)), dim=-1)  # porto:220x149x128 
                '''
                scores_a_o = torch.matmul(pooled_mlp_input_a, pooled_mlp_input_b.transpose(-2, -1))  # porto:220x149x149
                expand_scao = torch.zeros(220, 150, 150).cuda()       
                for e_z in range(220):
                    for e_j in range(150):
                        for e_i in range(150):
                            expand_scao[e_z][e_j][e_i] = scores_a_o[e_z][e_j / config.pooling_size][e_i / config.pooling_size]
                scores_a_o = expand_scao.masked_fill(mask_b == 0, -1e9).transpose(-2, -1)
                scores_a_o = scores_a_o.masked_fill(mask_a == 0, -1e9).transpose(-2, -1)
                scores_a = scores_a_o  # porto:220x149x149
                p_attn_a = F.softmax(scores_a, dim=-1)
                p_attn_a = p_attn_a.masked_fill(mask_b == 0, 0.0).transpose(-2, -1)
                p_attn_a = p_attn_a.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
                attn_ab = p_attn_a.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size / 2)
                sum_traj_b = mlp_input_b.unsqueeze(-3).expand(-1, time_steps_a, -1, -1).mul(attn_ab).sum(dim=-2)
                # sum_traj_b = mlp_input_b.unsqueeze(-2).expand(-1, -1, time_steps_a, -1).mul(attn_ab).sum(dim=-2)
                cell_input_a = torch.cat((mlp_input_a, (mlp_input_a-sum_traj_b)), dim=-1)  # porto:220x149x128'''
                '''
                scores_a_o = torch.matmul(pooled_mlp_input_a, pooled_mlp_input_b.transpose(-2, -1))  # porto:220x149x149       
                scores_a_o = scores_a_o.masked_fill(pooled_mask_b == 0, -1e9).transpose(-2, -1)
                scores_a_o = scores_a_o.masked_fill(pooled_mask_a == 0, -1e9).transpose(-2, -1)
                scores_a = scores_a_o  # porto:220x149x149
                p_attn_a = F.softmax(scores_a, dim=-1)
                p_attn_a = p_attn_a.masked_fill(pooled_mask_b == 0, 0.0).transpose(-2, -1)
                p_attn_a = p_attn_a.masked_fill(pooled_mask_a == 0, 0.0).transpose(-2, -1)
                attn_ab = p_attn_a.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size / 2)
                sum_traj_b = pooled_mlp_input_b.unsqueeze(-3).expand(-1, time_steps_a, -1, -1).mul(attn_ab).sum(dim=-2)
                # sum_traj_b = mlp_input_b.unsqueeze(-2).expand(-1, -1, time_steps_a, -1).mul(attn_ab).sum(dim=-2)
                cell_input_a_list = []
                for e_z in range(220):
                    traj_input_a_list = []
                    for e_j in range(150):
                        traj_input_a_list.append( torch.cat((mlp_input_a[e_z][e_j], (mlp_input_a[e_z][e_j]-sum_traj_b[e_z][e_j/config.pooling_size])), dim=-1).unsqueeze(0) )
                    cell_input_a_list.append(torch.cat(traj_input_a_list,dim=0).unsqueeze(0))
                cell_input_a = torch.cat(cell_input_a_list, dim=0)'''
                #cell_input_a = torch.cat((mlp_input_a, (mlp_input_a-sum_traj_b)), dim=-1)  # porto:220x149x128
            else:
                scores_a_o = torch.matmul(mlp_input_a, mlp_input_b.transpose(-2, -1))  # porto:220x149x149
                scores_a_o = scores_a_o.masked_fill(mask_b == 0, -1e9).transpose(-2, -1)
                scores_a_o = scores_a_o.masked_fill(mask_a == 0, -1e9).transpose(-2, -1)
                scores_a = scores_a_o  # porto:220x149x149
                p_attn_a = F.softmax(scores_a, dim=-1)
                p_attn_a = p_attn_a.masked_fill(mask_b == 0, 0.0).transpose(-2, -1)
                p_attn_a = p_attn_a.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
                attn_ab = p_attn_a.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size / 2)
                sum_traj_b = mlp_input_b.unsqueeze(-3).expand(-1, time_steps_a, -1, -1).mul(attn_ab).sum(dim=-2)
                # sum_traj_b = mlp_input_b.unsqueeze(-2).expand(-1, -1, time_steps_a, -1).mul(attn_ab).sum(dim=-2)
                cell_input_a = torch.cat((mlp_input_a, (mlp_input_a-sum_traj_b)), dim=-1)  # porto:220x149x128
        outputs_a, (hn_a, cn_a) = self.seq_model(cell_input_a.permute(1, 0, 2), (out_a, state_a))
        # outputs_a, (hn_a, cn_a) = self.seq_model(mlp_input_a.permute(1, 0, 2), (out_a, state_a))
        # outputs_a = self.batch_norm1(outputs_a.permute(1,2,0)).permute(2,0,1)
        outputs_ca = F.sigmoid(self.res_linear1(outputs_a)) * self.nonLeaky(self.res_linear2(outputs_a)) #F.tanh(self.res_linear2(outputs_a))
        # outputs_ca = self.batch_norm2(outputs_ca.permute(1,2,0)).permute(2,0,1)
        outputs_hata = F.sigmoid(self.res_linear3(outputs_a)) * self.nonLeaky(outputs_ca) #F.tanh(outputs_ca)
        outputs_fa = outputs_a + outputs_hata
        mask_out_a = []
        for b, v in enumerate(input_len_a):
            mask_out_a.append(outputs_fa[v - 1][b, :].view(1, -1))
        fa_outputs = torch.cat(mask_out_a, dim=0)

        out_b, state_b = self.init_hidden(self.hidden_size, input_b.size(0))
        if config.no_matching:
            cell_input_b = mlp_input_b
        else:
            if config.pooling_points:
                # scores_b = scores_a_o.permute(0, 2, 1)
                scores_b_o = torch.matmul(mlp_input_b, pooled_mlp_input_a.transpose(-2, -1))  # porto:220x149x149
                scores_b_o = scores_b_o.masked_fill(pooled_mask_a == 0, -1e9).transpose(-2, -1)
                scores_b_o = scores_b_o.masked_fill(mask_b == 0, -1e9).transpose(-2, -1)
                scores_b = scores_b_o  # porto:220x149x149
                ## scores_b = scores_b.masked_fill(mask_b == 0, -1e9)
                p_attn_b = F.softmax(scores_b, dim=-1)
                p_attn_b = p_attn_b.masked_fill(pooled_mask_a == 0, 0.0).transpose(-2, -1)
                p_attn_b = p_attn_b.masked_fill(mask_b == 0, 0.0).transpose(-2, -1)
                attn_ba = p_attn_b.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size / 2)
                # sum_traj_a = mlp_input_a.unsqueeze(-3).expand(-1, time_steps_b, -1, -1).mul(attn_ba).sum(dim=-2)
                sum_traj_a = mlp_input_a.unsqueeze(-2).expand(-1, -1, time_steps_b, -1).mul(attn_ba).sum(dim=-2)
                cell_input_b = torch.cat((mlp_input_b, (mlp_input_b - sum_traj_a)), dim=-1)  # porto:220x149x128
                '''
                scores_b = expand_scao.permute(0, 2, 1)
                p_attn_b = F.softmax(scores_b, dim=-1)
                p_attn_b = p_attn_b.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
                p_attn_b = p_attn_b.masked_fill(mask_b == 0, 0.0).transpose(-2, -1)
                attn_ba = p_attn_b.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size / 2)
                sum_traj_a = mlp_input_a.unsqueeze(-3).expand(-1, time_steps_b, -1, -1).mul(attn_ba).sum(dim=-2)
                cell_input_b = torch.cat((mlp_input_b, (mlp_input_b - sum_traj_a)), dim=-1)  # porto:220x149x128 '''
                '''scores_b = scores_a_o.permute(0, 2, 1)
                p_attn_b = F.softmax(scores_b, dim=-1)
                p_attn_b = p_attn_b.masked_fill(pooled_mask_a == 0, 0.0).transpose(-2, -1)
                p_attn_b = p_attn_b.masked_fill(pooled_mask_b == 0, 0.0).transpose(-2, -1)
                attn_ba = p_attn_b.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size / 2)
                sum_traj_a = pooled_mlp_input_a.unsqueeze(-3).expand(-1, time_steps_b, -1, -1).mul(attn_ba).sum(dim=-2)
                # cell_input_b = torch.cat((mlp_input_b, (mlp_input_b - sum_traj_a)), dim=-1)  # porto:220x149x128
                cell_input_b_list = []
                for e_z in range(220):
                    traj_input_b_list = []
                    for e_j in range(150):
                        traj_input_b_list.append( torch.cat((mlp_input_b[e_z][e_j], (mlp_input_b[e_z][e_j]-sum_traj_a[e_z][e_j/config.pooling_size])), dim=-1).unsqueeze(0) )
                    cell_input_b_list.append(torch.cat(traj_input_b_list,dim=0).unsqueeze(0))
                cell_input_b = torch.cat(cell_input_b_list, dim=0)'''
            else:
                scores_b = scores_a_o.permute(0, 2, 1)
                ## scores_b = scores_b.masked_fill(mask_b == 0, -1e9)
                p_attn_b = F.softmax(scores_b, dim=-1)
                p_attn_b = p_attn_b.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
                p_attn_b = p_attn_b.masked_fill(mask_b == 0, 0.0).transpose(-2, -1)
                attn_ba = p_attn_b.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size / 2)
                sum_traj_a = mlp_input_a.unsqueeze(-3).expand(-1, time_steps_b, -1, -1).mul(attn_ba).sum(dim=-2)
                # sum_traj_a = mlp_input_a.unsqueeze(-2).expand(-1, -1, time_steps_b, -1).mul(attn_ba).sum(dim=-2)
                cell_input_b = torch.cat((mlp_input_b, (mlp_input_b - sum_traj_a)), dim=-1)  # porto:220x149x128
        outputs_b, (hn_b, cn_b) = self.seq_model(cell_input_b.permute(1, 0, 2), (out_b, state_b))
        # outputs_b, (hn_b, cn_b) = self.seq_model(mlp_input_b.permute(1, 0, 2), (out_b, state_b))
        # outputs_b = self.batch_norm1(outputs_b.permute(1,2,0)).permute(2,0,1)
        outputs_cb = F.sigmoid(self.res_linear1(outputs_b)) * self.nonLeaky(self.res_linear2(outputs_b)) #F.tanh(self.res_linear2(outputs_b))
        # outputs_cb = self.batch_norm2(outputs_cb.permute(1,2,0)).permute(2,0,1)
        outputs_hatb = F.sigmoid(self.res_linear3(outputs_b)) * self.nonLeaky(outputs_cb) #F.tanh(outputs_cb)
        outputs_fb = outputs_b + outputs_hatb
        mask_out_b = []
        for b, v in enumerate(input_len_b):
            mask_out_b.append(outputs_b[v - 1][b, :].view(1, -1))
        fb_outputs = torch.cat(mask_out_b, dim=0)

        return fa_outputs, fb_outputs, outputs_fa, outputs_fb #, p_attn_a, p_attn_b

    def t2s_forward(self, inputs_a, inputs_b):
        input_a, input_len_a = inputs_a  # porto inputs:220x149x4 inputs_len:list
        input_b, input_len_b = inputs_b

        out_a, state_a = self.init_hidden(self.hidden_size, input_a.size(0))
        outputs_a, (hn_a, cn_a) = self.t2s_model(input_a[:, :, :-2].permute(1, 0, 2), (out_a, state_a))
        outputs_ca = F.sigmoid(self.res_linear1(outputs_a)) * F.tanh(self.res_linear2(outputs_a))
        outputs_hata = F.sigmoid(self.res_linear3(outputs_a)) * F.tanh(outputs_ca)
        outputs_fa = outputs_a + outputs_hata
        mask_out_a = []
        for b, v in enumerate(input_len_a):
            mask_out_a.append(outputs_fa[v - 1][b, :].view(1, -1))
        fa_outputs = torch.cat(mask_out_a, dim=0)

        out_b, state_b = self.init_hidden(self.hidden_size, input_b.size(0))
        outputs_b, (hn_b, cn_b) = self.t2s_model(input_b[:, :, :-2].permute(1, 0, 2), (out_b, state_b))
        outputs_cb = F.sigmoid(self.res_linear1(outputs_b)) * F.tanh(self.res_linear2(outputs_b))
        outputs_hatb = F.sigmoid(self.res_linear3(outputs_b)) * F.tanh(outputs_cb)
        outputs_fb = outputs_b + outputs_hatb
        mask_out_b = []
        for b, v in enumerate(input_len_b):
            mask_out_b.append(outputs_fb[v - 1][b, :].view(1, -1))
        fb_outputs = torch.cat(mask_out_b, dim=0)

        return fa_outputs, fb_outputs, outputs_fa, outputs_fb

    def forward(self, inputs_a, inputs_b, initial_state_a=None, initial_state_b=None, topn_p=None, neg_p=None):
        input_a, input_len_a = inputs_a  # porto inputs:220x149x4 inputs_len:list
        input_b, input_len_b = inputs_b
        time_steps_a = input_a.size(1)  # porto:149
        time_steps_b = input_b.size(1)
        out_a = None
        out_b = None
        mlp_input_a = self.nonLeaky(self.mlp_ele(input_a[:, :, :-2]))
        mlp_input_b = self.nonLeaky(self.mlp_ele(input_b[:, :, :-2]))

        if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
            out_a = initial_state_a
        else:
            out_a, state_a = initial_state_a 
        input_grid_a = ((input_a[:, :, 3]*1100 + input_a[:, :, 2] - 2202 + 1).clamp(0, 1100*1100)).type(torch.LongTensor) #porto:220x149
        mask_a = (input_grid_a != 0).unsqueeze(-2).cuda()  # porto:220x1x149
        scores_a_o = torch.matmul(mlp_input_a, mlp_input_b.transpose(-2, -1))  # porto:220x149x149
        ##scores_a = torch.cdist(mlp_input_a.unsqueeze(0), mlp_input_b.unsqueeze(0), p=2)
        ##scores_a = scores_a.squeeze()
        ##scores_a = fast_cdist(mlp_input_a, mlp_input_b)
        scores_a = scores_a_o.masked_fill(mask_a == 0, -1e9)  # porto:220x149x149
        ##sorted_a, index_a = torch.sort(scores_a)     
        #a_max_index = torch.max(scores_a, 2)[1] #porto:220x149
        #a_max = []
        #for iii in range(a_max_index.size(0)):
        #    a_max.append(mlp_input_b[iii].index_select(0, a_max_index[0, :]).unsqueeze(0))
        #a_max_b = torch.cat(a_max, dim=0)
        p_attn_a = F.softmax(scores_a, dim = -1)
        attn_ab = p_attn_a.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size/2)
        if topn_p is not None:
            topn_point = torch.gather(p_attn_a, 2, topn_p) #220x149x5
            topn_p_loss = 1.0 - (torch.sum(torch.sum(torch.sum(topn_point, dim=2), dim=1), dim=0)) / (220.0 * 149.0)
            neg_point = torch.gather(p_attn_a, 2, neg_p)
            neg_p_loss = (torch.sum(torch.sum(torch.sum(neg_point, dim=2), dim=1), dim=0)) / (220.0 * 149.0)
        sum_traj_b = mlp_input_b.unsqueeze(-3).expand(-1, time_steps_a, -1, -1).mul(attn_ab).sum(dim=-2)
        ##sum_a = p_attn_a.unsqueeze(-1).expand(-1,-1,-1,128)
        #sum_traj_b = mlp_input_a.mul(a_max_b)
        outputs_a = []
        ##lstm_result = []
        for t in range(time_steps_a):
            if self.stard_LSTM:
                #cell_input_a = input_a[:, t, :][:,:-2]
                #cell_input_a = self.mlp_ele(cell_input_a)
                cell_input_a = torch.cat((mlp_input_a[:, t, :], (mlp_input_a[:, t, :]-sum_traj_b[:, t, :])), 1)
                #cell_input_a = torch.cat((mlp_input_a[:, t, :], sum_traj_b[:, t, :]), 1)
            else:
                cell_input_a = input_a[:, t, :]
            if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
                out_a = self.cell(cell_input_a, out_a)
            else:
                out_a, state_a = self.cell(cell_input_a, (out_a, state_a))
            outputs_a.append(out_a.unsqueeze(0))
            # lstm_result.append(out.unsqueeze(1))
        mask_out_a = []
        for b, v in enumerate(input_len_a):
            mask_out_a.append(outputs_a[v-1][0][b,:].view(1,-1))
        fa_outputs = torch.cat(mask_out_a, dim=0)

        if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
            out_b = initial_state_b
        else:
            out_b, state_b = initial_state_b
        input_grid_b = ((input_b[:, :, 3]*1100 + input_b[:, :, 2] - 2202 + 1).clamp(0, 1100*1100)).type(torch.LongTensor) #porto:220x149
        mask_b = (input_grid_b != 0).unsqueeze(-2).cuda() #porto:220x1x149      
        #scores_b = torch.matmul(mlp_input_b, mlp_input_a.transpose(-2, -1))
        scores_b = scores_a_o.permute(0, 2, 1)
        scores_b = scores_b.masked_fill(mask_b == 0, -1e9)
        #b_max_index = torch.max(scores_b, 2)[1] #porto:220x149
        #b_max = []
        #for iii in range(b_max_index.size(0)):
        #    b_max.append(mlp_input_a[iii].index_select(0, b_max_index[0, :]).unsqueeze(0))
        #b_max_a = torch.cat(b_max, dim=0)
        ##p_attn_b = F.softmax(scores_b, dim = -1)
        p_attn_b = F.softmax(scores_b, dim = -1)
        attn_ba = p_attn_b.unsqueeze(-1).expand(-1,-1,-1,self.hidden_size/2)
        sum_traj_a = mlp_input_a.unsqueeze(-3).expand(-1, time_steps_b, -1, -1).mul(attn_ba).sum(dim=-2)
        #sum_traj_a = mlp_input_b.mul(b_max_a)
        outputs_b = []
        #lstm_result = []
        for t in range(time_steps_b):
            if self.stard_LSTM:
                #cell_input_a = input_a[:, t, :][:,:-2]
                #cell_input_a = self.mlp_ele(cell_input_a)
                #cell_input_b = torch.cat((mlp_input_b[:, t, :], sum_b, 1)
                cell_input_b = torch.cat((mlp_input_b[:, t, :], (mlp_input_b[:, t, :]-sum_traj_a[:, t, :])), 1)
                #cell_input_b = torch.cat((mlp_input_b[:, t, :], sum_traj_a[:, t, :]), 1)
            else:
                cell_input_b = input_b[:, t, :]
            if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
                out_b = self.cell(cell_input_b, out_b)
            else:
                out_b, state_b = self.cell(cell_input_b, (out_b, state_b))
            outputs_b.append(out_b.unsqueeze(0))
            # lstm_result.append(out.unsqueeze(1))
        mask_out_b = []
        for b, v in enumerate(input_len_b):
            mask_out_b.append(outputs_b[v-1][0][b,:].view(1,-1))
        fb_outputs = torch.cat(mask_out_b, dim=0)


        #w_clamped = self.sig(self.wweight)
        # print(w_clamped)
        #final_outputs = fi_outputs * (1.0-self.wweight) + f_outputs * self.wweight
        #final_outputs = f_outputs
        #print(self.wweight.data)
        #final_outputs = fi_outputs #* 0.5 + f_outputs * 0.5
        # final_outputs = self.final_l(torch.cat((f_outputs, fi_outputs), 1))

        # return f_outputs
        if topn_p is not None:
            return fa_outputs, fb_outputs, (1.0 - torch.exp(-(topn_p_loss + neg_p_loss)))
        else:
            # return fa_outputs, fb_outputs, 0.0
            # print(fa_outputs.shape)
            # print(fb_outputs.shape)
            return fa_outputs, fb_outputs, torch.cat(outputs_a, dim=0), torch.cat(outputs_b, dim=0)
        # return torch.cat(mask_out, dim = 0)


class Traj_Network(Module):
    def __init__(self, input_size, target_size, grid_size, batch_size, sampling_num, stard_LSTM = False, incell = True):
        super(Traj_Network, self).__init__()
        self.input_size = input_size
        self.target_size = target_size
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        # if config.recurrent_unit=='GRU' or config.recurrent_unit=='SimpleRNN':
        #     self.hidden = autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
        #                                      requires_grad=False).cuda()
        # else:
        #     self.hidden1 = (autograd.Variable(torch.zeros(self.batch_size*(1+self.sampling_num), self.target_size),requires_grad=False).cuda(),
        #               autograd.Variable(torch.zeros(self.batch_size*(1+self.sampling_num), self.target_size),requires_grad=False).cuda())
        #     self.hidden2 = (autograd.Variable(torch.zeros(self.batch_size * (1+self.sampling_num), self.target_size),
        #                                      requires_grad=False).cuda(),
        #                    autograd.Variable(torch.zeros(self.batch_size * (1+self.sampling_num), self.target_size),
        #                                      requires_grad=False).cuda())
        if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
            self.hidden = autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                            requires_grad=False).cuda()
        else:
            self.hidden = (autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                             requires_grad=False).cuda(),
                           autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                             requires_grad=False).cuda())
        self.rnn = RNNEncoder(self.input_size, self.target_size, self.grid_size, stard_LSTM=stard_LSTM,
                              incell=incell).cuda()
        self.smn = SMNEncoder(self.input_size, self.target_size, self.grid_size, stard_LSTM=stard_LSTM,
                              incell=incell).cuda()

    def forward(self, inputs_arrays, inputs_len_arrays):
        anchor_input = torch.Tensor(inputs_arrays[0])
        trajs_input = torch.Tensor(inputs_arrays[1])
        negative_input = torch.Tensor(inputs_arrays[2])

        anchor_input_len = inputs_len_arrays[0]
        trajs_input_len = inputs_len_arrays[1]
        negative_input_len = inputs_len_arrays[2]

        anchor_embedding = self.rnn([autograd.Variable(anchor_input, requires_grad=False).cuda(), anchor_input_len],
                                    self.hidden)
        trajs_embedding = self.rnn([autograd.Variable(trajs_input, requires_grad=False).cuda(), trajs_input_len],
                                   self.hidden)
        negative_embedding = self.rnn(
            [autograd.Variable(negative_input, requires_grad=False).cuda(), negative_input_len], self.hidden)

        trajs_loss = torch.exp(-F.pairwise_distance(anchor_embedding, trajs_embedding, p=2))
        negative_loss = torch.exp(-F.pairwise_distance(anchor_embedding, negative_embedding, p=2))
        return trajs_loss, negative_loss

    def matching_forward(self, inputs_arrays, inputs_len_arrays):
        anchor_input = torch.Tensor(inputs_arrays[0])
        trajs_input = torch.Tensor(inputs_arrays[1])
        negative_input = torch.Tensor(inputs_arrays[2])

        anchor_input_len = inputs_len_arrays[0]
        trajs_input_len = inputs_len_arrays[1]
        negative_input_len = inputs_len_arrays[2]

        #anchor_embedding = self.rnn([autograd.Variable(anchor_input,requires_grad=False).cuda(), anchor_input_len], self.hidden)
        #trajs_embedding = self.rnn([autograd.Variable(trajs_input,requires_grad=False).cuda(), trajs_input_len], self.hidden)
        #negative_embedding = self.rnn([autograd.Variable(negative_input,requires_grad=False).cuda(), negative_input_len], self.hidden)

        #topn_points, neg_points = topn_point_sampling(anchor_input, trajs_input)
        # anchor_embedding, trajs_embedding, outputs_ap, outputs_p = self.smn([autograd.Variable(anchor_input,requires_grad=False).cuda(), anchor_input_len], [autograd.Variable(trajs_input,requires_grad=False).cuda(), trajs_input_len], self.hidden1, self.hidden2, None, None) #topn_points.cuda(), neg_points.cuda())
        if config.trmModel == True:
            anchor_embedding, trajs_embedding, outputs_ap, outputs_p = self.smn.trm_forward(
                [autograd.Variable(anchor_input, requires_grad=False).cuda(), anchor_input_len],
                [autograd.Variable(trajs_input, requires_grad=False).cuda(), trajs_input_len])
            trajs_loss = torch.exp(-F.pairwise_distance(anchor_embedding, trajs_embedding, p=2))
            anchor_embedding, negative_embedding, outputs_an, outputs_n = self.smn.trm_forward(
                [autograd.Variable(anchor_input, requires_grad=False).cuda(), anchor_input_len],
                [autograd.Variable(negative_input, requires_grad=False).cuda(), negative_input_len])
            negative_loss = torch.exp(-F.pairwise_distance(anchor_embedding, negative_embedding, p=2))
        else:
            anchor_embedding, trajs_embedding, outputs_ap, outputs_p = self.smn.f(
                [autograd.Variable(anchor_input, requires_grad=False).cuda(), anchor_input_len],
                [autograd.Variable(trajs_input, requires_grad=False).cuda(), trajs_input_len])
            trajs_loss = torch.exp(-F.pairwise_distance(anchor_embedding, trajs_embedding, p=2))
            # trajs_loss = F.cosine_similarity(anchor_embedding, trajs_embedding)
            #anchor_embedding, negative_embedding, outputs_an, outputs_n = self.smn([autograd.Variable(anchor_input,requires_grad=False).cuda(), anchor_input_len], [autograd.Variable(negative_input,requires_grad=False).cuda(), negative_input_len], self.hidden1, self.hidden2, None, None)
            anchor_embedding, negative_embedding, outputs_an, outputs_n = self.smn.f(
                [autograd.Variable(anchor_input, requires_grad=False).cuda(), anchor_input_len],
                [autograd.Variable(negative_input, requires_grad=False).cuda(), negative_input_len])
            negative_loss = torch.exp(-F.pairwise_distance(anchor_embedding, negative_embedding, p=2))
            # negative_loss = F.cosine_similarity(anchor_embedding, negative_embedding)
        return trajs_loss, negative_loss, outputs_ap, outputs_p, outputs_an, outputs_n

    def trm_forward(self, inputs_arrays, inputs_len_arrays):
        anchor_input = torch.Tensor(inputs_arrays[0])
        trajs_input = torch.Tensor(inputs_arrays[1])
        negative_input = torch.Tensor(inputs_arrays[2])

        anchor_input_len = inputs_len_arrays[0]
        trajs_input_len = inputs_len_arrays[1]
        negative_input_len = inputs_len_arrays[2]

        #anchor_embedding = self.rnn([autograd.Variable(anchor_input,requires_grad=False).cuda(), anchor_input_len], self.hidden)
        #trajs_embedding = self.rnn([autograd.Variable(trajs_input,requires_grad=False).cuda(), trajs_input_len], self.hidden)
        #negative_embedding = self.rnn([autograd.Variable(negative_input,requires_grad=False).cuda(), negative_input_len], self.hidden)

        #topn_points, neg_points = topn_point_sampling(anchor_input, trajs_input)
        # anchor_embedding, trajs_embedding, outputs_ap, outputs_p = self.smn([autograd.Variable(anchor_input,requires_grad=False).cuda(), anchor_input_len], [autograd.Variable(trajs_input,requires_grad=False).cuda(), trajs_input_len], self.hidden1, self.hidden2, None, None) #topn_points.cuda(), neg_points.cuda())
        anchor_embedding, trajs_embedding, outputs_ap, outputs_p = self.smn.trm_forward(
            [autograd.Variable(anchor_input, requires_grad=False).cuda(), anchor_input_len],
            [autograd.Variable(trajs_input, requires_grad=False).cuda(), trajs_input_len])
        trajs_loss = torch.exp(-F.pairwise_distance(anchor_embedding, trajs_embedding, p=2))
        # trajs_loss = F.cosine_similarity(anchor_embedding, trajs_embedding)
        #anchor_embedding, negative_embedding, outputs_an, outputs_n = self.smn([autograd.Variable(anchor_input,requires_grad=False).cuda(), anchor_input_len], [autograd.Variable(negative_input,requires_grad=False).cuda(), negative_input_len], self.hidden1, self.hidden2, None, None)
        anchor_embedding, negative_embedding, outputs_an, outputs_n = self.smn.trm_forward(
            [autograd.Variable(anchor_input, requires_grad=False).cuda(), anchor_input_len],
            [autograd.Variable(negative_input, requires_grad=False).cuda(), negative_input_len])
        negative_loss = torch.exp(-F.pairwise_distance(anchor_embedding, negative_embedding, p=2))
        # negative_loss = F.cosine_similarity(anchor_embedding, negative_embedding)
        return trajs_loss, negative_loss, outputs_ap, outputs_p, outputs_an, outputs_n

    def t2s_forward(self, inputs_arrays, inputs_len_arrays):
        anchor_input = torch.Tensor(inputs_arrays[0])
        trajs_input = torch.Tensor(inputs_arrays[1])
        negative_input = torch.Tensor(inputs_arrays[2])

        anchor_input_len = inputs_len_arrays[0]
        trajs_input_len = inputs_len_arrays[1]
        negative_input_len = inputs_len_arrays[2]

        #anchor_embedding = self.rnn([autograd.Variable(anchor_input,requires_grad=False).cuda(), anchor_input_len], self.hidden)
        #trajs_embedding = self.rnn([autograd.Variable(trajs_input,requires_grad=False).cuda(), trajs_input_len], self.hidden)
        #negative_embedding = self.rnn([autograd.Variable(negative_input,requires_grad=False).cuda(), negative_input_len], self.hidden)

        #topn_points, neg_points = topn_point_sampling(anchor_input, trajs_input)
        # anchor_embedding, trajs_embedding, outputs_ap, outputs_p = self.smn([autograd.Variable(anchor_input,requires_grad=False).cuda(), anchor_input_len], [autograd.Variable(trajs_input,requires_grad=False).cuda(), trajs_input_len], self.hidden1, self.hidden2, None, None) #topn_points.cuda(), neg_points.cuda())
        anchor_embedding, trajs_embedding, outputs_ap, outputs_p = self.smn.t2s_forward(
            [autograd.Variable(anchor_input, requires_grad=False).cuda(), anchor_input_len],
            [autograd.Variable(trajs_input, requires_grad=False).cuda(), trajs_input_len])
        trajs_loss = torch.exp(-F.pairwise_distance(anchor_embedding, trajs_embedding, p=2))
        # trajs_loss = F.cosine_similarity(anchor_embedding, trajs_embedding)
        #anchor_embedding, negative_embedding, outputs_an, outputs_n = self.smn([autograd.Variable(anchor_input,requires_grad=False).cuda(), anchor_input_len], [autograd.Variable(negative_input,requires_grad=False).cuda(), negative_input_len], self.hidden1, self.hidden2, None, None)
        anchor_embedding, negative_embedding, outputs_an, outputs_n = self.smn.t2s_forward(
            [autograd.Variable(anchor_input, requires_grad=False).cuda(), anchor_input_len],
            [autograd.Variable(negative_input, requires_grad=False).cuda(), negative_input_len])
        negative_loss = torch.exp(-F.pairwise_distance(anchor_embedding, negative_embedding, p=2))
        # negative_loss = F.cosine_similarity(anchor_embedding, negative_embedding)
        return trajs_loss, negative_loss, outputs_ap, outputs_p, outputs_an, outputs_n

    def spatial_memory_update(self, inputs_arrays, inputs_len_arrays):
        batch_traj_input = torch.Tensor(inputs_arrays[3])
        batch_traj_len = inputs_len_arrays[3]
        batch_hidden = (autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size),requires_grad=False).cuda(),
                        autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size),requires_grad=False).cuda())
        self.rnn.batch_grid_state_gates([autograd.Variable(batch_traj_input).cuda(), batch_traj_len],batch_hidden)
