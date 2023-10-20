import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, GATConv
import torch_geometric

class FourierEncoder(torch.nn.Module):
    def __init__(self, level, include_self = True):
        super(FourierEncoder, self).__init__()
        self.level = level
        self.include_self = include_self
    def multiscale(self, x, scales):
        return torch.hstack([x / i for i in scales])
    def forward(self, x):
        device, dtype, orig_x = x.device, x.dtype, x
        scales = 2 ** torch.arange(-self.level/2, self.level/2, device = device, dtype = dtype)
        lifted_feature = torch.cat((torch.sin(self.multiscale(x, scales)), torch.cos(self.multiscale(x, scales))), 1)
        return lifted_feature

class PreNormException(Exception):
    pass

class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super(PreNormLayer, self).__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False
    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException
        if self.shift is not None:
            input_ = input_ + self.shift
        if self.scale is not None:
            input_ = input_ * self.scale
        return input_
    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False
    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input_.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."
        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size())/self.n_units
        delta = sample_avg - self.avg
        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)
        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1
    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg
        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)
        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False

class BipartiteGraphConv(torch_geometric.nn.MessagePassing):
    def __init__(self, emb_size):
        super(BipartiteGraphConv, self).__init__('add')
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size))
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False))
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False))
        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size))
        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(1, shift=False))
        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),)
    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))
    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i)
                                           + self.feature_module_edge(edge_features)
                                           + self.feature_module_right(node_features_j))
        return output

class Encoder(nn.Module):
    def __init__(self, input_dim_xs, input_dim_xt, input_dim_edge, num_layers, hidden_dim, mlp_hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        # Embedding of node feature
        self.xs_embedding = FourierEncoder(hidden_dim/(2*input_dim_xs))
        self.xt_embedding = FourierEncoder(hidden_dim/(2*input_dim_xt))
        
        # Embedding of edge feature
        #self.edge_embedding = FourierEncoder(hidden_dim/(2*input_dim_edge))
        self.conv_s_t = nn.ModuleList()
        self.conv_t_s = nn.ModuleList()
        for _ in range(num_layers):
            conv_s_t = BipartiteGraphConv(hidden_dim)
            self.conv_s_t.append(conv_s_t)
        for _ in range(num_layers):
            conv_t_s = BipartiteGraphConv(hidden_dim)
            self.conv_t_s.append(conv_t_s)
        self.mlp_xs_mu = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden_dim),
                                 nn.BatchNorm1d(mlp_hidden_dim),
                                 nn.ReLU())
        self.mlp_xt_mu = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden_dim),
                                 nn.BatchNorm1d(mlp_hidden_dim),
                                 nn.ReLU())
        
        self.mlp_xs_logsigma = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden_dim),
                                 nn.BatchNorm1d(mlp_hidden_dim),
                                 nn.ReLU())
        self.mlp_xt_logsigma = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden_dim),
                                 nn.BatchNorm1d(mlp_hidden_dim),
                                 nn.ReLU())
        
        self.fc_xs_mu = nn.Linear(mlp_hidden_dim, 1)
        self.fc_xt_mu = nn.Linear(mlp_hidden_dim, 1)
        self.fc_xs_logsigma = nn.Linear(mlp_hidden_dim, 1)
        self.fc_xt_logsigma = nn.Linear(mlp_hidden_dim, 1)

    def reparametrize(self, mu, log_sigma):
        if self.training:
            return mu + torch.randn_like(log_sigma) * torch.exp(log_sigma)
        else:
            return mu

    def forward(self, x_s, x_t, edge_index, edge_attr):

        x_s = self.xs_embedding(x_s)
        x_t = self.xt_embedding(x_t)
        #edge_attr = self.edge_embedding(edge_attr)

        inverse_edge_index = edge_index.clone()
        inverse_edge_index[[0,1]] = edge_index[[1,0]]
        for conv_s_t, conv_t_s in zip(self.conv_s_t, self.conv_t_s):
            new_x_t = conv_s_t(x_s, edge_index, edge_attr, x_t)
            new_x_s = conv_t_s(x_t, inverse_edge_index, edge_attr, x_s)
            x_t = new_x_t
            x_s = new_x_s


        xs_mu = self.mlp_xs_mu(x_s)
        #xs_mu = F.dropout(xs_mu, p = 0.1, training = self.training)
        xs_mu = self.fc_xs_mu(xs_mu)

        xt_mu = self.mlp_xt_mu(x_t)
        xt_mu = self.fc_xt_mu(xt_mu)

        xs_logsigma = self.mlp_xs_logsigma(x_s)
        #x2 = F.dropout(x2, p = 0.1, training = self.training)
        xs_logsigma = self.fc_xs_logsigma(xs_logsigma)

        xt_logsigma = self.mlp_xt_logsigma(x_t)
        xt_logsigma = self.fc_xt_logsigma(xt_logsigma)

        xs_z = self.reparametrize(xs_mu, xs_logsigma)
        xt_z = self.reparametrize(xt_mu, xt_logsigma)

        return xs_mu, xs_logsigma, xs_z, xt_mu, xt_logsigma, xt_z
    
class Decoder(nn.Module):
    def __init__(self, input_dim_xs, input_dim_xt, input_dim_edge, num_layers, hidden_dim, mlp_hidden_dim, mlp_out_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        # Embedding of node feature
        self.xs_embedding = FourierEncoder(hidden_dim/(2*input_dim_xs))
        self.xt_embedding = FourierEncoder(hidden_dim/(2*input_dim_xt))

        self.conv_s_t = nn.ModuleList()
        self.conv_t_s = nn.ModuleList()
        for _ in range(num_layers):
            conv_s_t = BipartiteGraphConv(hidden_dim)
            self.conv_s_t.append(conv_s_t)
        for _ in range(num_layers):
            conv_t_s = BipartiteGraphConv(hidden_dim)
            self.conv_t_s.append(conv_t_s)

        self.mlp_hxs = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden_dim),
                                 nn.BatchNorm1d(mlp_hidden_dim),
                                 nn.ReLU())
        self.mlp_hxt = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden_dim),
                                 nn.BatchNorm1d(mlp_hidden_dim),
                                 nn.ReLU())

        self.fc_hxs = nn.Linear(mlp_hidden_dim, mlp_out_dim)
        self.fc_hxt = nn.Linear(mlp_hidden_dim, mlp_out_dim)

        # predict degree
        '''self.mlp_degree = nn.Sequential(nn.Linear(mlp_out_dim+1, mlp_hidden_dim),
                                        nn.ReLU(), 
                                        nn.Linear(mlp_hidden_dim, 1),
                                 nn.Sigmoid())'''
        self.mlp_degree = nn.Sequential(nn.Linear(mlp_out_dim+1, mlp_hidden_dim),
                                        nn.ReLU(), 
                                        nn.Linear(mlp_hidden_dim, 1))
        # predict logits
        '''self.mlp_logits = nn.Sequential(nn.Linear(mlp_out_dim+1, mlp_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(mlp_hidden_dim, 1),
                                 nn.Sigmoid())'''
        self.mlp_logits = nn.Sequential(nn.Linear(mlp_out_dim+1, mlp_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(mlp_hidden_dim, 1))
        
        # predict number_x_1
        self.mlp_number_x = nn.Sequential(nn.Linear(mlp_out_dim+1, mlp_hidden_dim),
                                        nn.ReLU())
        self.fc_number_x = nn.Sequential(nn.Linear(mlp_hidden_dim, 1))

        # predict weights
        '''self.mlp_weights = nn.Sequential(nn.Linear(mlp_out_dim+1, mlp_hidden_dim),
                                         nn.ReLU(),
                                        nn.Linear(mlp_hidden_dim, 1),
                                        nn.Sigmoid())'''
        self.mlp_weights = nn.Sequential(nn.Linear(mlp_out_dim+1, mlp_hidden_dim),
                                         nn.ReLU(),
                                        nn.Linear(mlp_hidden_dim, 1))

        # predict x
        '''self.mlp_x = nn.Sequential(nn.Linear(mlp_out_dim+1, 
                                             mlp_hidden_dim),nn.ReLU(),
                                             nn.Linear(mlp_hidden_dim, 1),
                                 nn.Sigmoid())'''
        self.mlp_x = nn.Sequential(nn.Linear(mlp_out_dim+1, 
                                             mlp_hidden_dim),nn.ReLU(),
                                             nn.Linear(mlp_hidden_dim, 1))

        # predict ym
        '''self.mlp_ym = nn.Sequential(nn.Linear(mlp_out_dim+1, 
                                             mlp_hidden_dim),nn.ReLU(),
                                             nn.Linear(mlp_hidden_dim, 1),
                                 nn.Sigmoid())'''
        self.mlp_ym = nn.Sequential(nn.Linear(mlp_out_dim+1, 
                                             mlp_hidden_dim),nn.ReLU(),
                                             nn.Linear(mlp_hidden_dim, 1))
        
        # predict yn
        '''self.mlp_yn = nn.Sequential(nn.Linear(mlp_out_dim+1, 
                                             mlp_hidden_dim),nn.ReLU(),
                                             nn.Linear(mlp_hidden_dim, 1),
                                 nn.Sigmoid())'''
        self.mlp_yn = nn.Sequential(nn.Linear(mlp_out_dim+1, 
                                             mlp_hidden_dim),nn.ReLU(),
                                             nn.Linear(mlp_hidden_dim, 1))

        # predict s
        '''self.mlp_s = nn.Sequential(nn.Linear(mlp_out_dim+1, 
                                             mlp_hidden_dim),nn.ReLU(),
                                             nn.Linear(mlp_hidden_dim, 1),
                                 nn.Sigmoid())'''
        self.mlp_s = nn.Sequential(nn.Linear(mlp_out_dim+1, 
                                             mlp_hidden_dim),nn.ReLU(),
                                             nn.Linear(mlp_hidden_dim, 1))

        # predict rm
        '''self.mlp_rm = nn.Sequential(nn.Linear(mlp_out_dim+1, 
                                             mlp_hidden_dim),nn.ReLU(),
                                             nn.Linear(mlp_hidden_dim, 1),
                                 nn.Sigmoid())'''
        self.mlp_rm = nn.Sequential(nn.Linear(mlp_out_dim+1, 
                                             mlp_hidden_dim),nn.ReLU(),
                                             nn.Linear(mlp_hidden_dim, 1))
        
        # predict rn
        '''self.mlp_rn = nn.Sequential(nn.Linear(mlp_out_dim+1, 
                                             mlp_hidden_dim),nn.ReLU(),
                                             nn.Linear(mlp_hidden_dim, 1),
                                 nn.Sigmoid())'''
        self.mlp_rn = nn.Sequential(nn.Linear(mlp_out_dim+1, 
                                             mlp_hidden_dim),nn.ReLU(),
                                             nn.Linear(mlp_hidden_dim, 1))


    def forward(self, masked_x_s, masked_x_t, masked_edge_index, masked_edge_attr, xs_z, xt_z, batch):
        masked_x_s = self.xs_embedding(masked_x_s)
        masked_x_t = self.xt_embedding(masked_x_t)
        #edge_attr = self.edge_embedding(edge_attr)

        inverse_edge_index = masked_edge_index.clone()
        inverse_edge_index[[0,1]] = masked_edge_index[[1,0]]
        for conv_s_t, conv_t_s in zip(self.conv_s_t, self.conv_t_s):
            new_x_t = conv_s_t(masked_x_s, masked_edge_index, masked_edge_attr, masked_x_t)
            new_x_s = conv_t_s(masked_x_t, inverse_edge_index, masked_edge_attr, masked_x_s)
            masked_x_t = new_x_t
            masked_x_s = new_x_s

        h_xs = self.mlp_hxs(masked_x_s)
        h_xs = self.fc_hxs(h_xs)

        h_xt = self.mlp_hxt(masked_x_t)
        h_xt = self.fc_hxt(h_xt)

        hz_xs = torch.cat([h_xs, xs_z], 1)
        hz_xt = torch.cat([h_xt, xt_z], 1)
        # predict the degree mlp[hs,zs]
        predict_degree = self.mlp_degree(hz_xs)
        # predict the logits mlp[ht,zt]
        predict_logits = self.mlp_logits(hz_xt)
        # predict the weights mlp[ht,zt]
        predict_weights = self.mlp_weights(hz_xt)
        # predict the number of x that takes 1 linear(pooling(mlp[hz_xt]))
        predict_num_x = self.mlp_number_x(hz_xt)
        predict_num_x = global_mean_pool(predict_num_x, batch)
        predict_num_x = self.fc_number_x(predict_num_x)
        # predict x mlp[ht,zt]
        predict_x = self.mlp_x(hz_xt)
        # predict ym mlp[hs,zs]
        predict_ym = self.mlp_ym(hz_xs)
        # predict yn mlp[ht,zt]
        predict_yn = self.mlp_yn(hz_xt)
        # predict s mlp[ht,zt]
        predict_s = self.mlp_s(hz_xt)
        # predict rm mlp[hs,zs]
        predict_rm = self.mlp_rm(hz_xs)
        # predict rn mlp[ht,zt]
        predict_rn = self.mlp_rn(hz_xt)

        return predict_degree, predict_logits, predict_weights, predict_num_x, predict_x, predict_ym, predict_yn, predict_s, predict_rm, predict_rn