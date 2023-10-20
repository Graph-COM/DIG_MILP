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

class Predictor(nn.Module):
    def __init__(self, input_dim_xs, input_dim_xt, input_dim_edge, num_layers, hidden_dim, mlp_hidden_dim):
        super(Predictor, self).__init__()
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
        self.mlp_xs = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden_dim),
                                 nn.ReLU())
        self.mlp_xt = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden_dim),
                                 nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                                nn.ReLU(),
                                nn.Linear(mlp_hidden_dim, 1))

    def forward(self, x_s, x_t, edge_index, edge_attr, batch_xs, batch_xt):

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

        x_s = self.mlp_xs(x_s)
        x_t = self.mlp_xt(x_t)
        predict_y_s = global_mean_pool(x_s, batch_xs)
        predict_y_t = global_mean_pool(x_t, batch_xt)
        predict_y = (predict_y_s + predict_y_t) / 2
        predict_y = self.fc(predict_y)

        

        return predict_y