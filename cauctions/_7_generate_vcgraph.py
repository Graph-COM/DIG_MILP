import argparse
from tqdm import tqdm
import time
from pathlib import Path
import pickle
import numpy as np

import ecole
from ecole.observation import MilpBipartite


import torch
from torch_geometric.data import InMemoryDataset, Data

class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None, edge_attr = None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
        self.edge_attr = edge_attr
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)
    @property
    def num_nodes(self):
        return self.x_s.size(0) + self.x_t.size(0)

class CauctionsData(InMemoryDataset):
    def __init__(self, primal_format, num_instance, save_folder, primal_solution, primal_slack, dual_solution, dual_slack, normalize_statistics):
        self.save_folder = Path(save_folder)
        self.num_instance = num_instance
        self.primal_format = primal_format
        self.primal_solution = primal_solution
        self.primal_slack = primal_slack
        self.dual_solution = dual_solution
        self.dual_slack = dual_slack
        self.normalize_statistics = normalize_statistics
        super(CauctionsData, self).__init__(root = self.save_folder)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['data.pt']
    def download(self):
        pass
    def process(self):
        #load normalize statistics
        with open(self.normalize_statistics+'normalize_weight.pkl', 'rb') as f:
            weight_dict = pickle.load(f)
        #observation_function = MilpBipartite()
        data_list = []
        env = ecole.environment.Configuring(observation_function = MilpBipartite())
        for instance_idx in tqdm(range(self.num_instance)):
            obs, _, _, _, _ = env.reset(self.primal_format+str(instance_idx)+'.lp')
            # obs.variable_features: coefficient c | variable type | has_lower bound | has upper bound | lower bound | upper bound
            # obs.constraint_features:  b
            # obs.edge_features.indices: edge_index of the V-C bipartitr graph [0]constraints [1]variables
            
            num_constraints = obs.constraint_features.shape[0]
            num_variables = obs.variable_features.shape[0]
            edge_index = torch.from_numpy(obs.edge_features.indices.astype(np.int32)).long()
            edge_attr = torch.from_numpy(obs.edge_features.values.reshape(-1, 1).astype(np.float32))
            if weight_dict['equal'] == 1:
                toward1 = weight_dict['toward1']
                edge_attr = edge_attr + toward1
            else:
                edge_attr = (edge_attr - weight_dict['min_weight']) / (weight_dict['max_weight'] - weight_dict['min_weight'])
            # in the V-C bipartite graph, the features are as follows:
            # edge feature: aij
            # constraint feature: 0, y1-ym, r1-rm
            # variable feature: 1, x1-xn, ym+1-ym+n, s1-sn, rm+1-rm+n
            with open (self.primal_solution+str(instance_idx)+'.pkl', 'rb') as primal_solution_file:
                x = pickle.load(primal_solution_file)
            with open (self.primal_slack+str(instance_idx)+'.pkl', 'rb') as primal_slack_file:
                r = pickle.load(primal_slack_file)
            with open (self.dual_solution+str(instance_idx)+'.pkl', 'rb') as dual_solution_file:
                y = pickle.load(dual_solution_file)
            with open (self.dual_slack+str(instance_idx)+'.pkl', 'rb') as dual_slack_file:
                s = pickle.load(dual_slack_file)
            
            x_constraints = torch.zeros((num_constraints,3))
            x_variables  = torch.zeros((num_variables,5))

            for idx in range(num_constraints):
                x_constraints[idx,0] = 0.0
                x_constraints[idx,1] = float(y[idx+1])
                x_constraints[idx,2] = float(r[idx+1])
                #x_constraints[idx,3] = float(obs.constraint_features[idx])
            for idx in range(num_variables):
                x_variables[idx,0] = 1.0
                x_variables[idx,1] = float(x[idx+1])
                x_variables[idx,2] = float(y[idx+num_constraints+1])
                x_variables[idx,3] = float(s[idx+1])
                x_variables[idx,4] = float(r[idx+num_constraints+1])
                #x_variables[idx,5] = float(obs.variable_features[idx,0])
            data = BipartiteData(x_s=x_constraints, x_t=x_variables, edge_index=edge_index, edge_attr = edge_attr)
            data_list.append(data)
        # Concatenate the list of `Data` objects into a single `Data` object
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--primal_format', dest = 'primal_format', type = str, default = './data/primal_format/', help = 'which folder to get the primal instances')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 1000, help = 'the number of instances')
    parser.add_argument('--save_folder', dest = 'save_folder', type = str, default = './graph_dataset/', help = 'the folder to save the graph data')
    parser.add_argument('--primal_solution', dest = 'primal_solution', type = str, default = './data/normalize_primal_solution/', help = 'the folder to get the primal solution')
    parser.add_argument('--primal_slack', dest = 'primal_slack', type = str, default = './data/normalize_primal_slack/', help = 'the folder to get the primal slack')
    parser.add_argument('--dual_solution', dest = 'dual_solution', type = str, default = './data/normalize_dual_solution/', help = 'the folder to get the dual solution')
    parser.add_argument('--dual_slack', dest = 'dual_slack', type = str, default = './data/normalize_dual_slack/', help = 'the folder to get the dual slack')
    parser.add_argument('--normalize_statistics', dest = 'normalize_statistics', type = str, default = './data/normalize_statistics/', help = 'the folder to get the normalize data')
    args = parser.parse_args()
    

    dataset = CauctionsData(args.primal_format, args.num_instance, args.save_folder, args.primal_solution, args.primal_slack, args.dual_solution, args.dual_slack, args.normalize_statistics)
        
if __name__ == '__main__':
    main()