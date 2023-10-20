import argparse
from tqdm import tqdm
import time
from pathlib import Path
import pickle
import numpy as np
import random
import pyscipopt
from tqdm import tqdm

import ecole
from ecole.observation import MilpBipartite


import torch
from torch_geometric.data import InMemoryDataset, Data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None, edge_attr = None, y = None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
        self.edge_attr = edge_attr
        self.y = y
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)
    @property
    def num_nodes(self):
        return self.x_s.size(0) + self.x_t.size(0)

class SetCoverDownstreamVal(InMemoryDataset):
    def __init__(self, primal, num_instance, density_list, save_folder):
        self.save_folder = Path(save_folder)
        self.density = density_list
        self.num_instance = num_instance
        self.primal = primal
        super(SetCoverDownstreamVal, self).__init__(root = self.save_folder)
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
        # choose the instance index of the original primal and the generated primal instances
        
        # generate the data
        data_list = []
        env = ecole.environment.Configuring(observation_function = MilpBipartite())
        for instance_idx in tqdm(range(self.num_instance)):
            obs, _, _, _, _ = env.reset(self.primal+str(instance_idx)+'.lp')
            # obs.variable_features: coefficient c | variable type | has_lower bound | has upper bound | lower bound | upper bound
            # obs.constraint_features:  b
            # obs.edge_features.indices: edge_index of the V-C bipartitr graph [0]constraints [1]variables
            edge_index = torch.from_numpy(obs.edge_features.indices.astype(np.int32)).long()
            edge_attr = torch.from_numpy(obs.edge_features.values.reshape(-1, 1).astype(np.float32))  
            # in the V-C bipartite graph, the features are as follows:
            # edge feature: aij
            # constraint feature: 0, normalized_b
            # variable feature: 1, coefficient c  
            num_constraints = obs.constraint_features.shape[0]
            num_variables = obs.variable_features.shape[0]
            x_constraints = torch.zeros((num_constraints,2))
            x_variables  = torch.zeros((num_variables,2))

            for idx in range(num_constraints):
                x_constraints[idx,0] = 0.0
                x_constraints[idx,1] = obs.constraint_features[idx,0]
            for idx in range(num_variables):
                x_variables[idx,0] = 1.0
                x_variables[idx,1] = -obs.variable_features[idx,0]

            solver = pyscipopt.Model()
            solver.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
            solver.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
            solver.disablePropagation()
            solver.setIntParam('display/verblevel', 0)
            solver.readProblem(self.primal+str(instance_idx)+'.lp')
            solver.optimize()

            y = solver.getObjVal()
            print(y)

            data = BipartiteData(x_s=x_constraints, x_t=x_variables, edge_index=edge_index, edge_attr = edge_attr, y = y)
            data_list.append(data)

        
        # save the data and the normalize statistics
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--primal', dest = 'primal', type = str, default = './testset/m200n400/', help = 'which folder to get the primal instances')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 500, help = 'the number of instances')
    parser.add_argument('--seed', dest = 'seed', type = int, default = 123, help = 'the random seed')
    parser.add_argument('--density', dest = 'density', type = float, default = [0.15, 0.20, 0.25, 0.30,0.35], help = 'density')
    parser.add_argument('--save_folder', dest = 'save_folder', type = str, default = './testset/m200n400/', help = 'the folder to save the graph data')
    args = parser.parse_args()
    
    setup_seed(args.seed)
    primal = args.primal+'val/'
    save_folder = args.save_folder+'val/'
    dataset = SetCoverDownstreamVal(primal, args.num_instance, args.density, save_folder)
        
if __name__ == '__main__':
    main()