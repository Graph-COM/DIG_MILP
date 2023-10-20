import argparse
from tqdm import tqdm
import time
from pathlib import Path
import pickle
import numpy as np
import random
import pyscipopt


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

class SetCoverDownstream(InMemoryDataset):
    def __init__(self, primal, generate_primal, num_instance, ratio, save_folder, primal_solution, generate_primal_solution, normalize_statistics):
        self.save_folder = Path(save_folder)
        self.ratio = ratio
        self.num_instance = num_instance
        self.primal = primal
        self.generate_primal = generate_primal
        self.primal_solution = primal_solution
        self.generate_primal_solution = generate_primal_solution
        self.normalize_statistics = normalize_statistics
        super(SetCoverDownstream, self).__init__(root = self.save_folder)
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
        num_orginal = int(self.ratio * self.num_instance) 
        indices = list(range(self.num_instance))
        num_generate = self.num_instance - num_orginal
        #num_generate = 1000
        original_idx_list = random.sample(indices, num_orginal)
        generate_idx_list = random.sample(indices, num_generate)
        #generate_idx_list = range(500)

        #calculate normalize statistics
        max_b = -10000
        min_b = 10000
        max_c = -10000
        min_c = 10000
        max_weight = -10000
        min_weight = 10000
        max_y = -10000
        min_y = 10000
        y_dict = {}

        if len(original_idx_list) !=0:
            for instance_idx in tqdm(original_idx_list):
                c_list = []
                solver = pyscipopt.Model()
                solver.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
                solver.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
                solver.disablePropagation()
                solver.setIntParam('display/verblevel', 0)
                solver.readProblem(self.primal+str(instance_idx)+'.lp')
                constraints = solver.getConss()
                num_constraints = 0
                # get b and weight
                for constraint_idx in constraints:
                    if str(constraint_idx).startswith('C'):
                        rhs = solver.getRhs(constraint_idx)
                        if max_b < rhs:
                            max_b = rhs
                        if min_b > rhs:
                            min_b = rhs
                        coeff_dict = solver.getValsLinear(constraint_idx)
                        this_max = max(coeff_dict.values())
                        if this_max > max_weight:
                            max_weight = this_max
                        this_min = min(coeff_dict.values())
                        if this_min < min_weight:
                            min_weight = this_min
                # get c
                objective = solver.getObjective()
                for variable in objective:
                    this_c = objective[variable]
                    if max_c < this_c:
                        max_c = this_c
                    if min_c > this_c:
                        min_c = this_c
                    c_list.append(this_c)
                # get y
                with open (self.primal_solution+str(instance_idx)+'.pkl', 'rb') as primal_solution_file:
                    x = pickle.load(primal_solution_file)
                    this_obj = 0
                    for i in range(len(x)):
                        this_obj = this_obj + x[i+1] * c_list[i]
                    y_dict['original'+str(instance_idx)] = this_obj
                    if this_obj > max_y:
                        max_y = this_obj
                    if this_obj < min_y:
                        min_y = this_obj

        if len(generate_idx_list) != 0:
            for instance_idx in tqdm(generate_idx_list):
                c_list = []
                solver = pyscipopt.Model()
                solver.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
                solver.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
                solver.disablePropagation()
                solver.setIntParam('display/verblevel', 0)
                solver.readProblem(self.generate_primal+str(instance_idx)+'.lp')
                constraints = solver.getConss()
                num_constraints = 0
                # get b and weight
                for constraint_idx in constraints:
                    if str(constraint_idx).startswith('C'):
                        rhs = solver.getRhs(constraint_idx)
                        if max_b < rhs:
                            max_b = rhs
                        if min_b > rhs:
                            min_b = rhs
                        coeff_dict = solver.getValsLinear(constraint_idx)
                        this_max = max(coeff_dict.values())
                        if this_max > max_weight:
                            max_weight = this_max
                        this_min = min(coeff_dict.values())
                        if this_min < min_weight:
                            min_weight = this_min
                # get c
                objective = solver.getObjective()
                for variable in solver.getVars():
                    this_c = objective[variable]
                    if max_c < this_c:
                        max_c = this_c
                    if min_c > this_c:
                        min_c = this_c
                    c_list.append(this_c)
                # get y
                with open (self.generate_primal_solution+str(instance_idx)+'.pkl', 'rb') as generate_primal_solution_file:
                    x = pickle.load(generate_primal_solution_file)
                    this_obj = 0
                    for i in range(len(x)):
                        this_obj = this_obj + x[i+1] * c_list[i]
                    y_dict['generate'+str(instance_idx)] = this_obj
                    if this_obj > max_y:
                        max_y = this_obj
                    if this_obj < min_y:
                        min_y = this_obj


        if max_b == min_b:
            equal_b = 1
            b_toward1 = 1 - min_b
        else:
            equal_b = 0
        if min_c == max_c:
            equal_c = 1
            c_toward1 = 1 - min_c
        else:
            equal_c = 0
        if min_weight == max_weight:
            equal_weight = 1
            weight_toward1 = 1 - min_weight
        else:
            equal_weight = 0
        if min_y == max_y:
            equal_y = 1
            y_toward1 = 1 - min_y
        else:
            equal_y = 0

        normalize_dict = {}
        normalize_dict['max_b'] = max_b
        normalize_dict['min_b'] = min_b
        normalize_dict['max_c'] = max_c
        normalize_dict['min_c'] = min_c
        normalize_dict['max_y'] = max_y
        normalize_dict['min_y'] = min_y
        normalize_dict['max_weight'] = max_weight
        normalize_dict['min_weight'] = min_weight
        
        # generate the data
        data_list = []
        env = ecole.environment.Configuring(observation_function = MilpBipartite())
        if len(original_idx_list) !=0:
            for instance_idx in tqdm(original_idx_list):
                obs, _, _, _, _ = env.reset(self.primal+str(instance_idx)+'.lp')
                # obs.variable_features: coefficient c | variable type | has_lower bound | has upper bound | lower bound | upper bound
                # obs.constraint_features:  b
                # obs.edge_features.indices: edge_index of the V-C bipartitr graph [0]constraints [1]variables
                edge_index = torch.from_numpy(obs.edge_features.indices.astype(np.int32)).long()
                edge_attr = torch.from_numpy(obs.edge_features.values.reshape(-1, 1).astype(np.float32))  
                if equal_weight == 1:
                    edge_attr = edge_attr + weight_toward1
                else:
                    edge_attr = (edge_attr - min_weight) / (max_weight - min_weight)

                              
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

                # normalize b and c and y
                if equal_b == 1:
                    x_constraints[:,1] = x_constraints[:,1] + b_toward1
                else:
                    x_constraints[:,1] = (x_constraints[:,1] - min_b) / (max_b - min_b)
                
                if equal_c == 1:
                    x_variables[:,1] = x_variables[:,1] + c_toward1
                else:
                    x_variables[:,1] = (x_variables[:,1] - min_c) / (max_c - min_c)

                '''if equal_y == 1:
                    y = y_dict['original'+str(instance_idx)] + y_toward1
                else:
                    y = (y_dict['original'+str(instance_idx)] - min_y) / (max_y - min_y)'''
                y = y_dict['original'+str(instance_idx)]

                data = BipartiteData(x_s=x_constraints, x_t=x_variables, edge_index=edge_index, edge_attr = edge_attr, y = y)
                data_list.append(data)

        if len(generate_idx_list) !=0:
            for instance_idx in tqdm(generate_idx_list):
                obs, _, _, _, _ = env.reset(self.generate_primal+str(instance_idx)+'.lp')
                # obs.variable_features: coefficient c | variable type | has_lower bound | has upper bound | lower bound | upper bound
                # obs.constraint_features:  b
                # obs.edge_features.indices: edge_index of the V-C bipartitr graph [0]constraints [1]variables
                edge_index = torch.from_numpy(obs.edge_features.indices.astype(np.int32)).long()
                edge_attr = torch.from_numpy(obs.edge_features.values.reshape(-1, 1).astype(np.float32))  
                if equal_weight == 1:
                    edge_attr = edge_attr + weight_toward1
                else:
                    edge_attr = (edge_attr - min_weight) / (max_weight - min_weight)

                              
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

                # normalize b and c and y
                if equal_b == 1:
                    x_constraints[:,1] = x_constraints[:,1] + b_toward1
                else:
                    x_constraints[:,1] = (x_constraints[:,1] - min_b) / (max_b - min_b)
                
                if equal_c == 1:
                    x_variables[:,1] = x_variables[:,1] + c_toward1
                else:
                    x_variables[:,1] = (x_variables[:,1] - min_c) / (max_c - min_c)

                '''if equal_y == 1:
                    y = y_dict['generate'+str(instance_idx)] + y_toward1
                else:
                    y = (y_dict['generate'+str(instance_idx)] - min_y) / (max_y - min_y)'''
                y = y_dict['generate'+str(instance_idx)]
                print(y)
                data = BipartiteData(x_s=x_constraints, x_t=x_variables, edge_index=edge_index, edge_attr = edge_attr, y = y)
                data_list.append(data)

        
        # save the data and the normalize statistics

        with open(self.normalize_statistics+'normalize_dict.pkl', 'wb') as f:
            pickle.dump(normalize_dict, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--primal', dest = 'primal', type = str, default = '../data/primal_format/m200n400_mixed2/', help = 'which folder to get the primal instances')
    parser.add_argument('--generate_primal', dest = 'generate_primal', type = str, default = '../data/generate_primal/m200n400_mixed2/bowly2/', help = 'which folder to get the generated primal instances')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 1000, help = 'the number of instances')
    parser.add_argument('--seed', dest = 'seed', type = int, default = 123, help = 'the random seed')
    parser.add_argument('--ratio', dest = 'ratio', type = float, default = 0.5, help = 'the ratio of the original dataset in all the training data')
    parser.add_argument('--save_folder', dest = 'save_folder', type = str, default = './dataset/m200n400_mixed2/bowly2/', help = 'the folder to save the graph data')
    parser.add_argument('--primal_solution', dest = 'primal_solution', type = str, default = '../data/normalize_primal_solution/m200n400_mixed2/', help = 'the folder to get the primal solution')
    parser.add_argument('--generate_primal_solution', dest = 'generate_primal_solution', type = str, default = './dataset/generate_primal_solution/m200n400_mixed2/bowly2/', help = 'the folder to get the primal solution')
    parser.add_argument('--normalize_statistics', dest = 'normalize_statistics', type = str, default = './dataset/m200n400_mixed2/bowly2/', help = 'the folder to get the normalize data')
    args = parser.parse_args()
    
    setup_seed(args.seed)
    save_folder = args.save_folder + 'ratio0' + str(int(args.ratio * 10)) + '/'
    normalize_statistics = args.normalize_statistics + 'ratio0' + str(int(args.ratio * 10)) + '/'
    dataset = SetCoverDownstream(args.primal, args.generate_primal, args.num_instance, args.ratio, save_folder, args.primal_solution, args.generate_primal_solution, normalize_statistics)
        
if __name__ == '__main__':
    main()