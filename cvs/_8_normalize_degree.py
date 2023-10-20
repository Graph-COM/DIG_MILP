import argparse
import pickle
from tqdm import tqdm


from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from _7_generate_vcgraph import CVSData, BipartiteData



def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    # To load the dataset
    parser.add_argument('--primal_format', dest = 'primal_format', type = str, default = './data/primal_format/', help = 'which folder to get the primal instances')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 1000, help = 'the number of instances')
    parser.add_argument('--save_folder', dest = 'save_folder', type = str, default = './graph_dataset/', help = 'the folder to save the graph data')
    parser.add_argument('--primal_solution', dest = 'primal_solution', type = str, default = './data/primal_solution/', help = 'the folder to get the primal solution')
    parser.add_argument('--primal_slack', dest = 'primal_slack', type = str, default = './data/primal_slack/', help = 'the folder to get the primal slack')
    parser.add_argument('--dual_solution', dest = 'dual_solution', type = str, default = './data/dual_solution/', help = 'the folder to get the dual solution')
    parser.add_argument('--dual_slack', dest = 'dual_slack', type = str, default = './data/dual_slack/', help = 'the folder to get the dual slack')
    parser.add_argument('--normalize_statistics', dest = 'normalize_statistics', type = str, default = './data/normalize_statistics/', help = 'which folder to save the normalized statistics')
    args = parser.parse_args()

    # Load the dataset
    dataset = CVSData(args.primal_format, args.num_instance, args.save_folder, args.primal_solution, args.primal_slack, args.dual_solution, args.dual_slack, args.normalize_statistics)
    degree_dict = {}
    max_degree = 0
    min_degree = 10000

    for graph in dataset:
        num_nodes = graph.x_s.shape[0]
        degree_list = degree(graph.edge_index[0], num_nodes)
        this_max = max(degree_list)
        if this_max > max_degree:
            max_degree = this_max
        this_min = min(degree_list)
        if this_min < min_degree:
            min_degree = this_min
    degree_dict['max_degree'] = max_degree.item()
    degree_dict['min_degree'] = min_degree.item()
    print('max'+str(max_degree.item()))
    print('min'+str(min_degree.item()))
    with open(args.normalize_statistics+'normalize_degree.pkl', 'wb') as degree_f:
        pickle.dump(degree_dict, degree_f)

    '''weight_dict = {}
    max_weight = -10000
    min_weight = 10000
    for graph in tqdm(dataset):
        this_max = max(graph.edge_attr.reshape(-1))
        if this_max > max_weight:
            max_weight = this_max
        this_min = min(graph.edge_attr.reshape(-1))
        if this_min < min_weight:
            min_weight = this_min
    weight_dict['max_weight'] = max_weight.item()
    weight_dict['min_weight'] = min_weight.item()
    print('max'+str(max_weight.item()))
    print('min'+str(min_weight.item()))'''
    '''with open(args.normalize_statistics+'normalize_weight.pkl', 'wb') as weight_f:
        pickle.dump(weight_dict, weight_f)'''
    
    num_x_dict = {}
    min_num_x = 10000
    max_num_x = 0
    all_file_list = ['cvs08r139-94','cvs16r70-62','cvs16r89-60','cvs16r106-72','cvs16r128-89']
    for lp_file in tqdm(all_file_list):
        with open(args.primal_solution+lp_file+'.pkl', "rb") as x_file:    
            dict_x = pickle.load(x_file)
            this_num_x = sum(dict_x.values())
            if this_num_x > max_num_x:
                max_num_x = this_num_x
            if this_num_x < min_num_x:
                min_num_x = this_num_x
    num_x_dict['max_num_x'] = max_num_x
    num_x_dict['min_num_x'] = min_num_x
    if min_num_x == max_num_x:
        num_x_dict['equal'] = 1
        num_x_dict['toward1'] = 1 - min_num_x
    else:
        num_x_dict['equal'] = 0
        num_x_dict['toward1'] =  0
    with open(args.normalize_statistics+'normalize_num_x.pkl', 'wb') as num_x_f:
        pickle.dump(num_x_dict, num_x_f)

        


if __name__ == '__main__':
    main()