import argparse
from json import decoder
from platform import node
import random
import pyscipopt
from tqdm import tqdm
import os
import wandb
import shutil
import json
import numpy as np
import pickle

import torch
from torch_geometric.data import InMemoryDataset, Data
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing, global_add_pool

from _5_generate_test_graph import SetCoverDownstreamTest, BipartiteData
from _8_generate_val_data import SetCoverDownstreamVal, BipartiteData
from model import Predictor

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class MatrixMultiplication(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add') 
    def forward(self, x_from, x_to, edge_index, edge_attr):
        out = self.propagate(edge_index = edge_index, x = (x_from, x_to), edge_attr = edge_attr)
        return out
    def message(self, x_j, edge_attr):
        return x_j * edge_attr

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--batch', dest = 'batch', type=int, default = 256, help='training batch size')
    parser.add_argument('--epoch', dest = 'epoch', type=int, default = 1000, help='number of epochs')
    parser.add_argument('--gpu', dest = 'gpu', type = int, default = 5, help = 'the index of GPU')
    parser.add_argument('--lr', type = float, dest = 'lr', default = 1e-3, help = 'learning rate')
    parser.add_argument('--wandb', type = int, dest = 'wandb', default = 0, help = 'whether to use wandb')
    parser.add_argument('--train_folder', type = str, dest = 'train_folder', default = './train_files/010_random/', help = 'folder to save the trained models')
    parser.add_argument('--seed', type = int, dest = 'seed', default = 123, help = 'random seed')
    # To load the dataset
    parser.add_argument('--primal', dest = 'primal', type = str, default = '../data/primal_format/m200n400_mixed2/', help = 'which folder to get the primal instances')
    parser.add_argument('--generate_primal', dest = 'generate_primal', type = str, default = '../data/generate_primal/m200n400_mixed2/010_random/', help = 'which folder to get the generated primal instances')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 1000, help = 'the number of instances')
    parser.add_argument('--ratio', dest = 'ratio', type = float, default = 0.5, help = 'the ratio of the original dataset in all the training data')
    parser.add_argument('--save_folder', dest = 'save_folder', type = str, default = './dataset/m200n400_mixed2/010_random/', help = 'the folder to save the graph data')
    parser.add_argument('--primal_solution', dest = 'primal_solution', type = str, default = '../data/normalize_primal_solution/m200n400_mixed2/', help = 'the folder to get the primal solution')
    parser.add_argument('--generate_primal_solution', dest = 'generate_primal_solution', type = str, default = './dataset/generate_primal_solution/m200n400_mixed2/010_random/', help = 'the folder to get the primal solution')
    parser.add_argument('--normalize_statistics', dest = 'normalize_statistics', type = str, default = './dataset/m200n400_mixed2/010_random/', help = 'the folder to get the normalize data')
    # Model parameters
    # Predictor
    parser.add_argument('--predictor_input_dim_xs', type = int, dest = 'predictor_input_dim_xs', default = 2, help = 'predictor x_s input dimension')
    parser.add_argument('--predictor_input_dim_xt', type = int, dest = 'predictor_input_dim_xt', default = 2, help = 'predictor x_t input dimension')
    parser.add_argument('--predictor_input_dim_edge', type = int, dest = 'predictor_input_dim_edge', default = 1, help = 'predictor edge input dimension')
    parser.add_argument('--predictor_num_layers', type = int, dest = 'predictor_num_layers', default = 2, help = 'number of predictor convolutional layers')
    parser.add_argument('--predictor_hidden_dim', type = int, dest = 'predictor_hidden_dim', default = 24, help = 'dimension of the hidden layer in predictor')
    parser.add_argument('--predictor_mlp_hidden_dim', type = int, dest = 'predictor_mlp_hidden_dim', default = 24, help = 'dimension of the mlp hidden layer in predictor')
    args = parser.parse_args()

    # Set up seed
    setup_seed(args.seed)

    # Make train folder
    train_folder = args.train_folder + 'ratio0' + str(int(args.ratio * 10)) + '/'
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    # Define the device
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # loss
    regression_loss = nn.SmoothL1Loss()

    # Define models
    predictor = Predictor(args.predictor_input_dim_xs, args.predictor_input_dim_xt, args.predictor_input_dim_edge, args.predictor_num_layers, args.predictor_hidden_dim, args.predictor_mlp_hidden_dim)
    state_dict = torch.load(train_folder+'predictor980.pth', map_location = device)
    predictor.to(device)
    predictor.load_state_dict(state_dict)

    # Start testing
    predictor.eval()

    normalize_statistics = args.normalize_statistics + 'ratio0' + str(int(args.ratio * 10)) + '/'
    density_list = [0.03, 0.04, 0.05, 0.10, 0.15,0.20,0.25, 0.30,0.35]
    for density in density_list:
        test_primal = './testset/m200n400/density'+str(int(density*100))+'/'
        test_save_folder = './testset/m200n400/density'+str(int(density*100))+'/'
        testset = SetCoverDownstreamTest(test_primal, args.num_instance, density, test_save_folder)
        testloader = DataLoader(testset, batch_size=1)
        test_loss = 0
        test_mae_list = []
        test_rmae_list = []
        test_mse_list = []
        test_rmse_list = []
        for test_data in testloader:
            test_data = test_data.to(device)
            # Encode the complete graph into \mu, log\sigma

            with open(normalize_statistics+'normalize_dict.pkl', 'rb') as f:
                normalize_dict = pickle.load(f)
            if normalize_dict['max_b'] == normalize_dict['min_b']:
                test_data.x_s[:,1] = test_data.x_s[:,1] + (1 - normalize_dict['max_b'])
            else:
                test_data.x_s[:,1] = (test_data.x_s[:,1] - normalize_dict['min_b']) /(normalize_dict['max_b'] - normalize_dict['min_b'])
            if normalize_dict['max_c'] == normalize_dict['min_c']:
                test_data.x_t[:,1] = test_data.x_t[:,1] + (1 - normalize_dict['max_c'])
            else:
                test_data.x_t[:,1] = (test_data.x_t[:,1] - normalize_dict['min_c']) /(normalize_dict['max_c'] - normalize_dict['min_c'])
            if normalize_dict['max_weight'] == normalize_dict['min_weight']:
                test_data.edge_attr = test_data.edge_attr + (1 - normalize_dict['max_weight'])
            else:
                test_data.edge_attr = (test_data.edge_attr - normalize_dict['min_weight']) /(normalize_dict['max_weight'] - normalize_dict['min_weight'])


            num_constraints_per_graph = int(test_data.x_s.shape[0] / test_data.num_graphs)
            batch_xs = torch.arange(0,test_data.num_graphs).to(device)
            batch_xs = batch_xs.repeat_interleave(num_constraints_per_graph)
            with torch.no_grad():
                predict_y_test = predictor(test_data.x_s, test_data.x_t, test_data.edge_index, test_data.edge_attr, batch_xs, test_data.batch)
            unnormalized_y_label = test_data.y.reshape(-1, 1)
            
            loss_test = regression_loss(predict_y_test, unnormalized_y_label)
            test_loss = test_loss + loss_test.item()
            mae = torch.abs(predict_y_test - unnormalized_y_label).item()
            rmae = torch.abs(predict_y_test - unnormalized_y_label).item() / torch.abs(unnormalized_y_label).item()
            mse = torch.abs(predict_y_test - unnormalized_y_label).item()**2
            rmse = torch.abs(predict_y_test - unnormalized_y_label).item()**2 / torch.abs(unnormalized_y_label).item()**2
            test_mae_list.append(mae)
            test_rmae_list.append(rmae)
            test_mse_list.append(mse)
            test_rmse_list.append(rmse)
        print('ratio:'+str(args.ratio))
        print('density:'+str(density))
        print('test loss:'+str(test_loss / len(testloader)))
        print('test mae:'+str(np.mean(test_mae_list)))
        print('test rmae:'+str(np.mean(test_rmae_list)))
        print('test mse:'+str(np.mean(test_mse_list)))
        print('test msre:'+str(np.mean(test_rmse_list)))

if __name__ == '__main__':
    main()