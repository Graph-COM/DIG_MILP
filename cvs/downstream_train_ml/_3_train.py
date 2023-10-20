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

from _2_generate_train_data import CVSDownstream, BipartiteData
from _8_generate_val_data import CVSDownstreamVal, BipartiteData
from _5_generate_test_graph import CVSDownstreamTest, BipartiteData
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
    parser.add_argument('--gpu', dest = 'gpu', type = int, default = 2, help = 'the index of GPU')
    parser.add_argument('--lr', type = float, dest = 'lr', default = 1e-3, help = 'learning rate')
    parser.add_argument('--wandb', type = int, dest = 'wandb', default = 0, help = 'whether to use wandb')
    parser.add_argument('--train_folder', type = str, dest = 'train_folder', default = './train_files/new/', help = 'folder to save the trained models')
    parser.add_argument('--seed', type = int, dest = 'seed', default = 123, help = 'random seed')
    # To load the dataset
    parser.add_argument('--primal', dest = 'primal', type = str, default = '../data/primal_format/', help = 'which folder to get the primal instances')
    parser.add_argument('--generate_primal', dest = 'generate_primal', type = str, default = '../data/generate_primal/050_a150lr1e3/', help = 'which folder to get the generated primal instances')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 1000, help = 'the number of instances')
    parser.add_argument('--ratio', dest = 'ratio', type = float, default = 0.02, help = 'the ratio of the original dataset in all the training data')
    parser.add_argument('--save_folder', dest = 'save_folder', type = str, default = './dataset/050_a150lr1e3/', help = 'the folder to save the graph data')
    parser.add_argument('--primal_solution', dest = 'primal_solution', type = str, default = '../data/normalize_primal_solution/', help = 'the folder to get the primal solution')
    parser.add_argument('--generate_primal_solution', dest = 'generate_primal_solution', type = str, default = './dataset/generate_primal_solution/050_a150lr1e3/', help = 'the folder to get the primal solution')
    parser.add_argument('--normalize_statistics', dest = 'normalize_statistics', type = str, default = './dataset/050_a150lr1e3/', help = 'the folder to get the normalize data')
    # Model parameters
    # Predictor
    parser.add_argument('--predictor_input_dim_xs', type = int, dest = 'predictor_input_dim_xs', default = 2, help = 'predictor x_s input dimension')
    parser.add_argument('--predictor_input_dim_xt', type = int, dest = 'predictor_input_dim_xt', default = 2, help = 'predictor x_t input dimension')
    parser.add_argument('--predictor_input_dim_edge', type = int, dest = 'predictor_input_dim_edge', default = 1, help = 'predictor edge input dimension')
    parser.add_argument('--predictor_num_layers', type = int, dest = 'predictor_num_layers', default = 2, help = 'number of predictor convolutional layers')
    parser.add_argument('--predictor_hidden_dim', type = int, dest = 'predictor_hidden_dim', default = 24, help = 'dimension of the hidden layer in predictor')
    parser.add_argument('--predictor_mlp_hidden_dim', type = int, dest = 'predictor_mlp_hidden_dim', default = 24, help = 'dimension of the mlp hidden layer in predictor')
    args = parser.parse_args()

    torch.set_num_threads(10)
    # Init wandb
    if args.wandb:
        wandb.init(project="cvs_downstream")

    # Set up seed
    setup_seed(args.seed)

    # Make train folder
    train_folder = args.train_folder + 'ratio' + str("{:.2f}".format(args.ratio)).replace('.', '') + '/'
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    shutil.copy('./model.py', train_folder+str('model.py'))
    shutil.copy('./_3_train.py', train_folder+str('_3_train.py'))
    shutil.copy('./3_train.sh', train_folder+str('train.sh'))

    # Load the dataset
    save_folder = args.save_folder + 'ratio' + str("{:.2f}".format(args.ratio)).replace('.', '') + '/'
    normalize_statistics = args.save_folder + 'ratio' + str("{:.2f}".format(args.ratio)).replace('.', '') + '/'
    primal_list = ['cvs08r139-94','cvs16r70-62','cvs16r89-60']
    dataset = CVSDownstream(primal_list, args.primal, args.generate_primal, args.num_instance, args.ratio, save_folder, args.primal_solution, args.generate_primal_solution, normalize_statistics)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle = True, follow_batch = ['x_s', 'x_t'])

    primal_solution_path = '../data/primal_solution/'
    val_save_folder = './testset/'+'val/'
    test_list = ['cvs08r139-94','cvs16r70-62','cvs16r89-60']
    valset = CVSDownstreamVal(args.primal, primal_solution_path, test_list, val_save_folder)
    valloader = DataLoader(valset, batch_size=1, follow_batch = ['x_s', 'x_t'])


    # Define the device
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # Define models
    predictor = Predictor(args.predictor_input_dim_xs, args.predictor_input_dim_xt, args.predictor_input_dim_edge, args.predictor_num_layers, args.predictor_hidden_dim, args.predictor_mlp_hidden_dim)
    matrix_multiplication = MatrixMultiplication()
    # Define the loss criterion
    regression_loss = nn.SmoothL1Loss()
    
    # Define the optimizer
    params = list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # Start training
    #predictor.train()
    predictor.to(device)
    lowest_loss = 10000
    for epoch in tqdm(range(args.epoch)):
        predictor.train()
        loss_prediction_count = 0
        for batch_data in dataloader:
            # constraint feature x_s: 0, y1-ym, r1-rm
            # variable feature x_t: 1, x1-xn, ym+1-ym+n, s1-sn, rm+1-rm+n
            batch_data = batch_data.to(device)
            # Encode the complete graph into \mu, log\sigma
            predict_y = predictor(batch_data.x_s, batch_data.x_t, batch_data.edge_index, batch_data.edge_attr, batch_data.x_s_batch, batch_data.x_t_batch)
            y_label = batch_data.y.reshape(-1, 1)

            loss_prediction = regression_loss(predict_y, y_label)
            
            optimizer.zero_grad()
            loss_prediction.backward()
            optimizer.step()

            loss_prediction_count  = loss_prediction_count + loss_prediction.item()
        if args.wandb:
            wandb.log({"loss prediction": loss_prediction_count/len(dataloader)})
        if epoch > 0 and epoch%20 == 0:
            torch.save(predictor.state_dict(), train_folder+'predictor'+str(epoch)+'.pth')
        
        
        predictor.eval()
        val_loss = 0
        rae_list = []
        for val_data in valloader:
            val_data = val_data.to(device)
            # Encode the complete graph into \mu, log\sigma

            with open(normalize_statistics+'normalize_dict.pkl', 'rb') as f:
                normalize_dict = pickle.load(f)
            if normalize_dict['max_b'] == normalize_dict['min_b']:
                val_data.x_s[:,1] = val_data.x_s[:,1] + (1 - normalize_dict['max_b'])
            else:
                val_data.x_s[:,1] = (val_data.x_s[:,1] - normalize_dict['min_b']) /(normalize_dict['max_b'] - normalize_dict['min_b'])
            if normalize_dict['max_c'] == normalize_dict['min_c']:
                val_data.x_t[:,1] = val_data.x_t[:,1] + (1 - normalize_dict['max_c'])
            else:
                val_data.x_t[:,1] = (val_data.x_t[:,1] - normalize_dict['min_c']) /(normalize_dict['max_c'] - normalize_dict['min_c'])
            if normalize_dict['max_weight'] == normalize_dict['min_weight']:
                val_data.edge_attr = val_data.edge_attr + (1 - normalize_dict['max_weight'])
            else:
                val_data.edge_attr = (val_data.edge_attr - normalize_dict['min_weight']) /(normalize_dict['max_weight'] - normalize_dict['min_weight'])

            with torch.no_grad():
                predict_y_val = predictor(val_data.x_s, val_data.x_t, val_data.edge_index, val_data.edge_attr, val_data.x_s_batch, val_data.x_t_batch)
            unnormalized_y_label = val_data.y.reshape(-1, 1)
            '''if normalize_dict['max_y'] == normalize_dict['min_y']:
                y_label = unnormalized_y_label - (1 - normalize_dict['max_y'])
                unnormalized_predict_y = predict_y_val + (1 - normalize_dict['max_y'])
            else:
                y_label = (unnormalized_y_label - normalize_dict['min_y'] ) / (normalize_dict['max_y'] - normalize_dict['min_y'])
                unnormalized_predict_y = predict_y_val * (normalize_dict['max_y'] - normalize_dict['min_y']) + normalize_dict['min_y']'''
            loss_val = regression_loss(predict_y_val, unnormalized_y_label)
            val_loss = val_loss + loss_val.item()
            rae = torch.abs(predict_y_val - unnormalized_y_label).item() 
            rae_list.append(rae)
        if args.wandb:
            wandb.log({"loss validation": val_loss / len(valloader)})
            wandb.log({"rae validation": np.mean(rae_list)})
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            torch.save(predictor.state_dict(), train_folder+'lpredictor'+str(epoch)+'.pth')
    # Save the trained models
    torch.save(predictor.state_dict(), train_folder+'predictor.pth')
    if args.wandb:
        wandb.finish()

    # testing
    predictor.eval()
    test_list = ['cvs16r106-72','cvs16r128-89']
    primal_solution_path = '../data/primal_solution/'
    
    testset = CVSDownstreamTest(args.primal, primal_solution_path, test_list, save_folder)
    testloader = DataLoader(testset, batch_size=1, follow_batch = ['x_s', 'x_t'])
    test_loss = 0
    test_rae_list = []
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


        with torch.no_grad():
            predict_y_test = predictor(test_data.x_s, test_data.x_t, test_data.edge_index, test_data.edge_attr, test_data.x_s_batch, test_data.x_t_batch)
        unnormalized_y_label = test_data.y.reshape(-1, 1)
        '''if normalize_dict['max_y'] == normalize_dict['min_y']:
            y_label = unnormalized_y_label - (1 - normalize_dict['max_y'])
            unnormalized_predict_y = predict_y_test + (1 - normalize_dict['max_y'])
        else:
            y_label = (unnormalized_y_label - normalize_dict['min_y'] ) / (normalize_dict['max_y'] - normalize_dict['min_y'])
            unnormalized_predict_y = predict_y_test * (normalize_dict['max_y'] - normalize_dict['min_y']) + normalize_dict['min_y']'''
        
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
    print('test loss:'+str(test_loss / len(testloader)))
    print('test mae:'+str(np.mean(test_mae_list)))
    print('test rmae:'+str(np.mean(test_rmae_list)))
    print('test mse:'+str(np.mean(test_mse_list)))
    print('test msre:'+str(np.mean(test_rmse_list)))


if __name__ == '__main__':
    main()