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

from _7_generate_vcgraph import SetCoverData, BipartiteData
from model import Encoder, Decoder
from loss import kl_loss

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

#calculate Ax
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
    parser.add_argument('--epoch', dest = 'epoch', type=int, default = 200, help='number of epochs')
    parser.add_argument('--gpu', dest = 'gpu', type = int, default = 2, help = 'the index of GPU')
    parser.add_argument('--lr', type = float, dest = 'lr', default = 1e-3, help = 'learning rate')
    parser.add_argument('--wandb', type = int, dest = 'wandb', default = 0, help = 'whether to use wandb')
    parser.add_argument('--train_folder', type = str, dest = 'train_folder', default = './train_files/try/', help = 'folder to save the trained models')
    parser.add_argument('--alpha', type = float, dest = 'alpha', default = 5, help = 'alpha to balance the loss')
    parser.add_argument('--alpha_b', type = float, dest = 'alpha_b', default = 0.0, help = 'alpha_b to balance the loss of b')
    parser.add_argument('--seed', type = int, dest = 'seed', default = 123, help = 'random seed')
    # To load the dataset
    parser.add_argument('--primal_format', dest = 'primal_format', type = str, default = './data/primal_format/m200n400_mixed2/', help = 'which folder to get the primal instances')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 1000, help = 'the number of instances')
    parser.add_argument('--save_folder', dest = 'save_folder', type = str, default = './graph_dataset/m200n400_mixed2/', help = 'the folder to save the graph data')
    parser.add_argument('--primal_solution', dest = 'primal_solution', type = str, default = './data/normalize_primal_solution/m200n400_mixed2/', help = 'the folder to get the primal solution')
    parser.add_argument('--primal_slack', dest = 'primal_slack', type = str, default = './data/normalize_primal_slack/m200n400_mixed2/', help = 'the folder to get the primal slack')
    parser.add_argument('--dual_solution', dest = 'dual_solution', type = str, default = './data/normalize_dual_solution/m200n400_mixed2/', help = 'the folder to get the dual solution')
    parser.add_argument('--dual_slack', dest = 'dual_slack', type = str, default = './data/normalize_dual_slack/', help = 'the folder to get the dual slack')
    parser.add_argument('--normalize_statistics', dest = 'normalize_statistics', type = str, default = './data/normalize_statistics/m200n400_mixed2/', help = 'folder that save the statistics of the normalization')
    # Model parameters
    # Encoder
    parser.add_argument('--encoder_input_dim_xs', type = int, dest = 'encoder_input_dim_xs', default = 3, help = 'encoder x_s input dimension')
    parser.add_argument('--encoder_input_dim_xt', type = int, dest = 'encoder_input_dim_xt', default = 5, help = 'encoder x_t input dimension')
    parser.add_argument('--encoder_input_dim_edge', type = int, dest = 'encoder_input_dim_edge', default = 1, help = 'encoder edge input dimension')
    parser.add_argument('--encoder_num_layers', type = int, dest = 'encoder_num_layers', default = 2, help = 'number of encoder convolutional layers')
    parser.add_argument('--encoder_hidden_dim', type = int, dest = 'encoder_hidden_dim', default = 30, help = 'dimension of the hidden layer in encoder')
    parser.add_argument('--encoder_mlp_hidden_dim', type = int, dest = 'encoder_mlp_hidden_dim', default = 30, help = 'dimension of the mlp hidden layer in encoder')
    # Decoder
    parser.add_argument('--decoder_input_dim_xs', type = int, dest = 'decoder_input_dim_xs', default = 1, help = 'decoder x_s input dimension')
    parser.add_argument('--decoder_input_dim_xt', type = int, dest = 'decoder_input_dim_xt', default = 1, help = 'decoder x_t input dimension')
    parser.add_argument('--decoder_input_dim_edge', type = int, dest = 'decoder_input_dim_edge', default = 1, help = 'decoder edge input dimension')
    parser.add_argument('--decoder_num_layers', type = int, dest = 'decoder_num_layers', default = 2, help = 'number of decoder convolutional layers')
    parser.add_argument('--decoder_hidden_dim', type = int, dest = 'decoder_hidden_dim', default = 16, help = 'dimension of the hidden layer in decoder')
    parser.add_argument('--decoder_mlp_hidden_dim', type = int, dest = 'decoder_mlp_hidden_dim', default = 16, help = 'dimension of the mlp hidden layer in decoder')
    parser.add_argument('--decoder_mlp_out_dim', type = int, dest = 'decoder_mlp_out_dim', default = 1, help = 'dimension of the output of the mlp hidden layer in decoder')
    args = parser.parse_args()

    torch.set_num_threads(10)
    # Init wandb
    if args.wandb:
        wandb.init(project="generate_mip_setcover_m200n400_mixed")

    # Set up seed
    setup_seed(args.seed)

    # Make train folder
    if not os.path.exists(args.train_folder):
        os.mkdir(args.train_folder)

    shutil.copy('./model.py', args.train_folder+str('model.py'))
    shutil.copy('./_9_train.py', args.train_folder+str('_9_train.py'))
    shutil.copy('./9_train.sh', args.train_folder+str('train.sh'))

    # Load the dataset
    dataset = SetCoverData(args.primal_format, args.num_instance, args.save_folder, args.primal_solution, args.primal_slack, args.dual_solution, args.dual_slack, args.normalize_statistics)
    dataloader = DataLoader(dataset, batch_size=args.batch, follow_batch = ['x_s', 'x_t'])

    # get the statistics of min / max degree for normalization, weight
    with open(args.normalize_statistics+'normalize_degree.pkl','rb') as f:
        degree_dict = pickle.load(f)
    max_degree = degree_dict['max_degree']
    min_degree = degree_dict['min_degree']
    with open(args.normalize_statistics+'normalize_weight.pkl','rb') as weight_f:
        weight_dict = pickle.load(weight_f)
    with open(args.normalize_statistics+'statistics.pkl','rb') as node_f:
        node_dict = pickle.load(node_f)
    with open(args.normalize_statistics+'normalize_num_x.pkl','rb') as num_x_f:
        num_x_dict = pickle.load(num_x_f)

    # Define the device
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # Define models
    encoder = Encoder(args.encoder_input_dim_xs, args.encoder_input_dim_xt, args.encoder_input_dim_edge, args.encoder_num_layers, args.encoder_hidden_dim, args.encoder_mlp_hidden_dim)
    decoder = Decoder(args.decoder_input_dim_xs, args.decoder_input_dim_xt, args.decoder_input_dim_edge, args.decoder_num_layers, args.decoder_hidden_dim, args.decoder_mlp_hidden_dim, args.decoder_mlp_out_dim)
    
    matrix_multiplication = MatrixMultiplication()
    # Define the loss criterion
    regression_loss = nn.SmoothL1Loss()
    #regression_loss = nn.L1Loss()
    bce_loss = nn.BCELoss()

    # Define the optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # Start training
    encoder.train()
    encoder.to(device)
    decoder.train()
    decoder.to(device)
    lowest_loss = 10000
    for epoch in tqdm(range(args.epoch)):
        loss_degree_count = 0
        loss_logits_count = 0
        loss_weights_count = 0 
        loss_num_x_count = 0
        loss_x_count = 0
        loss_ym_count = 0
        loss_yn_count = 0
        loss_s_count = 0
        loss_rm_count = 0
        loss_rn_count = 0
        loss_kl_count = 0
        loss_b_count = 0
        loss_c_count = 0
        for batch_data in dataloader:
            # constraint feature x_s: 0, y1-ym, r1-rm
            # variable feature x_t: 1, x1-xn, ym+1-ym+n, s1-sn, rm+1-rm+n
            batch_data = batch_data.to(device)
            # Encode the complete graph into \mu, log\sigma
            xs_mu, xs_logsigma, xs_z, xt_mu, xt_logsigma, xt_z = encoder(batch_data.x_s, batch_data.x_t, batch_data.edge_index, batch_data.edge_attr)

            # generate the masked data
            masked_x_s = batch_data.x_s[:,0].reshape(-1, 1)
            masked_x_t = batch_data.x_t[:,0].reshape(-1, 1)

            # Decode the x1-n, y1-m, ym+1-ym+n, s1-n, r1-m
            # randomly select a constraint node
            num_constraint_node = batch_data.x_s.shape[0]
            num_variable_node = batch_data.x_t.shape[0]
            num_constraint_node_per_graph = int(num_constraint_node/batch_data.num_graphs)
            selected_constraint_indices = []
            for i in range(0, num_constraint_node, num_constraint_node_per_graph):
                index = random.randint(i, i + 199)
                selected_constraint_indices.append(index)
            selected_constraint_indices = torch.tensor(selected_constraint_indices).to(device)
            edge_to_delete = torch.where(torch.isin(batch_data.edge_index[0], selected_constraint_indices))[0]
            deleted_edge_index = batch_data.edge_index[:, edge_to_delete]
            # deleted_edge_index: the edge_index that is masked
            # masked_edge_index: the remaining edge index after masks
            # masked_edge_attr: the remaining edge attr after masks
            edge_to_delete_np = edge_to_delete.cpu()
            edge_index_np = batch_data.edge_index.cpu().numpy()
            edge_attr_np = batch_data.edge_attr.cpu().numpy()
            masked_edge_index_np = np.delete(edge_index_np, edge_to_delete_np, axis = 1)
            masked_edge_attr_np = np.delete(edge_attr_np, edge_to_delete_np, axis = 0)
            masked_edge_index = torch.tensor(masked_edge_index_np).to(device)
            masked_edge_attr = torch.tensor(masked_edge_attr_np).to(device)
            '''the codes are slow af
            masked_edge_index = batch_data.edge_index[:, [col for col in range(batch_data.edge_index.shape[1]) if col not in edge_to_delete]]
            masked_edge_attr = batch_data.edge_attr[[row for row in range(batch_data.edge_attr.shape[0]) if row not in edge_to_delete],:]'''
            
            # edge_attr_label: the label of the masked edge attrs
            edge_attr_label_values = batch_data.edge_attr[edge_to_delete].reshape(-1)
            
            edge_attr_label = torch.zeros(batch_data.x_t.shape[0]).to(device)
            edge_attr_mask = edge_attr_label.clone()
            edge_attr_label[deleted_edge_index[1]] = edge_attr_label_values
            edge_attr_label = edge_attr_label.reshape(-1, 1)
            edge_attr_mask[deleted_edge_index[1]] = 1.
            
            # degrees of each deleted constraint node
            degree_label = degree(batch_data.edge_index[0], batch_data.x_s.shape[0])
            degree_label = (degree_label - min_degree) / (max_degree - min_degree)
            degree_label = degree_label.reshape(-1,1)
            '''_, degree_label = torch.unique(deleted_edge_index[0], return_counts=True)
            degree_label = degree_label.reshape(-1, 1)'''
            # the logits that should be 1
            logits_label = torch.zeros(num_variable_node).to(device)
            logits_label[deleted_edge_index[1]] = 1.
            logits_label = logits_label.reshape(-1, 1)
            
            # the x, y, s, r labels
            x_label = batch_data.x_t[:,1].reshape(-1, 1)    
            ym_label = batch_data.x_s[:,1].reshape(-1, 1)
            yn_label = batch_data.x_t[:,2].reshape(-1, 1)
            sn_label = batch_data.x_t[:,3].reshape(-1, 1)
            rm_label = batch_data.x_s[:,2].reshape(-1, 1)
            rn_label = batch_data.x_t[:,4].reshape(-1, 1)
            
            num_x_label = global_add_pool(x_label, batch_data.x_t_batch).reshape(-1, 1)
            if num_x_dict['equal'] == 1:
                num_x_label = num_x_label + num_x_dict['toward1']
            else:
                num_x_label = (num_x_label - num_x_dict['min_num_x']) / (num_x_dict['max_num_x'] - num_x_dict['min_num_x'])

            # Decode from masked graph and get the output
            #import pdb; pdb.set_trace()
            predict_degree, predict_logits, predict_weights, predict_num_x, predict_x, predict_ym, predict_yn, predict_s, predict_rm, predict_rn = decoder(masked_x_s, masked_x_t, masked_edge_index, masked_edge_attr, xs_z, xt_z, batch_data.x_t_batch)
            # get the unnormalized datas
            unnormalized_xn_label = x_label * (node_dict['max_x']-node_dict['min_x']) + node_dict['min_x']
            unnormalized_predict_x = predict_x * (node_dict['max_x']-node_dict['min_x']) + node_dict['min_x']
            unnormalized_ym_label = ym_label * (node_dict['max_y']-node_dict['min_y']) + node_dict['min_y']
            unnormalized_predict_ym = predict_ym * (node_dict['max_y']-node_dict['min_y']) + node_dict['min_y']
            unnormalized_yn_label = yn_label * (node_dict['max_y']-node_dict['min_y']) + node_dict['min_y']
            unnormalized_predict_yn = predict_yn * (node_dict['max_y']-node_dict['min_y']) + node_dict['min_y']
            if weight_dict['equal'] == 1:
                unnormalized_weights_label = batch_data.edge_attr - weight_dict['toward1']
            else:
                unnormalized_weights_label = (batch_data.edge_attr - weight_dict['min_weight']) / (weight_dict['max_weight'] - weight_dict['min_weight'])
            
            unnormalized_rm_label = rm_label * (node_dict['max_r']-node_dict['min_r']) + node_dict['min_r']
            unnormalized_predict_rm = predict_rm * (node_dict['max_r']-node_dict['min_r']) + node_dict['min_r']
            unnormalized_rn_label = rn_label * (node_dict['max_r']-node_dict['min_r']) + node_dict['min_r']
            unnormalized_predict_rn = predict_rn * (node_dict['max_r']-node_dict['min_r']) + node_dict['min_r']
            unnormalized_s_label = sn_label * (node_dict['max_s']-node_dict['min_s']) + node_dict['min_s']
            unnormalized_predict_s = predict_s * (node_dict['max_s']-node_dict['min_s']) + node_dict['min_s']

            unnormalized_b_label = torch.zeros((batch_data.x_s.shape[0],1)).to(device)
            unnormalized_predict_b = unnormalized_b_label.clone()
            reverse_edge_index = batch_data.edge_index.clone()
            reverse_edge_index[[0,1]] = batch_data.edge_index[[1,0]]
            unnormalized_b_label = matrix_multiplication(unnormalized_xn_label, unnormalized_b_label, reverse_edge_index, unnormalized_weights_label)
            unnormalized_b_label = unnormalized_b_label + unnormalized_rm_label
            unnormalized_predict_b = matrix_multiplication(unnormalized_predict_x, unnormalized_predict_b, reverse_edge_index, unnormalized_weights_label)
            unnormalized_predict_b = unnormalized_predict_b + unnormalized_predict_rm
            loss_b = regression_loss(unnormalized_predict_b, unnormalized_b_label)

            unnormalized_c_label = torch.zeros((batch_data.x_t.shape[0],1)).to(device)
            unnormalized_predict_c = unnormalized_c_label.clone()
            unnormalized_c_label = matrix_multiplication(unnormalized_ym_label, unnormalized_c_label, batch_data.edge_index, unnormalized_weights_label)
            unnormalized_c_label = unnormalized_c_label + unnormalized_yn_label - unnormalized_s_label
            unnormalized_predict_c = matrix_multiplication(unnormalized_predict_ym, unnormalized_predict_c, batch_data.edge_index, unnormalized_weights_label)
            unnormalized_predict_c = unnormalized_predict_c + unnormalized_predict_yn - unnormalized_predict_s
            loss_c = regression_loss(unnormalized_predict_c, unnormalized_c_label)
            
            
            # recover b label (unnormalized)
            predict_weights = predict_weights * edge_attr_mask.reshape(-1,1)
            loss_degree = regression_loss(predict_degree, degree_label)
            loss_logits = regression_loss(predict_logits, logits_label)
            loss_weights = regression_loss(predict_weights, edge_attr_label)
            loss_num_x = regression_loss(predict_num_x, num_x_label)
            loss_x = regression_loss(predict_x, x_label)
            loss_ym = regression_loss(predict_ym, ym_label)
            loss_yn = regression_loss(predict_yn, yn_label)
            loss_y = loss_ym + loss_yn
            loss_s = regression_loss(predict_s, sn_label)
            loss_rm = regression_loss(predict_rm, rm_label)
            loss_rn = regression_loss(predict_rn, rn_label)
            loss_r = loss_rm + loss_rn


            loss_kl_xs = kl_loss(xs_mu, xs_logsigma)
            loss_kl_xt = kl_loss(xt_mu, xt_logsigma)
            loss_kl = loss_kl_xs + loss_kl_xt

            optimizer.zero_grad()
            loss = args.alpha * (loss_degree + loss_logits + loss_weights + loss_num_x + loss_x + loss_y + loss_s + loss_r) + loss_kl + args.alpha_b * loss_b
            loss.backward()
            optimizer.step()
            selected_constraint_indices = selected_constraint_indices.cpu()
            logits_label = logits_label.cpu()
            edge_attr_label = edge_attr_label.cpu()
            edge_attr_mask = edge_attr_mask.cpu()
            reverse_edge_index = reverse_edge_index.cpu()
            unnormalized_b_label = unnormalized_b_label.cpu()
            unnormalized_predict_b = unnormalized_predict_b.cpu()
            loss_degree_count  = loss_degree_count + loss_degree.item() / batch_data.num_graphs
            loss_logits_count  = loss_logits_count + loss_logits.item() / batch_data.num_graphs
            loss_weights_count  = loss_weights_count + loss_weights.item() / batch_data.num_graphs
            loss_num_x_count = loss_num_x_count + loss_num_x.item() / batch_data.num_graphs
            loss_x_count  = loss_x_count + loss_x.item() / batch_data.num_graphs
            loss_ym_count  = loss_ym_count + loss_ym.item() / batch_data.num_graphs
            loss_yn_count  = loss_yn_count + loss_yn.item() / batch_data.num_graphs
            loss_s_count  = loss_s_count + loss_s.item() / batch_data.num_graphs
            loss_rm_count  = loss_rm_count + loss_rm.item() / batch_data.num_graphs
            loss_rn_count  = loss_rn_count + loss_rn.item() / batch_data.num_graphs
            loss_kl_count  = loss_kl_count + loss_kl.item() / batch_data.num_graphs
            loss_b_count = loss_b_count + loss_b.item() / batch_data.num_graphs
            loss_c_count = loss_c_count + loss_c.item() / batch_data.num_graphs
        if args.wandb:
            wandb.log({"loss degree": loss_degree_count})
            wandb.log({"loss logits": loss_logits_count})
            wandb.log({"loss weights": loss_weights_count})
            wandb.log({"loss num of x": loss_num_x_count})
            wandb.log({"loss x": loss_x_count})
            wandb.log({"loss ym": loss_ym_count})
            wandb.log({"loss yn": loss_yn_count})
            wandb.log({"loss s": loss_s_count})
            wandb.log({"loss rm": loss_rm_count})
            wandb.log({"loss rn": loss_rn_count})
            wandb.log({"loss kl": loss_kl_count})
            wandb.log({"loss y": loss_ym_count+loss_yn_count})
            wandb.log({"loss r": loss_rm_count+loss_rn_count})
            wandb.log({"loss b": loss_b_count})
            wandb.log({"loss c": loss_c_count})
        if epoch > 0 and epoch%20 == 0:
            torch.save(encoder.state_dict(), args.train_folder+'encoder'+str(epoch)+'.pth')
            torch.save(decoder.state_dict(), args.train_folder+'decoder'+str(epoch)+'.pth')
        losses = loss_degree_count+loss_logits_count+loss_weights_count+loss_x+loss_ym+loss_yn+loss_s+loss_rm+loss_rn+loss_y+loss_kl
        if losses < lowest_loss:
            lowest_loss = losses
            torch.save(encoder.state_dict(), args.train_folder+'lencoder'+str(epoch)+'.pth')
            torch.save(decoder.state_dict(), args.train_folder+'ldecoder'+str(epoch)+'.pth')

    # Save the trained models
    torch.save(encoder.state_dict(), args.train_folder+'encoder.pth')
    torch.save(decoder.state_dict(), args.train_folder+'decoder.pth')

    if args.wandb:
        wandb.finish()


            
            


if __name__ == '__main__':
    main()