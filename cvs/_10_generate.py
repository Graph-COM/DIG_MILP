import argparse
import random
import pickle
from turtle import towards
import numpy as np
from tqdm import tqdm
import os


import torch
from torch_geometric.loader import DataLoader

from _7_generate_vcgraph import CVSData, BipartiteData
from model import Encoder, Decoder

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--batch', dest = 'batch', type=int, default = 1, help='training batch size')
    parser.add_argument('--eta', dest = 'eta', type=float, default = 0.01, help='ratio of the changed nodes')
    parser.add_argument('--gpu', dest = 'gpu', type = int, default = 0, help = 'the index of GPU')
    parser.add_argument('--train_folder', type = str, dest = 'train_folder', default = './train_files/a100_lr1e3/', help = 'folder to save the trained models')
    parser.add_argument('--seed', type = int, dest = 'seed', default = 123, help = 'random seed')
    parser.add_argument('--generate_folder', type = str, dest = 'generate_folder', default = './data/generate_primal/001/', help = 'the folder to save generated graph')
    parser.add_argument('--solution_folder', type = str, dest = 'solution_folder', default = './data/generate_primal_partial_solution/001/', help = 'the folder to save generated graph')
    # To load the dataset
    parser.add_argument('--primal_format', dest = 'primal_format', type = str, default = './data/primal_format/', help = 'which folder to get the primal instances')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 1000, help = 'the number of instances')
    parser.add_argument('--save_folder', dest = 'save_folder', type = str, default = './graph_dataset/', help = 'the folder to save the graph data')
    parser.add_argument('--primal_solution', dest = 'primal_solution', type = str, default = './data/normalize_primal_solution/', help = 'the folder to get the primal solution')
    parser.add_argument('--primal_slack', dest = 'primal_slack', type = str, default = './data/normalize_primal_slack/', help = 'the folder to get the primal slack')
    parser.add_argument('--dual_solution', dest = 'dual_solution', type = str, default = './data/normalize_dual_solution/', help = 'the folder to get the dual solution')
    parser.add_argument('--dual_slack', dest = 'dual_slack', type = str, default = './data/normalize_dual_slack/', help = 'the folder to get the dual slack')
    parser.add_argument('--normalize_statistics', dest = 'normalize_statistics', type = str, default = './data/normalize_statistics/', help = 'folder that save the statistics of the normalization')
    # Model parameters
    # Encoder
    parser.add_argument('--encoder_input_dim_xs', type = int, dest = 'encoder_input_dim_xs', default = 3, help = 'encoder x_s input dimension')
    parser.add_argument('--encoder_input_dim_xt', type = int, dest = 'encoder_input_dim_xt', default = 5, help = 'encoder x_t input dimension')
    parser.add_argument('--encoder_input_dim_edge', type = int, dest = 'encoder_input_dim_edge', default = 1, help = 'encoder edge input dimension')
    parser.add_argument('--encoder_num_layers', type = int, dest = 'encoder_num_layers', default = 2, help = 'number of encoder convolutional layers')
    parser.add_argument('--encoder_hidden_dim', type = int, dest = 'encoder_hidden_dim', default = 30, help = 'dimension of the hidden layer in encoder')
    parser.add_argument('--encoder_mlp_hidden_dim', type = int, dest = 'encoder_mlp_hidden_dim', default = 30, help = 'dimension of the mlp hidden layer in encoder')
    parser.add_argument('--encoder_dict', type = str, dest = 'encoder_dict', default = 'encoder.pth', help = 'file name of the encoder dict')
    # Decoder
    parser.add_argument('--decoder_input_dim_xs', type = int, dest = 'decoder_input_dim_xs', default = 1, help = 'decoder x_s input dimension')
    parser.add_argument('--decoder_input_dim_xt', type = int, dest = 'decoder_input_dim_xt', default = 1, help = 'decoder x_t input dimension')
    parser.add_argument('--decoder_input_dim_edge', type = int, dest = 'decoder_input_dim_edge', default = 1, help = 'decoder edge input dimension')
    parser.add_argument('--decoder_num_layers', type = int, dest = 'decoder_num_layers', default = 2, help = 'number of decoder convolutional layers')
    parser.add_argument('--decoder_hidden_dim', type = int, dest = 'decoder_hidden_dim', default = 16, help = 'dimension of the hidden layer in decoder')
    parser.add_argument('--decoder_mlp_hidden_dim', type = int, dest = 'decoder_mlp_hidden_dim', default = 16, help = 'dimension of the mlp hidden layer in decoder')
    parser.add_argument('--decoder_mlp_out_dim', type = int, dest = 'decoder_mlp_out_dim', default = 1, help = 'dimension of the output of the mlp hidden layer in decoder')
    parser.add_argument('--decoder_dict', type = str, dest = 'decoder_dict', default = 'decoder.pth', help = 'file name of the decoder dict')
    args = parser.parse_args()

    # create folder if not exists
    if not os.path.exists(args.generate_folder):
        os.mkdir(args.generate_folder)
    if not os.path.exists(args.solution_folder):
        os.mkdir(args.solution_folder)

    torch.set_num_threads(5)
    # Set up seed
    setup_seed(args.seed)
    # Load the dataset
    original_dataset = CVSData(args.primal_format, args.num_instance, args.save_folder, args.primal_solution, args.primal_slack, args.dual_solution, args.dual_slack, args.normalize_statistics)
    dataset = random.choices(original_dataset, k = 1000)
    dataloader = DataLoader(dataset, batch_size=args.batch, follow_batch = ['x_s', 'x_t'])

    # get the statistics of min / max degree for normalization, weight
    with open(args.normalize_statistics+'normalize_degree.pkl','rb') as degree_f:
        degree_dict = pickle.load(degree_f)
    with open(args.normalize_statistics+'normalize_weight.pkl','rb') as weight_f:
        weight_dict = pickle.load(weight_f)
    with open(args.normalize_statistics+'statistics.pkl','rb') as node_f:
        node_dict = pickle.load(node_f)
    with open(args.normalize_statistics+'normalize_num_x.pkl','rb') as num_x_f:
        num_x_dict = pickle.load(num_x_f)

    # Define the device
    #device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # Define models
    encoder = Encoder(args.encoder_input_dim_xs, args.encoder_input_dim_xt, args.encoder_input_dim_edge, args.encoder_num_layers, args.encoder_hidden_dim, args.encoder_mlp_hidden_dim)
    decoder = Decoder(args.decoder_input_dim_xs, args.decoder_input_dim_xt, args.decoder_input_dim_edge, args.decoder_num_layers, args.decoder_hidden_dim, args.decoder_mlp_hidden_dim, args.decoder_mlp_out_dim)


    # Start generating
    encoder.eval()
    decoder.eval()
    encoder.to(device)
    decoder.to(device)
    encoder_dict = torch.load(args.train_folder+args.encoder_dict, map_location = device)
    encoder.load_state_dict(encoder_dict)
    decoder_dict = torch.load(args.train_folder+args.decoder_dict, map_location = device)
    decoder.load_state_dict(decoder_dict)
    


    num_epochs = int(dataset[0].x_s.shape[0] * args.eta)
    num_epochs = max(num_epochs-1, 1)
    graph_idx_ = 0
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        for epoch in range(num_epochs):
            # Randomly sample xs_z and xt_z from N(0,I)
            shape_xs_z = (batch_data.x_s.shape[0],1)
            shape_xt_z = (batch_data.x_t.shape[0],1)
            xs_z = torch.randn(shape_xs_z).to(device)
            xt_z = torch.randn(shape_xt_z).to(device)
            # generate the masked data
            masked_x_s = batch_data.x_s[:,0].reshape(-1, 1)
            masked_x_t = batch_data.x_t[:,0].reshape(-1, 1)
            # Decode the x1-n, y1-m, ym+1-ym+n, s1-n, r1-m
            # randomly select a constraint node
            selected_constraint_indices = []
            for ii in range(batch_data.num_graphs):
                indices_tensor = torch.where(batch_data.x_s_batch==ii)[0]
                start_idx = torch.min(indices_tensor)
                end_idx = torch.max(indices_tensor)
                selected_number = random.randint(start_idx, end_idx - 1)
                selected_constraint_indices.append(selected_number)
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
            with torch.no_grad():
                #xs_mu, xs_logsigma, xs_z, xt_mu, xt_logsigma, xt_z = encoder(batch_data.x_s, batch_data.x_t, batch_data.edge_index, batch_data.edge_attr)
                predict_degree, predict_logits, predict_weights, predict_num_x, predict_x, predict_ym, predict_yn, predict_s, predict_rm, predict_rn = decoder(masked_x_s, masked_x_t, masked_edge_index, masked_edge_attr, xs_z, xt_z, batch_data.x_t_batch)
            
            # Reconstruct the graph from prediction
            # add edge_index and edge_attr
            predict_degree = predict_degree * (degree_dict['max_degree'] - degree_dict['min_degree']) + degree_dict['min_degree']
            predict_degree = torch.round(predict_degree).long()
            predict_degree_masked_nodes = predict_degree[selected_constraint_indices]
            # add the x
            if num_x_dict['equal'] == 1:
                predict_num_x = predict_num_x - num_x_dict['toward1']
            else:
                predict_num_x = predict_num_x * (num_x_dict['max_num_x'] - num_x_dict['min_num_x']) + num_x_dict['min_num_x']
            predict_num_x = torch.round(predict_num_x).long()
            topk_indices_degree = torch.zeros((1,1))
            topk_indices_num_x = torch.zeros((1,1))
            topk_indices_degree = topk_indices_degree.to(device)
            topk_indices_num_x = topk_indices_num_x.to(device)
            interval_tensor_degree = predict_logits
            print(predict_degree_masked_nodes[0].item())
            topk_values, topk_indices_degree_interval = torch.topk(interval_tensor_degree, k=predict_degree_masked_nodes[0].item(), dim = 0)
            topk_indices_degree = torch.cat((topk_indices_degree, topk_indices_degree_interval), 0)
            interval_tensor_num_x = predict_x
            #import pdb; pdb.set_trace()
            topk_values, topk_indices_num_x_interval = torch.topk(interval_tensor_num_x, k=predict_num_x[0].item(), dim = 0)
            topk_indices_num_x = torch.cat((topk_indices_num_x, topk_indices_num_x_interval), 0)
            
            topk_indices_degree = topk_indices_degree[1:,:].reshape(1,-1).long()
            topk_indices_degree_0 = selected_constraint_indices.repeat_interleave(predict_degree_masked_nodes.reshape(-1)).reshape(1,-1)
            
            add_edge_index = torch.cat((topk_indices_degree_0, topk_indices_degree), 0)
            add_edge_attr = predict_weights[topk_indices_degree.reshape(-1)]

            topk_indices_num_x = topk_indices_num_x[1:,:].reshape(1,-1).long()

            # change the graph information
            # constraint feature x_s: 0, y1-ym, r1-rm
            # variable feature x_t: 1, x1-xn, ym+1-ym+n, s1-sn, rm+1-rm+n
            batch_data.x_s[:,1] = predict_ym.reshape(-1)
            batch_data.x_s[:,2] = predict_rm.reshape(-1)
            batch_data.x_t[:,1] = batch_data.x_t[:,1] * 0.
            batch_data.x_t[topk_indices_num_x,1] = 1.
            batch_data.x_t[:,2] = predict_yn.reshape(-1)
            batch_data.x_t[:,3] = predict_s.reshape(-1)
            batch_data.x_t[:,4] = predict_rn.reshape(-1)
            batch_data.edge_index = torch.cat((masked_edge_index, add_edge_index), 1)
            batch_data.edge_attr = torch.cat((masked_edge_attr, add_edge_attr), 0)
            sorted_indices = torch.argsort(batch_data.edge_index[0])
            batch_data.edge_index = torch.index_select(batch_data.edge_index, 1, sorted_indices)
            batch_data.edge_attr = torch.index_select(batch_data.edge_attr, 0, sorted_indices)

            if epoch + 1 == num_epochs:
                if weight_dict['equal'] == 0:
                    batch_data.edge_attr = batch_data.edge_attr * (weight_dict['max_weight']-weight_dict['min_weight']) + weight_dict['min_weight']
                else:
                    batch_data.edge_attr = batch_data.edge_attr - weight_dict['toward1']
                batch_data.edge_attr = torch.round(batch_data.edge_attr)
                #batch_data.edge_attr = batch_data.edge_attr.long()
                # add x, y, s, r
                infer_x = torch.round(batch_data.x_t[:,1].reshape(-1,1) * (node_dict['max_x']-node_dict['min_x']) + node_dict['min_x'])
                infer_ym = torch.round(predict_ym * (node_dict['max_y']-node_dict['min_y']) + node_dict['min_y'])
                infer_yn = torch.round(predict_yn * (node_dict['max_y']-node_dict['min_y']) + node_dict['min_y'])
                infer_s = torch.round(predict_s * (node_dict['max_s']-node_dict['min_s']) + node_dict['min_s'])
                infer_rm = torch.clamp(torch.round(predict_rm * (node_dict['max_r']-node_dict['min_r']) + node_dict['min_r']), min = 0)
                infer_rn = torch.clamp(torch.round(predict_rn * (node_dict['max_r']-node_dict['min_r']) + node_dict['min_r']), max = 1)
                # constraint feature x_s: 0, y1-ym, r1-rm
                # variable feature x_t: 1, x1-xn, ym+1-ym+n, s1-sn, rm+1-rm+n
                batch_data.x_s[:,1] = infer_ym.reshape(-1)
                batch_data.x_s[:,2] = infer_rm.reshape(-1)
                batch_data.x_t[:,1] = infer_x.reshape(-1)
                batch_data.x_t[:,2] = infer_yn.reshape(-1)
                batch_data.x_t[:,3] = infer_s.reshape(-1)
                batch_data.x_t[:,4] = infer_rn.reshape(-1)

                #import pdb; pdb.set_trace()

                np_solution = infer_x.reshape(-1).cpu().numpy()
                np_solution_path = args.solution_folder + str(graph_idx_)+".lp"
                np.save(np_solution_path, np_solution)

                # and check if the solution is correct               
                # Write the instance in the folder for each generated graph:
                for batch_graph_idx in range(batch_data.num_graphs):
                    single_graph = batch_data#[batch_graph_idx]
                    # Write new mip file
                    with open(args.generate_folder+str(graph_idx_)+'.lp', 'w') as file:
                        file.write("maximize\nOBJ:")
                        for variable_idx in range(single_graph.x_t.shape[0]):
                            # get c
                            # get the edge id where have x_variable_idx
                            edge_id = torch.where(single_graph.edge_index[1]==variable_idx)
                            # get the id of ym connected with x_variable_idx 
                            ym_id = single_graph.edge_index[0][edge_id]
                            # get the weight of these edges
                            weights = single_graph.edge_attr[edge_id]
                            lhs = torch.matmul(single_graph.x_s[:,1][ym_id].reshape(1,-1),weights.float())
                            lhs = lhs + single_graph.x_t[:,2][variable_idx]
                            c_variable_idx = lhs - single_graph.x_t[:,3][variable_idx]
                            c_variable_idx = c_variable_idx.item()
                            file.write(' + '+str(c_variable_idx)+' x'+str(variable_idx+1))
                        file.write('\n')
                        file.write("\n\nsubject to\n")
                        
                        
                        for constraint_idx in range(single_graph.x_s.shape[0]):
                            # get b
                            # get the edge id where have y_constraint_idx
                            edge_id = torch.where(single_graph.edge_index[0]==constraint_idx)
                            if edge_id[0].size(0)==0:
                                continue
                            else:
                                # get the id of xn connected with y_constraint_idx
                                xn_id = single_graph.edge_index[1][edge_id]
                                # get the weight of these edges
                                weights = single_graph.edge_attr[edge_id]
                                lhs = torch.matmul(single_graph.x_t[:,1][xn_id].reshape(1,-1),weights.float())
                                b_constraint_idx = lhs + single_graph.x_s[:,2][constraint_idx]
                                b_constraint_idx = b_constraint_idx.item()
                                file.write('C'+str(constraint_idx+1)+': ')
                                for i in range(xn_id.shape[0]): 
                                    file.write('+'+str(weights[i].item())+'x'+str(xn_id[i].item()+1))
                                file.write(' <= '+str(b_constraint_idx))
                                #file.write(' <= '+str(-1.0))
                                file.write('\n')
                        file.write("\nbinary\n")
                        file.write("".join([f" x{j+1}" for j in range(single_graph.x_t.shape[0])]))
                    graph_idx_ = graph_idx_ + 1
                            
                        
                            
    #import pdb; pdb.set_trace()


    


if __name__ == '__main__':
    main()