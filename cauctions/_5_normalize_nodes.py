import argparse
import pickle
from tqdm import tqdm



def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 1000, help = 'the number of instances')
    parser.add_argument('--dual_solution', dest = 'dual_solution', type = str, default = './data/dual_solution/', help = 'which folder to save the dual solutions')
    parser.add_argument('--dual_slack', dest = 'dual_slack', type = str, default = './data/dual_slack/', help = 'which folder to save the dual solutions')
    parser.add_argument('--primal_solution', dest = 'primal_solution', type = str, default = './data/primal_solution/', help = 'which folder to save the primal solutions')
    parser.add_argument('--primal_slack', dest = 'primal_slack', type = str, default = './data/primal_slack/', help = 'which folder to save the primal solutions')
    parser.add_argument('--normalize_primal_solution', dest = 'normalize_primal_solution', type = str, default = './data/normalize_primal_solution/', help = 'which folder to save the normalized primal solutions')
    parser.add_argument('--normalize_primal_slack', dest = 'normalize_primal_slack', type = str, default = './data/normalize_primal_slack/', help = 'which folder to save the normalized primal solutions')
    parser.add_argument('--normalize_dual_solution', dest = 'normalize_dual_solution', type = str, default = './data/normalize_dual_solution/', help = 'which folder to save the normalized dual solutions')
    parser.add_argument('--normalize_dual_slack', dest = 'normalize_dual_slack', type = str, default = './data/normalize_dual_slack/', help = 'which folder to save the normalized dual solutions')
    parser.add_argument('--normalize_statistics', dest = 'normalize_statistics', type = str, default = './data/normalize_statistics/', help = 'which folder to save the normalized statistics')
    parser.add_argument('--count', dest = 'count', type = int, default = 1, help = 'whether to count the max and min')
    args = parser.parse_args()
    if args.count == 1:
        max_x = 0
        min_x = 10000
        max_y = 0
        min_y = 10000
        max_s = 0
        min_s = 10000
        max_r = 0
        min_r = 10000
        for instance_idx in range(args.num_instance):
            with open(args.primal_solution+str(instance_idx)+'.pkl', "rb") as x_file:    
                dict_x = pickle.load(x_file)
                max_this_dict = max(dict_x.values())
                if max_this_dict > max_x:
                    max_x = max_this_dict
                min_this_dict = min(dict_x.values())
                if min_this_dict < min_x:
                    min_x = min_this_dict
            with open(args.dual_solution+str(instance_idx)+'.pkl', "rb") as y_file:    
                dict_y = pickle.load(y_file)
                max_this_dict = max(dict_y.values())
                if max_this_dict > max_y:
                    max_y = max_this_dict
                min_this_dict = min(dict_y.values())
                if min_this_dict < min_y:
                    min_y = min_this_dict  
            with open(args.primal_slack+str(instance_idx)+'.pkl', "rb") as r_file:    
                dict_r = pickle.load(r_file)
                max_this_dict = max(dict_r.values())
                if max_this_dict > max_r:
                    max_r = max_this_dict
                min_this_dict = min(dict_r.values())
                if min_this_dict < min_r:
                    min_r = min_this_dict  
            with open(args.dual_slack+str(instance_idx)+'.pkl', "rb") as s_file:    
                dict_s = pickle.load(s_file)
                max_this_dict = max(dict_s.values())
                if max_this_dict > max_s:
                    max_s = max_this_dict
                min_this_dict = min(dict_s.values())
                if min_this_dict < min_s:
                    min_s = min_this_dict  
        normalize_statistics = {}
        normalize_statistics['max_x'] = max_x
        normalize_statistics['max_y'] = max_y
        normalize_statistics['max_s'] = max_s
        normalize_statistics['max_r'] = max_r
        normalize_statistics['min_x'] = min_x
        normalize_statistics['min_y'] = min_y
        normalize_statistics['min_s'] = min_s
        normalize_statistics['min_r'] = min_r
        with open (args.normalize_statistics+'statistics.pkl','wb') as f:
            pickle.dump(normalize_statistics, f)
    else:
        with open (args.normalize_statistics+'statistics.pkl','rb') as f:
            normalize_statistics = pickle.load(f)
        max_x = normalize_statistics['max_x']
        max_y = normalize_statistics['max_y']
        max_s = normalize_statistics['max_s']
        max_r = normalize_statistics['max_r']
        min_x = normalize_statistics['min_x']
        min_y = normalize_statistics['min_y']
        min_s = normalize_statistics['min_s']
        min_r = normalize_statistics['min_r']
    
    for instance_idx in tqdm(range(args.num_instance)):
        with open(args.primal_solution+str(instance_idx)+'.pkl', "rb") as x_file:    
            dict_x = pickle.load(x_file)
            for key in dict_x:
                dict_x[key] = (dict_x[key] - min_x) / (max_x - min_x)
        with open(args.normalize_primal_solution+str(instance_idx)+'.pkl', "wb") as x_write_file:
            pickle.dump(dict_x, x_write_file)
        with open(args.dual_solution+str(instance_idx)+'.pkl', "rb") as y_file:    
            dict_y = pickle.load(y_file)
            for key in dict_y:
                dict_y[key] = (dict_y[key] - min_y) / (max_y - min_y)
        with open(args.normalize_dual_solution+str(instance_idx)+'.pkl', "wb") as y_write_file:
            pickle.dump(dict_y, y_write_file)
        with open(args.primal_slack+str(instance_idx)+'.pkl', "rb") as r_file:    
            dict_r = pickle.load(r_file)
            for key in dict_r:
                dict_r[key] = (dict_r[key] - min_r) / (max_r - min_r)
        with open(args.normalize_primal_slack+str(instance_idx)+'.pkl', "wb") as r_write_file:
            pickle.dump(dict_r, r_write_file)
        with open(args.dual_slack+str(instance_idx)+'.pkl', "rb") as s_file:    
            dict_s = pickle.load(s_file)
            for key in dict_s:
                dict_s[key] = (dict_s[key] - min_s) / (max_s - min_s)
        with open(args.normalize_dual_slack+str(instance_idx)+'.pkl', "wb") as s_write_file:
            pickle.dump(dict_s, s_write_file)

    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()