import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import spearmanr
import matplotlib

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 17})

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--ratio_list', dest = 'ratio_list', type = list, default = [0.10,0.20,0.30], help = 'which folder to get the lp instances')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 200, help = 'the number of instances')
    parser.add_argument('--seed', dest = 'seed', type = int, default = 123, help = 'random seed')
    parser.add_argument('--save_folder', dest = 'save_folder', type = str, default = '.', help = 'which folder to get the lp instances')
    parser.add_argument('--paint_folder', dest = 'paint_folder', type = str, default = './paint/', help = 'which folder to get the lp instances')
    args = parser.parse_args()

    for trial_id in range(0,1):
        seed_idx = 0
        for seed in range(45):
            save_folder = args.save_folder + '/time_log' + str(trial_id)+'/' + str(seed) + '/' 
            save_folder2 = args.save_folder + '/time_log' + str(trial_id)+'/' + str(seed) + '/' 
            primal_path = save_folder + 'primal.npy'    
            generate_path_1 = save_folder2 + 'generate1.npy'
            generate_path_2 = save_folder2 + 'generate2.npy'
            generate_path_3 = save_folder2 + 'generate3.npy'
            
            if seed_idx == 0:
                primal_list = np.array(np.load(primal_path)).reshape(1,-1)[:,:args.num_instance]
                generate_list1 = np.array(np.load(generate_path_1)).reshape(1,-1)[:,:args.num_instance]
                generate_list2 = np.array(np.load(generate_path_2)).reshape(1,-1)[:,:args.num_instance]
                generate_list3 = np.array(np.load(generate_path_3)).reshape(1,-1)[:,:args.num_instance]
            else:
                primal_list = np.vstack((primal_list, np.array(np.load(primal_path)).reshape(1,-1)[:,:args.num_instance]))
                generate_list1 = np.vstack((generate_list1, np.array(np.load(generate_path_1)).reshape(1,-1)[:,:args.num_instance]))
                generate_list2 = np.vstack((generate_list2, np.array(np.load(generate_path_2)).reshape(1,-1)[:,:args.num_instance]))
                generate_list3 = np.vstack((generate_list3, np.array(np.load(generate_path_3)).reshape(1,-1)[:,:args.num_instance]))
            seed_idx = seed_idx + 1
            #import pdb; pdb.set_trace()
        if trial_id == 0:
            whole_primal_list = primal_list / 3
            whole_generate_list1 = generate_list1 / 3
            whole_generate_list2 = generate_list2 / 3
            whole_generate_list3 = generate_list3 / 3
        else:
            whole_primal_list = whole_primal_list + primal_list / 3
            whole_generate_list1 = whole_generate_list1 + generate_list1 / 3
            whole_generate_list2 = whole_generate_list2 + generate_list2 / 3
            whole_generate_list3 = whole_generate_list3 + generate_list3 / 3


    primal_list_mean = np.mean(whole_primal_list, 1)
    generate_list_mean1 = np.mean(whole_generate_list1, 1)
    generate_list_mean2 = np.mean(whole_generate_list2, 1)
    generate_list_mean3 = np.mean(whole_generate_list3, 1)

    result1 = scipy.stats.linregress(primal_list_mean, generate_list_mean1)
    print('ratio0.1:')
    print('r'+str(result1.rvalue))
    print('p'+str(result1.pvalue))
    result2 = scipy.stats.linregress(primal_list_mean, generate_list_mean2)
    print('ratio0.2:')
    print('r'+str(result2.rvalue))
    print('p'+str(result2.pvalue))
    result3 = scipy.stats.linregress(primal_list_mean, generate_list_mean3)
    print('ratio0.3:')
    print('r'+str(result3.rvalue))
    print('p'+str(result3.pvalue))
    plt.figure(figsize=(8, 8))  # Set the figure size
    plt.scatter(primal_list_mean, generate_list_mean1, color='#1a6840', marker='o', label='xy')
    plt.xlabel('original')
    plt.ylabel('ours (ratio = 0.1)')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(args.paint_folder+'ratio010.pdf')
    plt.clf()
    plt.figure(figsize=(8, 8))  # Set the figure size
    plt.scatter(primal_list_mean, generate_list_mean2, color='#1a6840', marker='o', label='xy')
    plt.xlabel('original')
    plt.ylabel('ours (ratio = 0.2)')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(args.paint_folder+'ratio020.pdf')

    plt.clf()
    plt.figure(figsize=(8, 8))  # Set the figure size
    plt.scatter(primal_list_mean, generate_list_mean3, color='#1a6840', marker='o', label='xy')
    plt.xlabel('original')
    plt.ylabel('ours (ratio = 0.3)')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(args.paint_folder+'ratio030.pdf')

    
    #import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()