import pyscipopt
import argparse
import time
import numpy as np
from tqdm import tqdm
import pickle
import os
import random


def setup_seed(seed):
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--primal_folder', dest = 'primal_folder', type = str, default = '../data/primal_format/', help = 'which folder to get the lp instances')
    parser.add_argument('--generate_primal_folder', dest = 'generate_primal_folder', type = str, default = '../data/generate_primal/', help = 'which folder to get the lp instances')
    parser.add_argument('--ratio_list', dest = 'ratio_list', type = list, default = [0.10,0.20,0.30], help = 'which folder to get the lp instances')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 200, help = 'the number of instances')
    parser.add_argument('--seed', dest = 'seed', type = int, default = 123, help = 'random seed')
    parser.add_argument('--save_folder', dest = 'save_folder', type = str, default = './time_log/', help = 'which folder to get the lp instances')
    args = parser.parse_args()

    if not os.path.exists(args.save_folder + str(args.seed)+'/'):
        os.mkdir(args.save_folder + str(args.seed)+'/')
    setup_seed(args.seed)

    scorefunc = random.choice(['s', 'p', 'q'])
    scorefac = random.random()
    preferbinary = random.choice([True, False])
    clamp = random.uniform(0,0.5)
    midpull = random.random()
    midpullreldomtrig = random.random()
    lpgainnormalize = random.choice(['s', 'd', 'l'])
    pricing = random.choice(['l','a','f','p','s','q','d'])
    colagelimit = random.randint(0,100)
    rowagelimit = random.randint(0,100)
    childsel = random.choice(['d','u','p','i','l','r','h'])
    minortho = random.random()
    minorthoroot = random.random()
    maxcuts = random.randint(0,1000)
    maxcutsroot = random.randint(0,10000)
    cutagelimit = random.randint(0,200)
    poolfreq = random.randint(0,100)
    
    parameter_dict = {
    "branching/scorefunc": scorefunc,  #s, p, q
    "branching/scorefac": scorefac, # [0, 1]
    "branching/preferbinary": preferbinary, # True False
    "branching/clamp": clamp, #[0,0.5]
    "branching/midpull": midpull, #[0,1]
    "branching/midpullreldomtrig": midpullreldomtrig,#[0,1]
    "branching/lpgainnormalize": lpgainnormalize,# d,l,s
    "lp/pricing": pricing,# lafpsqd
    "lp/colagelimit": colagelimit,#[-1,2147483647]
    "lp/rowagelimit": rowagelimit,#[-1,2147483647]
    "nodeselection/childsel": childsel,# dupilrh
    "separating/minortho": minortho,#[0,1],
    "separating/minorthoroot": minorthoroot,#[0,1]
    "separating/maxcuts": maxcuts,#[0,2147483647]
    "separating/maxcutsroot": maxcutsroot,#[0,2147483647]
    "separating/cutagelimit": cutagelimit,#[-1,2147483647]
    "separating/poolfreq": poolfreq#[-1,65534]
    }

    primal_time_list = []
    generate_time_list1 = []
    generate_time_list2 = []
    generate_time_list3 = []
    # get the primal solutions x
    for instance_idx in tqdm(range(args.num_instance)):

        solver = pyscipopt.Model()
        #solver.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
        solver.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        #solver.disablePropagation()
        solver.setRealParam('limits/time', 300)
        solver.setIntParam('display/verblevel', 0)
        solver.setParams(parameter_dict)
        solver.readProblem(args.primal_folder+str(instance_idx)+'.lp')
        start_primal_time = time.time()
        solver.optimize()
        end_primal_time = time.time()

        time_cost = end_primal_time - start_primal_time
        primal_time_list.append(time_cost)
        
        for ratio in args.ratio_list:
            problem_instance = args.generate_primal_folder+"{:.2f}".format(ratio).replace('.','')+'_a150lr1e3/'+str(instance_idx)+'.lp'
            solver.readProblem(problem_instance)
            start_generate_time = time.time()
            solver.optimize()
            end_generate_time = time.time()
            if ratio==0.1:
                generate_time_list1.append(end_generate_time - start_generate_time)
            elif ratio==0.2:
                generate_time_list2.append(end_generate_time - start_generate_time)
            elif ratio==0.3:
                generate_time_list3.append(end_generate_time - start_generate_time)
    print('mean solution time primal:'+str(np.mean(primal_time_list)))
    print('std solution time primal:'+str(np.std(primal_time_list)))
    np.save(args.save_folder + str(args.seed)+'/primal.npy', primal_time_list)
    print('ratio 0.1 mean:'+str(np.mean(generate_time_list1)))
    print('ratio 0.1 std:'+str(np.std(generate_time_list1)))
    np.save(args.save_folder + str(args.seed)+'/generate1.npy', generate_time_list1)
    print('ratio 0.2 mean:'+str(np.mean(generate_time_list2)))
    print('ratio 0.2 std:'+str(np.std(generate_time_list2)))
    np.save(args.save_folder + str(args.seed)+'/generate2.npy', generate_time_list2)
    print('ratio 0.3 mean:'+str(np.mean(generate_time_list3)))
    print('ratio 0.3 std:'+str(np.std(generate_time_list3)))
    np.save(args.save_folder + str(args.seed)+'/generate3.npy', generate_time_list3)

    

if __name__ == '__main__':
    main()