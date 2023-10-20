import pyscipopt
import argparse
import time
import numpy as np
from tqdm import tqdm
import pickle

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--dual_folder', dest = 'dual_folder', type = str, default = './data/dual_format/', help = 'which folder to get the dual instances')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 1000, help = 'the number of instances')
    parser.add_argument('--solution_folder', dest = 'solution_folder', type = str, default = './data/dual_solution/', help = 'which folder to save the dual solutions')
    parser.add_argument('--slack_folder', dest = 'slack_folder', type = str, default = './data/dual_slack/', help = 'which folder to save the dual solutions')
    args = parser.parse_args()


    time_list = []
    # get the dual solutions y
    all_file_list = ['cvs08r139-94','cvs16r70-62','cvs16r89-60','cvs16r106-72','cvs16r128-89']
    for lp_file in tqdm(all_file_list):
        solver = pyscipopt.Model()
        #solver.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
        #solver.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        #solver.disablePropagation()
        solver.setIntParam('display/verblevel', 0)
        solver.readProblem(args.dual_folder+lp_file+'.lp')
        start_time = time.time()
        solver.optimize()
        end_time = time.time()
        solutions = {}

        # Retrieve the solution
        if solver.getStatus() == "optimal":
            solution = solver.getBestSol()
            for variable in solver.getVars():
                var_name = int(str(variable.name)[1:])
                var_value = solver.getVal(variable)
                solutions[var_name] = round(var_value)
            time_cost = end_time - start_time
            time_list.append(time_cost)
            solution_file_path = args.solution_folder +lp_file+'.pkl'
            with open(solution_file_path, 'wb') as f:
                pickle.dump(solutions, f)
        else:
            print('instance'+lp_file+'is infesible')

    print('mean solution time:'+str(np.mean(time_list)))
    print('std solution time:'+str(np.std(time_list)))

if __name__ == '__main__':
    main()