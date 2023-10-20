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
    for instance_idx in tqdm(range(args.num_instance)):
        solver = pyscipopt.Model()
        #solver.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
        #solver.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        #solver.disablePropagation()
        solver.setIntParam('display/verblevel', 0)
        solver.readProblem(args.dual_folder+str(instance_idx)+'.lp')
        start_time = time.time()
        solver.optimize()
        end_time = time.time()
        solutions = {}

        # Retrieve the solution
        '''if solver.getNVars() != 600:
            print( solver.getNVars())
            print('wrong')'''
        if solver.getStatus() == "optimal":
            solution = solver.getBestSol()
            for variable in solver.getVars():
                var_name = int(str(variable.name)[1:])
                var_value = solver.getVal(variable)
                solutions[var_name] = round(var_value)
            time_cost = end_time - start_time
            time_list.append(time_cost)
            solution_file_path = args.solution_folder +str(instance_idx)+'.pkl'
            with open(solution_file_path, 'wb') as f:
                pickle.dump(solutions, f)
        else:
            print('instance'+str(instance_idx)+'is infesible')

    print('mean solution time:'+str(np.mean(time_list)))
    print('std solution time:'+str(np.std(time_list)))

if __name__ == '__main__':
    main()