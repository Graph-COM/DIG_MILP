import pyscipopt
import argparse
import time
import numpy as np
from tqdm import tqdm
import pickle

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--generate_primal_folder', dest = 'generate_primal_folder', type = str, default = '../data/generate_primal/m200n400_mixed2/050_a5wobnumx_regression_nosigmoid/', help = 'which folder to get the lp instances')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 1000, help = 'the number of instances')
    parser.add_argument('--solution_folder', dest = 'solution_folder', type = str, default = './dataset/generate_primal_solution/m200n400_mixed2/050_a5wobnumx_regression_nosigmoid/', help = 'which folder to save the lp solutions')
    args = parser.parse_args()


    time_list = []
    # get the primal solutions x
    for instance_idx in tqdm(range(args.num_instance)):
        solver = pyscipopt.Model()
        #solver.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
        #solver.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        #solver.disablePropagation()
        #solver.setIntParam('display/verblevel', 0)
        solver.setRealParam('limits/time', 300)
        solver.readProblem(args.generate_primal_folder+str(instance_idx)+'.lp')
        start_time = time.time()
        solver.optimize()
        end_time = time.time()
        solutions = {}
        time_cost = end_time - start_time
        time_list.append(time_cost)
        if solver.getStatus() == 'optimal':
            # Retrieve the solution
            solution = solver.getBestSol()
            for variable in solver.getVars():
                var_name = int(str(variable.name)[1:])
                var_value = solver.getVal(variable)
                solutions[var_name] = var_value
            time_cost = end_time - start_time
            time_list.append(time_cost)
            solution_file_path = args.solution_folder +str(instance_idx)+'.pkl'
            with open(solution_file_path, 'wb') as f:
                pickle.dump(solutions, f)
            print(solver.getObjVal())
        else:
            print('The problem is infeasible')

    print('mean solution time:'+str(np.mean(time_list)))
    print('std solution time:'+str(np.std(time_list)))
    #print(time_list)

if __name__ == '__main__':
    main()