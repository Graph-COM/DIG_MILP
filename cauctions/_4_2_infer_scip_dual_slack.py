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
        solver.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
        solver.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        solver.disablePropagation()
        solver.setIntParam('display/verblevel', 0)
        solver.readProblem(args.dual_folder+str(instance_idx)+'.lp')
        
        solution_file_path = args.solution_folder +str(instance_idx)+'.pkl'
        with open (solution_file_path, 'rb') as f:
            solutions = pickle.load(f)

        # get the slack s, [A^T I]y - c = s
        num_variables = solver.getNVars()
        s_dict = {}
        constraints = solver.getConss()
        num_constraints = 0
        for constraint_idx in constraints:
            if str(constraint_idx).startswith('C'):
                num_constraints = num_constraints + 1
        for constraint_idx in constraints:
            if str(constraint_idx).startswith('C') and int(str(constraint_idx)[1:])<=num_constraints - num_variables:
                coeff_dict = solver.getValsLinear(constraint_idx)
                lhs = 0
                for key, values in coeff_dict.items():
                    x_key = solutions[int(key[1:])]
                    lhs = lhs + x_key * values
                rhs = solver.getLhs(constraint_idx)
                s = lhs - rhs
                if s<0:
                    print('something went wrong, not feasible at all')
                s_dict[int(str(constraint_idx)[1:])] = s
        slack_file_path = args.slack_folder + str(instance_idx)+'.pkl'
        with open(slack_file_path, 'wb') as f:
            pickle.dump(s_dict, f)

if __name__ == '__main__':
    main()