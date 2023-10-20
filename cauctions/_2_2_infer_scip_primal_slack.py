import pyscipopt
import argparse
import time
import numpy as np
from tqdm import tqdm
import pickle

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--primal_folder', dest = 'primal_folder', type = str, default = './data/primal_format/', help = 'which folder to get the lp instances')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 1000, help = 'the number of instances')
    parser.add_argument('--solution_folder', dest = 'solution_folder', type = str, default = './data/primal_solution/', help = 'which folder to save the lp solutions')
    parser.add_argument('--slack_folder', dest = 'slack_folder', type = str, default = './data/primal_slack/', help = 'which folder to save the lp solutions')
    args = parser.parse_args()


    time_list = []
    # get the primal solutions x
    for instance_idx in tqdm(range(args.num_instance)):
        solver = pyscipopt.Model()
        solver.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
        solver.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        solver.disablePropagation()
        solver.setIntParam('display/verblevel', 0)
        solver.readProblem(args.primal_folder+str(instance_idx)+'.lp')


        with open(args.solution_folder+str(instance_idx)+'.pkl', 'rb') as solution_file:
            solutions = pickle.load(solution_file)

        # get the slack r, Ax + rm = b, x+rn = 1
        r_dict = {}
        constraints = solver.getConss()
        num_constraints = 0
        for constraint_idx in constraints:
            if str(constraint_idx).startswith('C'):
                coeff_dict = solver.getValsLinear(constraint_idx)
                lhs = 0
                for key, values in coeff_dict.items():
                    x_key = solutions[int(key[1:])]
                    lhs = lhs + x_key * values
                rhs = solver.getRhs(constraint_idx)
                r = rhs - lhs
                r_dict[int(str(constraint_idx)[1:])] = r
                num_constraints = num_constraints + 1
        num_variables = solver.getNVars()
        for solution_idx in range(num_variables):
            r_dict[num_constraints + solution_idx + 1] = 1 - solutions[solution_idx+1]
        slack_file_path = args.slack_folder + str(instance_idx)+'.pkl'
        with open(slack_file_path, 'wb') as f:
            pickle.dump(r_dict, f)

if __name__ == '__main__':
    main()