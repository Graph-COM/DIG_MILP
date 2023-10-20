import pyscipopt
import argparse
import time
import numpy as np
from tqdm import tqdm
import pickle

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--primal_folder', dest = 'primal_folder', type = str, default = './data/primal_format/', help = 'which folder to get the lp instances')
    parser.add_argument('--raw_solution_folder', dest = 'raw_solution_folder', type = str, default = './data/raw_solution1/', help = 'which folder to get the lp instances')
    parser.add_argument('--solution_folder', dest = 'solution_folder', type = str, default = './data/primal_solution/', help = 'which folder to save the lp solutions')
    parser.add_argument('--slack_folder', dest = 'slack_folder', type = str, default = './data/primal_slack/', help = 'which folder to save the lp solutions')
    args = parser.parse_args()


    time_list = []
    all_file_list = ['cvs08r139-94','cvs16r70-62','cvs16r89-60','cvs16r106-72','cvs16r128-89']
    #file_list = [args.selected_file]
    # get the primal solutions x
    for lp_file in all_file_list:
        solver = pyscipopt.Model()
        solver.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
        solver.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        solver.disablePropagation()
        solver.setIntParam('display/verblevel', 0)
        solver.readProblem(args.primal_folder+lp_file+'.lp')
        solutions = {}
        obj_claim = 0
        # load the solution, start from x0
        with open(args.raw_solution_folder+lp_file+'.sol') as sol_file:
            for line in sol_file:
                elements = line.strip().split(' ')
                if elements[0].startswith('=obj='):
                    obj_claim = int(float(elements[-1]))
                elif elements[0].startswith('x'):
                    solutions[int(elements[0][1:])] = int(float(elements[-1]))
        num_variables = solver.getNVars()
        for variable_idx in range(num_variables):
            solutions.setdefault(variable_idx+1, 0)
        with open(args.solution_folder+lp_file+'.pkl', "wb") as solution_f:
            pickle.dump(solutions, solution_f)

        # verify the objective
        obj_verify = 0
        objective_dict = solver.getObjective()
        for variable in objective_dict:
            this_c = objective_dict[variable]
            obj_verify = obj_verify + this_c * solutions[int(str(variable[0])[1:])]

        if obj_claim == -obj_verify:
            print('solution verified')
        else:
            print('wrong solution!!!')
        
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
        slack_file_path = args.slack_folder + lp_file +'.pkl'
        with open(slack_file_path, 'wb') as f:
            pickle.dump(r_dict, f)

if __name__ == '__main__':
    main()