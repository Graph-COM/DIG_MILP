import pyscipopt
import argparse
from tqdm import tqdm

import ecole
from ecole.observation import MilpBipartite


def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--primal_folder', dest = 'primal_folder', type = str, default = './data/primal_format/', help = 'which folder to get the lp instances')
    parser.add_argument('--dual_folder', dest = 'dual_folder', type = str, default = './data/dual_format/', help = 'which folder to save the dual instances')
    args = parser.parse_args()

    env = ecole.environment.Configuring(observation_function = MilpBipartite())   
    
    all_file_list = ['iis-glass-cov','iis-hc-cov']
    for lp_file in tqdm(all_file_list):
        env.reset(instance=args.primal_folder + lp_file+'.lp')
        obs, _, _, _, _ = env.reset(args.primal_folder+lp_file+'.lp')
        # constraint_features: constraint vertex, b
        # edge_features: the coefficient of the adjacency matrix aij
        # edge_features.indices: (edge index in VC bipartite graph: variable - constraint idx) 
        # variable_features: c, variable type (binary: 1,0,0,0), upper, has_upper, lower, haslower

        solver = pyscipopt.Model()
        solver.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
        solver.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        solver.setIntParam('display/verblevel', 0)
        solver.disablePropagation()

        solver.readProblem(args.primal_folder+lp_file+'.lp')
        write_path = args.dual_folder+lp_file+'.lp'

        b_dict = {}
        c_dict = {}
        # Get the constraints
        constraints = solver.getConss()

        with open(write_path, "w") as file:
            file.write('Minimize\n')
            file.write(' obj: ')

        # Get b
        num_constraints = 0
        for constraint_idx in constraints:
            if str(constraint_idx).startswith('C'):
                rhs = solver.getRhs(constraint_idx)
                num_constraints = num_constraints + 1
                b_dict[num_constraints] = rhs
                
        num_variables = solver.getNVars()

        # Write objective
        with open(write_path, "a") as file:
            for key, value in b_dict.items():
                if value != 0:
                    file.write('+ '+ str(value)+' y'+str(key)+' ')
            for i in range(num_variables):
                file.write('+ '+' 1y'+str(i+num_constraints+1)+' ')
            file.write('\n')
            file.write('Subject to\n')
        print(num_constraints)
        # Get c
        objective_function = solver.getObjective()
        for term, coefficient in objective_function.terms.items():
            c_dict[int(str(term[0])[1:])] = coefficient
        for variable_idx in range(num_variables):
            c_dict.setdefault(variable_idx+1, 0)
        # Write the constraints
        for idx in range(num_variables):
            with open(write_path, "a") as file:
                file.write('C'+str(idx+1)+': ')
            for constraint_idx in constraints:
                if str(constraint_idx).startswith('C'):
                    i_th = 'x'+str(idx+1)
                    coeff_dict = solver.getValsLinear(constraint_idx) 
                    if i_th in coeff_dict:
                        value = coeff_dict[i_th]
                        with open(write_path, "a") as file:
                            file.write('+ '+ str(value)+' y'+str(constraint_idx)[1:]+' ')
            with open(write_path, "a") as file:
                file.write('+ '+ ' 1y'+str(int(str(num_constraints))+idx+1)+' ')
                tmp_rhs = c_dict[idx+1]
                file.write('>= '+str(tmp_rhs)+'\n')
        for idx in range(num_constraints+num_variables):
            with open(write_path, "a") as file:
                file.write('C'+str(idx+num_variables+1)+ ': 1y'+str(1+idx)+' >= 0\n')
        
        # Write variables
        with open(write_path, "a") as file:
            file.write("\nGeneral Integer\n")
            file.write("".join([f" y{j+1}" for j in range(num_variables+ num_constraints)]))
        #import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()