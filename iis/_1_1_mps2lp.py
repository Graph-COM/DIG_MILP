import argparse
import pyscipopt


def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--mps_folder', dest = 'mps_folder', type = str, default = './raw_data/', help = 'which folder to get the mps instances')
    parser.add_argument('--lp_folder', dest = 'lp_folder', type = str, default = './data/primal_format1/', help = 'which folder to solve the lp instances')
    args = parser.parse_args()

    file_list = ['iis-glass-cov','iis-hc-cov']
    for mps_name in file_list:
        mps_file = args.mps_folder + mps_name +'.mps'
        solver = pyscipopt.Model()
        solver.readProblem(mps_file)
        # the original problem is minimize, change it into maximize
        objective_expr = solver.getObjective()
        
        #solver.setObjective(pyscipopt.quicksum(-objective_expr[term]*term[0]  for term in objective_expr), "maximize")
        #solver.setObjective(new_obj, 'maximize')
        
        #solver.setMaximize()
        lp_file = args.lp_folder + mps_name +'.lp'
        solver.writeProblem(lp_file)
        objective = solver.getObjective()
        constraints = solver.getConss()
        #import pdb; pdb.set_trace()




if __name__ == '__main__':
    main()