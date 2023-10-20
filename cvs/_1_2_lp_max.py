import argparse
import pyscipopt


def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--lp_folder', dest = 'lp_folder', type = str, default = './data/primal_format1/', help = 'which folder to get the mps instances')
    parser.add_argument('--new_lp_folder', dest = 'new_lp_folder', type = str, default = './data/primal_format_max/', help = 'which folder to solve the lp instances')
    args = parser.parse_args()

    file_list = ['cvs08r139-94','cvs16r70-62','cvs16r89-60','cvs16r106-72','cvs16r128-89']
    for lp_name in file_list:
        lp_file = args.lp_folder + lp_name +'.lp'
        solver = pyscipopt.Model()
        solver.readProblem(lp_file)
        # the original problem is minimize, change it into maximize
        objective_expr = solver.getObjective()
        
        solver.setObjective(pyscipopt.quicksum(-objective_expr[term]*term[0]  for term in objective_expr), "maximize")
        solver.setMaximize()

        lp_file = args.new_lp_folder + lp_name +'.lp'
        solver.writeProblem(lp_file)
        objective = solver.getObjective()
        constraints = solver.getConss()
        #import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
