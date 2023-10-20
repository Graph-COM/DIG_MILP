import argparse
import pyscipopt
import pickle


def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 1000, help = 'the number of instances')
    parser.add_argument('--primal_folder', dest = 'primal_folder', type = str, default = './data/primal_format/', help = 'which folder to get the lp instances')
    parser.add_argument('--normalize_statistics', dest = 'normalize_statistics', type = str, default = './data/normalize_statistics/', help = 'which folder to save the normalized statistics')
    args = parser.parse_args()

    max_weight = -10000
    min_weight = 10000
    weight_dict = {}
    for instance_idx in range(args.num_instance):
        solver = pyscipopt.Model()
        solver.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
        solver.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        solver.disablePropagation()
        solver.setIntParam('display/verblevel', 0)
        solver.readProblem(args.primal_folder+str(instance_idx)+'.lp')
        constraints = solver.getConss()
        for constraint_idx in constraints:
            coeff_dict = solver.getValsLinear(constraint_idx)
            this_max = max(coeff_dict.values())
            if this_max > max_weight:
                max_weight = this_max
            this_min = min(coeff_dict.values())
            if this_min < min_weight:
                min_weight = this_min
        
    weight_dict['max_weight'] = max_weight
    weight_dict['min_weight'] = min_weight
    if min_weight == max_weight:
        weight_dict['equal'] = 1
        weight_dict['toward1'] = 1 - min_weight
    else:
        weight_dict['equal'] = 0
        weight_dict['toward1'] = 0
    with open(args.normalize_statistics+str('normalize_weight.pkl'), 'wb') as f:
        pickle.dump(weight_dict, f)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()