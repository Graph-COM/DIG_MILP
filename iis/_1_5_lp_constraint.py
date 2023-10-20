import argparse
import pyscipopt
import re

def modify_lp_file(lp_file_path, output_file_path):
    with open(lp_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
        for line in input_file:
            if line.startswith(" C"):
                # Parse the constraint line
                parts = line.split()
                constraint_name = parts[0]
                

                constraint_coeffs = [coeff for coeff in parts[1:-2]]
                constraint_rhs = float(parts[-1])
                
                # Modify the coefficients' signs and update the RHS
                for item_idx in range(len(constraint_coeffs)):
                    item = constraint_coeffs[item_idx]
                    if item.startswith('+') or item.startswith('-'):
                        modified_item = -float(item)
                        constraint_coeffs[item_idx] = modified_item
                modified_rhs = -constraint_rhs

                # Construct the modified constraint line
                modified_line = f"{constraint_name} {' '.join(map(str, constraint_coeffs))} <= {modified_rhs}\n"

                # Write the modified line to the output file
                output_file.write(modified_line)
            else:
                # Write other lines (non-constraint lines) as they are
                output_file.write(line)

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--lp_folder', dest = 'lp_folder', type = str, default = './data/primal_format3/', help = 'which folder to get the mps instances')
    parser.add_argument('--new_lp_folder', dest = 'new_lp_folder', type = str, default = './data/primal_format/', help = 'which folder to solve the lp instances')
    args = parser.parse_args()

    file_list = ['iis-glass-cov','iis-hc-cov']
    for lp_name in file_list:
        old_file = args.lp_folder + lp_name +'.lp'
        new_file = args.new_lp_folder + lp_name +'.lp'
        modify_lp_file(old_file, new_file)


if __name__ == '__main__':
    main()
