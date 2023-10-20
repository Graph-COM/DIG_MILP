import argparse
import pyscipopt
import re

def modify_sol_file(sol_file_path, output_file_path):
    with open(sol_file_path, 'r') as file:
        sol_content = file.read()

    # Use regex to find variable names starting with 'x' followed by a number
    variable_pattern1 = re.compile(r'x(\d+)')

    # Use regex substitution to replace 'x' followed by a number with 'x0' followed by the same number
    def replace_var1(match):
        return 'x' + str(int(match.group(1))+1)

    modified_content = re.sub(variable_pattern1, replace_var1, sol_content)

    with open(output_file_path, 'w') as file:
        file.write(modified_content)

def main():
    parser = argparse.ArgumentParser(description='sol file parser')
    parser.add_argument('--sol_folder', dest = 'sol_folder', type = str, default = './data/raw_solution/', help = 'which folder to get the mps instances')
    parser.add_argument('--new_sol_folder', dest = 'new_sol_folder', type = str, default = './data/raw_solution1/', help = 'which folder to solve the sol instances')
    args = parser.parse_args()

    file_list = ['cvs08r139-94','cvs16r70-62','cvs16r89-60','cvs16r106-72','cvs16r128-89']
    for sol_name in file_list:
        old_file = args.sol_folder + sol_name +'.sol'
        new_file = args.new_sol_folder + sol_name +'.sol'
        modify_sol_file(old_file, new_file)


if __name__ == '__main__':
    main()
