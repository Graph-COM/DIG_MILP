import argparse
import pyscipopt
import re

def modify_lp_file(lp_file_path, output_file_path):
    with open(lp_file_path, 'r') as file:
        lp_content = file.read()

    # Use regex to find variable names starting with 'x' followed by a number
    variable_pattern2 = re.compile(r'c(\d+)')

    # Use regex substitution to replace 'x' followed by a number with 'x0' followed by the same number

    def replace_var2(match):
        return 'C' + str(int(match.group(1)))

    modified_content = re.sub(variable_pattern2, replace_var2, lp_content)

    with open(output_file_path, 'w') as file:
        file.write(modified_content)

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--lp_folder', dest = 'lp_folder', type = str, default = './data/primal_format_max/', help = 'which folder to get the mps instances')
    parser.add_argument('--new_lp_folder', dest = 'new_lp_folder', type = str, default = './data/primal_format2/', help = 'which folder to solve the lp instances')
    args = parser.parse_args()

    file_list = ['iis-glass-cov','iis-hc-cov']
    for lp_name in file_list:
        old_file = args.lp_folder + lp_name +'.lp'
        new_file = args.new_lp_folder + lp_name +'.lp'
        modify_lp_file(old_file, new_file)


if __name__ == '__main__':
    main()
