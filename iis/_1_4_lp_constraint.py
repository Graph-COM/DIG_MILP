import argparse
import pyscipopt
import re

def modify_lp_file(lp_file_path, output_file_path):
    with open(lp_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
        merged_line = ''
        last_c_flag = 0
        for line in input_file:
            if line.startswith(" C"):
                if last_c_flag == 1:
                    output_file.write(' ' + merged_line + "\n")
                    merged_line = line.strip()
                else:
                    merged_line = line.strip()
                last_c_flag = 1
            else:
                merged_line += line.strip()
                # Write the merged line to the output file
                output_file.write(' ' + merged_line + "\n")
                merged_line = ''
                last_c_flag = 0
            

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--lp_folder', dest = 'lp_folder', type = str, default = './data/primal_format2/', help = 'which folder to get the mps instances')
    parser.add_argument('--new_lp_folder', dest = 'new_lp_folder', type = str, default = './data/primal_format3/', help = 'which folder to solve the lp instances')
    args = parser.parse_args()

    file_list = ['iis-hc-cov']
    for lp_name in file_list:
        old_file = args.lp_folder + lp_name +'.lp'
        new_file = args.new_lp_folder + lp_name +'.lp'
        modify_lp_file(old_file, new_file)


if __name__ == '__main__':
    main()
