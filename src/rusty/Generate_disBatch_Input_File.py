##################################################################
## Generate input file for disBatch
## Run DMRG for different magnetic field
##################################################################

import numpy as np

def generate_input_file(input_field, input_delta, task_file):
    '''Generate corresponding folders and input files based on chemical potential'''

    folder_name = "j" + "{:.2f}".format(input_field) + "/delta" + "{:.2f}".format(input_delta) + "/"; # print(folder_name)
    task_file.write("cd " + folder_name \
        + " &&  julia --threads=1 Heisenberg_Dimerized_TEBD.jl" + " &> heisenberg_tebd" \
        + "{:.2f}".format(input_field) + "{:.2f}".format(input_delta) + ".log" + "\n")
    

def main():
    field_strength = np.around(np.arange(0.05, 0.6, 0.05),  decimals = 2)    
    delta_strength = np.around(np.arange(0.02, 0.24, 0.02), decimals = 2)

    submit_file = open("heisenberg", "a")
    for tmpField in field_strength:
        for tmpDelta in delta_strength:
            generate_input_file(tmpField, tmpDelta, submit_file)
    submit_file.close() 


if __name__ == "__main__":
    main()