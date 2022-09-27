import sys 
import os 


import matplotlib.pyplot as plt
import numpy as np

PATH_TO_REPOSITORIES_FOLDER = "foo"
PATH_TO_DATA = "bar"

NORM_BOX_COORDINATES = None
ROI_COORDINATES = None

sys.path.insert(0, PATH_TO_REPOSITORIES_FOLDER) 

from BEC1_Analysis.code.measurement import Measurement
from BEC1_Analysis.code import image_processing_functions

def main():
    my_measurement = Measurement(measurement_directory_path = PATH_TO_DATA, hold_images_in_memory = False, run_parameters_verbose = True)
    print("Initializing")
    my_measurement._initialize_runs_dict(use_saved_params = False)
    my_measurement.dump_runs_dict()
    run_to_use = 0
    box_set = False
    while (not box_set) and run_to_use < len(my_measurement.runs_dict):
        try:
            my_measurement.set_box("ROI", box_coordinates = ROI_COORDINATES, run_to_use = run_to_use)
        except TypeError:
            pass 
        else:
            box_set = True
        run_to_use += 1    
    nominal_resonance_frequency_1 = my_measurement.experiment_parameters["state_1_unitarity_res_freq_MHz"]
    nominal_resonance_frequency_3 = my_measurement.experiment_parameters["state_3_unitarity_res_freq_MHz"]
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    pixel_area = my_measurement.experiment_parameters["top_um_per_pixel"]
    polrot_phase_sign = my_measurement.experiment_parameters["polrot_phase_sign"]
    #Other code as necessary
    for run_id in my_measurement.runs_dict:
        current_run = my_measurement.runs_dict[run_id]
        print("Checking run: " + str(run_id))
        current_nominal_freq_A = current_run.parameters["ImagFreq1"] 
        current_nominal_freq_B = current_run.parameters["ImagFreq2"] 
        detuning_1A = frequency_multiplier * (current_nominal_freq_A - nominal_resonance_frequency_1) 
        detuning_1B = frequency_multiplier * (current_nominal_freq_B - nominal_resonance_frequency_1) 
        detuning_2A = frequency_multiplier * (current_nominal_freq_A - nominal_resonance_frequency_3) 
        detuning_2B = frequency_multiplier * (current_nominal_freq_B - nominal_resonance_frequency_3)

        current_run_image_array_A = current_run.get_image('TopA', memmap = True) 
        current_run_image_array_B = current_run.get_image('TopB', memmap = True) 
        current_run_abs_image_A = image_processing_functions.get_absorption_image(current_run_image_array_A, 
                                                                    ROI = my_measurement.measurement_parameters["ROI"])
        current_run_abs_image_B = image_processing_functions.get_absorption_image(current_run_image_array_B, ROI = my_measurement.measurement_parameters["ROI"])
        current_density_1, current_density_3 = image_processing_functions.get_atom_density_from_polrot_images(current_run_abs_image_A, current_run_abs_image_B,
                                                                    detuning_1A, detuning_1B, detuning_2A, detuning_2B)
        #Process the densities however you like
    #Do whatever code afterwards



if __name__ == "__main__":
    main()