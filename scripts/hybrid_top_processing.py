import datetime
import os 
import sys 

import numpy as np 
import matplotlib.pyplot as plt

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_repo_folder = os.path.abspath(path_to_file + "/../../")


sys.path.insert(0, path_to_repo_folder)

from BEC1_Analysis.code.measurement import Measurement
from BEC1_Analysis.code import image_processing_functions, data_fitting_functions, loading_functions, science_functions

ROI_COORDINATES = None
NORM_BOX_COORDINATES = None 

#Syntax: First two numbers denote the index of the initial (transferred from) and 
#final (transferred to) state for spectroscopy. The key 'AB' indicates that the initial 
#state is imaged in topA and the final in topB, whereas 'BA' indicates the reverse.
ALLOWED_IMAGING_MODES = ['polrot', 'abs']

def main():
    measurement_directory_path = get_measurement_directory_input()
    imaging_mode = get_imaging_mode_input()
    main_after_inputs(measurement_directory_path, imaging_mode)

# main() version without command line input, compatible with portal
def main_after_inputs(measurement_directory_path, imaging_mode):
    workfolder_pathname = get_workfolder_path()
    if(not os.path.isdir(workfolder_pathname)):
        os.makedirs(workfolder_pathname)
    my_measurement = setup_measurement(workfolder_pathname, measurement_directory_path)
    counts_1_array, counts_3_array, energies_1_array, energies_3_array = get_hybrid_trap_data(my_measurement, imaging_mode)
    save_and_plot_data(workfolder_pathname, counts_1_array, counts_3_array, energies_1_array, energies_3_array)


def save_and_plot_data(my_measurement, workfolder_pathname, counts_1_array, counts_3_array, energies_1_array, energies_3_array):
    counts_data_saving_path = os.path.join(workfolder_pathname, "Hybrid_Exp_Counts.npy")
    np.save(counts_data_saving_path, np.stack((counts_1_array, counts_3_array)))
    energies_data_saving_path = os.path.join(workfolder_pathname, "Hybrid_Exp_Energies.npy")
    energies_array = (counts_1_array * energies_1_array + counts_3_array * energies_3_array) / (counts_1_array + counts_3_array)
    np.save(energies_data_saving_path, np.stack((energies_1_array, energies_3_array, energies_array))) 
    average_counts_1 = sum(counts_1_array) / len(counts_1_array) 
    counts_1_deviation = np.sqrt(sum(np.square(counts_1_array - average_counts_1)) / len(counts_1_array))
    average_counts_3 = sum(counts_3_array) / len(counts_3_array) 
    counts_3_deviation = np.sqrt(sum(np.square(counts_3_array - average_counts_3)) / len(counts_3_array))
    imbalances = (counts_3_array - counts_1_array) / (counts_3_array + counts_1_array)
    average_imbalance = sum(imbalances) / len(imbalances) 
    imbalances_deviation = np.sqrt(sum(np.square(imbalances - average_imbalance)) / len(imbalances))
    average_energy = sum(energies_array) / len(energies_array)
    energy_deviation = np.sqrt(sum(np.square(energies_array - average_energy)) / len(energies_array))
    average_energy_1 = sum(energies_1_array) / len(energies_1_array)
    average_energy_3 = sum(energies_3_array) / len(energies_3_array)
    report_string = ''
    report_string += "Average Counts State 1: {0:.0f}\n".format(average_counts_1)
    report_string += "State 1 Counts Deviation: {0:.0f}\n".format(counts_1_deviation) 
    report_string += "Average Counts State 3: {0:.0f}\n".format(average_counts_3)
    report_string += "State 3 Counts Deviation: {0:.0f}\n".format(counts_3_deviation)
    report_string += "Average Imbalance: {0:.3f}\n".format(average_imbalance)
    report_string += "Imbalance Deviation: {0:.3f}\n".format(imbalances_deviation)
    report_string += "Average Energy State 1: {0:.1f}\n".format(average_energy_1)
    report_string += "Average Energy State 3: {0:.1f}\n".format(average_energy_3)
    report_string += "Average Total Energy: {0:.1f}\n".format(average_energy)
    report_string += "Total Energy Deviation: {0:.1f}\n".format(energy_deviation)
    loading_functions.universal_clipboard_copy(report_string)
    print(report_string)
    plt.plot(counts_1_array, 'x', label = "State 1 Counts") 
    plt.plot(counts_3_array, 'o', label = "State 3 Counts")
    plt.xlabel("Iteration") 
    plt.ylabel("Atom counts")
    plt.legend()
    measurement_directory_folder_name = os.path.basename(os.path.normpath(my_measurement.measurement_directory_path))
    plt.suptitle("Hybrid Trap Counts: " + measurement_directory_folder_name)
    plt.savefig("Hybrid_Trap_Counts_Figure.png")
    plt.show()
    plt.plot(energies_1_array, 'x', label = "State 1 Energy")
    plt.plot(energies_3_array, 'o', label = "State 3 Energy")
    plt.plot(energies_array, 'd', label = "Average Energy") 
    plt.legend() 
    plt.xlabel("Iteration") 
    plt.ylabel("Energy (Hz)") 
    plt.suptitle("Hybrid Trap Energies: " + measurement_directory_folder_name) 
    plt.savefig("Hybrid_Trap_Energies_Figure.png")
    plt.show()
    

def get_hybrid_trap_data(my_measurement, imaging_mode):
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    hybrid_cross_section_um = np.pi * np.square(
        my_measurement.experiment_parameters["top_um_per_pixel"] * my_measurement.experiment_parameters["axicon_diameter_pix"] / 2)
    hybrid_trap_frequency = my_measurement.experiment_parameters["axial_trap_frequency_hz"]
    counts_1_list = []
    counts_3_list = []
    energies_1_list = [] 
    energies_3_list = [] 
    for run_id in my_measurement.runs_dict:
        print(str(run_id))
        current_run = my_measurement.runs_dict[run_id]
        if(imaging_mode == "abs"):
            atom_density_1, atom_density_3 = get_density_from_abs_image(my_measurement, current_run)
        elif(imaging_mode == "polrot"):
            atom_density_1, atom_density_3 = get_density_from_polrot_image(my_measurement, current_run)
        counts_1 = image_processing_functions.atom_count_pixel_sum(atom_density_1, pixel_area)
        counts_3 = image_processing_functions.atom_count_pixel_sum(atom_density_3, pixel_area)
        counts_1_list.append(counts_1)
        counts_3_list.append(counts_3)
        state_1_harmonic_positions, state_1_harmonic_densities = image_processing_functions.get_hybrid_trap_densities_along_harmonic_axis(atom_density_1)
        state_3_harmonic_positions, state_3_harmonic_densities = image_processing_functions.get_hybrid_trap_densities_along_harmonic_axis(atom_density_3)
        state_1_average_energy = science_functions.get_hybrid_trap_average_energy(
            state_1_harmonic_positions, state_1_harmonic_densities, 
            hybrid_cross_section_um, hybrid_trap_frequency
        )
        state_3_average_energy = science_functions.get_hybrid_trap_average_energy(
            state_3_harmonic_positions, state_3_harmonic_densities, 
            hybrid_cross_section_um, hybrid_trap_frequency
        )
        energies_1_list.append(state_1_average_energy)
        energies_3_list.append(state_3_average_energy)
    counts_1_array = np.array(counts_1_list) 
    counts_3_array = np.array(counts_3_list)
    energies_1_array = np.array(energies_1_list)
    energies_3_array = np.array(energies_3_list)
    return (counts_1_array, counts_3_array, energies_1_array, energies_3_array)    

def get_density_from_abs_image(my_measurement, current_run):
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    res_freq_state_1 = my_measurement.experiment_parameters["state_1_unitarity_res_freq_MHz"]
    res_freq_state_3 = my_measurement.experiment_parameters["state_3_unitarity_res_freq_MHz"]
    image_stack_A = current_run.get_image("TopA", memmap = True) 
    image_stack_B = current_run.get_image("TopB", memmap = True)
    imaging_freq_A = current_run.parameters["ImagFreq1"]
    imaging_freq_B = current_run.parameters["ImagFreq2"] 
    detuning_1 = frequency_multiplier * (imaging_freq_A - res_freq_state_1)
    detuning_3 = frequency_multiplier * (imaging_freq_B - res_freq_state_3)
    atom_density_image_state_1 = image_processing_functions.get_atom_density_absorption(image_stack_A, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                            detuning = detuning_1)
    atom_density_image_state_3 = image_processing_functions.get_atom_density_absorption(image_stack_B, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                            detuning = detuning_3)
    return (atom_density_image_state_1, atom_density_image_state_3)


def get_density_from_polrot_image(my_measurement, current_run):
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    res_freq_state_1 = my_measurement.experiment_parameters["state_1_unitarity_res_freq_MHz"]
    res_freq_state_3 = my_measurement.experiment_parameters["state_3_unitarity_res_freq_MHz"]
    image_stack_A = current_run.get_image("TopA", memmap = True) 
    image_stack_B = current_run.get_image("TopB", memmap = True)
    imaging_freq_A = current_run.parameters["ImagFreq1"]
    imaging_freq_B = current_run.parameters["ImagFreq2"] 
    detuning_1A = frequency_multiplier * (imaging_freq_A - res_freq_state_1)
    detuning_2A = frequency_multiplier * (imaging_freq_A - res_freq_state_3)
    detuning_1B = frequency_multiplier * (imaging_freq_B - res_freq_state_1) 
    detuning_2B = frequency_multiplier * (imaging_freq_B - res_freq_state_3)
    abs_image_A = image_processing_functions.get_absorption_image(image_stack_A, 
                                                                    ROI = my_measurement.measurement_parameters["ROI"])
    abs_image_B = image_processing_functions.get_absorption_image(image_stack_B, 
                                                                    ROI = my_measurement.measurement_parameters["ROI"])
    atom_density_image_state_1, atom_density_image_state_3 = image_processing_functions.get_atom_density_from_polrot_images(abs_image_A, abs_image_B, 
                                                                    detuning_1A, detuning_1B, detuning_2A, detuning_2B, 
                                                                    phase_sign = my_measurement.experiment_parameters["polrot_phase_sign"])
    return (atom_density_image_state_1, atom_density_image_state_3)


def get_measurement_directory_input():
    print("Please enter the path to the measurement directory containing the runs to analyze.")
    user_input = input()
    if(not os.path.isdir(user_input)):
        raise RuntimeError("Unable to find the measurement directory.")
    return user_input
    

def get_imaging_mode_input():
    print("Please enter the imaging mode for hybrid top. Supported options are:") 
    for imaging_mode in ALLOWED_IMAGING_MODES:
        print(imaging_mode)
    user_input = input()
    if(not user_input in ALLOWED_IMAGING_MODES):
        raise RuntimeError("Specified imaging mode not supported.") 
    return user_input


def setup_measurement(workfolder_pathname, measurement_directory_path):
    my_measurement = Measurement(measurement_directory_path, hold_images_in_memory = False, run_parameters_verbose = True, imaging_type = "top_double")
    print("Initializing")
    saved_params_filename = os.path.join(workfolder_pathname, "rf_spect_runs_dump.json")
    if(os.path.isfile(saved_params_filename)):
        try:
            my_measurement._initialize_runs_dict(use_saved_params = True, saved_params_filename = saved_params_filename)
        except RuntimeError:
            my_measurement._initialize_runs_dict(use_saved_params = False)
            my_measurement.dump_runs_dict(dump_filename = saved_params_filename)
    else:
        my_measurement._initialize_runs_dict(use_saved_params = False) 
        my_measurement.dump_runs_dict(dump_filename = saved_params_filename)
    run_to_use = 0
    box_set = False
    while (not box_set) and run_to_use < len(my_measurement.runs_dict):
        try:
            my_measurement.set_ROI(box_coordinates = ROI_COORDINATES, run_to_use = run_to_use)
            my_measurement.set_norm_box(box_coordinates = NORM_BOX_COORDINATES, run_to_use = run_to_use)
        except TypeError:
            pass 
        else:
            box_set = True
        run_to_use += 1
    return my_measurement

def get_workfolder_path():
    PRIVATE_DIRECTORY_REPO_NAME = "Private_BEC1_Analysis"
    path_to_private_directory_repo = os.path.join(path_to_repo_folder, PRIVATE_DIRECTORY_REPO_NAME)
    current_datetime = datetime.datetime.now() 
    current_year = current_datetime.strftime("%Y")
    current_year_month = current_datetime.strftime("%Y-%m")
    current_year_month_day = current_datetime.strftime("%Y-%m-%d")
    workfolder_pathname = os.path.join(path_to_private_directory_repo, current_year, current_year_month, current_year_month_day + "_RF_Spectroscopy")
    return workfolder_pathname


if __name__ == "__main__":
    main()