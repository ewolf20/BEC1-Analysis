import datetime
import os 
import sys 

import numpy as np 
import matplotlib.pyplot as plt

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_repo_folder = os.path.abspath(path_to_file + "/../../")


sys.path.insert(0, path_to_repo_folder)

from BEC1_Analysis.code.measurement import Measurement
from BEC1_Analysis.code import image_processing_functions, loading_functions, analysis_functions

ROI_COORDINATES = None
NORM_BOX_COORDINATES = None 

ALLOWED_IMAGING_MODES = ['polrot', 'abs']

def main():
    measurement_directory_path = get_measurement_directory_input()
    imaging_mode = get_imaging_mode_input()
    main_after_inputs(measurement_directory_path, imaging_mode)

# main() version without command line input, compatible with portal
def main_after_inputs(measurement_directory_path, imaging_mode):
    my_measurement = setup_measurement(measurement_directory_path)
    workfolder_pathname = my_measurement.initialize_workfolder(descriptor = "Hybrid_Trap_Analysis")
    counts_1_array, counts_3_array, energies_1_array, energies_3_array, energies_array = get_hybrid_trap_data(my_measurement, imaging_mode)
    save_and_plot_data(my_measurement, workfolder_pathname, counts_1_array, counts_3_array, energies_1_array, energies_3_array, energies_array)


def save_and_plot_data(my_measurement, workfolder_pathname, counts_1_array, counts_3_array, energies_1_array, energies_3_array, energies_array):
    counts_data_saving_path = os.path.join(workfolder_pathname, "Hybrid_Exp_Counts.npy")
    np.save(counts_data_saving_path, np.stack((counts_1_array, counts_3_array)))
    energies_data_saving_path = os.path.join(workfolder_pathname, "Hybrid_Exp_Energies.npy")
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
    with open(os.path.join(workfolder_pathname, "Results.txt"), 'w') as f:
        f.write(report_string)
    print(report_string)
    plt.plot(counts_1_array, 'x', label = "State 1 Counts") 
    plt.plot(counts_3_array, 'o', label = "State 3 Counts")
    plt.xlabel("Iteration") 
    plt.ylabel("Atom counts")
    plt.legend()
    measurement_directory_folder_name = os.path.basename(os.path.normpath(my_measurement.measurement_directory_path))
    plt.suptitle("Hybrid Trap Counts: " + measurement_directory_folder_name)
    plt.savefig(os.path.join(workfolder_pathname, "Hybrid_Trap_Counts_Figure.png"))
    plt.show()
    plt.plot(energies_1_array, 'x', label = "State 1 Energy")
    plt.plot(energies_3_array, 'o', label = "State 3 Energy")
    plt.plot(energies_array, 'd', label = "Average Energy") 
    plt.legend() 
    plt.xlabel("Iteration") 
    plt.ylabel("Energy (Hz)") 
    plt.suptitle("Hybrid Trap Energies: " + measurement_directory_folder_name) 
    plt.savefig(os.path.join(workfolder_pathname, "Hybrid_Trap_Energies_Figure.png"))
    plt.show()
    

def get_hybrid_trap_data(my_measurement, imaging_mode):
    if imaging_mode == "polrot":
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_polrot, ("density_1", "density_3"), print_progress = True, catch_errors = True)
        my_measurement.analyze_runs(analysis_functions.get_atom_counts_top_polrot, ("counts_1", "counts_3"),
                                    fun_kwargs = {"first_stored_density_name":"density_1", "second_stored_density_name":"density_3"}, 
                                    catch_errors = True)
    elif imaging_mode == "abs":
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_abs, ("density_1", "density_3"), print_progress = True, catch_errors = True)
        my_measurement.analyze_runs(analysis_functions.get_atom_counts_top_AB_abs, ("counts_1", "counts_3"),
                                    fun_kwargs = {"first_stored_density_name":"density_1", "second_stored_density_name":"density_3"}, 
                                    catch_errors = True)
    my_measurement.analyze_runs(analysis_functions.get_hybrid_trap_average_energy, ("average_energy", "average_energy_1", "average_energy_3"), 
                                fun_kwargs = {"first_stored_density_name":"density_1", "second_stored_density_name":"density_3", 
                                            "return_sub_energies": True}, catch_errors = True)
    def energy_fitted_filter(my_measurement, my_run):
        return not my_run.analysis_results["average_energy"] == Measurement.ANALYSIS_ERROR_INDICATOR_STRING
    counts_1 = my_measurement.get_analysis_value_from_runs("counts_1", run_filter = energy_fitted_filter)
    counts_3 = my_measurement.get_analysis_value_from_runs("counts_3", run_filter = energy_fitted_filter)
    average_energy_1 = my_measurement.get_analysis_value_from_runs("average_energy_1", run_filter = energy_fitted_filter) 
    average_energy_3 = my_measurement.get_analysis_value_from_runs("average_energy_3", run_filter = energy_fitted_filter) 
    average_energy = my_measurement.get_analysis_value_from_runs("average_energy", run_filter = energy_fitted_filter) 
    return (counts_1, counts_3, average_energy_1, average_energy_3, average_energy)


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


def setup_measurement(measurement_directory_path):
    my_measurement = Measurement(measurement_directory_path, hold_images_in_memory = False, run_parameters_verbose = True, imaging_type = "top_double")
    print("Initializing")
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

if __name__ == "__main__":
    main()