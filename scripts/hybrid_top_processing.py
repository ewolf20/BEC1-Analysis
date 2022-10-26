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
    center_guess_MHz, rabi_freq_guess = get_guesses_input()
    main_after_inputs(measurement_directory_path, resonance_key, center_guess_MHz = center_guess_MHz, rabi_freq_guess = rabi_freq_guess)

# main() version without command line input, compatible with portal
def main_after_inputs(measurement_directory_path, imaging_mode):
    workfolder_pathname = get_workfolder_path()
    if(not os.path.isdir(workfolder_pathname)):
        os.makedirs(workfolder_pathname)
    my_measurement = setup_measurement(workfolder_pathname, measurement_directory_path)
    if(imaging_mode == "abs"):
        counts_1_array, counts_3_array, energies_1_array, energies_3_array = get_hybrid_trap_data_abs(my_measurement)
    elif(imaging_mode == "polrot"):
        counts_1_array, counts_3_array, energies_1_array, energies_3_array = get_hybrid_trap_data_polrot(my_measurement)
    save_fit_and_plot_data(workfolder_pathname, rf_frequencies_array, counts_A_array, counts_B_array, tau_value, resonance_key, 
                            center_guess_MHz = center_guess_MHz, rabi_freq_guess = rabi_freq_guess)

def save_fit_and_plot_data(workfolder_pathname, rf_frequencies_array, counts_A_array, counts_B_array, tau_value, resonance_key, 
                            center_guess_MHz = None, rabi_freq_guess = None):
    counts_data_saving_path = os.path.join(workfolder_pathname, "RF_Spect_Counts.npy")
    initial_final_key = resonance_key.split("_")[1] 
    if(initial_final_key == "AB"):
        initial_counts_array = counts_A_array 
        final_counts_array = counts_B_array 
    elif(initial_final_key == "BA"):
        initial_counts_array = counts_B_array 
        final_counts_array = counts_A_array
    np.save(counts_data_saving_path, np.stack((rf_frequencies_array, initial_counts_array, final_counts_array)))
    transfers = final_counts_array / (initial_counts_array + final_counts_array)
    if(center_guess_MHz):
        center_guess = center_guess_MHz * 1000 
    else:
        center_guess = None
    try:
        fit_results, inlier_indices = data_fitting_functions.fit_rf_spect_detuning_scan(rf_frequencies_array * 1000, transfers, tau_value, 
                                                                    center = center_guess, rabi_freq = rabi_freq_guess,
                                                                    filter_outliers = True, report_inliers = True)
    except RuntimeError as e:
        plt.plot(rf_frequencies_array, transfers, 'x', label = "Data")
        fit_report = "Fit failed. " + str(e)
    else:
        overall_indices = np.arange(len(rf_frequencies_array)) 
        outlier_indices = overall_indices[~np.isin(overall_indices, inlier_indices)]
        popt, pcov = fit_results
        center, rabi_freq = popt
        fit_plotting_frequencies = np.linspace(min(rf_frequencies_array), max(rf_frequencies_array), 100)
        plt.plot(rf_frequencies_array[inlier_indices], transfers[inlier_indices], 'x', label = "Data") 
        plt.plot(rf_frequencies_array[outlier_indices], transfers[outlier_indices], 'rd', label = "Outliers")
        plt.plot(fit_plotting_frequencies, data_fitting_functions.rf_spect_detuning_scan(fit_plotting_frequencies * 1000, tau_value, *popt)) 
        plt.suptitle("RF Spectroscopy {0}: Center = {1:0.5e} MHz, Rabi Freq = {2:0.2e} kHz".format(resonance_key, center, rabi_freq))
        fit_report = data_fitting_functions.fit_report(data_fitting_functions.rf_spect_detuning_scan, fit_results, precision = 5)
    with open(os.path.join(workfolder_pathname, "RF_Spectroscopy_Fit_Report.txt"), 'w') as f:
        f.write("RF Spect: " + fit_report) 
    print("RF Spect: ")
    print(fit_report)
    loading_functions.universal_clipboard_copy("RF Spect: \n" + fit_report)
    plt.xlabel("RF Frequency (MHz)")
    plt.ylabel("Transfer")
    plt.legend()
    plt.savefig(os.path.join(workfolder_pathname, "RF_Spectroscopy_Curve.png"))
    plt.show()
    



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
        image_stack_A = current_run.get_image("TopA", memmap = True)
        image_stack_B = current_run.get_image("TopB", memmap = True)
        imaging_freq_A = current_run.parameters["ImagFreq1"]
        imaging_freq_B = current_run.parameters["ImagFreq2"]
        detuning_1 = frequency_multiplier * (imaging_freq_A - res_freq_state_1)
        detuning_3 = frequency_multiplier * (imaging_freq_B - res_freq_state_3)
        atom_density_1 = image_processing_functions.get_atom_density_absorption(image_stack_A, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                    detuning = detuning_1)
        atom_density_3 = image_processing_functions.get_atom_density_absorption(image_stack_B, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                    detuning = detuning_3)
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
    frequency_multiplier = my_measurement.experiment_parameters[""]
    image_stack_A = current_run.get_image("TopA", memmap = True) 
    image_stack_B = current_run.get_image("TopB", memmap = True)
    imaging_freq_A = current_run.parameters["ImagFreq1"]
    imaging_freq_B = current_run.parameters["ImagFreq2"] 


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