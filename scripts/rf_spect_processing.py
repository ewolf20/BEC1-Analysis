import datetime
import os 
import sys 

import numpy as np 
import matplotlib.pyplot as plt

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_repo_folder = os.path.abspath(path_to_file + "/../../")


sys.path.insert(0, path_to_repo_folder)

from BEC1_Analysis.code.measurement import Measurement
from BEC1_Analysis.code import image_processing_functions, data_fitting_functions, loading_functions

ROI_COORDINATES = None
NORM_BOX_COORDINATES = None 

#Syntax: First two numbers denote the index of the initial (transferred from) and 
#final (transferred to) state for spectroscopy. The key 'AB' indicates that the initial 
#state is imaged in topA and the final in topB, whereas 'BA' indicates the reverse.
ALLOWED_RESONANCE_TYPES = ["12_AB", "12_BA",  "21_AB", "21_BA", 
                         "13_AB", "13_BA", "31_AB", "31_BA", 
                         "23_AB", "23_BA", "32_AB", "32_BA"]

def main():
    measurement_directory_path = get_measurement_directory_input()
    resonance_key = get_resonance_key_input()
    center_guess_MHz, rabi_freq_guess = get_guesses_input()
    main_after_inputs(measurement_directory_path, resonance_key, center_guess_MHz = center_guess_MHz, rabi_freq_guess = rabi_freq_guess)

# main() version without command line input, compatible with portal
def main_after_inputs(measurement_directory_path, resonance_key, center_guess_MHz = None, rabi_freq_guess = None):
    workfolder_pathname = get_workfolder_path()
    if(not os.path.isdir(workfolder_pathname)):
        os.makedirs(workfolder_pathname)
    my_measurement = setup_measurement(workfolder_pathname, measurement_directory_path)
    rf_frequencies_array, counts_A_array, counts_B_array, tau_value = get_rf_frequencies_and_counts(my_measurement, resonance_key)
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

def get_resonance_key_input():
    print("Please enter the resonance which you wish to fit. Supported options are:") 
    for resonance_type in ALLOWED_RESONANCE_TYPES:
        print(resonance_type)
    user_input = input()
    if(not user_input in ALLOWED_RESONANCE_TYPES):
        raise RuntimeError("Specified resonance not supported.") 
    return user_input


def get_guesses_input():
    print("If desired, please enter guesses for the RF resonance center (in MHz) and the rabi frequency (in kHz).")
    print("To omit a guess, simply press enter with no input.")
    print("Center Guess (MHz):")
    center_guess_user_input = input() 
    if(center_guess_user_input == ''):
        center_guess = None 
    else:
        center_guess = float(center_guess_user_input)
    print("Rabi Freq Guess (kHz):")
    rabi_freq_guess_user_input = input() 
    if(rabi_freq_guess_user_input == ''):
        rabi_freq_guess = None 
    else:
        rabi_freq_guess = float(rabi_freq_guess_user_input)
    return (center_guess, rabi_freq_guess)


def get_rf_frequencies_and_counts(my_measurement, resonance_key):
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    rf_frequencies_list = []
    counts_A_list = []
    counts_B_list = []
    for i, run_id in enumerate(my_measurement.runs_dict):
        print(str(run_id))
        current_run = my_measurement.runs_dict[run_id]
        if i == 0:
            tau_value = current_run.parameters["SpectPulseTime"]
        else:
            check_tau = current_run.parameters["SpectPulseTime"]
            if check_tau != tau_value:
                raise RuntimeError("The tau values should be constant over the RF spectroscopy sequence")
        image_stack_A = current_run.get_image("TopA", memmap = True)
        image_stack_B = current_run.get_image("TopB", memmap = True)
        imaging_freq_A = current_run.parameters["ImagFreq1"] 
        imaging_freq_B = current_run.parameters["ImagFreq2"] 
        detuning_A, detuning_B = get_detunings_from_resonance_key(my_measurement, resonance_key, imaging_freq_A, imaging_freq_B)
        atom_density_A = image_processing_functions.get_atom_density_absorption(image_stack_A, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                    detuning = detuning_A)
        atom_density_B = image_processing_functions.get_atom_density_absorption(image_stack_B, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                    detuning = detuning_B)
        counts_A = image_processing_functions.atom_count_pixel_sum(atom_density_A, pixel_area)
        counts_B = image_processing_functions.atom_count_pixel_sum(atom_density_B, pixel_area)
        counts_A_list.append(counts_A)
        counts_B_list.append(counts_B)
        rf_frequency = current_run.parameters["RF_Box_Center"]
        rf_frequencies_list.append(rf_frequency)
    rf_frequencies_array = np.array(rf_frequencies_list)
    counts_A_array = np.array(counts_A_list) 
    counts_B_array = np.array(counts_B_list)
    return (rf_frequencies_array, counts_A_array, counts_B_array, tau_value)

def get_detunings_from_resonance_key(my_measurement, resonance_key, freqA, freqB):
    res_freq_state_1 = my_measurement.experiment_parameters["state_1_unitarity_res_freq_MHz"]
    res_freq_state_2 = my_measurement.experiment_parameters["state_2_unitarity_res_freq_MHz"]
    res_freq_state_3 = my_measurement.experiment_parameters["state_3_unitarity_res_freq_MHz"]
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    initial_final_key, imaging_key = resonance_key.split("_")
    initial_state_label, final_state_label = initial_final_key
    if(initial_state_label == "1"):
        initial_res_freq = res_freq_state_1
    elif(initial_state_label == "2"):
        initial_res_freq = res_freq_state_2 
    elif(initial_state_label == "3"):
        initial_res_freq = res_freq_state_3 
    if(final_state_label == "1"):
        final_res_freq = res_freq_state_1 
    elif(final_state_label == "2"):
        final_res_freq = res_freq_state_2 
    elif(final_state_label == "3"):
        final_res_freq = res_freq_state_3
    if(imaging_key == "AB"):
        detuning_A = frequency_multiplier * (freqA - initial_res_freq)
        detuning_B = frequency_multiplier * (freqB - final_res_freq)
    elif(imaging_key == "BA"):
        detuning_B = frequency_multiplier * (freqB - initial_res_freq)
        detuning_A = frequency_multiplier * (freqA - final_res_freq)
    return (detuning_A, detuning_B)    




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