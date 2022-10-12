import datetime
import os 
import sys 

import numpy as np 
import matplotlib.pyplot as plt

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_repo_folder = os.path.abspath(path_to_file + "/../../")


sys.path.insert(0, path_to_repo_folder)

from BEC1_Analysis.code.measurement import Run, Measurement
from BEC1_Analysis.code import image_processing_functions, data_fitting_functions

ROI_COORDINATES = None
NORM_BOX_COORDINATES = None 

def main():
    measurement_directory_path = get_measurement_directory_input()
    workfolder_pathname = get_workfolder_path()
    if(not os.path.isdir(workfolder_pathname)):
        os.makedirs(workfolder_pathname)
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
    res_freq_state_1 = my_measurement.experiment_parameters["state_1_unitarity_res_freq_MHz"]
    res_freq_state_2 = my_measurement.experiment_parameters["state_2_unitarity_res_freq_MHz"]
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    rf_frequencies_list = []
    counts_1_list = []
    counts_2_list = []
    for i, run_id in enumerate(my_measurement.runs_dict):
        print(str(run_id))
        current_run = my_measurement.runs_dict[run_id]
        if i == 0:
            tau_value = current_run.parameters["SpectPulseTime"]
        else:
            check_tau = current_run.parameters["SpectPulseTime"]
            if check_tau != tau_value:
                raise RuntimeError("The tau values should be constant over the RF spectroscopy sequence")
        image_stack_state_1 = current_run.get_image("TopA", memmap = True)
        image_stack_state_2 = current_run.get_image("TopB", memmap = True)
        detuning_1 = frequency_multiplier * (current_run.parameters["ImagFreq1"] - res_freq_state_1) 
        detuning_2 = frequency_multiplier * (current_run.parameters["ImagFreq2"] - res_freq_state_2)
        atom_density_1 = image_processing_functions.get_atom_density_absorption(image_stack_state_1, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                    detuning = detuning_1)
        atom_density_2 = image_processing_functions.get_atom_density_absorption(image_stack_state_2, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                    detuning = detuning_2)
        counts_1 = image_processing_functions.atom_count_pixel_sum(atom_density_1, pixel_area)
        counts_2 = image_processing_functions.atom_count_pixel_sum(atom_density_2, pixel_area)
        counts_1_list.append(counts_1) 
        counts_2_list.append(counts_2) 
        rf_frequency = current_run.parameters["RF12_Box_Center"]
        rf_frequencies_list.append(rf_frequency)
    rf_frequencies_array = np.array(rf_frequencies_list)
    counts_1_array = np.array(counts_1_list) 
    counts_2_array = np.array(counts_2_list)
    title = "RF_Spect_Counts"
    data_saving_path = os.path.join(workfolder_pathname, title + ".npy")
    np.save(data_saving_path, np.stack((rf_frequencies_array, counts_1_array, counts_2_array)))
    transfers = counts_2_array / (counts_2_array + counts_1_array)
    fit_results, inlier_indices = data_fitting_functions.fit_rf_spect_detuning_scan(rf_frequencies_array * 1000, transfers, tau_value, 
                                                                    filter_outliers = True, report_inliers = True)
    overall_indices = np.arange(len(rf_frequencies_array)) 
    outlier_indices = overall_indices[~np.isin(overall_indices, inlier_indices)] 
    popt, pcov = fit_results
    center, rabi_freq = popt
    fit_plotting_frequencies = np.linspace(min(rf_frequencies_array), max(rf_frequencies_array), 100)
    plt.plot(rf_frequencies_array[inlier_indices], transfers[inlier_indices], 'x', label = "Data") 
    plt.plot(rf_frequencies_array[outlier_indices], transfers[outlier_indices], 'rd', label = "Outliers")
    plt.plot(fit_plotting_frequencies, data_fitting_functions.rf_spect_detuning_scan(fit_plotting_frequencies * 1000, tau_value, *popt)) 
    plt.xlabel("RF Frequency (MHz)")
    plt.ylabel("Transfer")
    plt.legend()
    plt.suptitle("RF Spectroscopy: Center = " + str(center / 1000) + " MHz")
    plt.savefig(os.path.join(workfolder_pathname, title + ".png"))
    plt.show()
    fit_report = data_fitting_functions.fit_report(data_fitting_functions.rf_spect_detuning_scan, fit_results, precision = 5)
    with open(os.path.join(workfolder_pathname, title + "_Fit_Report.txt"), 'w') as f:
        f.write(fit_report) 
    print(title + ":")
    print(fit_report)

# main() version compatible with portal
def main_portal(measurement_directory_path):
    workfolder_pathname = get_workfolder_path()
    if(not os.path.isdir(workfolder_pathname)):
        os.makedirs(workfolder_pathname)
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
    res_freq_state_1 = my_measurement.experiment_parameters["state_1_unitarity_res_freq_MHz"]
    res_freq_state_2 = my_measurement.experiment_parameters["state_2_unitarity_res_freq_MHz"]
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    rf_frequencies_list = []
    counts_1_list = []
    counts_2_list = []
    for i, run_id in enumerate(my_measurement.runs_dict):
        print(str(run_id))
        current_run = my_measurement.runs_dict[run_id]
        if i == 0:
            tau_value = current_run.parameters["SpectPulseTime"]
        else:
            check_tau = current_run.parameters["SpectPulseTime"]
            if check_tau != tau_value:
                raise RuntimeError("The tau values should be constant over the RF spectroscopy sequence")
        image_stack_state_1 = current_run.get_image("TopA", memmap = True)
        image_stack_state_2 = current_run.get_image("TopB", memmap = True)
        detuning_1 = frequency_multiplier * (current_run.parameters["ImagFreq1"] - res_freq_state_1) 
        detuning_2 = frequency_multiplier * (current_run.parameters["ImagFreq2"] - res_freq_state_2)
        atom_density_1 = image_processing_functions.get_atom_density_absorption(image_stack_state_1, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                    detuning = detuning_1)
        atom_density_2 = image_processing_functions.get_atom_density_absorption(image_stack_state_2, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                    detuning = detuning_2)
        counts_1 = image_processing_functions.atom_count_pixel_sum(atom_density_1, pixel_area)
        counts_2 = image_processing_functions.atom_count_pixel_sum(atom_density_2, pixel_area)
        counts_1_list.append(counts_1) 
        counts_2_list.append(counts_2) 
        rf_frequency = current_run.parameters["RF12_Box_Center"]
        rf_frequencies_list.append(rf_frequency)
    rf_frequencies_array = np.array(rf_frequencies_list)
    counts_1_array = np.array(counts_1_list) 
    counts_2_array = np.array(counts_2_list)
    title = "RF_Spect_Counts"
    data_saving_path = os.path.join(workfolder_pathname, title + ".npy")
    np.save(data_saving_path, np.stack((rf_frequencies_array, counts_1_array, counts_2_array)))
    transfers = counts_2_array / (counts_2_array + counts_1_array)
    fit_results, inlier_indices = data_fitting_functions.fit_rf_spect_detuning_scan(rf_frequencies_array * 1000, transfers, tau_value, 
                                                                    filter_outliers = True, report_inliers = True)
    overall_indices = np.arange(len(rf_frequencies_array)) 
    outlier_indices = overall_indices[~np.isin(overall_indices, inlier_indices)] 
    popt, pcov = fit_results
    center, rabi_freq = popt
    fit_plotting_frequencies = np.linspace(min(rf_frequencies_array), max(rf_frequencies_array), 100)
    plt.plot(rf_frequencies_array[inlier_indices], transfers[inlier_indices], 'x', label = "Data") 
    plt.plot(rf_frequencies_array[outlier_indices], transfers[outlier_indices], 'rd', label = "Outliers")
    plt.plot(fit_plotting_frequencies, data_fitting_functions.rf_spect_detuning_scan(fit_plotting_frequencies * 1000, tau_value, *popt)) 
    plt.xlabel("RF Frequency (MHz)")
    plt.ylabel("Transfer")
    plt.legend()
    plt.suptitle("RF Spectroscopy: Center = " + str(center / 1000) + " MHz")
    plt.savefig(os.path.join(workfolder_pathname, title + ".png"))
    plt.show()
    fit_report = data_fitting_functions.fit_report(data_fitting_functions.rf_spect_detuning_scan, fit_results, precision = 5)
    with open(os.path.join(workfolder_pathname, title + "_Fit_Report.txt"), 'w') as f:
        f.write(fit_report) 
    print(title + ":")
    print(fit_report)


def get_measurement_directory_input():
    print("Please enter the path to the measurement directory containing the runs to analyze.")
    user_input = input()
    if(not os.path.isdir(user_input)):
        raise RuntimeError("Unable to find the measurement directory.")
    return user_input

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