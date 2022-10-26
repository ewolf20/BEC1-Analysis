import datetime
import os 
import sys 

import numpy as np 
import matplotlib.pyplot as plt

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_repo_folder = os.path.abspath(path_to_file + "/../../")


sys.path.insert(0, path_to_repo_folder)

from satyendra.code.image_watchdog import ImageWatchdog
from BEC1_Analysis.code.measurement import Run, Measurement
from BEC1_Analysis.code import image_processing_functions, data_fitting_functions, loading_functions

SUPPORTED_IMAGE_TYPES = ['Side_lf', 'Side_hf', 'TopA', 'TopB', 'TopAB']


ROI_COORDINATES = None
NORM_BOX_COORDINATES = None 

def main():
    measurement_directory_path = get_measurement_directory_input()
    imaging_mode_string = get_imaging_mode()
    main_after_inputs(measurement_directory_path, imaging_mode_string)

def main_after_inputs(measurement_directory_path, imaging_mode_string):
    workfolder_pathname = initialize_workfolder(measurement_directory_path)
    run_image_name_list, frequency_key_list, imaging_type_key = imaging_mode_decoder(imaging_mode_string)
    my_measurement = setup_measurement(workfolder_pathname, measurement_directory_path, imaging_type_key, run_image_name_list)
    clipboard_string = ""
    for run_image_name, frequency_key in zip(run_image_name_list, frequency_key_list):
        frequency_multiplier = get_frequency_multiplier(my_measurement, imaging_mode_string)
        frequencies_array, counts_array = get_frequency_counts_data(my_measurement, run_image_name, frequency_key, frequency_multiplier)
        if(not imaging_mode_string == "TopAB"):
            title = "Data_" + imaging_mode_string
        else:
            title = "Data_" + run_image_name
        clipboard_string = save_fit_and_plot_data(workfolder_pathname, frequencies_array, counts_array, title, run_image_name,
                                                 frequency_multiplier, clipboard_string)
    loading_functions.universal_clipboard_copy(clipboard_string)

def setup_measurement(workfolder_pathname, measurement_directory_path, imaging_type_key, run_image_name_list):
    my_measurement = Measurement(measurement_directory_path, hold_images_in_memory = False, run_parameters_verbose = True, imaging_type = imaging_type_key)
    print("Initializing")
    saved_params_filename = os.path.join(workfolder_pathname, imaging_type_key + "_runs_dump.json")
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
            my_measurement.set_ROI(box_coordinates = ROI_COORDINATES, run_to_use = run_to_use, image_to_use = run_image_name_list[0])
            my_measurement.set_norm_box(box_coordinates = NORM_BOX_COORDINATES, run_to_use = run_to_use, image_to_use = run_image_name_list[0])
        except TypeError:
            pass 
        else:
            box_set = True
        run_to_use += 1
    return my_measurement


def get_frequency_counts_data(my_measurement, run_image_name, frequency_key, frequency_multiplier):
    frequencies_list = []
    counts_list = []
    for run_id in my_measurement.runs_dict:
        print(str(run_id))
        current_run = my_measurement.runs_dict[run_id]
        current_image_stack = current_run.get_image(run_image_name, memmap = True)
        current_od_image = image_processing_functions.get_absorption_od_image(current_image_stack, ROI = my_measurement.measurement_parameters['ROI'])
        current_counts = image_processing_functions.pixel_sum(current_od_image)
        counts_list.append(current_counts)
        nominal_frequency = current_run.parameters[frequency_key] 
        frequencies_list.append(nominal_frequency * frequency_multiplier)
    frequencies_array = np.array(frequencies_list)
    counts_array = np.array(counts_list)
    return (frequencies_array, counts_array)


def save_fit_and_plot_data(workfolder_pathname, frequencies_array, counts_array, title, run_image_name, frequency_multiplier, clipboard_string):
    data_saving_path = os.path.join(workfolder_pathname, title + ".npy")
    np.save(data_saving_path, np.stack((frequencies_array, counts_array)))
    fit_results, inlier_indices = data_fitting_functions.fit_imaging_resonance_lorentzian(frequencies_array, counts_array, filter_outliers = True, 
                                                                        report_inliers = True)
    overall_indices = np.arange(len(frequencies_array))
    outlier_indices = overall_indices[~np.isin(overall_indices, inlier_indices)] 
    outlier_frequencies = frequencies_array[outlier_indices]
    outlier_counts = counts_array[outlier_indices]
    inlier_frequencies = frequencies_array[inlier_indices] 
    inlier_counts = counts_array[inlier_indices]
    popt, pcov = fit_results
    fit_report = data_fitting_functions.fit_report(data_fitting_functions.imaging_resonance_lorentzian, fit_results, precision = 4)
    with open(os.path.join(workfolder_pathname, title + "_Fit_Report.txt"), 'w') as f:
        f.write(fit_report) 
        amp, center, gamma, offset = popt 
        nominal_center = center / frequency_multiplier
        nominal_resonance_string = "\nNominal Resonance: {0:.2f}\n\n".format(nominal_center)
        f.write(nominal_resonance_string)
        clipboard_string += title + ":\n" + fit_report + nominal_resonance_string
    print(title + ":")
    print(fit_report)
    plt.plot(inlier_frequencies, inlier_counts, 'x', label = "Data")
    plt.plot(outlier_frequencies, outlier_counts, 'rd', label = "Outliers")
    frequencies_plotting_range = np.linspace(min(frequencies_array), max(frequencies_array), 100)
    plt.plot(frequencies_plotting_range, data_fitting_functions.imaging_resonance_lorentzian(frequencies_plotting_range, *popt), label = "Fit")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Counts")
    plt.suptitle("Imaging Resonance: " + str(run_image_name))
    plt.legend()
    plt.savefig(os.path.join(workfolder_pathname, title + "_Graph.png"))
    plt.show()
    return clipboard_string

def get_measurement_directory_input():
    print("Please enter the path to the measurement directory containing the runs to analyze.")
    user_input = input()
    if(not os.path.isdir(user_input)):
        raise RuntimeError("Unable to find the measurement directory.")
    return user_input


def get_imaging_mode():
    print("Please enter the imaging mode. Supported options are: ") 
    for image_type in SUPPORTED_IMAGE_TYPES:
        print(image_type)
    user_input = input() 
    if(not user_input in SUPPORTED_IMAGE_TYPES):
        raise RuntimeError("Specified imaging type is not supported.")
    return user_input


def imaging_mode_decoder(imaging_mode_string):
    SIDE_RUN_IMAGE_NAME = "Side" 
    TOP_A_RUN_IMAGE_NAME = "TopA"
    TOP_B_RUN_IMAGE_NAME = "TopB" 
    SIDE_LF_FREQ_KEY = "LFImgFreq"
    SIDE_HF_FREQ_KEY = "ImagFreq0"
    TOP_A_FREQ_KEY = "ImagFreq1"
    TOP_B_FREQ_KEY = "ImagFreq2" 
    SIDE_LF_MEASUREMENT_KEY = "side_low_mag"
    SIDE_HF_MEASUREMENT_KEY = "side_high_mag"
    TOP_MEASUREMENT_KEY = "top_double"
    if(imaging_mode_string == "Side_lf"):
        return ([SIDE_RUN_IMAGE_NAME], [SIDE_LF_FREQ_KEY], SIDE_LF_MEASUREMENT_KEY)
    elif(imaging_mode_string == "Side_hf"):
        return ([SIDE_RUN_IMAGE_NAME], [SIDE_HF_FREQ_KEY], SIDE_HF_MEASUREMENT_KEY)
    elif(imaging_mode_string == "TopA"):
        return ([TOP_A_RUN_IMAGE_NAME], [TOP_A_FREQ_KEY], TOP_MEASUREMENT_KEY)
    elif(imaging_mode_string == "TopB"):
        return ([TOP_B_RUN_IMAGE_NAME], [TOP_B_FREQ_KEY], TOP_MEASUREMENT_KEY)
    elif(imaging_mode_string == "TopAB"):
        return ([TOP_A_RUN_IMAGE_NAME, TOP_B_RUN_IMAGE_NAME], [TOP_A_FREQ_KEY, TOP_B_FREQ_KEY], TOP_MEASUREMENT_KEY)


def get_frequency_multiplier(my_measurement, imaging_mode_string):
    if(imaging_mode_string == "side_lf"):
        frequency_multiplier = my_measurement.experiment_parameters["li_lf_freq_multiplier"]
    else:
        frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    return frequency_multiplier

def initialize_workfolder(measurement_directory_path):
    PRIVATE_DIRECTORY_REPO_NAME = "Private_BEC1_Analysis"
    path_to_private_directory_repo = os.path.join(path_to_repo_folder, PRIVATE_DIRECTORY_REPO_NAME)
    current_datetime = datetime.datetime.now()
    current_year = current_datetime.strftime("%Y")
    current_year_month = current_datetime.strftime("%Y-%m")
    current_year_month_day = current_datetime.strftime("%Y-%m-%d")
    measurement_directory_folder_name = os.path.basename(os.path.normpath(measurement_directory_path))
    workfolder_descriptor = "_Imaging_Resonance_" + measurement_directory_folder_name
    workfolder_pathname = os.path.join(path_to_private_directory_repo, current_year, current_year_month, current_year_month_day + workfolder_descriptor)
    if(not os.path.isdir(workfolder_pathname)):
        os.makedirs(workfolder_pathname)
    with open(os.path.join(workfolder_pathname, "Source.txt"), 'w') as f:
        f.write("Data source: " + measurement_directory_path)
    return workfolder_pathname


if __name__ == "__main__":
    main()