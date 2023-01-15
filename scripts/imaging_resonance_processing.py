import datetime
import os 
import sys 

import numpy as np 
import matplotlib.pyplot as plt

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_repo_folder = os.path.abspath(path_to_file + "/../../")


sys.path.insert(0, path_to_repo_folder)

from BEC1_Analysis.code.measurement import Measurement
from BEC1_Analysis.code import analysis_functions, data_fitting_functions, loading_functions

SUPPORTED_IMAGE_TYPES = ['Side_lf', 'Side_hf', 'TopA', 'TopB', 'TopAB']

def main():
    measurement_directory_path = get_measurement_directory_input()
    imaging_mode_string = get_imaging_mode()
    main_after_inputs(measurement_directory_path, imaging_mode_string)

def main_after_inputs(measurement_directory_path, imaging_mode_string):
    workfolder_pathname = initialize_workfolder(measurement_directory_path)
    my_measurement = setup_measurement(measurement_directory_path, imaging_mode_string)
    frequency_multiplier = get_frequency_multiplier(my_measurement, imaging_mode_string)
    if imaging_mode_string == "TopAB":
        my_measurement.analyze_runs(analysis_functions.get_od_pixel_sums_top_double, ("counts_1", "counts_3"), print_progress = True)
        frequencies_1, counts_1 = my_measurement.get_parameter_analysis_value_pair_from_runs("ImagFreq1", "counts_1")
        frequencies_3, counts_3 = my_measurement.get_parameter_analysis_value_pair_from_runs("ImagFreq2", "counts_3")
        clipboard_string_topA = save_fit_and_plot_data(workfolder_pathname, frequencies_1, counts_1, "TopA", frequency_multiplier)
        clipboard_string_topB = save_fit_and_plot_data(workfolder_pathname, frequencies_3, counts_3, "TopB", frequency_multiplier)
        clipboard_string = clipboard_string_topA + clipboard_string_topB
    elif imaging_mode_string == "TopA":
        my_measurement.analyze_runs(analysis_functions.get_od_pixel_sum_top_A, "counts", print_progress = True)
        frequencies, counts = my_measurement.get_parameter_analysis_value_pair_from_runs("ImagFreq1", "counts")
        clipboard_string = save_fit_and_plot_data(workfolder_pathname, frequencies, counts, "TopA", frequency_multiplier)
    elif imaging_mode_string == "TopB":
        my_measurement.analyze_runs(analysis_functions.get_od_pixel_sum_top_B, "counts", print_progress = True)
        frequencies, counts = my_measurement.get_parameter_analysis_value_pair_from_runs("ImagFreq2", "counts")
        clipboard_string = save_fit_and_plot_data(workfolder_pathname, frequencies, counts, "TopB", frequency_multiplier)
    elif imaging_mode_string == "Side_hf":
        my_measurement.analyze_runs(analysis_functions.get_od_pixel_sum_side, "counts", print_progress = True)
        frequencies, counts = my_measurement.get_parameter_analysis_value_pair_from_runs("ImagFreq0", "counts")
        clipboard_string = save_fit_and_plot_data(workfolder_pathname, frequencies, counts, "Side", frequency_multiplier)
    elif imaging_mode_string == "Side_lf":
        my_measurement.analyze_runs(analysis_functions.get_od_pixel_sum_side, "counts", print_progress = True)
        frequencies, counts = my_measurement.get_parameter_analysis_value_pair_from_runs("LFImgFreq", "counts")
        clipboard_string = save_fit_and_plot_data(workfolder_pathname, frequencies, counts, "Side", frequency_multiplier)
    loading_functions.universal_clipboard_copy(clipboard_string)

def setup_measurement(measurement_directory_path, imaging_mode_string):
    run_image_to_use, imaging_type_key = imaging_mode_decoder(imaging_mode_string)
    my_measurement = Measurement(measurement_directory_path, hold_images_in_memory = False, run_parameters_verbose = True, imaging_type = imaging_type_key)
    print("Initializing")
    run_to_use = 0
    box_set = False
    while (not box_set) and run_to_use < len(my_measurement.runs_dict):
        try:
            my_measurement.set_ROI(run_to_use = run_to_use, image_to_use = run_image_to_use)
            my_measurement.set_norm_box(run_to_use = run_to_use, image_to_use = run_image_to_use)
        except TypeError:
            pass 
        else:
            box_set = True
        run_to_use += 1
    return my_measurement


def save_fit_and_plot_data(workfolder_pathname, nominal_frequencies_array, counts_array, run_image_name, frequency_multiplier):
    title = "Data_" + run_image_name
    frequencies_array = nominal_frequencies_array * frequency_multiplier
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
        clipboard_string = title + ":\n" + fit_report + nominal_resonance_string
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


def get_frequency_multiplier(my_measurement, imaging_mode_string):
    if imaging_mode_string == "Side_lf":
        return my_measurement.experiment_parameters["li_lf_freq_multiplier"] 
    else:
        return my_measurement.experiment_parameters["li_hf_freq_multiplier"]


def imaging_mode_decoder(imaging_mode_string):
    SIDE_RUN_IMAGE_NAME = "Side" 
    TOP_A_RUN_IMAGE_NAME = "TopA"
    TOP_B_RUN_IMAGE_NAME = "TopB" 
    SIDE_LF_MEASUREMENT_KEY = "side_low_mag"
    SIDE_HF_MEASUREMENT_KEY = "side_high_mag"
    TOP_MEASUREMENT_KEY = "top_double"
    if(imaging_mode_string == "Side_lf"):
        return (SIDE_RUN_IMAGE_NAME, SIDE_LF_MEASUREMENT_KEY)
    elif(imaging_mode_string == "Side_hf"):
        return (SIDE_RUN_IMAGE_NAME, SIDE_HF_MEASUREMENT_KEY)
    elif(imaging_mode_string == "TopA"):
        return (TOP_A_RUN_IMAGE_NAME, TOP_MEASUREMENT_KEY)
    elif(imaging_mode_string == "TopB"):
        return (TOP_B_RUN_IMAGE_NAME, TOP_MEASUREMENT_KEY)
    elif(imaging_mode_string == "TopAB"):
        return (TOP_A_RUN_IMAGE_NAME, TOP_MEASUREMENT_KEY)


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