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
from BEC1_Analysis.code import image_processing_functions, data_fitting_functions

SUPPORTED_IMAGE_TYPES = ['Side_lf', 'Side_hf', 'TopA', 'TopB', 'TopAB']


ROI_COORDINATES = None
NORM_BOX_COORDINATES = None 

def main():
    measurement_directory_path = get_measurement_directory_input()
    imaging_mode_string = get_imaging_mode()
    workfolder_pathname = get_workfolder_path()
    if(not os.path.isdir(workfolder_pathname)):
        os.makedirs(workfolder_pathname)
    run_image_name_list, frequency_key_list, imaging_type_key = imaging_mode_decoder(imaging_mode_string)
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
    for run_image_name, frequency_key in zip(run_image_name_list, frequency_key_list):
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
            if(imaging_mode_string == "side_lf"):
                frequency_multiplier = my_measurement.experiment_parameters["li_lf_freq_multiplier"]
            else:
                frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
            frequencies_list.append(nominal_frequency * frequency_multiplier)
        frequencies_array = np.array(frequencies_list)
        counts_array = np.array(counts_list)
        if(not imaging_mode_string == "TopAB"):
            title = "Data_" + imaging_mode_string
        else:
            title = "Data_" + run_image_name
        data_saving_path = os.path.join(workfolder_pathname, title + ".npy")
        np.save(data_saving_path, np.stack((frequencies_array, counts_array)))
        fit_results = data_fitting_functions.fit_imaging_resonance_lorentzian(frequencies_array, counts_array)
        popt, pcov = fit_results 
        fit_report = data_fitting_functions.fit_report(data_fitting_functions.imaging_resonance_lorentzian, fit_results)
        with open(os.path.join(get_workfolder_path(), title + "_Fit_Report.txt"), 'w') as f:
            f.write(fit_report) 
        print(title + ":")
        print(fit_report)
        plt.plot(frequencies_array, counts_array, 'x', label = "Data")
        frequencies_plotting_range = np.linspace(min(frequencies_array), max(frequencies_array), 100)
        plt.plot(frequencies_plotting_range, data_fitting_functions.imaging_resonance_lorentzian(frequencies_plotting_range, *popt), label = "Fit")
        plt.xlabel("Frequency (MHz)") 
        plt.ylabel("Counts") 
        plt.suptitle("Imaging Resonance: " + str(run_image_name))
        plt.legend()
        plt.savefig(os.path.join(get_workfolder_path(), title + "_Graph.png"))
        plt.show()



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


def get_workfolder_path():
    PRIVATE_DIRECTORY_REPO_NAME = "Private_BEC1_Analysis"
    path_to_private_directory_repo = os.path.join(path_to_repo_folder, PRIVATE_DIRECTORY_REPO_NAME)
    current_datetime = datetime.datetime.now() 
    current_year = current_datetime.strftime("%Y")
    current_year_month = current_datetime.strftime("%Y-%m")
    current_year_month_day = current_datetime.strftime("%Y-%m-%d")
    workfolder_pathname = os.path.join(path_to_private_directory_repo, current_year, current_year_month, current_year_month_day + "_Imaging_Resonance")
    return workfolder_pathname


if __name__ == "__main__":
    main()