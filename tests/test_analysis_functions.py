import hashlib
import json
import os 
import sys 
import shutil


import astropy
import numpy as np
#Temp import 
import matplotlib.pyplot as plt


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)
from BEC1_Analysis.code import measurement
from BEC1_Analysis.code import analysis_functions


TEMP_WORKSPACE_PATH = "./resources"
TEMP_MEASUREMENT_FOLDER_NAME = "analysis_test_temp_measurement"

EPOCH_TIMESTRING_PARAMETERS = "1970-01-01T00:00:00"
EPOCH_TIMESTRING_FILENAME = "1970-01-01--00-00-00"


def create_side_lf_measurement():
    pass 

def create_side_hf_measurement():
    pass 


def create_top_measurement():
    pass

def create_catch_measurement():
    pass


#Generate a test image, suitable for use in most analysis functions
#The image pattern is a square, with -ln(abs) = 1 for a grid of pixels centered 
#on the origin and -ln(abs) = 0 for all others.
def get_default_absorption_image():
    DEFAULT_ABS_IMAGE_SHAPE = (512, 512)     
    SQUARE_WIDTH = 128
    SQUARE_CENTER_INDICES = (256, 256)
    center_y_index, center_x_index = SQUARE_CENTER_INDICES
    y_indices, x_indices = np.indices(DEFAULT_ABS_IMAGE_SHAPE)
    default_absorption_image = np.where(
        np.logical_and(
            np.abs(y_indices - center_y_index) < SQUARE_WIDTH //2, 
            np.abs(x_indices - center_x_index) < SQUARE_WIDTH // 2
        ),
        1.0/np.e, 
        1.0
    )
    return default_absorption_image

#Create a dummy measurement folder with a single (simulated) image, 
#plus experiment_parameters.json file and run_params_dump.json file.
#If the measurement is of a type that has multiple images per run, the same image stack is used for each.
def create_dummy_measurement_folder(image_stack, run_param_values, experiment_param_values, measurement_type):
    dummy_measurement_folder_pathname = os.path.join(TEMP_WORKSPACE_PATH, TEMP_MEASUREMENT_FOLDER_NAME)
    if not os.path.exists(dummy_measurement_folder_pathname):
        os.mkdir(dummy_measurement_folder_pathname)
    else:
        raise RuntimeError("Dummy folder already exists")
    #Create fake images
    DUMMY_RUN_ID = 1
    run_names = measurement.MEASUREMENT_IMAGE_NAME_DICT[measurement_type]
    for run_name in run_names:
        dummy_run_image_filename = "{0:d}_{1}_{2}.fits".format(DUMMY_RUN_ID, EPOCH_TIMESTRING_FILENAME, run_name)
        dummy_run_image_pathname = os.path.join(dummy_measurement_folder_pathname, dummy_run_image_filename)
        save_run_image(image_stack, dummy_run_image_pathname)
    #Create fake experiment parameters json
    update_time_dict = {} 
    for key in experiment_param_values:
        update_time_dict[key] = EPOCH_TIMESTRING_FILENAME
    experiment_parameters_dict = {}
    experiment_parameters_dict["Values"] = experiment_param_values 
    experiment_parameters_dict["Update_Times"] = update_time_dict
    experiment_parameters_pathname = os.path.join(dummy_measurement_folder_pathname, measurement.Measurement.MEASUREMENT_FOLDER_EXPERIMENT_PARAMS_FILENAME)
    with open(experiment_parameters_pathname, 'w') as f:
        json.dump(experiment_parameters_dict, f)
    #Create fake run parameters json
    if not "id" in run_param_values:
        run_param_values["id"] = DUMMY_RUN_ID 
    if not "runtime" in run_param_values:
        run_param_values["runtime"] = EPOCH_TIMESTRING_PARAMETERS
    run_parameters_dict = {}
    run_parameters_dict[str(DUMMY_RUN_ID)] = run_param_values 
    run_parameters_pathname = os.path.join(dummy_measurement_folder_pathname, measurement.Measurement.MEASUREMENT_FOLDER_RUN_PARAMS_FILENAME)
    with open(run_parameters_pathname, 'w') as f:
        json.dump(run_parameters_dict, f)
    
#Given an array of floats representing the absorption of an image (with - dark)/(without - dark), generate 
#a dummy image stack to replicate the absorption image.
def generate_image_stack_from_absorption(absorption_array):
    BASE_LIGHT_LEVEL = 5000
    BASE_DARK_LEVEL = 100
    abs_array_shape = absorption_array.shape 
    dark_image = np.full(abs_array_shape, BASE_DARK_LEVEL, dtype = np.ushort)
    without_image = np.full(abs_array_shape, BASE_LIGHT_LEVEL + BASE_DARK_LEVEL, dtype = np.ushort)
    with_image_unrounded = np.ones(abs_array_shape) * BASE_DARK_LEVEL + np.ones(abs_array_shape) * BASE_LIGHT_LEVEL * absorption_array 
    with_image = np.round(with_image_unrounded).astype(np.ushort)
    return np.stack((with_image, without_image, dark_image))


def save_run_image(image_stack, pathname):
    hdu = astropy.io.fits.PrimaryHDU(image_stack)
    hdu.writeto(pathname)



