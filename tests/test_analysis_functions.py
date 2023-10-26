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

BASE_LIGHT_LEVEL = 5000
BASE_DARK_LEVEL = 100

DEFAULT_ABSORPTION = 1.0/np.e
DEFAULT_ABS_IMAGE_SHAPE = (512, 512)     
DEFAULT_ABS_SQUARE_WIDTH = 127
DEFAULT_ABS_SQUARE_CENTER_INDICES = (256, 256)

DEFAULT_ABSORPTION_IMAGE_ROI = [170, 170, 340, 340]
DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE = (170, 170)
DEFAULT_ABSORPTION_IMAGE_NORM_BOX = [10, 10, 160, 160]
DEFAULT_ABSORPTION_IMAGE_NORM_BOX_SHAPE = (150, 150)



def _get_raw_pixels_test_helper(type_name, function_to_test):
    try:
        measurement_pathname, my_measurement, my_run = create_measurement(type_name, ROI = DEFAULT_ABSORPTION_IMAGE_ROI)
        returned_array_no_roi = function_to_test(my_measurement, my_run, crop_roi = False) 
        returned_array_roi = function_to_test(my_measurement, my_run, crop_roi = True) 

        with_atoms_no_roi, without_atoms_no_roi, dark_no_roi = returned_array_no_roi 
        assert with_atoms_no_roi.dtype == np.ushort
        assert without_atoms_no_roi.dtype == np.ushort 
        assert dark_no_roi.dtype == np.ushort
        assert with_atoms_no_roi.shape == DEFAULT_ABS_IMAGE_SHAPE 
        assert without_atoms_no_roi.shape == DEFAULT_ABS_IMAGE_SHAPE 
        assert dark_no_roi.shape == DEFAULT_ABS_IMAGE_SHAPE

        expected_without_atoms_sum = np.prod(DEFAULT_ABS_IMAGE_SHAPE) * float(BASE_LIGHT_LEVEL + BASE_DARK_LEVEL) 
        expected_dark_sum = np.prod(DEFAULT_ABS_IMAGE_SHAPE) * float(BASE_DARK_LEVEL)
        #Not exact because of rounding
        expected_with_atoms_sum = (np.prod(DEFAULT_ABS_IMAGE_SHAPE) * float(BASE_LIGHT_LEVEL + BASE_DARK_LEVEL)
         - np.square(DEFAULT_ABS_SQUARE_WIDTH) * (1.0 - DEFAULT_ABSORPTION) * BASE_LIGHT_LEVEL)

        assert np.isclose(np.sum(without_atoms_no_roi.astype(float)), expected_without_atoms_sum)
        assert np.isclose(np.sum(dark_no_roi.astype(float)), expected_dark_sum)
        #Increased rtol to allow for rounding errors
        assert np.isclose(np.sum(with_atoms_no_roi.astype(float)), expected_with_atoms_sum, rtol = 1e-3)

        with_atoms_roi, without_atoms_roi, dark_roi = returned_array_roi   

        assert with_atoms_roi.dtype == np.ushort
        assert without_atoms_roi.dtype == np.ushort 
        assert dark_roi.dtype == np.ushort
        assert with_atoms_roi.shape == DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE 
        assert without_atoms_roi.shape == DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE 
        assert dark_roi.shape == DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE

        expected_without_atoms_sum_roi = np.prod(DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE) * float(BASE_LIGHT_LEVEL + BASE_DARK_LEVEL) 
        expected_dark_sum_roi = np.prod(DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE) * float(BASE_DARK_LEVEL)
        #Not exact because of rounding
        expected_with_atoms_sum_roi = (np.prod(DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE) * float(BASE_LIGHT_LEVEL + BASE_DARK_LEVEL)
         - np.square(DEFAULT_ABS_SQUARE_WIDTH) * (1.0 - DEFAULT_ABSORPTION) * BASE_LIGHT_LEVEL)

        assert np.isclose(np.sum(without_atoms_roi.astype(float)), expected_without_atoms_sum_roi)
        assert np.isclose(np.sum(dark_roi.astype(float)), expected_dark_sum_roi)
        #Increased rtol to allow for rounding errors
        assert np.isclose(np.sum(with_atoms_roi.astype(float)), expected_with_atoms_sum_roi, rtol = 1e-3)
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_raw_pixels_na_catch():
    type_name = "na_catch"
    function_to_test = analysis_functions.get_raw_pixels_na_catch 
    _get_raw_pixels_test_helper(type_name, function_to_test)

def test_get_raw_pixels_side():
    type_name = "side_low_mag" 
    function_to_test = analysis_functions.get_raw_pixels_side 
    _get_raw_pixels_test_helper(type_name, function_to_test)


def test_get_raw_pixels_top_A():
    type_name = "top_double" 
    function_to_test = analysis_functions.get_raw_pixels_top_A
    _get_raw_pixels_test_helper(type_name, function_to_test)


def test_get_raw_pixels_top_B():
    type_name = "top_double" 
    function_to_test = analysis_functions.get_raw_pixels_top_B
    _get_raw_pixels_test_helper(type_name, function_to_test)


def _get_abs_image_test_helper(type_name, function_to_test):
    try:
        measurement_pathname, my_measurement, my_run = create_measurement(type_name, ROI = DEFAULT_ABSORPTION_IMAGE_ROI, 
                                                        norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        absorption_image = function_to_test(my_measurement, my_run)
        default_absorption_image = get_default_absorption_image()
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = DEFAULT_ABSORPTION_IMAGE_ROI
        cropped_default_absorption_image = default_absorption_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        assert np.all(np.isclose(cropped_default_absorption_image, absorption_image, rtol = 1e-3))
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_abs_image_na_catch():
    type_name = "na_catch"
    function_to_test = analysis_functions.get_abs_image_na_catch
    _get_abs_image_test_helper(type_name, function_to_test)

def test_get_abs_image_side():
    type_name = "side_low_mag"
    function_to_test = analysis_functions.get_abs_image_side
    _get_abs_image_test_helper(type_name, function_to_test)

def test_get_abs_image_top_A():
    type_name = "top_double"
    function_to_test = analysis_functions.get_abs_image_top_A
    _get_abs_image_test_helper(type_name, function_to_test)

def test_get_abs_image_top_B():
    type_name = "top_double" 
    function_to_test = analysis_functions.get_abs_image_top_B
    _get_abs_image_test_helper(type_name, function_to_test)


def _get_od_image_test_helper(type_name, function_to_test):
    try:
        measurement_pathname, my_measurement, my_run = create_measurement(type_name, ROI = DEFAULT_ABSORPTION_IMAGE_ROI, 
                                                                 norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        od_image = function_to_test(my_measurement, my_run) 
        default_od_image = -np.log(get_default_absorption_image())
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = DEFAULT_ABSORPTION_IMAGE_ROI
        cropped_default_od_image = default_od_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        assert np.all(np.isclose(cropped_default_od_image, od_image, rtol = 1e-3))
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_od_image_na_catch():
    type_name = "na_catch"
    function_to_test = analysis_functions.get_od_image_na_catch
    _get_od_image_test_helper(type_name, function_to_test) 

def test_get_od_image_side():
    type_name = "side_low_mag"
    function_to_test = analysis_functions.get_od_image_side 
    _get_od_image_test_helper(type_name, function_to_test)

def test_get_od_image_top_A():
    type_name = "top_double" 
    function_to_test = analysis_functions.get_od_image_top_A 
    _get_od_image_test_helper(type_name, function_to_test)

def test_get_od_image_top_B():
    type_name = "top_double" 
    function_to_test = analysis_functions.get_od_image_top_B 
    _get_od_image_test_helper(type_name, function_to_test)


def _get_od_pixel_sum_test_helper(type_name, function_to_test):
    try:
        measurement_pathname, my_measurement, my_run = create_measurement(type_name, ROI = DEFAULT_ABSORPTION_IMAGE_ROI, 
                                                                          norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        pixel_sum = function_to_test(my_measurement, my_run) 
        expected_pixel_sum = np.square(DEFAULT_ABS_SQUARE_WIDTH)
        assert np.isclose(expected_pixel_sum, pixel_sum, rtol = 1e-3)
    finally:
        shutil.rmtree(measurement_pathname)


def test_get_od_pixel_sum_na_catch():
    type_name = "na_catch" 
    function_to_test = analysis_functions.get_od_pixel_sum_na_catch
    _get_od_pixel_sum_test_helper(type_name, function_to_test)

def test_get_od_pixel_sum_side():
    type_name = "side_low_mag" 
    function_to_test = analysis_functions.get_od_pixel_sum_side
    _get_od_pixel_sum_test_helper(type_name, function_to_test) 

def test_get_od_pixel_sum_top_A():
    type_name = "top_double" 
    function_to_test = analysis_functions.get_od_pixel_sum_top_A 
    _get_od_pixel_sum_test_helper(type_name, function_to_test)

def test_get_od_pixel_sum_top_B():
    type_name = "top_double" 
    function_to_test = analysis_functions.get_od_pixel_sum_top_A 
    _get_od_pixel_sum_test_helper(type_name, function_to_test)


def create_measurement(type_name, image_stack = None, run_param_values= None, experiment_param_values = None, ROI = None, norm_box = None):
    if image_stack is None:
        default_abs_image = get_default_absorption_image()
        image_stack = generate_image_stack_from_absorption(default_abs_image)
    if experiment_param_values is None:
        experiment_param_values = {} 
    if run_param_values is None:
        run_param_values = {}         
    measurement_pathname = create_dummy_measurement_folder(image_stack, run_param_values,
                                experiment_param_values, type_name)
    my_measurement = measurement.Measurement(measurement_directory_path = measurement_pathname, 
                                    imaging_type = type_name)
    if not ROI is None:
        my_measurement.set_ROI(box_coordinates = ROI)
    if not norm_box is None:
        my_measurement.set_norm_box(box_coordinates = norm_box)
    for run_id in my_measurement.runs_dict:
        my_run = my_measurement.runs_dict[run_id] 
        break
    return (measurement_pathname, my_measurement, my_run)


def create_side_low_mag_measurement(image_stack = None, run_param_values = None, experiment_param_values = None, ROI = None, norm_box = None):
    return create_measurement("side_low_mag", image_stack = image_stack, run_param_values = run_param_values, 
                    experiment_param_values = experiment_param_values, ROI = ROI, norm_box = norm_box)



def create_side_high_mag_measurement(image_stack = None, run_param_values = None, experiment_param_values = None, ROI = None, norm_box = None):
    return create_measurement("side_high_mag", image_stack = image_stack, run_param_values = run_param_values, 
                    experiment_param_values = experiment_param_values, ROI = ROI, norm_box = norm_box)

def create_top_measurement(image_stack = None, run_param_values = None, experiment_param_values = None, ROI = None, norm_box = None):
    return create_measurement("top_double", image_stack = image_stack, run_param_values = run_param_values, 
                    experiment_param_values = experiment_param_values, ROI = ROI, norm_box = norm_box)


def create_catch_measurement(image_stack = None, run_param_values = None, experiment_param_values = None, ROI = None, norm_box = None):
    return create_measurement("na_catch", image_stack = image_stack, run_param_values = run_param_values, 
                    experiment_param_values = experiment_param_values, ROI = ROI, norm_box = norm_box)


#Generate a test image, suitable for use in most analysis functions
#The image pattern is a square, with -ln(abs) = 1 for a grid of pixels centered 
#on the origin and -ln(abs) = 0 for all others.
def get_default_absorption_image():
    center_y_index, center_x_index = DEFAULT_ABS_SQUARE_CENTER_INDICES
    y_indices, x_indices = np.indices(DEFAULT_ABS_IMAGE_SHAPE)
    default_absorption_image = np.where(
        np.logical_and(
            np.abs(y_indices - center_y_index) < (DEFAULT_ABS_SQUARE_WIDTH + 1) //2, 
            np.abs(x_indices - center_x_index) < (DEFAULT_ABS_SQUARE_WIDTH + 1) // 2
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
    return dummy_measurement_folder_pathname
    
#Given an array of floats representing the absorption of an image (with - dark)/(without - dark), generate 
#a dummy image stack to replicate the absorption image.

def generate_image_stack_from_absorption(absorption_array):
    abs_array_shape = absorption_array.shape 
    dark_image = np.full(abs_array_shape, BASE_DARK_LEVEL, dtype = np.ushort)
    without_image = np.full(abs_array_shape, BASE_LIGHT_LEVEL + BASE_DARK_LEVEL, dtype = np.ushort)
    with_image_unrounded = np.ones(abs_array_shape) * BASE_DARK_LEVEL + np.ones(abs_array_shape) * BASE_LIGHT_LEVEL * absorption_array 
    with_image = np.round(with_image_unrounded).astype(np.ushort)
    return np.stack((with_image, without_image, dark_image))


def save_run_image(image_stack, pathname):
    hdu = astropy.io.fits.PrimaryHDU(image_stack)
    hdu.writeto(pathname)



