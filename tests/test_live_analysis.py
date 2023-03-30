import datetime
import json
import os 
import sys 
import shutil
import time

import numpy as np
from astropy.io import fits 

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

RESOURCES_FOLDER_PATH = "./resources"
TEMP_LIVE_ANALYSIS_FOLDER_NAME = "live_analysis_temp"
EXPERIMENT_PARAMETERS_FILENAME = "experiment_parameters.json"


FILENAME_DATETIME_FORMAT_STRING = "%Y-%m-%d--%H-%M-%S"
PARAMETERS_DATETIME_FORMAT_STRING = "%Y-%m-%dT%H:%M:%SZ"
PARAMETERS_RUN_TIME_NAME = "runtime"
FILENAME_DELIMITER_CHAR = "_"
TEMP_FILE_MARKER = "TEMP"

DATA_DUMP_PARAMS_FILENAME = "run_params_dump.json" 
DATA_DUMP_PARAMS_TEMP_FILENAME = "run_params_dumpTEMP.json"


IMAGE_FILE_EXTENSION = ".fits"

UINT_16_CUTOFF = 65536

from BEC1_Analysis.code.measurement import Run, Measurement



def put_image_in_live_analysis_folder_fake(run_id, live_analysis_folder_pathname, pixel_size = 2160, mode = "Top", rng = None, 
                                            val = None):
    if val is None and rng is None:
        rng = np.random.default_rng()
    TOP_IMAGE_NAMES = ["TopA", "TopB"]
    SIDE_IMAGE_NAMES = ["Side"]
    if mode == "Top":
        image_names = TOP_IMAGE_NAMES
    elif mode == "Side":
        image_names = SIDE_IMAGE_NAMES
    image_filename_list_sans_extension = [_make_image_filename(run_id, f) for f in image_names]
    for image_filename_sans_extension in image_filename_list_sans_extension:
        if val is None:
            image_data = rng.integers(0, UINT_16_CUTOFF, size = (3, pixel_size, pixel_size), dtype = np.uint16)
        else:
            image_data = np.zeros((3, pixel_size, pixel_size)) + val
        _save_image_to_live_analysis_folder(image_data, live_analysis_folder_pathname, image_filename_sans_extension, IMAGE_FILE_EXTENSION)
    

def _make_image_filename(run_id, image_type_string):
    current_datetime = datetime.datetime.now()
    current_datetime_string = current_datetime.strftime(FILENAME_DATETIME_FORMAT_STRING)
    return FILENAME_DELIMITER_CHAR.join((str(run_id), current_datetime_string, image_type_string))


def _save_image_to_live_analysis_folder(image_numpy_array, live_analysis_folder_pathname, base_filename, file_extension):
    temp_filename = base_filename + FILENAME_DELIMITER_CHAR + file_extension 
    temp_pathname = os.path.join(live_analysis_folder_pathname, temp_filename)
    final_filename = base_filename + file_extension 
    final_pathname = os.path.join(live_analysis_folder_pathname, final_filename)
    hdu = fits.PrimaryHDU(image_numpy_array)
    hdu.writeto(temp_pathname)
    os.rename(temp_pathname, final_pathname)


def initialize_measurement(live_analysis_folder_pathname, imaging_type):
    my_measurement = Measurement(measurement_directory_path = live_analysis_folder_pathname, 
                                image_format = IMAGE_FILE_EXTENSION, imaging_type = imaging_type)
    return my_measurement


def update_live_analysis_parameters_json(live_analysis_folder_pathname, run_id, run_parameters = None):
    if run_parameters is None:
        run_parameters = {}
    current_datetime = datetime.datetime.now() 
    current_datetime_string = current_datetime.strftime(PARAMETERS_DATETIME_FORMAT_STRING)
    run_parameters["runtime"] = current_datetime_string 
    run_parameters["id"] = run_id
    run_parameters["ListBoundVariables"] = []
    parameters_pathname = os.path.join(live_analysis_folder_pathname, DATA_DUMP_PARAMS_FILENAME)
    parameters_temp_pathname = os.path.join(live_analysis_folder_pathname, DATA_DUMP_PARAMS_TEMP_FILENAME)
    if os.path.exists(parameters_pathname):
        with open(parameters_pathname, 'r') as f:
            initial_dict = json.load(f)
    else:
        initial_dict = {}
    initial_dict[str(run_id)] = run_parameters 
    _save_run_parameters(initial_dict, parameters_pathname, parameters_temp_pathname)
    


def _save_run_parameters(parameters_dict, parameters_pathname, parameters_temp_pathname):
    with open(parameters_temp_pathname, 'w') as f:
        json.dump(parameters_dict, f)
    SAVING_PATIENCE = 3 
    counter = 0 
    while counter < SAVING_PATIENCE:
        try:
            os.replace(parameters_temp_pathname, parameters_pathname)
        except OSError as e:
            if counter < SAVING_PATIENCE:
                counter += 1
                time.sleep(0.1)
            else:
                raise e


def initialize_live_analysis_folder_fake():
    #Save experiment parameters and run_parameters_dump files
    live_analysis_folder_pathname = os.path.join(RESOURCES_FOLDER_PATH, TEMP_LIVE_ANALYSIS_FOLDER_NAME)
    if not os.path.exists(live_analysis_folder_pathname):
        os.mkdir(live_analysis_folder_pathname)
    fake_experiment_parameters_dict = {"Values":{}, "Update_Times":{}}
    experiment_parameters_pathname = os.path.join(live_analysis_folder_pathname, EXPERIMENT_PARAMETERS_FILENAME)
    with open(experiment_parameters_pathname, 'w') as f:
        json.dump(fake_experiment_parameters_dict, f)
    params_dump_pathname = os.path.join(live_analysis_folder_pathname, DATA_DUMP_PARAMS_FILENAME)
    with open(params_dump_pathname, 'w') as f:
        json.dump({}, f) 
    return live_analysis_folder_pathname



#================Begin tests=================

LIVE_ANALYSIS_TEST_NUMBER = 1347
LIVE_ANALYSIS_TIME_INCREMENT_SECS = 1.0

def fake_live_analysis_function(my_measurement, my_run):
    total_sum = 0.0
    for image_name in my_run.image_dict:
        total_sum += my_run.get_image(image_name)[0][0][0]
    return "Run id: {0:.1f}, sum: {1:.1f}".format(my_run.parameters["id"], total_sum % UINT_16_CUTOFF)

def test_live_analysis_top_double_normal_ordering():
    RNG_SEED = 1337
    rng = np.random.default_rng(seed = RNG_SEED)
    live_analysis_folder_pathname = initialize_live_analysis_folder_fake()

    try:
        my_measurement = initialize_measurement(live_analysis_folder_pathname, "top_double")
        my_measurement.add_to_live_analyses(fake_live_analysis_function, "foo")
        counter = 1337
        while counter < LIVE_ANALYSIS_TEST_NUMBER:
            run_id = counter 
            update_live_analysis_parameters_json(live_analysis_folder_pathname, run_id)
            put_image_in_live_analysis_folder_fake(run_id, live_analysis_folder_pathname, mode = "Top", val = counter)
            my_measurement.update(catch_errors = False)
            print(my_measurement.get_analysis_value_from_runs("foo"))
            print([my_measurement.runs_dict[key].image_dict for key in my_measurement.runs_dict])
            counter += 1
    finally:
        shutil.rmtree(live_analysis_folder_pathname)

def test_live_analysis_top_double_params_meas_image_ordering():
    RNG_SEED = 1337
    rng = np.random.default_rng(seed = RNG_SEED)
    live_analysis_folder_pathname = initialize_live_analysis_folder_fake()

    try:
        my_measurement = initialize_measurement(live_analysis_folder_pathname, "top_double")
        my_measurement.add_to_live_analyses(fake_live_analysis_function, "foo")
        counter = 1337
        while counter < LIVE_ANALYSIS_TEST_NUMBER:
            run_id = counter 
            update_live_analysis_parameters_json(live_analysis_folder_pathname, run_id)
            my_measurement.update(catch_errors = False)
            put_image_in_live_analysis_folder_fake(run_id, live_analysis_folder_pathname, mode = "Top", val = counter)
            print(my_measurement.get_analysis_value_from_runs("foo"))
            counter += 1
    finally:
        shutil.rmtree(live_analysis_folder_pathname)


def test_live_analysis_top_double_image_meas_params_ordering():
    RNG_SEED = 1337
    rng = np.random.default_rng(seed = RNG_SEED)
    live_analysis_folder_pathname = initialize_live_analysis_folder_fake()
    try:
        my_measurement = initialize_measurement(live_analysis_folder_pathname, "top_double")
        my_measurement.add_to_live_analyses(fake_live_analysis_function, "foo")
        counter = 0
        while counter < LIVE_ANALYSIS_TEST_NUMBER:
            run_id = counter 
            put_image_in_live_analysis_folder_fake(run_id, live_analysis_folder_pathname, mode = "Top", val = counter)
            my_measurement.update(catch_errors = False)
            update_live_analysis_parameters_json(live_analysis_folder_pathname, run_id)
            print(my_measurement.get_analysis_value_from_runs("foo"))
            counter += 1
    finally:
        shutil.rmtree(live_analysis_folder_pathname)









