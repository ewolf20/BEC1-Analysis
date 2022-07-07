import hashlib
import json
import os 
import sys 

import matplotlib.pyplot as plt 

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

TEST_MEASUREMENT_DIRECTORY_PATH = "./resources"

TEST_IMAGE_FILE_PATH = "resources/805277_2022-04-06--8-49-08_Side.fits"
TEST_IMAGE_FILE_NAME = "805277_2022-04-06--8-49-08_Side.fits"

TEST_IMAGE_RUN_ID = 805277
TEST_IMAGE_PATHNAME_DICT = {'side_image': TEST_IMAGE_FILE_PATH}

TEST_IMAGE_ARRAY_SHA_256_HEX_STRING =  '8995013339fed807810ad04c32f5f0db96ea34ca4e3d924d408c35f22da8facb'
RUN_PARAMS_SHA_CHECKSUM = '9693e102bc60e7a8944743c883e51d154a7527a705c07af6df6cc5f7fc96ecec'

from BEC1_Analysis.code.measurement import Run, Measurement
from satyendra.code.breadboard_functions import load_breadboard_client

def check_sha_hash(my_bytes, checksum_string):
    m = hashlib.sha256() 
    m.update(my_bytes) 
    return m.hexdigest() == checksum_string

def get_sha_hash_string(my_bytes):
    m = hashlib.sha256() 
    m.update(my_bytes) 
    return m.hexdigest()

class TestMeasurement:

    @staticmethod 
    def initialize_measurement():
        return Measurement(measurement_directory_path = TEST_MEASUREMENT_DIRECTORY_PATH, imaging_type = 'side_low_mag')

    @staticmethod 
    def test__init__():
        my_measurement = TestMeasurement.initialize_measurement() 
        assert True

    @staticmethod 
    def test_initialize_runs_dict():
        my_measurement = TestMeasurement.initialize_measurement() 
        my_measurement._initialize_runs_dict()
        my_runs_dict = my_measurement.runs_dict
        assert list(my_runs_dict)[0] == TEST_IMAGE_RUN_ID
        my_run = my_runs_dict[TEST_IMAGE_RUN_ID]
        my_run_image = my_run.get_image('Side')
        my_run_params = my_run.get_parameters()
        my_run_params_bytes = json.dumps(my_run_params).encode("ASCII")
        assert check_sha_hash(my_run_image.data.tobytes(), TEST_IMAGE_ARRAY_SHA_256_HEX_STRING)
        assert check_sha_hash(my_run_params_bytes, RUN_PARAMS_SHA_CHECKSUM)

    @staticmethod 
    def test_runs_dict_dump_and_load():
        RUN_PARAMS_SHA_CHECKSUM = '9693e102bc60e7a8944743c883e51d154a7527a705c07af6df6cc5f7fc96ecec'
        my_measurement = TestMeasurement.initialize_measurement() 
        my_measurement._initialize_runs_dict()
        try:
            my_measurement.dump_runs_dict(dump_filename = 'foo.txt')
            my_measurement._initialize_runs_dict(use_saved_params = True, saved_params_filename = 'foo.txt')
        finally:
            os.remove('foo.txt')
        my_runs_dict = my_measurement.runs_dict
        assert list(my_runs_dict)[0] == TEST_IMAGE_RUN_ID
        my_run = my_runs_dict[TEST_IMAGE_RUN_ID]
        my_run_image = my_run.get_image('Side')
        my_run_params = my_run.get_parameters()
        my_run_params_bytes = json.dumps(my_run_params).encode("ASCII")
        print(get_sha_hash_string(my_run_image.data.tobytes()))
        assert check_sha_hash(my_run_image.data.tobytes(), TEST_IMAGE_ARRAY_SHA_256_HEX_STRING)
        assert check_sha_hash(my_run_params_bytes, RUN_PARAMS_SHA_CHECKSUM)

    @staticmethod
    def test_label_badshots():
        my_measurement = TestMeasurement.initialize_measurement() 
        my_measurement._initialize_runs_dict() 
        my_run = my_measurement.runs_dict[TEST_IMAGE_RUN_ID]
        assert not my_run.is_badshot
        my_measurement.label_badshots(lambda f: list(f))
        assert my_run.is_badshot

    #Does not test the interactive box setting.
    @staticmethod
    def test_set_box():
        my_measurement = TestMeasurement.initialize_measurement() 
        my_measurement._initialize_runs_dict() 
        box_coordinates = [1, 2, 3, 4]
        my_measurement.set_box('foo', box_coordinates = box_coordinates)
        assert my_measurement.measurement_parameters['foo'] == box_coordinates



    @staticmethod
    def test_parse_run_id_from_filename():
        assert TEST_IMAGE_RUN_ID == Measurement._parse_run_id_from_filename(TEST_IMAGE_FILE_NAME)


class TestRun:

    #Class variable runs
    my_run_with_memory = Run(TEST_IMAGE_RUN_ID, TEST_IMAGE_PATHNAME_DICT, breadboard_client = load_breadboard_client())
    my_run_without_memory = Run(TEST_IMAGE_RUN_ID, TEST_IMAGE_PATHNAME_DICT, breadboard_client = load_breadboard_client())

    @staticmethod
    def test__init__():
        my_run = Run(TEST_IMAGE_RUN_ID, TEST_IMAGE_PATHNAME_DICT, breadboard_client = load_breadboard_client())
        assert True

    @staticmethod
    def test_load_image():
        my_image = TestRun.my_run_without_memory.load_image(TEST_IMAGE_FILE_PATH)
        assert check_sha_hash(my_image.data.tobytes(), TEST_IMAGE_ARRAY_SHA_256_HEX_STRING)

    @staticmethod 
    def test_load_parameters():
        my_parameters_dict = TestRun.my_run_without_memory.get_parameters()
        my_parameters_json_string = json.dumps(my_parameters_dict)
        my_parameters_json_bytes = my_parameters_json_string.encode("ASCII") 
        assert check_sha_hash(my_parameters_json_bytes, RUN_PARAMS_SHA_CHECKSUM)
        

    @staticmethod
    def test_get_image():
        my_image_with_memory = TestRun.my_run_with_memory.get_image('side_image') 
        my_image_without_memory = TestRun.my_run_without_memory.get_image('side_image')
        assert check_sha_hash(my_image_with_memory.data.tobytes(), TEST_IMAGE_ARRAY_SHA_256_HEX_STRING)
        assert check_sha_hash(my_image_without_memory.data.tobytes(), TEST_IMAGE_ARRAY_SHA_256_HEX_STRING)
