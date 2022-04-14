import hashlib
import json
import os 
import sys 

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

TEST_MEASUREMENT_DIRECTORY_PATH = "./resources"

TEST_IMAGE_FILE_PATH = "resources/805277_2022-04-06--8-49-08_Side.fits"
TEST_IMAGE_FILE_NAME = "805277_2022-04-06--8-49-08_Side.fits"

TEST_IMAGE_RUN_ID = 805277
TEST_IMAGE_PATHNAME_DICT = {'side_image': TEST_IMAGE_FILE_PATH}

TEST_IMAGE_ARRAY_SHA_256_HEX_STRING =  '8995013339fed807810ad04c32f5f0db96ea34ca4e3d924d408c35f22da8facb'
TEST_IMAGE_PARAMETERS_SHA_256_HEX_STRING = '8501acefd3c455d5712c5e569d2cf66e6259f912cb564e7a586ffe456ce16733'

from BEC1_Analysis.code.measurement import Run, Measurement
from satyendra.code.breadboard_functions import load_breadboard_client



def check_sha_hash(bytes, checksum_string):
    m = hashlib.sha256() 
    m.update(bytes) 
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
        RUN_PARAMS_SHA_CHECKSUM = '9693e102bc60e7a8944743c883e51d154a7527a705c07af6df6cc5f7fc96ecec'
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
    def test_label_badshots():
        my_measurement = TestMeasurement.initialize_measurement() 
        my_measurement._initialize_runs_dict() 
        my_run = my_measurement.runs_dict[TEST_IMAGE_RUN_ID]
        assert not my_run.is_badshot
        my_measurement.label_badshots(lambda f: True)
        assert my_run.is_badshot
        




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
        my_parameters_df = TestRun.my_run_without_memory.get_parameters()
        my_parameters_json_string = my_parameters_df.to_json() 
        my_parameters_json_bytes = my_parameters_json_string.encode("ASCII") 
        assert check_sha_hash(my_parameters_json_bytes, TEST_IMAGE_PARAMETERS_SHA_256_HEX_STRING)
        

    @staticmethod
    def test_get_image():
        my_image_with_memory = TestRun.my_run_with_memory.get_image('side_image') 
        my_image_without_memory = TestRun.my_run_without_memory.get_image('side_image')
        assert check_sha_hash(my_image_with_memory.data.tobytes(), TEST_IMAGE_ARRAY_SHA_256_HEX_STRING)
        assert check_sha_hash(my_image_without_memory.data.tobytes(), TEST_IMAGE_ARRAY_SHA_256_HEX_STRING)


    @staticmethod
    def check_sha_hash(bytes, checksum_string):
        m = hashlib.sha256() 
        m.update(bytes) 
        return m.hexdigest() == checksum_string
    

