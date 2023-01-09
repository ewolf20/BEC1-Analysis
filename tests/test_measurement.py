import hashlib
import json
import os 
import sys 
import shutil

import numpy as np

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

TEST_MEASUREMENT_DIRECTORY_PATH = "./resources"
TEST_MEASUREMENT_DIRECTORY_PATH_WITH_DUMP = "./resources/Test_Dump_Directory"

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
        my_measurement_params_from_dump = Measurement(measurement_directory_path = TEST_MEASUREMENT_DIRECTORY_PATH, imaging_type = "side_low_mag")
        my_measurement_params_from_dump._initialize_runs_dict()
        assert list(my_runs_dict)[0] == TEST_IMAGE_RUN_ID 
        my_run = my_runs_dict[TEST_IMAGE_RUN_ID]
        my_run_params = my_run.get_parameters() 
        my_run_params_bytes = json.dumps(my_run_params).encode("ASCII") 
        assert check_sha_hash(my_run_params_bytes, RUN_PARAMS_SHA_CHECKSUM)


    @staticmethod
    def test_update_runs_dict():
        pass


    @staticmethod 
    def test_analysis_dump_and_load():
        RUN_ANALYSIS_CHECKSUM = None
        my_measurement = TestMeasurement.initialize_measurement()
        def analysis_function_scalar_zero(my_measurement, my_run):
            return 0.0
        def analysis_function_scalar_one(my_measurement, my_run):
            return 1.0
        def analysis_function_array_zero(my_measurement, my_run):
            return np.array([0.0])
        my_measurement.analyze_runs(analysis_function_scalar_zero, "bar")
        my_measurement.analyze_runs(analysis_function_array_zero, "baz")
        try:
            TEST_DUMP_FOLDERNAME = "Temp"
            os.mkdir(TEST_DUMP_FOLDERNAME)
            TEST_DUMP_FILENAME = "foo.json"
            test_dump_pathname = os.path.join(TEST_DUMP_FOLDERNAME, TEST_DUMP_FILENAME)
            my_measurement.dump_run_analysis_dict(dump_pathname = test_dump_pathname)
            my_measurement.analyze_runs(analysis_function_scalar_one, "bar", overwrite_existing = True)
            my_measurement.load_run_analysis_dict(dump_pathname = test_dump_pathname)
        finally:
            shutil.rmtree(TEST_DUMP_FOLDERNAME)
        assert my_measurement.get_analysis_value_from_runs("bar") == [0.0]
        assert my_measurement.get_analysis_value_from_runs("baz") == [np.array([0.0])]

    @staticmethod
    def test_get_parameter_value_from_runs():
        VALUE_NAME_TO_CHECK = "id"
        EXPECTED_VALUES = [TEST_IMAGE_RUN_ID]
        my_measurement = TestMeasurement.initialize_measurement() 
        my_measurement._initialize_runs_dict()
        values = my_measurement.get_parameter_value_from_runs("id")
        assert values == EXPECTED_VALUES


    @staticmethod 
    def test_get_analysis_value_from_runs():
        VALUE_NAME_TO_CHECK = "foo"
        VALUE = 3 
        VALUE_AS_LIST = [3]
        my_measurement = TestMeasurement.initialize_measurement() 
        my_measurement._initialize_runs_dict()
        for run_id in my_measurement.runs_dict:
            current_run = my_measurement.runs_dict[run_id] 
            current_run.analysis_results[VALUE_NAME_TO_CHECK] = VALUE 
        assert my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK) == VALUE_AS_LIST


    @staticmethod 
    def test_get_parameter_analysis_result_pair_from_runs():
        VALUE_NAME_TO_CHECK = "foo"
        VALUE = 3
        VALUE_AS_LIST = [VALUE]
        EXPECTED_PARAMS = [TEST_IMAGE_RUN_ID]
        my_measurement = TestMeasurement.initialize_measurement() 
        my_measurement._initialize_runs_dict()
        for run_id in my_measurement.runs_dict:
            current_run = my_measurement.runs_dict[run_id] 
            current_run.analysis_results[VALUE_NAME_TO_CHECK] = VALUE 
        param_list, analysis_list = my_measurement.get_parameter_analysis_result_pair_from_runs("id", "foo")
        assert param_list == EXPECTED_PARAMS
        assert analysis_list == VALUE_AS_LIST
        for run_id in my_measurement.runs_dict:
            current_run = my_measurement.runs_dict[run_id] 
            current_run.analysis_results[VALUE_NAME_TO_CHECK] = Measurement.ANALYSIS_ERROR_INDICATOR_STRING
        param_list, analysis_list = my_measurement.get_parameter_analysis_result_pair_from_runs("id", "foo", ignore_errors = True)
        assert param_list == []
        assert analysis_list == []
        param_list, analysis_list = my_measurement.get_parameter_analysis_result_pair_from_runs("id", "foo", ignore_errors = False)
        assert param_list == EXPECTED_PARAMS
        assert analysis_list == [Measurement.ANALYSIS_ERROR_INDICATOR_STRING]




    @staticmethod 
    def test_analyze_runs():
        VALUE_NAME_TO_CHECK = "foo"
        VALUE_1 = 1
        VALUE_2 = 2
        VALUE_1_AS_LIST = [1] 
        VALUE_2_AS_LIST = [2]
        my_measurement = TestMeasurement.initialize_measurement() 
        my_measurement._initialize_runs_dict()
        def analysis_func_1(my_measurement, my_run):
            return (VALUE_1,)
        def analysis_func_2(my_measurement, my_run):
            return (VALUE_2,)
        def analysis_func_1_scalar(my_measurement, my_run):
            return VALUE_1
        def analysis_func_error(my_measurement, my_run):
            raise RuntimeError("Intended error for measurement analysis testing.")
        my_measurement.analyze_runs(analysis_func_1, (VALUE_NAME_TO_CHECK,))
        assert my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK) == VALUE_1_AS_LIST
        my_measurement.analyze_runs(analysis_func_2, (VALUE_NAME_TO_CHECK,), overwrite_existing = False)
        assert my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK) == VALUE_1_AS_LIST
        my_measurement.analyze_runs(analysis_func_2, (VALUE_NAME_TO_CHECK,), overwrite_existing = True)
        assert my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK) == VALUE_2_AS_LIST
        my_measurement.analyze_runs(analysis_func_1_scalar, VALUE_NAME_TO_CHECK, overwrite_existing = True)
        assert my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK) == VALUE_1_AS_LIST
        ERR_NAME_TO_CHECK = "bar"
        try:
            my_measurement.analyze_runs(analysis_func_error, ERR_NAME_TO_CHECK, catch_errors = False)
        except Exception as e:
            pass
        else:
            raise ValueError("There was supposed to be an error here.")
        my_measurement.analyze_runs(analysis_func_error, ERR_NAME_TO_CHECK, catch_errors = True)
        assert my_measurement.get_analysis_value_from_runs(ERR_NAME_TO_CHECK, ignore_errors = False) == [Measurement.ANALYSIS_ERROR_INDICATOR_STRING]

    @staticmethod
    def test_label_badshots_custom():
        my_measurement = TestMeasurement.initialize_measurement() 
        my_measurement._initialize_runs_dict() 
        my_run = my_measurement.runs_dict[TEST_IMAGE_RUN_ID]
        assert not my_run.is_badshot
        def badshot_function_true(my_measurement, my_run):
            return True
        def badshot_function_false(my_measurement, my_run):
            return False
        my_measurement.label_badshots_custom(badshot_function = badshot_function_true)
        assert my_run.is_badshot
        my_measurement.label_badshots_custom(badshot_function = badshot_function_false)
        assert my_run.is_badshot 
        my_measurement.label_badshots_custom(badshot_function = badshot_function_false, override_existing_badshots = True)
        assert not my_run.is_badshot
        my_measurement.label_badshots_custom(badshots_list = [TEST_IMAGE_RUN_ID])
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
    my_run_with_memory = Run(TEST_IMAGE_RUN_ID, TEST_IMAGE_PATHNAME_DICT, {})
    my_run_without_memory = Run(TEST_IMAGE_RUN_ID, TEST_IMAGE_PATHNAME_DICT, {})

    @staticmethod
    def test__init__():
        my_run = Run(TEST_IMAGE_RUN_ID, TEST_IMAGE_PATHNAME_DICT, {})
        assert True

    @staticmethod
    def test_load_image():
        my_image = TestRun.my_run_without_memory.load_image(TEST_IMAGE_FILE_PATH)
        assert check_sha_hash(my_image.data.tobytes(), TEST_IMAGE_ARRAY_SHA_256_HEX_STRING)

    @staticmethod
    def test_get_image():
        my_image_with_memory = TestRun.my_run_with_memory.get_image('side_image') 
        my_image_without_memory = TestRun.my_run_without_memory.get_image('side_image')
        assert check_sha_hash(my_image_with_memory.data.tobytes(), TEST_IMAGE_ARRAY_SHA_256_HEX_STRING)
        assert check_sha_hash(my_image_without_memory.data.tobytes(), TEST_IMAGE_ARRAY_SHA_256_HEX_STRING)
