from collections import namedtuple
import hashlib
import os 
import sys 
import shutil

import numpy as np

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

TEST_MEASUREMENT_DIRECTORY_PATH = "./resources/Test_Measurement_Directory"
TEST_MEASUREMENT_DIRECTORY_PATH_WITH_DUMP = "./resources/Test_Dump_Directory"

TEST_IMAGE_FILE_PATH = "resources/Test_Measurement_Directory/805277_2022-04-06--8-49-08_Side.fits"
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
    def initialize_measurement(connected_mode = True):
        return Measurement(measurement_directory_path = TEST_MEASUREMENT_DIRECTORY_PATH, imaging_type = 'side_low_mag', 
                            connected_mode = connected_mode)

    @staticmethod 
    def test__init__():
        my_measurement = TestMeasurement.initialize_measurement() 
        assert True

    @staticmethod 
    def test_override_experiment_parameter():
        my_measurement = TestMeasurement.initialize_measurement()
        my_measurement.experiment_parameters = {"foo":1} 
        my_measurement.override_experiment_parameter("foo", 2) 
        assert len(my_measurement.experiment_parameters) == 2
        assert my_measurement.experiment_parameters["foo"] == 2
        assert Measurement.EXPERIMENT_PARAMETER_OVERRIDE_PREFIX + "foo" in my_measurement.experiment_parameters

    @staticmethod
    def test_undo_experiment_parameter_overrides():
        my_measurement = TestMeasurement.initialize_measurement()
        my_measurement.experiment_parameters = {"foo":1} 
        my_measurement.override_experiment_parameter("foo", 2) 
        assert len(my_measurement.experiment_parameters) == 2
        my_measurement.undo_experiment_parameter_overrides() 
        assert len(my_measurement.experiment_parameters) == 1
        assert my_measurement.experiment_parameters["foo"] == 1


    @staticmethod
    def test_combine_runs():
        my_measurement = TestMeasurement.initialize_measurement() 
        FakeRun = namedtuple("FakeRun", ("val", "is_badshot"))
        def make_fake_run(val, is_badshot = False):
            return FakeRun(val = val, is_badshot = is_badshot)
        my_measurement.runs_dict = {1:make_fake_run("Oh"), 2:make_fake_run("hi"),
                             3:make_fake_run("there"), 4:make_fake_run("neighbor")}
        pre_combine_dict = my_measurement.runs_dict

        def trivial_run_hash_function(my_run):
            return 0
        
        def run_combine_function(my_run_list):
            my_val_list = [run.val for run in my_run_list]
            overall_string = ' '.join(my_val_list)
            return make_fake_run(overall_string)

        my_measurement.combine_runs(trivial_run_hash_function, run_combine_function)
        assert len(my_measurement.runs_dict) == 1
        assert my_measurement.pre_combine_runs_dict == pre_combine_dict
        assert my_measurement.runs_dict[0].val == "Oh hi there neighbor"

        my_measurement.undo_combine_runs()

        #Test combination 
        def parity_run_hash_function(my_run):
            return len(my_run.val) % 2
        
        my_measurement.combine_runs(parity_run_hash_function, run_combine_function)
        assert len(my_measurement.runs_dict) == 2
        vals_list = [my_measurement.runs_dict[key].val for key in my_measurement.runs_dict]
        assert "Oh hi neighbor" in vals_list 
        assert "there" in vals_list


    @staticmethod 
    def test_undo_combine_runs():
        my_measurement = TestMeasurement.initialize_measurement()
        post_dict = {1:"foo", 2:"bar"}
        pre_dict = {3:"hi"}
        my_measurement.runs_dict = post_dict
        my_measurement.pre_combine_runs_dict = pre_dict
        my_measurement.undo_combine_runs()
        assert my_measurement.pre_combine_runs_dict is None
        assert my_measurement.runs_dict == pre_dict        


    @staticmethod
    def test_update_runs_dict():
        my_measurement = TestMeasurement.initialize_measurement()
        my_measurement.runs_dict = {3:"hi"} 
        my_measurement._update_runs_dict()
        assert len(my_measurement.runs_dict) == 1 
        assert TEST_IMAGE_RUN_ID in my_measurement.runs_dict

    @staticmethod 
    def test_measurement_dump_and_load():
        RUN_ANALYSIS_CHECKSUM = None
        spoofed_measurement = TestMeasurement._dump_and_load_measurement_spoofer()
        try:
            TEST_DUMP_FOLDERNAME = "Temp"
            #Modify the dumped measurement via an override of an experiment parameter
            spoofed_measurement.override_experiment_parameter("swallow_species", "coconut")
            spoofed_measurement.dump_measurement(dump_foldername = TEST_DUMP_FOLDERNAME)
            #Twiddle the spoofed measurement to be different
            spoofed_measurement.analyze_runs(lambda my_measurement, my_run: 0.0, "bar", overwrite_existing = True)
            spoofed_measurement.measurement_analysis_results["horse"] = "horse of course"
            spoofed_measurement.experiment_parameters["swallow_species"] = "European"
            spoofed_measurement.measurement_parameters["has_soul"] = True
            for run_id in spoofed_measurement.runs_dict:
                current_run = spoofed_measurement.runs_dict[run_id] 
                current_run.parameters["count_von_count"] = "ah ah ah"
            #Test equality
            spoofed_measurement.load_measurement(dump_foldername = TEST_DUMP_FOLDERNAME)
            TestMeasurement._dump_and_load_measurement_validator(spoofed_measurement)
            disconnected_measurement = TestMeasurement.initialize_measurement(connected_mode = False)
            disconnected_measurement.load_measurement(dump_foldername = TEST_DUMP_FOLDERNAME)
            TestMeasurement._dump_and_load_measurement_validator(disconnected_measurement)
        finally:
            shutil.rmtree(TEST_DUMP_FOLDERNAME)


    @staticmethod
    def _dump_and_load_measurement_spoofer():
        my_measurement = TestMeasurement.initialize_measurement()
        def analysis_function_scalar_zero(my_measurement, my_run):
            return 0.0
        def analysis_function_scalar_one(my_measurement, my_run):
            return 1.0
        def analysis_function_array_zero(my_measurement, my_run):
            return np.array([0.0])
        my_measurement.analyze_runs(analysis_function_scalar_zero, "bar")
        my_measurement.analyze_runs(analysis_function_array_zero, "baz")
        my_measurement.measurement_analysis_results["horse"] = "noble animal"
        my_measurement.experiment_parameters["swallow_species"] = "African"
        my_measurement.measurement_parameters["has_soul"] = False
        for i, run_id in enumerate(my_measurement.runs_dict):
            current_run = my_measurement.runs_dict[run_id]
            current_run.parameters["count_von_count"] = i
        return my_measurement

    @staticmethod 
    def _dump_and_load_measurement_validator(my_measurement):
        assert np.array_equal(my_measurement.get_analysis_value_from_runs("bar"), np.array([0.0])) 
        assert np.array_equal(my_measurement.get_analysis_value_from_runs("baz"), np.array([np.array([0.0])]))
        assert my_measurement.measurement_analysis_results["horse"] == "noble animal"
        assert my_measurement.experiment_parameters["swallow_species"] == "African" 
        assert my_measurement.measurement_parameters["has_soul"] == False
        for i, run_id in enumerate(my_measurement.runs_dict):
            current_run = my_measurement.runs_dict[run_id] 
            assert current_run.parameters["count_von_count"] == i

    @staticmethod
    def test_get_parameter_value_from_runs():
        VALUE_NAME_TO_CHECK = "id"
        OTHER_VALUE_NAME = "runtime"
        EXPECTED_VALUES = np.array([TEST_IMAGE_RUN_ID])
        EXPECTED_OTHER_VALUES = np.array(['2022-04-06T08:49:10Z']) 
        my_measurement = TestMeasurement.initialize_measurement() 
        values = my_measurement.get_parameter_value_from_runs("id")
        assert np.array_equal(values, EXPECTED_VALUES)
        #Test run filtering
        filtered_values_false = my_measurement.get_parameter_value_from_runs("id", run_filter = lambda my_measurement, my_run: False)
        assert np.array_equal(filtered_values_false, np.array([]))
        filtered_values_true = my_measurement.get_parameter_value_from_runs("id", run_filter = lambda my_measurement, my_run: True)
        assert np.array_equal(filtered_values_true, EXPECTED_VALUES)
        #Test non-numpy output
        filtered_values_true_list = my_measurement.get_parameter_value_from_runs("id", run_filter = lambda my_measurement, my_run: True, 
                                                            numpyfy = False)
        assert isinstance(filtered_values_true_list, list) 
        assert filtered_values_true_list == [TEST_IMAGE_RUN_ID]
        #Test retrieving two vals at once
        id_vals, runtime_vals = my_measurement.get_parameter_value_from_runs(("id", "runtime"))
        assert np.array_equal(id_vals, EXPECTED_VALUES) 
        assert np.array_equal(runtime_vals, EXPECTED_OTHER_VALUES)



    @staticmethod 
    def test_get_analysis_value_from_runs():
        VALUE_NAME_TO_CHECK = "foo"
        OTHER_VALUE_NAME = "bar"
        VALUE = 3 
        VALUE_AS_ARRAY = np.array([3])
        OTHER_VALUE = 2
        OTHER_VALUE_AS_ARRAY = np.array([2])
        my_measurement = TestMeasurement.initialize_measurement() 
        for run_id in my_measurement.runs_dict:
            current_run = my_measurement.runs_dict[run_id] 
            current_run.analysis_results[VALUE_NAME_TO_CHECK] = VALUE 
            current_run.analysis_results[OTHER_VALUE_NAME] = OTHER_VALUE
        assert np.array_equal(my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK), VALUE_AS_ARRAY)
        #Test run filtering
        filtered_values_false = my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK, run_filter = lambda my_measurement, my_run: False)
        assert np.array_equal(filtered_values_false, np.array([]))
        filtered_values_true = my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK, run_filter = lambda my_measurement, my_run: True)
        assert np.array_equal(filtered_values_true, VALUE_AS_ARRAY)
        def combined_filter_part_1(my_measurement, my_run):
            return True 
        def combined_filter_part_2(my_measurement, my_run):
            return False
        filtered_values_combined = my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK, run_filter = (combined_filter_part_1, 
                                                                                                            combined_filter_part_2))
        assert np.array_equal(filtered_values_combined, np.array([]))
        #Test global filtering 
        my_measurement.set_global_run_filter(lambda my_measurement, my_run: False)
        filtered_values_global = my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK)
        assert np.array_equal(filtered_values_global, np.array([]))
        my_measurement.set_global_run_filter(None)
        filtered_values_global = my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK)
        assert np.array_equal(filtered_values_global, VALUE_AS_ARRAY)
        #Test non-numpy output 
        filtered_values_true_list = my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK, run_filter = lambda my_measurement, my_run: True, 
                                                                                numpyfy = False)
        assert isinstance(filtered_values_true_list, list)
        assert filtered_values_true_list == [VALUE]
        #Test retrieving two values at once
        foo_vals, bar_vals = my_measurement.get_analysis_value_from_runs((VALUE_NAME_TO_CHECK, OTHER_VALUE_NAME))
        assert np.array_equal(VALUE_AS_ARRAY, foo_vals)
        assert np.array_equal(OTHER_VALUE_AS_ARRAY, bar_vals)



    @staticmethod 
    def test_get_parameter_analysis_value_pair_from_runs():
        VALUE_NAME_TO_CHECK = "foo"
        VALUE = 3
        OTHER_VALUE_NAME = "bar"
        OTHER_VALUE = 2
        OTHER_VALUE_AS_ARRAY = np.array([OTHER_VALUE])
        VALUE_AS_ARRAY = np.array([VALUE])
        EXPECTED_PARAMS = np.array([TEST_IMAGE_RUN_ID])
        EXPECTED_RUNTIME_ARRAY = np.array(['2022-04-06T08:49:10Z']) 

        my_measurement = TestMeasurement.initialize_measurement() 
        for run_id in my_measurement.runs_dict:
            current_run = my_measurement.runs_dict[run_id] 
            current_run.analysis_results[VALUE_NAME_TO_CHECK] = VALUE 
            current_run.analysis_results[OTHER_VALUE_NAME] = OTHER_VALUE
        param_array, analysis_array = my_measurement.get_parameter_analysis_value_pair_from_runs("id", "foo")
        assert np.array_equal(param_array, EXPECTED_PARAMS)
        assert np.array_equal(analysis_array, VALUE_AS_ARRAY)
        #Test multiple return values
        id_array, runtime_array, foo_array, bar_array = my_measurement.get_parameter_analysis_value_pair_from_runs(("id", "runtime"), ("foo", "bar")) 
        assert np.array_equal(id_array, EXPECTED_PARAMS)
        assert np.array_equal(foo_array, VALUE_AS_ARRAY)
        assert np.array_equal(bar_array, OTHER_VALUE_AS_ARRAY)
        assert np.array_equal(runtime_array, EXPECTED_RUNTIME_ARRAY)
        #Test error filtering
        for run_id in my_measurement.runs_dict:
            current_run = my_measurement.runs_dict[run_id] 
            current_run.analysis_results[VALUE_NAME_TO_CHECK] = Measurement.ANALYSIS_ERROR_INDICATOR_STRING
        param_array, analysis_array = my_measurement.get_parameter_analysis_value_pair_from_runs("id", "foo", ignore_errors = True)
        assert np.array_equal(param_array, np.array([]))
        assert np.array_equal(analysis_array, np.array([]))
        param_array, analysis_array = my_measurement.get_parameter_analysis_value_pair_from_runs("id", "foo", ignore_errors = False)
        assert np.array_equal(param_array, EXPECTED_PARAMS)
        assert np.array_equal(analysis_array, np.array([Measurement.ANALYSIS_ERROR_INDICATOR_STRING]))
        #Test run filtering
        param_array_false, analysis_array_false = my_measurement.get_parameter_analysis_value_pair_from_runs("id", "foo", 
                                                                        run_filter = lambda my_measurement, my_run: False)
        assert np.array_equal(param_array_false, np.array([]))
        assert np.array_equal(analysis_array_false, np.array([]))
        param_array_true, analysis_array_true = my_measurement.get_parameter_analysis_value_pair_from_runs("id", "foo", 
                                                                        run_filter = lambda my_measurement, my_run: True, ignore_errors = False)
        assert np.array_equal(param_array_true, EXPECTED_PARAMS)
        assert np.array_equal(analysis_array_true, np.array([Measurement.ANALYSIS_ERROR_INDICATOR_STRING]))
        #Test non-numpy returns 
        param_list_true, analysis_list_true = my_measurement.get_parameter_analysis_value_pair_from_runs("id", "foo", 
                                                                        run_filter = lambda my_measurement, my_run: True, ignore_errors = False, 
                                                                        numpyfy = False)
        assert isinstance(param_list_true, list) 
        assert param_list_true == [TEST_IMAGE_RUN_ID] 
        assert isinstance(analysis_list_true, list) 
        assert analysis_list_true == [Measurement.ANALYSIS_ERROR_INDICATOR_STRING]


    @staticmethod
    def test_get_constant_parameter_value_from_runs():
        my_measurement = TestMeasurement.initialize_measurement()
        #Convenient hack to get another, distinct run
        my_other_measurement = TestMeasurement.initialize_measurement()
        for run_id in my_measurement.runs_dict:
            my_run_id = run_id
        my_measurement.runs_dict[my_run_id + 1] = my_other_measurement.runs_dict[my_run_id]
        #Check correct behavior in case of identical values
        for run_id in my_measurement.runs_dict:
            my_run = my_measurement.runs_dict[run_id] 
            my_run.parameters["foo"] = 1 
        assert my_measurement.get_constant_parameter_value_from_runs("foo") == 1
        #Check for error with distinct values
        for i, run_id in enumerate(my_measurement.runs_dict):
            my_run = my_measurement.runs_dict[run_id] 
            my_run.parameters["foo"] = i 
        try:
            my_measurement.get_constant_parameter_value_from_runs("foo")
        except RuntimeError:
            assert True
        else:
            raise RuntimeError("Failed to throw expected error.")

    @staticmethod 
    def test_analyze_runs():
        VALUE_NAME_TO_CHECK = "foo"
        VALUE_1 = 1
        VALUE_2 = 2
        VALUE_1_AS_LIST = [1] 
        VALUE_2_AS_LIST = [2]
        my_measurement = TestMeasurement.initialize_measurement() 
        def analysis_func_1(my_measurement, my_run):
            return (VALUE_1,)
        def analysis_func_2(my_measurement, my_run):
            return (VALUE_2,)
        def analysis_func_1_scalar(my_measurement, my_run):
            return VALUE_1
        def analysis_func_error(my_measurement, my_run):
            raise RuntimeError("Intended error for measurement analysis testing.")
        def analysis_func_kwargs(my_measurement, my_run, input = ""):
            return input
        #Test analyze_runs general functionality
        my_measurement.analyze_runs(analysis_func_1, (VALUE_NAME_TO_CHECK,))
        assert my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK) == VALUE_1_AS_LIST
        my_measurement.analyze_runs(analysis_func_2, (VALUE_NAME_TO_CHECK,), overwrite_existing = False)
        assert my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK) == VALUE_1_AS_LIST
        my_measurement.analyze_runs(analysis_func_2, (VALUE_NAME_TO_CHECK,), overwrite_existing = True)
        assert my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK) == VALUE_2_AS_LIST
        my_measurement.analyze_runs(analysis_func_1_scalar, VALUE_NAME_TO_CHECK, overwrite_existing = True)
        assert my_measurement.get_analysis_value_from_runs(VALUE_NAME_TO_CHECK) == VALUE_1_AS_LIST
        #Test error handling
        ERR_NAME_TO_CHECK = "bar"
        try:
            my_measurement.analyze_runs(analysis_func_error, ERR_NAME_TO_CHECK, catch_errors = False)
        except Exception as e:
            pass
        else:
            raise ValueError("There was supposed to be an error here.")
        my_measurement.analyze_runs(analysis_func_error, ERR_NAME_TO_CHECK, catch_errors = True)
        assert my_measurement.get_analysis_value_from_runs(ERR_NAME_TO_CHECK, ignore_errors = False) == [Measurement.ANALYSIS_ERROR_INDICATOR_STRING]
        #Test functions with kwargs
        my_measurement.analyze_runs(analysis_func_kwargs, "baz", fun_kwargs = {'input':'hi'})
        assert my_measurement.get_analysis_value_from_runs("baz") == ['hi']
        #Test filtering
        my_measurement.analyze_runs(analysis_func_1_scalar, "oof", run_filter = lambda my_measurement, my_run: False)
        for run_id in my_measurement.runs_dict:
            current_run = my_measurement.runs_dict[run_id] 
            assert not "oof" in current_run.analysis_results
        my_measurement.analyze_runs(analysis_func_1_scalar, "oof", run_filter = lambda my_measurement, my_run: True)
        assert my_measurement.get_analysis_value_from_runs("oof") == [1]
        #Test filtering with global 

    @staticmethod
    def test_label_badshots_custom():
        my_measurement = TestMeasurement.initialize_measurement() 
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



    @staticmethod
    def test_update():
        my_measurement = TestMeasurement.initialize_measurement() 
        my_measurement.runs_dict = {}
        def badshot_function_false(my_measurement, my_run):
            return False 
        def my_analysis_function_zero(my_measurement, my_run):
            return 0.0
        my_measurement.set_badshot_function(badshot_function_false)
        my_measurement.add_to_live_analyses(my_analysis_function_zero, 'foo')
        my_measurement.update()
        assert len(my_measurement.runs_dict) == 1
        assert np.array_equal(my_measurement.get_analysis_value_from_runs('foo'), np.array([0.0]))
        def badshot_function_true(my_measurement, my_run):
            return True 
        my_measurement.set_badshot_function(badshot_function_true)
        my_measurement.update(override_existing_badshots = True) 
        assert np.array_equal(my_measurement.get_analysis_value_from_runs('foo'), [] )


    @staticmethod 
    def test_get_outlier_filter():
        my_measurement = TestMeasurement.initialize_measurement()
        randoms = np.load(os.path.join("./resources", "Sample_Normal_Randoms.npy"))
        my_measurement.runs_dict = {}
        OUTLIER_INDEX_FOO = 46
        OUTLIER_VALUE_FOO = 5
        OUTLIER_INDEX_BAR = 37
        OUTLIER_VALUE_BAR = 6
        for i in range(len(randoms)):
            current_run = Run(i, TEST_IMAGE_PATHNAME_DICT, {"id":i})
            if not i == OUTLIER_INDEX_FOO:
                current_run.analysis_results["foo"] = randoms[i] 
            else:
                current_run.analysis_results["foo"] = OUTLIER_VALUE_FOO
            if not i == OUTLIER_INDEX_BAR:
                current_run.analysis_results["bar"] = randoms[i] 
            else:
                current_run.analysis_results["bar"] = OUTLIER_VALUE_BAR
            my_measurement.runs_dict[i] = current_run
        outlier_filter = my_measurement.get_outlier_filter("foo", confidence_interval = 0.9999)
        outlier_filtered_ids = my_measurement.get_parameter_value_from_runs("id", run_filter = outlier_filter)
        assert len(outlier_filtered_ids) == len(randoms) - 1
        assert not OUTLIER_INDEX_FOO in outlier_filtered_ids
        #Test extra filtering
        def even_id_filter(my_measurement, my_run):
            return my_run.parameters["id"] %2 == 0
        outlier_filter_even_id = my_measurement.get_outlier_filter("foo", confidence_interval = 0.9999, run_filter = even_id_filter)
        outlier_filtered_even_ids = my_measurement.get_parameter_value_from_runs("id", run_filter = outlier_filter_even_id)
        assert len(outlier_filtered_even_ids) == len(randoms) // 2 - 1
        assert not OUTLIER_INDEX_FOO in outlier_filtered_even_ids
        #Test minimum and maximum returning 
        outlier_filter, interval = my_measurement.get_outlier_filter("foo", confidence_interval = 0.9999, return_interval = True)
        interval_lower, interval_upper = interval 
        filtered_values = my_measurement.get_analysis_value_from_runs("foo", run_filter = outlier_filter)
        assert np.min(filtered_values) == interval_lower 
        assert np.max(filtered_values) == interval_upper
        #Test filtering on multiple values simultaneously
        outlier_filter_multiple_values, interval_foo, interval_bar = my_measurement.get_outlier_filter(("foo", "bar"), confidence_interval = 0.9999, 
                                                                                                        return_interval = True)
        outlier_filtered_multiple_value_ids = my_measurement.get_parameter_value_from_runs("id", run_filter = outlier_filter_multiple_values)
        assert len(outlier_filtered_multiple_value_ids) == len(randoms) - 2
        assert not OUTLIER_INDEX_FOO in outlier_filtered_multiple_value_ids
        assert not OUTLIER_INDEX_BAR in outlier_filtered_multiple_value_ids
        outlier_filtered_multiple_values_foo = my_measurement.get_analysis_value_from_runs("foo", run_filter = outlier_filter_multiple_values)
        min_foo, max_foo = interval_foo 
        assert min(outlier_filtered_multiple_values_foo) == min_foo 
        assert max(outlier_filtered_multiple_values_foo) == max_foo 
        outlier_filtered_multiple_values_bar = my_measurement.get_analysis_value_from_runs("bar", run_filter = outlier_filter_multiple_values) 
        min_bar, max_bar = interval_bar 
        assert min(outlier_filtered_multiple_values_bar) == min_bar 
        assert max(outlier_filtered_multiple_values_bar) == max_bar




    #Does not test the interactive box setting.
    @staticmethod
    def test_set_box():
        my_measurement = TestMeasurement.initialize_measurement() 
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
        my_image = TestRun.my_run_without_memory._load_image(TEST_IMAGE_FILE_PATH)
        assert check_sha_hash(my_image.data.tobytes(), TEST_IMAGE_ARRAY_SHA_256_HEX_STRING)

    @staticmethod
    def test_get_image():
        my_image_with_memory = TestRun.my_run_with_memory.get_image('side_image') 
        my_image_without_memory = TestRun.my_run_without_memory.get_image('side_image')
        assert check_sha_hash(my_image_with_memory.data.tobytes(), TEST_IMAGE_ARRAY_SHA_256_HEX_STRING)
        assert check_sha_hash(my_image_without_memory.data.tobytes(), TEST_IMAGE_ARRAY_SHA_256_HEX_STRING)
