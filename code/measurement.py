import datetime
import json
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import numpy as np
from astropy.io import fits

from .image_processing_functions import get_absorption_image
from . import loading_functions, statistics_functions

IMAGE_FORMATS_LIST = ['.fits']
IMAGING_TYPES_LIST = ['top_double', 'side_low_mag', 'side_high_mag']
MEASUREMENT_IMAGE_NAME_DICT = {'top_double': ['TopA', 'TopB'],
                                'side_low_mag':['Side'], 'side_high_mag':['Side']}
FILENAME_DATETIME_FORMAT_STRING = "%Y-%m-%d--%H-%M-%S"
PARAMETERS_DATETIME_FORMAT_STRING = "%Y-%m-%dT%H:%M:%SZ"
PARAMETERS_RUN_TIME_NAME = "runtime"
FILENAME_DELIMITER_CHAR = "_"
TEMP_FILE_MARKER = "TEMP"

RUN_MISMATCH_PATIENCE_TIME_SECS = 30

class Measurement():

    ANALYSIS_DUMPFILE_NAME = "analysis_dump.json"
    PARAMETERS_DUMPFILE_NAME = "parameters_dump.json"

    """Initialization method.
    
    Parameters:
    
    measurement_directory_path: str, The path to a directory containing the labeled images to process.

    imaging_type: str, The type of imaging, e.g. top or side, low mag or high mag.

    experiment_parameters: dict {parname:value} of experiment-level parameters not saved within the run parameters, e.g. trapping frequencies

    image_format: str, the filetype of the images being processed
    hold_images_in_memory: bool, Whether images are kept loaded in memory, or loaded on an as-needed basis and then released.

    measurement_parameters: dict {parname:value} of measurement-level params, e.g. a list of run ids which are flagged as bad shots or 
    the coordinates of a background box.

    run_parameters_verbose: Whether the runs store the entire set of experiment parameters, or just those being scanned.

    badshot_function: A badshot function, to be used in badshot labeling.

    disconnected_mode: A boolean indicating whether the measurement is initialized in "connected mode", i.e. with access to the underlying 
    measurement directory. If not in this mode, run images cannot be accessed, and various data are initialized by loading from a pre-existing dump. 
    """
    def __init__(self, measurement_directory_path = None, imaging_type = 'top_double', experiment_parameters = None, image_format = ".fits", 
                    hold_images_in_memory = False, measurement_parameters = None, run_parameters_verbose = False,
                    badshot_function = None, analyses_list = None, 
                    global_run_filter = None, is_live = False, connected_mode = True):
        if(measurement_parameters):   
            self.measurement_parameters = measurement_parameters
        else:
            self.measurement_parameters = {}
        self.badshot_function = badshot_function
        self.badshot_checked_list = []
        if analyses_list is None:
            self.analyses_list = []
        self.measurement_analysis_results = {}
        self.global_run_filter = global_run_filter
        self.runs_dict = {}
        self.connected_mode = connected_mode
        #Specific to connected mode
        if connected_mode:
            if(not measurement_directory_path):
                measurement_directory_path = os.getcwd()
            self.measurement_directory_path = measurement_directory_path 
            self.imaging_type = imaging_type
            self.image_format = image_format
            self.hold_images_in_memory = hold_images_in_memory
            if(not experiment_parameters):
                experiment_parameters_path = os.path.join(measurement_directory_path, "experiment_parameters.json")
                with open(experiment_parameters_path, 'r') as json_file:
                    experiment_parameters_dict = json.load(json_file)
                    self.experiment_parameters = experiment_parameters_dict["Values"]
            else:
                self.experiment_parameters = experiment_parameters
            self.run_parameters_verbose = run_parameters_verbose
            self.is_live = is_live
            self._update_runs_dict()

    def set_badshot_function(self, badshot_function):
        self.badshot_function = badshot_function

    def set_global_run_filter(self, run_filter):
        self.global_run_filter = run_filter


    def _get_run_ids_and_parameters_from_measurement_folder(self):
        DATA_DUMP_PARAMS_FILENAME = "run_params_dump.json" 
        path_to_dump_file_in_measurement_folder = os.path.join(self.measurement_directory_path, DATA_DUMP_PARAMS_FILENAME)
        run_parameters_list = loading_functions.load_run_parameters_from_json(path_to_dump_file_in_measurement_folder, 
                                                                    make_raw_parameters_terse = (not self.run_parameters_verbose), 
                                                                    have_patience = self.is_live)
        unique_run_ids_list = self._get_unique_run_ids_from_folder()
        sorted_run_ids_list = sorted(unique_run_ids_list)
        matched_run_ids_and_parameters_list = []
        #O(n^2) naive search, but it's fine...
        for run_id in sorted_run_ids_list: 
            for parameters in run_parameters_list:
                if parameters['id'] == run_id:
                    matched_run_ids_and_parameters_list.append((run_id, parameters))
                    break
            else:
                warnings.warn("Unable to find data for run id: " + str(run_id))
        return matched_run_ids_and_parameters_list


    def _get_unique_run_ids_from_folder(self):
        image_filenames_list = [f for f in os.listdir(self.measurement_directory_path) if self.image_format in f]
        unique_run_ids_list = []
        for image_filename in image_filenames_list:
            run_id = Measurement._parse_run_id_from_filename(image_filename)
            if not run_id in unique_run_ids_list:
                for other_filename in image_filenames_list:
                    if str(run_id) in other_filename and TEMP_FILE_MARKER in other_filename:
                        break
                else:
                    unique_run_ids_list.append(run_id)
        return unique_run_ids_list
            



    def _add_run(self, run_id_and_parameters):
        run_id, run_parameters = run_id_and_parameters
        run_image_pathname_dict = {}
        run_id_image_filenames = [f for f in os.listdir(self.measurement_directory_path) if (self.image_format in f and 
                                                                                run_id == Measurement._parse_run_id_from_filename(f))]
        for run_id_image_filename in run_id_image_filenames:
            for image_name in MEASUREMENT_IMAGE_NAME_DICT[self.imaging_type]:
                if image_name in run_id_image_filename:
                    run_id_image_pathname = os.path.join(self.measurement_directory_path, run_id_image_filename)
                    run_image_pathname_dict[image_name] = run_id_image_pathname 
                    break
            else:
                raise RuntimeError("Run image does not match specification. Is the imaging type correct?")
        if not len(run_image_pathname_dict) == len(MEASUREMENT_IMAGE_NAME_DICT[self.imaging_type]):
            raise RuntimeError("A run image appears to be missing...")
        generated_run = Run(run_id, run_image_pathname_dict, hold_images_in_memory= self.hold_images_in_memory, 
                            parameters = run_parameters, image_format = self.image_format)
        self.runs_dict[run_id] = generated_run


    """
    Scans the measurement folder for new runs to add to self.runs_dict; 
    ignores errors from runs more recent than some given time, to avoid issues with live analysis."""
    def _update_runs_dict(self):
        matched_run_ids_and_parameters_list = self._get_run_ids_and_parameters_from_measurement_folder()
        for run_id_and_parameters in matched_run_ids_and_parameters_list:
            run_id, parameters = run_id_and_parameters
            if not run_id in self.runs_dict:
                try:
                    self._add_run(run_id_and_parameters)
                except RuntimeError as e:
                    current_time = datetime.datetime.now()
                    run_time = datetime.datetime.strptime(parameters[PARAMETERS_RUN_TIME_NAME], PARAMETERS_DATETIME_FORMAT_STRING)
                    if not np.abs((current_time - run_time).total_seconds()) < RUN_MISMATCH_PATIENCE_TIME_SECS:
                        raise e
        current_run_ids_list = [f[0] for f in matched_run_ids_and_parameters_list]
        saved_run_ids_list = [f for f in self.runs_dict]
        for saved_run_id in saved_run_ids_list:
            if not saved_run_id in current_run_ids_list:
                self.runs_dict.pop(saved_run_id)


    """
    JSON serialize the data in a measurement. 
    
    When called, serialize most of the data from a measurement object, including each run's analysis_results and parameters dicts, as well as the 
    measurement-wide measurement_analysis_results, measurement_parameters, and experiment_parameters dicts. The resulting folder contains an independent 
    store of information capable of "reconstituting" all of the data in the measurement object.
    
    NOTE: This method relies on the objects in the respective dictionaries being JSON serializable, with a _single_ exception: because of 
    their ubiquity in the analysis_results dictionaries, numpy arrays have been manually accommodated as well. 
    To do this, a separate directory is created wherein the numpy arrays are saved as .npy files.
    
    NOTE: This method only stores JSON serializable data, and e.g. does not restore saved live analysis functions, global run filters, or badshot functions.
    These must be manually re-initialized if needed."""

    def dump_measurement(self, dump_foldername = "dump_folder"):
        if not os.path.exists(dump_foldername):
            os.mkdir(dump_foldername)
        analysis_dump_pathname = os.path.join(dump_foldername, Measurement.ANALYSIS_DUMPFILE_NAME)
        with open(analysis_dump_pathname, 'w') as analysis_dump_file:
            analysis_dump_dict = {} 
            analysis_dump_dict["Measurement"] = Measurement._digest_analysis_results("Measurement", analysis_dump_pathname, self.measurement_analysis_results)
            for run_id in self.runs_dict:
                current_run = self.runs_dict[run_id] 
                analysis_dump_dict[str(run_id)] = Measurement._digest_analysis_results(str(run_id), analysis_dump_pathname, current_run.analysis_results)
            json.dump(analysis_dump_dict, analysis_dump_file)
        #Default assumption is that parameters values will be vanilla JSON serializable.
        parameters_dump_pathname = os.path.join(dump_foldername, Measurement.PARAMETERS_DUMPFILE_NAME)
        with open(parameters_dump_pathname, 'w') as parameters_dump_file:
            parameters_dump_dict = {} 
            parameters_dump_dict["Measurement"] = self.measurement_parameters 
            parameters_dump_dict["Experiment"] = self.experiment_parameters 
            for run_id in self.runs_dict:
                current_run = self.runs_dict[run_id] 
                parameters_dump_dict[str(run_id)] = current_run.parameters 
            json.dump(parameters_dump_dict, parameters_dump_file)


    @staticmethod
    def _digest_analysis_results(analysis_label, dump_pathname, analysis_results):
        dump_foldername = os.path.dirname(dump_pathname)
        dump_filename_sans_extension = os.path.basename(dump_pathname).split('.')[0]
        numpy_storage_folder_name = dump_filename_sans_extension + "_auxiliary_numpy_storage"
        numpy_storage_folder_path = os.path.join(dump_foldername, numpy_storage_folder_name)
        return_dict = {}
        for key in analysis_results:
            analysis_object = analysis_results[key] 
            if not isinstance(analysis_object, np.ndarray):
                return_dict[key] = analysis_object 
            else:
                if not os.path.isdir(numpy_storage_folder_path):
                    os.mkdir(numpy_storage_folder_path)
                numpy_filename = "{0}_{1}.npy".format(analysis_label, key)
                numpy_pathname = os.path.join(numpy_storage_folder_path, numpy_filename)
                np.save(numpy_pathname, analysis_object)
                return_dict_numpy_key = "{0}_Numpy_Pathname".format(key)
                return_dict[return_dict_numpy_key] = numpy_pathname
        return return_dict




    """Load a measurement from the dump folder created by dump_measurement. Should be called after measurement is initialized, 
    but before any other manipulations."""

    def load_measurement(self, dump_foldername = "dump_folder"):
        parameters_dump_pathname = os.path.join(dump_foldername, Measurement.PARAMETERS_DUMPFILE_NAME)
        with open(parameters_dump_pathname, 'r') as parameters_dump_file:
            parameters_dump_dict = json.load(parameters_dump_file) 
            for key in parameters_dump_dict:
                current_parameters = parameters_dump_dict[key]
                if key == "Measurement":
                    self.measurement_parameters = current_parameters 
                elif key == "Experiment":
                    print("Loading experiment params")
                    self.experiment_parameters = current_parameters
                    print(current_parameters)
                else:
                    run_id_as_integer = int(key) 
                    if self.connected_mode:
                        if run_id_as_integer in self.runs_dict:
                            current_run = self.runs_dict[run_id_as_integer]
                            current_run.parameters = current_parameters 
                    else:
                        current_run = Run(run_id_as_integer, None, parameters = current_parameters, connected_mode = False)
                        self.runs_dict[run_id_as_integer] = current_run
        analysis_dump_pathname = os.path.join(dump_foldername, Measurement.ANALYSIS_DUMPFILE_NAME)
        with open(analysis_dump_pathname, 'r') as analysis_dump_file:
            analysis_dump_dict = json.load(analysis_dump_file)
            for key in analysis_dump_dict:
                current_analysis_results = Measurement._undigest_analysis_results(analysis_dump_dict[key])
                if key == "Measurement":
                    self.measurement_analysis_results = current_analysis_results
                else:
                    run_id_as_integer = int(key)
                    if run_id_as_integer in self.runs_dict:
                        current_run = self.runs_dict[run_id_as_integer] 
                        current_run.analysis_results = current_analysis_results


    @staticmethod 
    def _undigest_analysis_results(loaded_json_analysis_results):
        return_dict = {}
        for key in loaded_json_analysis_results:
            if not "_Numpy_Pathname" in key:
                return_dict[key] = loaded_json_analysis_results[key] 
            else:
                original_analysis_results_key = key.split("_Numpy_Pathname")[0] 
                numpy_pathname = loaded_json_analysis_results[key] 
                numpy_array = np.load(numpy_pathname) 
                return_dict[original_analysis_results_key] = numpy_array 
        return return_dict



    def initialize_workfolder(self, descriptor = "Default", workfolder_pathname = None):
        #If workfolder pathname not specified, measurement will create a workfolder inside a directory labeled 
        #"Private_BEC1_Analysis" within the same directory that contains the directory for "BEC1_Analysis" 
        if workfolder_pathname is None:
            PRIVATE_DIRECTORY_REPO_NAME = "Private_BEC1_Analysis"
            path_to_file = os.path.dirname(os.path.abspath(__file__))
            path_to_repo_folder = os.path.abspath(path_to_file + "/../../")
            path_to_private_directory_repo = os.path.join(path_to_repo_folder, PRIVATE_DIRECTORY_REPO_NAME)
            current_datetime = datetime.datetime.now()
            current_year = current_datetime.strftime("%Y")
            current_year_month = current_datetime.strftime("%Y-%m")
            current_year_month_day = current_datetime.strftime("%Y-%m-%d")
            measurement_directory_folder_name = os.path.basename(os.path.normpath(self.measurement_directory_path))
            workfolder_label = "_{0}_{1}".format(descriptor, measurement_directory_folder_name)
            workfolder_pathname = os.path.join(path_to_private_directory_repo, current_year, current_year_month, current_year_month_day + workfolder_label)
        if(not os.path.isdir(workfolder_pathname)):
            os.makedirs(workfolder_pathname)
        with open(os.path.join(workfolder_pathname, "Source.txt"), 'w') as f:
            f.write("Data source: " + self.measurement_directory_path)
        return workfolder_pathname

    """
    Set a rectangular box with user input.
    
    run_to_use: The run to use for setting the box position. If not passed, the interactive prompt cycles through runs 
    until a box is selected on one.
    
    image_to_use: The name of the image to use for setting the box position. If not specified, the default image for the run is used.

    overwrite_existing: If False, the box will only be set if the key label does not exist in self.measurement_parameters; otherwise, 
    nothing is done.
    """
    def set_box(self, label, run_to_use = None, image_to_use = None, box_coordinates = None, overwrite_existing = True):
        if not overwrite_existing and label in self.measurement_parameters:
            return None
        if(not box_coordinates):
            if run_to_use is None:
                id_try_list = list(self.runs_dict) 
            else:
                id_try_list = [run_to_use]
            for id_to_try in id_try_list:
                my_run = self.runs_dict[id_to_try]
                if(image_to_use):
                    my_image_array = my_run.get_image(image_to_use)
                else:
                    my_image_array = my_run.get_default_image()
                my_with_atoms_image = get_absorption_image(my_image_array)
                title = str(id_to_try)
                try:
                    x_1, x_2, y_1, y_2 = Measurement._draw_box(my_with_atoms_image, label, title)
                    assert not any([x_1 is None, x_2 is None, y_1 is None, y_2 is None])
                except AssertionError:
                    pass
                else:  
                    x_min = int(min(x_1, x_2))
                    y_min = int(min(y_1, y_2))
                    x_max = int(max(x_1, x_2))
                    y_max = int(max(y_1, y_2))
                    break
            self.measurement_parameters[label] = [x_min, y_min, x_max, y_max]
        else:
            self.measurement_parameters[label] = box_coordinates


    """
    Alias to set_box('norm_box'), for convenience."""
    def set_norm_box(self, run_to_use=None, image_to_use = None, box_coordinates = None, overwrite_exisiting = True):
        self.set_box('norm_box', run_to_use = run_to_use, image_to_use = image_to_use, box_coordinates = box_coordinates, 
                        overwrite_existing = overwrite_exisiting)

    """
    Alias to set_box('ROI'), for convenience."""
    def set_ROI(self, run_to_use = None, image_to_use = None, box_coordinates = None, overwrite_existing = True):
        self.set_box('ROI', run_to_use = run_to_use, image_to_use = image_to_use, box_coordinates = box_coordinates, 
                        overwrite_existing = overwrite_existing)

    @staticmethod
    def _draw_box(my_image, label, title):
        ax = plt.gca()
        ax.imshow(my_image, cmap = 'gray')
        ax.set_title(title)
        x_1 = None
        y_1 = None
        x_2 = None
        y_2 = None
        def line_select_callback(eclick, erelease):
            nonlocal x_1
            nonlocal x_2
            nonlocal y_1
            nonlocal y_2
            x_1, y_1 = eclick.xdata, eclick.ydata
            x_2, y_2 = erelease.xdata, erelease.ydata 
        props = {'facecolor':'none', 'edgecolor':'red', 'linewidth':1}
        rect = RectangleSelector(ax, line_select_callback, props = props)
        plt.suptitle("Set box: " + label)
        plt.show()
        return((x_1, x_2, y_1, y_2))

    
    def check_box(self, label, run_to_use = 0):
        for i, key in enumerate(self.runs_dict):
            if(i == run_to_use):
                my_run = self.runs_dict[key] 
                break 
        my_image_array = my_run.get_default_image()
        my_with_atoms_image = my_image_array[0]
        box_coordinates = self.measurement_parameters[label]
        x_min, y_min, x_max, y_max = box_coordinates
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth = 1, edgecolor = 'r', facecolor = 'none') 
        ax = plt.gca()
        ax.imshow(my_with_atoms_image, cmap = 'gray')
        ax.add_patch(rect)
        plt.show()

    """
    Returns parameter values from each run as a list. 
    
    Given a value name value_name, returns a list with the value of that parameter for each run. 
    
    Params:

    Value name: A string or tuple of strings identifying the values to be returned. If a single string is passed, method returns 
    an array/list of values; if a tuple is passed, a tuple of arrays/lists is returned, one for each value.
    
    Ignore badshots: If True, does not return the parameter value for runs which are flagged as bad shots. 
    
    run_filter: As with analyze_runs, if passed, only returns parameter values for which the function run_filter returns True.
    
    Numpyfy: If True, returned object is a numpy array of the queried values. Otherwise, a list is returned."""
    def get_parameter_value_from_runs(self, value_name, ignore_badshots = True, run_filter = None, numpyfy = True):
        filtered_dict = self.filter_run_dict(ignore_badshots = ignore_badshots, run_filter = run_filter)
        value_name_is_tuple = isinstance(value_name, tuple)
        value_names_tuple = Measurement._condition_string_tuple(value_name)
        combined_values_list = []
        for value_name in value_names_tuple:
            values_list = []
            for run_id in filtered_dict:
                current_run = self.runs_dict[run_id] 
                values_list.append(current_run.parameters[value_name])
            if not numpyfy:
                combined_values_list.append(values_list)
            else:
                combined_values_list.append(np.array(values_list))
        if value_name_is_tuple:
            return tuple(combined_values_list) 
        else:
            return combined_values_list[0]
        


    def get_analysis_value_from_runs(self, value_name, ignore_badshots = True, ignore_errors = True, run_filter = None, numpyfy = True):
        filtered_dict = self.filter_run_dict(ignore_badshots = ignore_badshots, run_filter = run_filter, ignore_errors = ignore_errors, 
                                                analysis_value_err_check_name=value_name)
        value_name_is_tuple = isinstance(value_name, tuple)
        value_names_tuple = Measurement._condition_string_tuple(value_name)
        combined_values_list = []
        for value_name in value_names_tuple:
            values_list = []
            for run_id in filtered_dict:
                current_run = filtered_dict[run_id]
                values_list.append(current_run.analysis_results[value_name])
            if not numpyfy:
                combined_values_list.append(values_list)
            else:
                combined_values_list.append(np.array(values_list))
        if value_name_is_tuple:
            return tuple(combined_values_list) 
        else:
            return combined_values_list[0] 
    
    """Convenience function which returns a pair of parameter values and analysis results. Convenient for plotting, and also convenient where 
    some runs have errors in the analysis."""
    def get_parameter_analysis_value_pair_from_runs(self, parameter_name, analysis_value_name, ignore_badshots = True, ignore_errors = True, 
                                                    run_filter = None, numpyfy = True):
        filtered_dict = self.filter_run_dict(ignore_badshots = ignore_badshots, run_filter = run_filter, ignore_errors = ignore_errors, 
                                                analysis_value_err_check_name=analysis_value_name)
        parameter_value_names_tuple = Measurement._condition_string_tuple(parameter_name)
        analysis_value_names_tuple = Measurement._condition_string_tuple(analysis_value_name)
        combined_param_values_list = [[] for x in range(len(parameter_value_names_tuple))]
        combined_analysis_result_list = [[] for x in range(len(analysis_value_names_tuple))]
        for run_id in filtered_dict:
            current_run = filtered_dict[run_id] 
            for param_value_name, param_list in zip(
                                        parameter_value_names_tuple, 
                                        combined_param_values_list):
                param_list.append(current_run.parameters[param_value_name])
            for analysis_value_name, analysis_list in zip(
                                                        analysis_value_names_tuple, 
                                                        combined_analysis_result_list):
                analysis_list.append(current_run.analysis_results[analysis_value_name])
        if numpyfy:
            combined_param_values_list = [np.array(l) for l in combined_param_values_list]
            combined_analysis_result_list = [np.array(l) for l in combined_analysis_result_list]
        return (*combined_param_values_list, *combined_analysis_result_list)
        
    """
    Performs arbitrary user-specified analysis on the underlying runs dict. 
    
    Given a function analysis_function which takes in the measurement and runs and returns 
    some output, apply the function to all runs and store the results in the analysis_dicts of each run 
    under the names specified in result_varnames.
    
    Parameters:
    
    analysis_function: A function with signature fun(measurement, run) that, given the overall measurement and 
    the specific run of interest as input, outputs some arbitrary results either singly as result or in tuple form as (result_1, result_2, ...)
    
    result_varnames: A string varname or a tuple (varname_1, varname_2, ...) of names under which the output of analysis_function should be 
    stored. 
    
    ignore_badshots: If True, analysis is not applied to runs which have been labeled as bad shots.
    
    overwrite_existing: If False, values already present in the analysis_dict of a run will not be changed; if there is any novel 
    value name in result_varnames that is not a key in analysis_dict, that value will be added to analysis_dict.

    catch_errors: Where true, the function will catch errors raised by analysis function, raising their messages as warnings and storing the 
    analysis error indicator string as the value in the analysis_results dict.

    run_filter: A function fun(my_measurement, my_run) which determines which runs to analyze; if passed, only those runs for which filter returns 
    True are analyzed.
    
    NOTE: Because the output of analysis_function can be arbitrary, it is the form of the argument result_varname that determines whether the 
    function is treated as having a return which is in single or tuple form. The former form is allowed for notational convenience, the latter 
    to accommodate e.g. numerically demanding analyses that inherently return two different results of individual interest."""

    ANALYSIS_ERROR_INDICATOR_STRING = "ERR"

    def analyze_runs(self, analysis_function, result_varnames, fun_kwargs = None, ignore_badshots = True, overwrite_existing = True, catch_errors = False, 
                    run_filter = None, print_progress = False):
        if fun_kwargs is None:
            fun_kwargs = {}
        results_in_tuple_form = isinstance(result_varnames, tuple)
        if not results_in_tuple_form:
            result_varnames = (result_varnames,)
        run_id_to_analyze_list = []
        filtered_dict = self.filter_run_dict(ignore_badshots=ignore_badshots, run_filter = run_filter)
        for run_id in filtered_dict:
            current_run = filtered_dict[run_id]
            if(not overwrite_existing):
                result_varnames_not_all_present = False 
                for varname in result_varnames:
                    if not varname in current_run.analysis_results:
                        result_varnames_not_all_present = True
                        break 
                if result_varnames_not_all_present:
                    run_id_to_analyze_list.append(run_id)
            else:
                run_id_to_analyze_list.append(run_id)
        if print_progress:
            analysis_len = len(run_id_to_analyze_list)
            counter = 0
        for run_id in run_id_to_analyze_list:
            if(print_progress):
                print("Analyzing run {0:d} ({1:.1%})".format(run_id, counter / analysis_len))
                counter += 1
            current_run = filtered_dict[run_id]
            if catch_errors:
                try:
                    results = analysis_function(self, current_run, **fun_kwargs)
                except Exception as e:
                    results = [Measurement.ANALYSIS_ERROR_INDICATOR_STRING] * len(result_varnames)
                    warnings.warn(repr(e))
                else:
                    if not results_in_tuple_form:
                        results = (results,)
            else:
                results = analysis_function(self, current_run, **fun_kwargs)
                if not results_in_tuple_form:
                    results = (results,)
            for varname, result in zip(result_varnames, results):
                    if overwrite_existing or not varname in current_run.analysis_results:
                        current_run.analysis_results[varname] = result


    """
    Convenience function for automatically excluding data points above a certain level of significance.
    
    Given a result_name, return a filter function which returns True for all runs where run.analysis_results[result_name] 
    is within a confidence interval of the mean specified by confidence_interval, using the Student T test. 

    Params:

    result_name: a string or tuple of strings indicating the analysis result which is to be filtered on. If a tuple is passed, 
    the filter function returned is True only when the T test is passed for all strings in result_name

    If run filter is specified, only those runs for which run_filter returns True will be examined, and the filter 
    which is returned will also incorporate the stipulated run_filter.

    If return_interval is True, returns a tuple (filter_function, confidence_range_1, confidence_range_2, ...), 
    where confidence_range_i = (lower_bound_i, upper_bound_i) for each result_name_i in the passed result_name. 
    """
    def get_outlier_filter(self, result_name, confidence_interval = 0.95, return_interval = False, run_filter = None, 
                            ignore_errors = True, ignore_badshots = True, iterative = True):
        result_name_tuple = Measurement._condition_string_tuple(result_name)
        overall_valid_ids = self.get_parameter_value_from_runs("id", ignore_badshots = ignore_badshots, run_filter = run_filter)
        interval_tuples_list = []
        for current_result_name in result_name_tuple:
            ids, values = self.get_parameter_analysis_value_pair_from_runs("id", current_result_name, ignore_badshots = ignore_badshots,
                                                     ignore_errors = ignore_errors, run_filter = run_filter, numpyfy = True)
            inlier_indices = statistics_functions.filter_mean_outliers(values, alpha = (1 - confidence_interval) / 2, iterative = iterative)
            inlier_ids = ids[inlier_indices]
            inlier_values = values[inlier_indices]
            min_inlier_value = np.min(inlier_values, initial = np.inf) 
            max_inlier_value = np.max(inlier_values, initial = -np.inf)
            interval_tuples_list.append((min_inlier_value, max_inlier_value))
            overall_valid_ids = np.intersect1d(overall_valid_ids, inlier_ids)
        def inlier_run_filter(my_measurement, my_run):
            return my_run.parameters["id"] in overall_valid_ids
        combined_run_filter = Measurement._condition_run_filter((inlier_run_filter, run_filter))
        if not return_interval:
            return combined_run_filter
        else:
            return (combined_run_filter, *interval_tuples_list)
    

    def add_to_live_analyses(self, analysis_function, result_varnames, fun_kwargs = None, run_filter = None):
        if fun_kwargs is None:
            fun_kwargs = {}
        self.analyses_list.append((analysis_function, result_varnames, fun_kwargs, run_filter))


    def _apply_live_analyses(self, ignore_badshots = True, overwrite_existing = False, catch_errors = False, print_progress = False):
        for analysis_tuple in self.analyses_list:
            fun, varnames, fun_kwargs, run_filter = analysis_tuple
            self.analyze_runs(fun, varnames, fun_kwargs = fun_kwargs, run_filter = run_filter, 
                                ignore_badshots = ignore_badshots, overwrite_existing=overwrite_existing,
                                catch_errors=catch_errors, print_progress=print_progress)


    """
    Labels runs as bad shots using user-specified input.

    Parameters:
    
    badshot_function: Function with calling signature (measurement, run), 
    as with any analysis function, but assumed to return a boolean True or False, corresponding to whether the run is or is not 
    a bad shot. 

    badshots_list (optional): If passed, badshot_function is completely ignored; instead, the run_ids appearing in badshots_list are 
    labeled as bad shots.

    override_existing_badshots: If True, then runs which are _already_ labeled as bad shots but for which either badshot_function or badshots_list 
    indicate that they are not bad shots will be re-labeled as good shots. 

    use_badshot_checked_list: If True and override_existing is false, run_ids are appended to a checklist once they have been checked for bad shot status,
    and will not be checked on subsequent invocations of the function which also have use_checklist = True. If False, runs are neither appended to this checklist for future invocations nor 
    exempted from badshot checking on the current invocation.
    """
    def label_badshots_custom(self, badshot_function = None, badshots_list = None, override_existing_badshots = False, use_badshot_checked_list = False):
        for run_id in self.runs_dict:
            if not override_existing_badshots and use_badshot_checked_list:
                if run_id in self.badshot_checked_list:
                    continue 
                else:
                    self.badshot_checked_list.append(run_id)
            current_run = self.runs_dict[run_id]
            if override_existing_badshots:
                #Explicit None comparison to eliminate edge case where badshots list is the empty list []
                if not badshots_list is None:
                    current_run.is_badshot = (run_id in badshots_list)
                elif badshot_function:
                    current_run.is_badshot = badshot_function(self, current_run) 
            elif not current_run.is_badshot:
                if not badshots_list is None:
                    current_run.is_badshot = (run_id in badshots_list)
                elif badshot_function:
                    current_run.is_badshot = badshot_function(self, current_run)
            current_run.parameters["badshot"] = current_run.is_badshot


    def filter_run_dict(self, ignore_badshots = True, run_filter = None, ignore_errors = False, analysis_value_err_check_name = None):
        filtered_dict = {}
        conditioned_run_filter = Measurement._condition_run_filter(run_filter)
        conditioned_global_run_filter = Measurement._condition_run_filter(self.global_run_filter)
        conditioned_overall_run_filter = Measurement._condition_run_filter((conditioned_run_filter, conditioned_global_run_filter))
        if ignore_errors:
            conditioned_analysis_err_check_name_tuple = Measurement._condition_string_tuple(analysis_value_err_check_name)
        for run_id in self.runs_dict:
            current_run = self.runs_dict[run_id]
            filter_run = False
            if ignore_badshots and current_run.is_badshot:
                filter_run = True 
            if filter_run or not conditioned_overall_run_filter(self, current_run):
                filter_run = True 
            if not filter_run and ignore_errors:
                for err_check_name in conditioned_analysis_err_check_name_tuple:
                    analysis_result = current_run.analysis_results[err_check_name]
                    if isinstance(analysis_result, str) and analysis_result == Measurement.ANALYSIS_ERROR_INDICATOR_STRING:
                        filter_run = True
                        break
            if not filter_run:
                filtered_dict[run_id] = current_run
        return filtered_dict 


    #Helper function for handling cases where run filter is None, a filter, or a tuple of filters
    @staticmethod
    def _condition_run_filter(run_filter):
        if run_filter is None:
            return lambda my_measurement, my_run: True
        else:
            try:
                for filter in run_filter:
                    break
                def combined_filter_func(my_measurement, my_run):
                    for filter in run_filter:
                        if (not filter is None ) and (not filter(my_measurement, my_run)):
                            return False 
                    return True
                return combined_filter_func
            except TypeError:
                return run_filter


    #Helper function for functions which take inputs of either strings or tuples of strings
    @staticmethod
    def _condition_string_tuple(string_or_tuple):
        if isinstance(string_or_tuple, tuple):
            return string_or_tuple 
        else:
            return (string_or_tuple,)


    def _label_badshots_default(self, override_existing_badshots = False):
        self.label_badshots_custom(badshot_function = self.badshot_function, use_badshot_checked_list = (not override_existing_badshots), 
                                override_existing_badshots=override_existing_badshots)



    def get_badshots_list(self):
        badshots_list = []
        for run_id in self.runs_dict:
            current_run = self.runs_dict[run_id]
            if current_run.is_badshot:
                badshots_list.append(run_id) 
        return badshots_list


    def update(self, update_runs = True, update_badshots = True, update_analyses = True, overwrite_existing_analysis = False, 
                override_existing_badshots = False, ignore_badshots = True, catch_errors = True, print_progress = False):
        if update_runs:
            self._update_runs_dict()
        if update_badshots:
            self._label_badshots_default(override_existing_badshots=override_existing_badshots)
        if update_analyses:
            self._apply_live_analyses(ignore_badshots=ignore_badshots, catch_errors=catch_errors, print_progress=print_progress, 
                                overwrite_existing = overwrite_existing_analysis)
        
        


    @staticmethod
    def _parse_run_id_from_filename(image_filename):
        run_id_string = image_filename.split(FILENAME_DELIMITER_CHAR)[0]
        return int(run_id_string)

    
    @staticmethod
    def _parse_datetime_from_filename(filename):
        datetime_string = filename.split(FILENAME_DELIMITER_CHAR)[1]
        return datetime.datetime.strptime(datetime_string, FILENAME_DATETIME_FORMAT_STRING)
        
class Run():
    """Initialization method
    
    Params:
    
    run_id: int, the run id
    image_pathnames_dict: A dict {image_name:image_pathname} of paths to each image associated with the given run. The names image_name are taken 
    from the list in MEASUREMENT_IMAGE_NAME_DICT which corresponds to the imaging_type of the overarching measurement. 
    image_format: The file extension of the image files
    parameters: The run parameters.
    
    Optimal params:

    hold_images_in_memory: If true, the underlying images of the run are loaded into memory at run creation; if false, they are loaded when 
        requested by the get_image. 

    image_format: The format of the underlying image files. 

    analysis_results: A dict containing the results of analyses on the runs. Generally this will be empty at creation, unless runs are being 
        re-loaded from a previous measurement session.

    connected_mode: A boolean indicating whether the run is initialized in connected mode. If not in this mode, access to the run images is unavailable; 
    only the analysis_results and parameters objects are present. 
    """
    def __init__(self, run_id, image_pathnames_dict, parameters, hold_images_in_memory = True, image_format = ".fits", analysis_results = None, 
                connected_mode = True):
        self.run_id = run_id
        self.parameters = parameters
        if(not analysis_results):
            self.analysis_results = {}
        else:
            self.analysis_results = analysis_results
        if('badshot' in self.analysis_results):
            self.is_badshot = self.analysis_results['badshot']
        else:
            self.is_badshot = False
            self.analysis_results["badshot"] = False
        self.connected_mode = connected_mode
        if connected_mode:
        #Specific to connected mode
            self.hold_images_in_memory = hold_images_in_memory
            self.image_dict = {}
            if not image_format in IMAGE_FORMATS_LIST:
                raise RuntimeError("Image format is not supported.")
            self.image_format = image_format
            for key in image_pathnames_dict:
                image_pathname = image_pathnames_dict[key] 
                if(hold_images_in_memory):
                    self.image_dict[key] = self._load_image(image_pathname)
                else:
                    self.image_dict[key] = image_pathname


    def get_image(self, image_name, memmap = False):
        if not self.connected_mode:
            raise RuntimeError("Access to run images unavailable when not in connected mode.")
        if(self.hold_images_in_memory):
            return self.image_dict[image_name]
        else:
            return self._load_image(self.image_dict[image_name], memmap = memmap)

    """
    Gives the first image in the run's imagedict; returns for any imaging type."""
    def get_default_image(self, memmap = False):
        if not self.connected_mode:
            raise RuntimeError("Access to run images unavailable when not in connected mode.")
        for image_name in self.image_dict:
            return self.get_image(image_name, memmap = memmap)

    def get_parameter_value(self, value_name):
        return self.parameters[value_name]

    def get_parameters(self):
        return self.parameters


    """
    Loads the image located at a pathname.

    Where memmap is true, loads a reference to the image location, rather than 
    the whole image into memory. To do so requires image to be unscaled.

    WARNING: An unscaled image is offset by -32768 thanks to unsigned integer issues. This 
    is safe for typical use, because this offset cancels when dark counts are subtracted.
    """
    def _load_image(self, image_pathname, memmap = False):
        if(self.image_format == ".fits"):
            with fits.open(image_pathname, memmap = memmap, do_not_scale_image_data = memmap) as hdul:
                return hdul[0].data
        else:
            raise RuntimeError("The image format is not supported.")
