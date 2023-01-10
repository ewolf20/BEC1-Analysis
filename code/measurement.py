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
from . import loading_functions 

loading_functions.load_satyendra()

IMAGE_FORMATS_LIST = ['.fits']
IMAGING_TYPES_LIST = ['top_double', 'side_low_mag', 'side_high_mag']
MEASUREMENT_IMAGE_NAME_DICT = {'top_double': ['TopA', 'TopB'],
                                'side_low_mag':['Side'], 'side_high_mag':['Side']}
DATETIME_FORMAT_STRING = "%Y-%m-%d--%H-%M-%S"
FILENAME_DELIMITER_CHAR = "_"

class Measurement():

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
    badshot_function: A badshot function, to be used by 
    """
    def __init__(self, measurement_directory_path = None, imaging_type = 'top_double', experiment_parameters = None, image_format = ".fits", 
                    hold_images_in_memory = True, measurement_parameters = None, run_parameters_verbose = False, use_saved_analysis = False, 
                    saved_analysis_filename = "measurement_run_analysis_dump.json", badshot_function = None, analyses_list = None):
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
        if(measurement_parameters):   
            self.measurement_parameters = measurement_parameters
        else:
            self.measurement_parameters = {}
        self.run_parameters_verbose = run_parameters_verbose
        self.badshot_function = badshot_function
        self.badshot_checked_list = []
        self.runs_dict = {}
        self._initialize_runs_dict()
        if analyses_list is None:
            self.analyses_list = []

    def set_badshot_function(self, badshot_function):
        self.badshot_function = badshot_function

    """Initializes the runs dict.
    
    Creates a dictionary {run_id:Run} of runs in the measurement. Each individual run is an object containing the run parameters and images."""
    def _initialize_runs_dict(self):
        matched_run_ids_and_parameters_list = self._get_run_ids_and_parameters_from_measurement_folder()
        for run_id_and_parameters in matched_run_ids_and_parameters_list:
            self._add_run(run_id_and_parameters)


    def _get_run_ids_and_parameters_from_measurement_folder(self):
        DATA_DUMP_PARAMS_FILENAME = "run_params_dump.json" 
        path_to_dump_file_in_measurement_folder = os.path.join(self.measurement_directory_path, DATA_DUMP_PARAMS_FILENAME)
        if(os.path.exists(path_to_dump_file_in_measurement_folder)):
            run_parameters_list = loading_functions.load_run_parameters_from_json(path_to_dump_file_in_measurement_folder, 
                                                                    make_raw_parameters_terse = (not self.run_parameters_verbose))
        else:
            raise RuntimeError("""Drawing parameters directly from breadboard is deprecated. You may use ImageWatchdog.get_run_metadata()
                                to generate a run params json for legacy datasets.""")
        unique_run_ids_list = list(set([Measurement._parse_run_id_from_filename(f) for f in os.listdir(self.measurement_directory_path) if self.image_format in f]))
        sorted_run_ids_list = sorted(unique_run_ids_list)
        matched_run_ids_and_parameters_list = []
        #O(n^2) naive search, but it's fine...
        for run_id in sorted_run_ids_list: 
            for parameters in run_parameters_list:
                if parameters['id'] == run_id:
                    matched_run_ids_and_parameters_list.append((run_id, parameters))
                    break
            else:
                raise RuntimeError("Unable to find data for run id: " + str(run_id))
        return matched_run_ids_and_parameters_list

    def _add_run(self, run_id_and_parameters):
        run_id, run_parameters = run_id_and_parameters
        run_image_pathname_dict = {}
        run_id_image_filenames = [f for f in os.listdir(self.measurement_directory_path) if str(run_id) in f]
        for run_id_image_filename in run_id_image_filenames:
            for image_name in MEASUREMENT_IMAGE_NAME_DICT[self.imaging_type]:
                if image_name in run_id_image_filename:
                    run_id_image_pathname = os.path.join(self.measurement_directory_path, run_id_image_filename)
                    run_image_pathname_dict[image_name] = run_id_image_pathname 
                    break
            else:
                raise RuntimeError("Run image does not match specification. Is the imaging type correct?")
        generated_run = Run(run_id, run_image_pathname_dict, hold_images_in_memory= self.hold_images_in_memory, 
                            parameters = run_parameters, image_format = self.image_format)
        self.runs_dict[run_id] = generated_run



    def _update_runs_dict(self):
        matched_run_ids_and_parameters_list = self._get_run_ids_and_parameters_from_measurement_folder()
        for run_id_and_parameters in matched_run_ids_and_parameters_list:
            run_id, _ = run_id_and_parameters
            if not run_id in self.runs_dict:
                self._add_run(run_id_and_parameters)
        current_run_ids_list = [f[0] for f in matched_run_ids_and_parameters_list]
        saved_run_ids_list = [f for f in self.runs_dict]
        for saved_run_id in saved_run_ids_list:
            if not saved_run_id in current_run_ids_list:
                self.runs_dict.pop(saved_run_id)


    """
    Dumps the parameters of the run analysis_results dicts to a .json file.
    
    When called, save a dictionary {run_ID, analysis_results} of the analysis results of each run in the current 
    runs dict as a json file. Allows analyses to be recalled from previous sessions. 
    
    NOTE: This method relies on the objects in analysis_results being JSON serializable, with a _single_ exception: because of 
    their ubiquity, numpy arrays have been manually accommodated as well. To do this, a separate directory is created wherein the 
    numpy arrays are saved as .npy files."""

    def dump_run_analysis_dict(self, dump_pathname = "measurement_run_analysis_dump.json"):
        with open(dump_pathname, 'w') as dump_file:
            dump_dict = {} 
            for run_id in self.runs_dict:
                current_run = self.runs_dict[run_id] 
                dump_dict[run_id] = Measurement._digest_analysis_results(run_id, dump_pathname, current_run.analysis_results)
            json.dump(dump_dict, dump_file)

    @staticmethod
    def _digest_analysis_results(run_id, dump_pathname, analysis_results):
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
                numpy_filename = "{0:d}_{1}.npy".format(run_id, key)
                numpy_pathname = os.path.join(numpy_storage_folder_path, numpy_filename)
                np.save(numpy_pathname, analysis_object)
                return_dict_numpy_key = "{0}_Numpy_Pathname".format(key)
                return_dict[return_dict_numpy_key] = numpy_pathname
        return return_dict





    def load_run_analysis_dict(self, dump_pathname = "measurement_run_analysis_dump.json"):
        with open(dump_pathname, 'r') as dump_file:
            dump_dict = json.load(dump_file)
            for run_id in dump_dict:
                run_id_as_integer = int(run_id)
                current_run = self.runs_dict[run_id_as_integer] 
                current_run.analysis_results = Measurement._undigest_analysis_results(dump_dict[run_id])


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


    """
    Set a rectangular box with user input.
    
    run_to_use: The run to use for setting the box position. Default 0, i.e. the first run 
    in the runs_dict, but if this is a bad shot, can be changed."""
    def set_box(self, label, run_to_use = 0, image_to_use = None, box_coordinates = None):
        if(not box_coordinates):
            for i, key in enumerate(self.runs_dict):
                if(i == run_to_use):
                    my_run = self.runs_dict[key] 
                    break
            if(not image_to_use):
                for key in my_run.image_dict:
                    my_image_array = my_run.get_image(key)
                    break
            else:
                my_image_array = my_run.get_image(image_to_use)
            my_with_atoms_image = get_absorption_image(my_image_array)
            title = str(my_run.run_id)
            x_1, x_2, y_1, y_2 = Measurement._draw_box(my_with_atoms_image, label, title)
            x_min = int(min(x_1, x_2))
            y_min = int(min(y_1, y_2))
            x_max = int(max(x_1, x_2))
            y_max = int(max(y_1, y_2))
            self.measurement_parameters[label] = [x_min, y_min, x_max, y_max]
        else:
            self.measurement_parameters[label] = box_coordinates


    """
    Alias to set_box('norm_box'), for convenience."""
    def set_norm_box(self, run_to_use=0, image_to_use = None, box_coordinates = None):
        self.set_box('norm_box', run_to_use = run_to_use, image_to_use = image_to_use, box_coordinates = box_coordinates)

    """
    Alias to set_box('ROI'), for convenience."""
    def set_ROI(self, run_to_use = 0, image_to_use = None, box_coordinates = None):
        self.set_box('ROI', run_to_use = run_to_use, image_to_use = image_to_use, box_coordinates = box_coordinates)

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


    def get_parameter_value_from_runs(self, value_name, ignore_badshots = True):
        return_list = []
        for run_id in self.runs_dict:
            current_run = self.runs_dict[run_id] 
            if not ignore_badshots or not current_run.is_badshot:
                return_list.append(current_run.parameters[value_name])
        return return_list


    def get_analysis_value_from_runs(self, value_name, ignore_badshots = True, ignore_errors = True):
        return_list = []
        for run_id in self.runs_dict:
            current_run = self.runs_dict[run_id]
            if not ignore_badshots or not current_run.is_badshot:
                result_value = current_run.analysis_results[value_name] 
                if not ignore_errors or not (isinstance(result_value, str) and result_value == Measurement.ANALYSIS_ERROR_INDICATOR_STRING):
                    return_list.append(current_run.analysis_results[value_name])
        return return_list
    
    """Convenience function which returns a pair of parameter values and analysis results. Convenient for plotting, and also convenient where 
    some runs have errors in the analysis."""
    def get_parameter_analysis_result_pair_from_runs(self, parameter_name, analysis_value_name, ignore_badshots = True, ignore_errors = True):
        param_list = [] 
        analysis_result_list = [] 
        for run_id in self.runs_dict:
            current_run = self.runs_dict[run_id] 
            if not ignore_badshots or not current_run.is_badshot:
                result_value = current_run.analysis_results[analysis_value_name]
                if not ignore_errors or not (isinstance(result_value, str) and result_value == Measurement.ANALYSIS_ERROR_INDICATOR_STRING):
                    param_list.append(current_run.parameters[parameter_name])
                    analysis_result_list.append(result_value)
        return (param_list, analysis_result_list)

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
    
    NOTE: Because the output of analysis_function can be arbitrary, it is the form of the argument result_varname that determines whether the 
    function is treated as having a return which is in single or tuple form. The former form is allowed for notational convenience, the latter 
    to accommodate e.g. numerically demanding analyses that inherently return two different results of individual interest."""

    ANALYSIS_ERROR_INDICATOR_STRING = "ERR"

    def analyze_runs(self, analysis_function, result_varnames, ignore_badshots = True, overwrite_existing = False, catch_errors = False, 
                    print_progress = False):
        results_in_tuple_form = isinstance(result_varnames, tuple)
        if not results_in_tuple_form:
            result_varnames = (result_varnames,)
        for run_id in self.runs_dict:
            if(print_progress):
                print("Analyzing run {0:d}".format(run_id))
            current_run = self.runs_dict[run_id]
            if not ignore_badshots or not current_run.is_badshot:
                if(not overwrite_existing):
                    result_varnames_not_all_present = False 
                    for varname in result_varnames:
                        if not varname in current_run.analysis_results:
                            result_varnames_not_all_present = True
                            break 
                    if not result_varnames_not_all_present:
                        continue
                if catch_errors:
                    try:
                        results = analysis_function(self, current_run)
                    except Exception as e:
                        results = [Measurement.ANALYSIS_ERROR_INDICATOR_STRING] * len(result_varnames)
                        warnings.warn(repr(e))
                    else:
                        if not results_in_tuple_form:
                            results = (results,)
                else:
                    results = analysis_function(self, current_run)
                    if not results_in_tuple_form:
                        results = (results,)
                for varname, result in zip(result_varnames, results):
                        if overwrite_existing or not varname in current_run.analysis_results:
                            current_run.analysis_results[varname] = result 
    

    def add_default_analysis(self, analysis_function, result_varnames):
        self.analyses_list.append((analysis_function, result_varnames))


    def _apply_default_analyses(self, ignore_badshots = True, overwrite_existing = False, catch_errors = False, print_progress = False):
        for fun_and_varnames_tuple in self.analyses_list:
            fun, varnames = fun_and_varnames_tuple
            self.analyze_runs(fun, varnames, ignore_badshots = ignore_badshots, overwrite_existing=overwrite_existing, catch_errors=catch_errors, 
                                    print_progress=print_progress)


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
            self._apply_default_analyses(ignore_badshots=ignore_badshots, catch_errors=catch_errors, print_progress=print_progress, 
                                overwrite_existing = overwrite_existing_analysis)
        
        


    @staticmethod
    def _parse_run_id_from_filename(image_filename):
        run_id_string = image_filename.split(FILENAME_DELIMITER_CHAR)[0]
        return int(run_id_string)

    
    @staticmethod
    def _parse_datetime_from_filename(filename):
        datetime_string = filename.split(FILENAME_DELIMITER_CHAR)[1]
        return datetime.datetime.strptime(datetime_string, DATETIME_FORMAT_STRING)
        
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
    """
    def __init__(self, run_id, image_pathnames_dict, parameters, hold_images_in_memory = True, image_format = ".fits", analysis_results = None):
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
        self.hold_images_in_memory = hold_images_in_memory
        self.image_dict = {}
        if not image_format in IMAGE_FORMATS_LIST:
            raise RuntimeError("Image format is not supported.")
        self.image_format = image_format
        for key in image_pathnames_dict:
            image_pathname = image_pathnames_dict[key] 
            if(hold_images_in_memory):
                self.image_dict[key] = self.load_image(image_pathname)
            else:
                self.image_dict[key] = image_pathname


    def get_image(self, image_name, memmap = False):
        if(self.hold_images_in_memory):
            return self.image_dict[image_name]
        else:
            return self.load_image(self.image_dict[image_name], memmap = memmap)

    """
    Gives the first image in the run's imagedict; returns for any imaging type."""
    def get_default_image(self, memmap = False):
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
    def load_image(self, image_pathname, memmap = False):
        if(self.image_format == ".fits"):
            with fits.open(image_pathname, memmap = memmap, do_not_scale_image_data = memmap) as hdul:
                return hdul[0].data
        else:
            raise RuntimeError("The image format is not supported.")
