import datetime
import importlib.resources as pkg_resources
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import numpy as np
from astropy.io import fits

from .image_processing_functions import get_absorption_image


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_satyendra = path_to_file + "/../../"
sys.path.insert(0, path_to_satyendra)

from satyendra.code import breadboard_functions

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
    run_parameters_verbose: Whether the runs store """

    def __init__(self, measurement_directory_path = None, imaging_type = 'top_double', experiment_parameters = None, image_format = ".fits", 
                    hold_images_in_memory = True, measurement_parameters = None, run_parameters_verbose = False):
        self.breadboard_client = breadboard_functions.load_breadboard_client() 
        if(not measurement_directory_path):
            measurement_directory_path = os.getcwd() 
        self.measurement_directory_path = measurement_directory_path 
        self.imaging_type = imaging_type
        self.image_format = image_format
        self.hold_images_in_memory = hold_images_in_memory
        if(not experiment_parameters):
            self.experiment_parameters = Measurement.load_experiment_parameters()
        else:
            self.experiment_parameters = experiment_parameters
        if(measurement_parameters):   
            self.measurement_parameters = measurement_parameters
        else:
            self.measurement_parameters = {}
        self.run_parameters_verbose = run_parameters_verbose
        

    """Initializes the runs dict.
    
    Creates a dictionary {run_id:Run} of runs in the measurement. Each individual run is an object containing the run parameters and images."""
    #TODO: Update so that the error is more descriptive when the wrong measurement type is specified
    def _initialize_runs_dict(self, use_saved_params = False, saved_params_filename = "run_params_dump.json"):
        unique_run_ids_list = list(set([Measurement._parse_run_id_from_filename(f) for f in os.listdir(self.measurement_directory_path) if self.image_format in f]))
        datetimes_list = list([Measurement._parse_datetime_from_filename(f) for f in os.listdir(self.measurement_directory_path) if self.image_format in f])
        min_datetime = min(datetimes_list) 
        max_datetime = max(datetimes_list)
        sorted_run_ids_list = sorted(unique_run_ids_list)
        if(not use_saved_params):
            run_parameters_list = breadboard_functions.get_run_parameter_dicts_from_ids(self.breadboard_client, sorted_run_ids_list,
                                                                                    start_datetime = min_datetime, end_datetime = max_datetime, 
                                                                                    verbose = self.run_parameters_verbose)
        else:
            with open(saved_params_filename) as run_params_json:
                run_parameters_dict = json.load(run_params_json)
            unsorted_run_parameters_list = [(int(key), run_parameters_dict[key]) for key in run_parameters_dict]
            run_parameters_list = [f[1] for f in sorted(unsorted_run_parameters_list, key = lambda x: x[0])]
        runs_dict = {}
        for run_id, run_parameters in zip(sorted_run_ids_list, run_parameters_list):
            run_image_pathname_dict = {}
            run_id_image_filenames = [f for f in os.listdir(self.measurement_directory_path) if str(run_id) in f]
            for run_id_image_filename in run_id_image_filenames:
                for image_name in MEASUREMENT_IMAGE_NAME_DICT[self.imaging_type]:
                    if image_name in run_id_image_filename:
                        run_id_image_pathname = os.path.join(self.measurement_directory_path, run_id_image_filename)
                        run_image_pathname_dict[image_name] = run_id_image_pathname 
                        break
            current_run = Run(run_id, run_image_pathname_dict, self.breadboard_client, hold_images_in_memory= self.hold_images_in_memory, 
                                parameters = run_parameters, image_format = self.image_format)
            runs_dict[run_id] = current_run
        self.runs_dict = runs_dict



    """
    Dumps the parameters of the runs dict to a .json file.
    
    When called, save a dictionary {run_ID, params} of the parameters of each run in the current 
    runs dict. Avoids repeating calls to breadboard."""
    def dump_runs_dict(self, dump_filename = "run_params_dump.json"):
        with open(dump_filename, 'w') as dump_file:
            dump_dict = {} 
            for run_id in self.runs_dict:
                current_run = self.runs_dict[run_id] 
                dump_dict[run_id] = current_run.get_parameters() 
            dump_file.write(json.dumps(dump_dict))

    
    """
    Labels runs as bad shots.
    
    Uses the function badshot_function to label runs as bad shots. badshot function has calling signature (runs_dict, **kwargs), 
    with **kwargs intended for passing in self.measurement_parameters, and returns a list of run_ids which are bad shots.

    If badshots_list is passed, instead labels the run_ids in badshots_array as bad shots."""
    def label_badshots(self, badshot_function = None, badshots_list = None):
        if(not badshots_list and badshot_function):
            badshots_list = badshot_function(self.runs_dict, **self.measurement_parameters)
        for run_id in self.runs_dict:
            if run_id in badshots_list:
                current_run = self.runs_dict[run_id]
                current_run.is_badshot = True
                current_run.parameters['badshot'] = True


    def get_badshots_list(self):
        badshots_list = []
        for run_id in self.runs_dict:
            current_run = self.runs_dict[run_id]
            if current_run.is_badshot:
                badshots_list.append(run_id) 
        return badshots_list



    """
    Set a rectangular box with user input.
    
    run_to_use: The run to use for setting the box position. Default 0, i.e. the first run 
    in the runs_dict, but if this is a bad shot, can be changed."""
    def set_box(self, label, run_to_use = 0, box_coordinates = None):
        if(not box_coordinates):
            for i, key in enumerate(self.runs_dict):
                if(i == run_to_use):
                    my_run = self.runs_dict[key] 
                    break 
            for key in my_run.image_dict:
                my_image_array = my_run.get_image(key)
                break
            my_with_atoms_image = get_absorption_image(my_image_array)
            x_1, x_2, y_1, y_2 = Measurement._draw_box(my_with_atoms_image, label)
            x_min = int(min(x_1, x_2))
            y_min = int(min(y_1, y_2))
            x_max = int(max(x_1, x_2))
            y_max = int(max(y_1, y_2))      
            self.measurement_parameters[label] = [x_min, y_min, x_max, y_max]
        else:
            self.measurement_parameters[label] = box_coordinates


    """
    Alias to set_box('norm_box'), for convenience."""
    def set_norm_box(self, run_to_use=0, box_coordinates = None):
        self.set_box('norm_box', run_to_use = run_to_use, box_coordinates = box_coordinates)

    @staticmethod
    def _draw_box(my_image, label):
        ax = plt.gca()
        ax.imshow(my_image, cmap = 'gray')
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
        
    

    @staticmethod
    def _parse_run_id_from_filename(image_filename):
        run_id_string = image_filename.split(FILENAME_DELIMITER_CHAR)[0]
        return int(run_id_string)

    
    @staticmethod
    def _parse_datetime_from_filename(filename):
        datetime_string = filename.split(FILENAME_DELIMITER_CHAR)[1]
        return datetime.datetime.strptime(datetime_string, DATETIME_FORMAT_STRING)

    @staticmethod 
    def load_experiment_parameters():
        from .. import secrets as s 
        with pkg_resources.path(s, "experiment_parameters_secret.json") as parameters_path:
            with open(parameters_path) as parameters_file:
                return json.load(parameters_file)


        
class Run():
    """Initialization method
    
    Params:
    
    run_id: int, the run id
    image_pathnames_dict: A dict {image_name:image_pathname} of paths to each image associated with the given run. The names image_name are taken 
    from the list in MEASUREMENT_IMAGE_NAME_DICT which corresponds to the imaging_type of the overarching measurement. 
    breadboard_client: A client for querying breadboard to obtain the run parameters.
    image_format: The file extension of the image files
    parameters: The run parameters. If None, these are initialized by querying breadboard.
    parameters_verbose: Only used if parameters is None. If True, the parameters as queried from breadboard include all cicero information, not just list bound variables. 
    """
    def __init__(self, run_id, image_pathnames_dict, breadboard_client = None, hold_images_in_memory = True, image_format = ".fits", parameters = None, 
                parameters_verbose = False):
        self.run_id = run_id
        self.parameters_verbose = parameters_verbose
        self.breadboard_client = breadboard_client
        if(not parameters):
            self.parameters = self.load_parameters() 
        else:
            self.parameters = parameters
        if('badshot' in self.parameters):
            self.is_badshot = self.parameters['badshot']
        else:
            self.is_badshot = False
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


    #TODO check formatting of returned dict from breadboard
    #TODO add support for recently uploaded runs
    def load_parameters(self):
        return breadboard_functions.get_run_parameter_dict_from_id(self.breadboard_client, self.run_id, verbose = self.parameters_verbose)

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
