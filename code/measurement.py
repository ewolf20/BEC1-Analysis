import datetime
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import numpy as np
from astropy.io import fits


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
    def _initialize_runs_dict(self):
        unique_run_ids_list = list(set([Measurement._parse_run_id_from_filename(f) for f in os.listdir(self.measurement_directory_path) if self.image_format in f]))
        datetimes_list = list([Measurement._parse_datetime_from_filename(f) for f in os.listdir(self.measurement_directory_path) if self.image_format in f])
        min_datetime = min(datetimes_list) 
        max_datetime = max(datetimes_list)
        sorted_run_ids_list = sorted(unique_run_ids_list)
        run_parameters_list = breadboard_functions.get_run_parameter_dicts_from_ids(self.breadboard_client, sorted_run_ids_list,
                                                                                    start_datetime = min_datetime, end_datetime = max_datetime)
        runs_dict = {}
        for run_id, run_parameters in zip(sorted_run_ids_list, run_parameters_list):
            run_image_pathname_dict = {}
            run_id_image_pathnames = [os.path.join(self.measurement_directory_path, f) for f in os.listdir(self.measurement_directory_path) if str(run_id) in f]
            for run_id_image_pathname in run_id_image_pathnames:
                for image_name in MEASUREMENT_IMAGE_NAME_DICT[self.imaging_type]:
                    if image_name in run_id_image_pathname:
                        run_image_pathname_dict[image_name] = run_id_image_pathname 
                        break
            current_run = Run(run_id, run_image_pathname_dict, self.breadboard_client, hold_images_in_memory= self.hold_images_in_memory, 
                                parameters = run_parameters, image_format = self.image_format)
            runs_dict[run_id] = current_run
        self.runs_dict = runs_dict

    
    """
    Labels runs as bad shots.
    
    Uses the function badshot_function to label runs as bad shots. badshot function has calling signature (Run, **kwargs), 
    with **kwargs intended for passing in self.measurement_parameters. Runs which are already labeled as bad shots are unchanged."""
    def label_badshots(self, badshot_function):
        for run_id in self.runs_dict:
            current_run = self.runs_dict[run_id]
            if not current_run.is_badshot:
                current_run.is_badshot = badshot_function(current_run, **self.measurement_parameters)


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
            my_with_atoms_image = my_image_array[0]
            x_1, x_2, y_1, y_2 = Measurement._draw_box(my_with_atoms_image)
            x_min = int(min(x_1, x_2))
            y_min = int(min(y_1, y_2))
            x_max = int(max(x_1, x_2))
            y_max = int(max(y_1, y_2))      
            self.measurement_parameters[label] = [x_min, y_min, x_max, y_max]
        else:
            self.measurement_parameters[label] = box_coordinates


    """
    Alias to set_box('norm_box'), for convenience."""
    def set_norm_box(self, run_to_use, box_coordinates = None):
        self.set_box('norm_box', run_to_use = run_to_use, box_coordinates = box_coordinates)

    @staticmethod
    def _draw_box(my_image):
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
        plt.show()
        return((x_1, x_2, y_1, y_2))




    
    def check_box(self, label, run_to_use = 0):
        for i, key in enumerate(self.runs_dict):
            if(i == run_to_use):
                my_run = self.runs_dict[key] 
                break 
        for key in my_run.image_dict:
            my_image_array = my_run.get_image(key)
            break
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
        return None
            


        
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


    def get_image(self, image_name):
        if(self.hold_images_in_memory):
            return self.image_dict[image_name]
        else:
            return self.load_image(self.image_dict[image_name])


    #TODO check formatting of returned dict from breadboard
    #TODO add support for recently uploaded runs
    def load_parameters(self):
        return self.breadboard_client.get_runs_df_from_ids(self.run_id, verbose = self.parameters_verbose)

    def get_parameter_value(self, value_name):
        return self.parameters[value_name]

    def get_parameters(self):
        return self.parameters

    def load_image(self, image_pathname):
        if(self.image_format == ".fits"):
            with fits.open(image_pathname) as hdul:
                return hdul[0].data
        else:
            raise RuntimeError("The image format is not supported.")
