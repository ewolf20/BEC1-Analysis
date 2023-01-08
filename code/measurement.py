import datetime
import json
import os

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
    """
    def __init__(self, measurement_directory_path = None, imaging_type = 'top_double', experiment_parameters = None, image_format = ".fits", 
                    hold_images_in_memory = True, measurement_parameters = None, run_parameters_verbose = False):
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
        self.runs_dict = {}
        

    """Initializes the runs dict.
    
    Creates a dictionary {run_id:Run} of runs in the measurement. Each individual run is an object containing the run parameters and images."""
    def _initialize_runs_dict(self, use_saved_params = False, saved_params_filename = "measurement_run_params_dump.json"):
        if(use_saved_params):
            run_parameters_list = loading_functions.load_run_parameters_from_json(saved_params_filename)
        else:
            run_parameters_list = None
        matched_run_ids_and_parameters_list = self._get_run_ids_and_parameters_in_measurement_folder(run_parameters_list = run_parameters_list)
        for run_id_and_parameters in matched_run_ids_and_parameters_list:
            self._add_run(run_id_and_parameters)


    def _get_run_parameters_from_measurement_folder(self):
        DATA_DUMP_PARAMS_FILENAME = "run_params_dump.json" 
        path_to_dump_file_in_measurement_folder = os.path.join(self.measurement_directory_path, DATA_DUMP_PARAMS_FILENAME)
        if(os.path.exists(path_to_dump_file_in_measurement_folder)):
            run_parameters_list = loading_functions.load_run_parameters_from_json(path_to_dump_file_in_measurement_folder, 
                                                                    make_raw_parameters_terse = (not self.run_parameters_verbose))
        else:
            raise RuntimeError("""Drawing parameters directly from breadboard is deprecated. You may use ImageWatchdog.get_run_metadata()
                                to generate a run params json for legacy datasets.""")
        return run_parameters_list


    def _get_run_ids_and_parameters_in_measurement_folder(self, run_parameters_list = None):
        if not run_parameters_list:
            run_parameters_list = self._get_run_parameters_from_measurement_folder()
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
        pass


    """
    Dumps the parameters of the runs dict to a .json file.
    
    When called, save a dictionary {run_ID, params} of the parameters of each run in the current 
    runs dict. Allows measurement-specific updates to the run parameters."""
    def dump_runs_dict(self, dump_filename = "measurement_run_params_dump.json"):
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
    """
    def __init__(self, run_id, image_pathnames_dict, parameters, hold_images_in_memory = True, image_format = ".fits"):
        self.run_id = run_id
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
