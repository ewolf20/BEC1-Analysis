import os

import numpy as np
from astropy.io import fits

from BEC1_Analysis.code.breadboard_functions import load_breadboard_client

IMAGE_FORMATS_LIST = ['.fits']
IMAGING_TYPES_LIST = ['top_double', 'side_low_mag', 'side_high_mag']
MEASUREMENT_IMAGE_NAME_DICT = {'top_double': ['TopA', 'TopB'],
                                'side_low_mag':['Side'], 'side_high_mag':['Side']}

class Measurement():

    DATETIME_FORMAT_STRING = "%Y-%m-%d--%H-%M-%S"

    """Initialization method.
    
    Parameters:
    
    measurement_directory_path: str, The path to a directory containing the labeled images to process.
    imaging_type: str, The type of imaging, e.g. top or side, low mag or high mag.
    experiment_parameters: dict {parname:value} of experiment-level parameters not saved within the run parameters, e.g. trapping frequencies
    image_format: str, the filetype of the images being processed
    hold_images_in_memory: bool, Whether images are kept loaded in memory, or loaded on an as-needed basis and then released.
    analysis_parameters: dict {parname:value} of analysis-level params, e.g. a list of run ids which are flagged as bad shots or 
    the coordinates of a background box.
    run_parameters_verbose: Whether the runs store """

    def __init__(self, measurement_directory_path = None, imaging_type = 'top_double', experiment_parameters = None, image_format = ".fits", 
                    hold_images_in_memory = True, measurement_parameters = None, run_parameters_verbose = False):
        self.breadboard_client = load_breadboard_client() 
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
        self.measurement_parameters = measurement_parameters
        self.run_parameters_verbose = run_parameters_verbose
        

    """Initializes the runs dict.
    
    Creates a dictionary {run_id:Run} of runs in the measurement. Each individual run is an object containing the run parameters and images."""
    def _initialize_runs_dict(self):
        unique_run_ids_list = list(set([Measurement._parse_run_id_from_filename(f) for f in os.listdir(self.measurement_directory_path) if self.image_format in f]))
        sorted_run_ids_list = sorted(unique_run_ids_list)
        runs_dict = {}
        for run_id in sorted_run_ids_list:
            run_image_pathname_dict = {}
            run_id_image_pathnames = [os.path.join(self.measurement_directory_path, f) for f in os.listdir(self.measurement_directory_path) if str(run_id) in f]
            for run_id_image_pathname in run_id_image_pathnames:
                for image_name in MEASUREMENT_IMAGE_NAME_DICT[self.imaging_type]:
                    if image_name in run_id_image_pathname:
                        run_image_pathname_dict[image_name] = run_id_image_pathname 
                        break
            current_run = Run(run_id, run_image_pathname_dict, self.breadboard_client, hold_images_in_memory= self.hold_images_in_memory, 
                                image_format = self.image_format)
            runs_dict[run_id] = current_run 
        self.runs_dict = runs_dict


    """
    Returns a list of the unique run ids in the measurement folder, plus optionally the parameters for each.
    Parameters:

    return_parameters: Whether the method should query breadboard for the run information, or leave it to be filled by the individual runs. 

    Returns: a tuple (sorted_run_ids_list, sorted_run_parameters_list). sorted_run_ids_list is a list of run_ids sorted in ascending order; sorted_run_parameters_list is the corresponding parameters dicts """
    def _get_run_information(self, return_parameters = False, verbose_parameters = False):
        unique_run_ids_list = list(set([Measurement._parse_run_id_from_filename(f) for f in os.listdir(self.measurement_directory_path) if self.image_format in f]))
        sorted_run_ids_list = sorted(unique_run_ids_list)
        if(return_parameters):
            for sorted_run_id in sorted_run_ids_list:
                for filename in os.listdir(self.measurement_directory_path):
                    if(sorted_run_id in filename):
                        run_datetime = Measurement.parse_datetime_from_filename(filename)

        




    @staticmethod
    def _parse_run_id_from_filename(image_filename):
        run_id_string = image_filename.split("_")[0]
        return int(run_id_string)

    
    @staticmethod
    def _parse_datetime_from_filename(filename):
        pass
        


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
        self.breadboard_client = breadboard_client
        if(not parameters):
            self.parameters = self.load_parameters() 
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
        self.parameters_verbose = parameters_verbose
    

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
