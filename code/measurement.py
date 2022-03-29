import numpy as np
from astropy.io import fits

from breadboard_functions import load_breadboard_client

class Measurement():

    def __init__(self):
        pass



IMAGE_FORMATS_LIST = ['.fits']

class Run():

    def __init__(self, run_id, image_pathnames_dict, breadboard_client, hold_images_in_memory = True, image_format = ".fits", parameters = None):
        self.run_id = run_id
        self.breadboard_client = breadboard_client
        if(not parameters):
            self.parameters = self.load_parameters() 
        self.load_images = load_images 
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
        if(hold_images_in_memory):
            return self.image_dict[image_name] 
        else:
            return self.load_image(self.image_dict[image_name]) 


    #TODO check formatting of returned dict from breadboard
    def load_parameters(self):
        return self.bc.get_runs_df_from_ids(self.run_id)


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
