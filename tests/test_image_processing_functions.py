import hashlib
import os 
import sys

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np 


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

TEST_MEASUREMENT_DIRECTORY_PATH = "./resources"

TEST_IMAGE_FILE_PATH = "resources/805277_2022-04-06--8-49-08_Side.fits"
TEST_IMAGE_FILE_NAME = "805277_2022-04-06--8-49-08_Side.fits"


from BEC1_Analysis.code import image_processing_functions 

def load_test_image():
    with fits.open(TEST_IMAGE_FILE_PATH) as hdul:
        return hdul[0].data

def check_sha_hash(bytes, checksum_string):
    m = hashlib.sha256() 
    m.update(bytes) 
    return m.hexdigest() == checksum_string

def get_sha_hash_string(my_bytes):
    m = hashlib.sha256() 
    m.update(my_bytes) 
    return m.hexdigest()

def test_get_absorption_image():
    DEFAULT_FULL_SHA_CHECKSUM = 'ac99dcb7ba1e002c79aa0f55fde1204254fdd377a34c2e897a7eb79a569c57e2'
    DEFAULT_CLIPPED_FULL_SHA_CHECKSUM = 'cf07ab0015c1408a22255399d85a2873da1bcf9b0867acf5f1fbf8e225d670fb'
    ROI_SHA_CHECKSUM = '50ec4965eb71198577fc260efae986aefbdfc39b31249ba402226a08c9eb0567'
    ROI = [270, 0, 480, 180] 
    test_image_array = load_test_image()
    absorption_image_full_default = image_processing_functions.get_absorption_image(test_image_array, clean_strategy = 'default')
    absorption_image_full_default_clipped = image_processing_functions.get_absorption_image(test_image_array, clean_strategy = 'default_clipped')
    assert check_sha_hash(absorption_image_full_default.data.tobytes(), DEFAULT_FULL_SHA_CHECKSUM)
    assert check_sha_hash(absorption_image_full_default_clipped.data.tobytes(), DEFAULT_CLIPPED_FULL_SHA_CHECKSUM) 
    absorption_image_ROI = image_processing_functions.get_absorption_image(test_image_array, ROI = ROI)
    assert check_sha_hash(absorption_image_ROI, ROI_SHA_CHECKSUM)


def test_get_absorption_od_image():
    OD_IMAGE_CHECKSUM = 'eba885a09e45b672613b13e27776319bc7231362c18e20e11c64b5d6de8a193a'
    test_image_array = load_test_image() 
    od_image_full = image_processing_functions.get_absorption_od_image(test_image_array)
    assert check_sha_hash(od_image_full.data.tobytes(), OD_IMAGE_CHECKSUM)

def test_pixel_sum():
    ROI = [270, 0, 480, 180] 
    ROI_TARGET_SUM = 44731.2998268131
    FULL_IMAGE_TARGET_SUM = 169928.56906968268
    test_image_array = load_test_image() 
    od_image_full = image_processing_functions.get_absorption_od_image(test_image_array)
    full_image_sum = image_processing_functions.pixel_sum(od_image_full)
    roi_sum = image_processing_functions.pixel_sum(od_image_full, sum_region = ROI)
    assert np.abs(full_image_sum - FULL_IMAGE_TARGET_SUM) < 1.0
    assert np.abs(roi_sum - ROI_TARGET_SUM) < 1.0