import hashlib
import os 
import sys

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np 


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

RESOURCES_DIRECTORY_PATH = "./resources"

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
    ROI_SHA_CHECKSUM = '10f2035f854f69996ec888e758ae3a9dbbd92b8fa781111f7206261e32fdda85'
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


def test_atom_count_pixel_sum():
    test_array = np.ones((20, 20)) 
    test_pixel_size = 3
    assert np.abs(image_processing_functions.atom_count_pixel_sum(test_array, test_pixel_size) - 1200) < 1.0


def test_get_atom_density_absorption():
    test_image_array = load_test_image()
    IMAGE_SHA_CHECKSUM = '48a07f3605fee705cdad24be32ea91f0d04a212a12bacaba43f91d3db29ab10f'
    IMAGE_DETUNED_SHA_CHECKSUM = '51c2741068278bdd5bc8fb9c3af128037823460e1cab45247d29597e36ee33d5'
    IMAGE_SATURATED_SHA_CHECKSUM = '63d3ba5bef27273df22244c14277e2a2e81bea3a3038854c1de5a5ba28d85801'
    atom_number_image_full = image_processing_functions.get_atom_density_absorption(test_image_array)
    atom_number_image_full_detuned = image_processing_functions.get_atom_density_absorption(test_image_array, detuning = 3)
    atom_number_image_full_sat = image_processing_functions.get_atom_density_absorption(test_image_array, flag = 'sat_beer-lambert', saturation_counts = 1000000)
    assert check_sha_hash(atom_number_image_full.data.tobytes(), IMAGE_SHA_CHECKSUM)
    assert check_sha_hash(atom_number_image_full_detuned.data.tobytes(), IMAGE_DETUNED_SHA_CHECKSUM)
    assert check_sha_hash(atom_number_image_full_sat.data.tobytes(), IMAGE_SATURATED_SHA_CHECKSUM)


POLROT_DETUNING_1A = 5
POLROT_DETUNING_2A = -10
POLROT_DETUNING_1B = 10
POLROT_DETUNING_2B = -5

def _generate_fake_polrot_images():
    IMAGE_PIXEL_SIZE = 300
    SIGMA_1 = 2.0 
    SIGMA_2 = 1.3
    li_cross_section = image_processing_functions._get_res_cross_section_from_species('6Li')
    li_linewidth = image_processing_functions._get_linewidth_from_species('6Li')
    fake_image_x = np.linspace(-5, 5, IMAGE_PIXEL_SIZE)
    fake_image_y = np.linspace(-5, 5, IMAGE_PIXEL_SIZE)
    fake_image_x_grid, fake_image_y_grid = np.meshgrid(fake_image_x, fake_image_y) 
    def gaussian_density_function(x, y, sigma):
        return np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma))) 
    fake_density_1 = gaussian_density_function(fake_image_x_grid, fake_image_y_grid, SIGMA_1)
    fake_density_2 = gaussian_density_function(fake_image_x_grid, fake_image_y_grid, SIGMA_2)
    fake_od_naught_1 = fake_density_1 * li_cross_section
    fake_od_naught_2 = fake_density_2 * li_cross_section
    polrot_image_function = image_processing_functions.wrapped_polrot_image_function
    image_A = np.zeros(fake_image_x_grid.shape) 
    image_B = np.zeros(fake_image_x_grid.shape) 
    for i in range(IMAGE_PIXEL_SIZE):
        for j in range(IMAGE_PIXEL_SIZE):
            fake_od_naught_1_pixel = fake_od_naught_1[i][j] 
            fake_od_naught_2_pixel = fake_od_naught_2[i][j]
            fake_od_pixel_array = np.array([fake_od_naught_1_pixel, fake_od_naught_2_pixel])
            image_A_pixel, image_B_pixel = polrot_image_function(fake_od_pixel_array, POLROT_DETUNING_1A, POLROT_DETUNING_1B, 
                                                            POLROT_DETUNING_2A, POLROT_DETUNING_2B, li_linewidth, 0, 0, np.inf)
            image_A[i][j] = image_A_pixel
            image_B[i][j] = image_B_pixel
    return (image_A, image_B)

"""
Makes sure that the polrot _generation_, and thus the base polrot image function, hasn't changed"""
def test_polrot_images_function():
    image_A, image_B = _generate_fake_polrot_images()
    saved_image_A = np.load(os.path.join(RESOURCES_DIRECTORY_PATH, "Fake_Polrot_Image_A.npy"))
    saved_image_B = np.load(os.path.join(RESOURCES_DIRECTORY_PATH, "Fake_Polrot_Image_B.npy"))
    assert np.all(np.abs(saved_image_A - image_A) < 1e-6)
    assert np.all(np.abs(saved_image_B - image_B) < 1e-6)


    

def test_get_atom_density_from_polrot_images():
    fake_image_A, fake_image_B = _generate_fake_polrot_images()
    reconstructed_density_1, reconstructed_density_2 = image_processing_functions.get_atom_density_from_polrot_images(fake_image_A, fake_image_B, 
                                                                                                                    POLROT_DETUNING_1A, POLROT_DETUNING_1B,
                                                                                                                    POLROT_DETUNING_2A, POLROT_DETUNING_2B)
    saved_density_1 = np.load(os.path.join(RESOURCES_DIRECTORY_PATH, "Fake_Polrot_Atom_Density_1.npy"))
    saved_density_2 = np.load(os.path.join(RESOURCES_DIRECTORY_PATH, "Fake_Polrot_Atom_Density_2.npy"))
    assert np.all(np.abs(saved_density_1 - reconstructed_density_1) < 1e-4) 
    assert np.all(np.abs(saved_density_2 - reconstructed_density_2) < 1e-4)
    

