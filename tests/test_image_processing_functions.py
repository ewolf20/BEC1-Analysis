import hashlib
import os
import sys

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np 
import scipy


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

RESOURCES_DIRECTORY_PATH = "./resources"

TEST_IMAGE_FILE_PATH = "resources/Test_Measurement_Directory/805277_2022-04-06--8-49-08_Side.fits"
TEST_IMAGE_FILE_NAME = "805277_2022-04-06--8-49-08_Side.fits"
ABSORPTION_NUMPY_ARRAY_FILEPATH = "resources/Test_Image_Absorption.npy" 
OD_NUMPY_ARRAY_FILEPATH = "resources/Test_Image_OD.npy"


from BEC1_Analysis.code import image_processing_functions, data_fitting_functions

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


def test_get_pixel_variance():
    rng_seed = 1337 
    ARRAY_SIZE = 300
    EXPECTED_VARIANCE = 1.01004351474
    rng = np.random.default_rng(seed = rng_seed)
    normals = rng.standard_normal(size = (ARRAY_SIZE, ARRAY_SIZE))
    variance_array = image_processing_functions.get_pixel_variance(normals)
    average_variance = np.average(variance_array) 
    assert np.isclose(average_variance, EXPECTED_VARIANCE)



def test_subcrop():
    overall_image_array = np.arange(25).reshape((5, 5))
    first_crop_coordinates = [1, 1, 4, 4]
    first_crop_xmin, first_crop_ymin, first_crop_xmax, first_crop_ymax = first_crop_coordinates
    first_crop_array = overall_image_array[first_crop_ymin:first_crop_ymax, first_crop_xmin:first_crop_xmax]
    second_crop_coordinates = [2, 1, 4, 3]
    second_crop_xmin, second_crop_ymin, second_crop_xmax, second_crop_ymax = second_crop_coordinates
    second_crop_expected_array = overall_image_array[second_crop_ymin:second_crop_ymax, second_crop_xmin:second_crop_xmax]
    subcrop_array = image_processing_functions.subcrop(first_crop_array, second_crop_coordinates, first_crop_coordinates)
    assert np.array_equal(subcrop_array, second_crop_expected_array)


def test_get_absorption_image():
    ROI = [270, 0, 480, 180]
    norm_box = [300, 250, 400, 300]
    test_image_array = load_test_image()
    absorption_image_full = image_processing_functions.get_absorption_image(test_image_array)
    saved_absorption_image_full = np.load(ABSORPTION_NUMPY_ARRAY_FILEPATH)
    assert np.all(np.abs(absorption_image_full - saved_absorption_image_full) < 1e-4)
    absorption_image_ROI = image_processing_functions.get_absorption_image(test_image_array, ROI = ROI)
    xmin, ymin, xmax, ymax = ROI 
    saved_absorption_image_ROI = saved_absorption_image_full[ymin:ymax, xmin:xmax]
    assert np.all(np.abs(absorption_image_ROI - saved_absorption_image_ROI) < 1e-4)
    absorption_image_ROI_norm = image_processing_functions.get_absorption_image(test_image_array, ROI = ROI, norm_box_coordinates = norm_box)
    norm_xmin, norm_ymin, norm_xmax, norm_ymax = norm_box 
    with_atoms = test_image_array[0] 
    without_atoms = test_image_array[1] 
    dark = test_image_array[2]
    norm_with_atoms_subtracted = with_atoms[norm_ymin:norm_ymax, norm_xmin:norm_xmax] - dark[norm_ymin:norm_ymax, norm_xmin:norm_xmax]
    norm_without_atoms_subtracted = without_atoms[norm_ymin:norm_ymax, norm_xmin:norm_xmax] - dark[norm_ymin:norm_ymax, norm_xmin:norm_xmax]
    norm_ratio = np.sum(norm_with_atoms_subtracted) / np.sum(norm_without_atoms_subtracted)
    unnorm_to_norm_ratio_array = (absorption_image_ROI / absorption_image_ROI_norm).flatten()
    unnorm_to_norm_ratio_array = unnorm_to_norm_ratio_array[~np.isnan(unnorm_to_norm_ratio_array)]
    assert np.all(np.isclose(norm_ratio, unnorm_to_norm_ratio_array))


def test_get_absorption_od_image():
    test_image_array = load_test_image() 
    od_image_full = image_processing_functions.get_absorption_od_image(test_image_array)
    saved_od_image = np.load(OD_NUMPY_ARRAY_FILEPATH)
    assert np.all(np.abs(od_image_full - saved_od_image) < 1e-4)

def test_pixel_sum():
    ROI = [270, 0, 480, 180] 
    ROI_TARGET_SUM = 44835.131246112694
    FULL_IMAGE_TARGET_SUM = 266372.87341004313
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
    ROI = [270, 0, 480, 180] 
    EXPECTED_SUM = 5738896.80
    EXPECTED_DETUNED_SUM = 11734801.39
    EXPECTED_SAT_SUM = 5756960.91
    test_image_array = load_test_image()
    atom_number_image_full = image_processing_functions.get_atom_density_absorption(test_image_array)
    atom_number_image_full_detuned = image_processing_functions.get_atom_density_absorption(test_image_array, detuning = 3)
    atom_number_image_full_sat = image_processing_functions.get_atom_density_absorption(test_image_array, flag = 'sat_beer-lambert', saturation_counts = 1000000)
    atom_number_image_full_geo_adjusted = image_processing_functions.get_atom_density_absorption(test_image_array, cross_section_imaging_geometry_factor = 0.5)
    atom_count = image_processing_functions.atom_count_pixel_sum(atom_number_image_full, 27.52, sum_region = ROI)
    atom_count_detuned = image_processing_functions.atom_count_pixel_sum(atom_number_image_full_detuned, 27.52, sum_region = ROI) 
    atom_count_sat = image_processing_functions.atom_count_pixel_sum(atom_number_image_full_sat, 27.52, sum_region = ROI)
    atom_count_geo_adjusted = image_processing_functions.atom_count_pixel_sum(atom_number_image_full_geo_adjusted, 27.52, sum_region = ROI)
    assert np.abs(atom_count - EXPECTED_SUM) < 0.01 
    assert np.abs(atom_count_detuned - EXPECTED_DETUNED_SUM) < 0.01 
    assert np.abs(atom_count_sat - EXPECTED_SAT_SUM) < 0.01
    assert np.abs(atom_count_geo_adjusted - 2 *EXPECTED_SUM) < 0.01


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
    return image_processing_functions.get_polrot_images_from_atom_density(fake_density_1, fake_density_2, POLROT_DETUNING_1A, POLROT_DETUNING_1B,
                                                        POLROT_DETUNING_2A, POLROT_DETUNING_2B, phase_sign = -1.0)
"""
Makes sure that the polrot _generation_, and thus the base polrot image function, hasn't changed"""
def test_polrot_images_function():
    image_A, image_B = _generate_fake_polrot_images()
    saved_image_A = np.load(os.path.join(RESOURCES_DIRECTORY_PATH, "Fake_Polrot_Image_A.npy"))
    saved_image_B = np.load(os.path.join(RESOURCES_DIRECTORY_PATH, "Fake_Polrot_Image_B.npy"))
    assert np.all(np.abs(saved_image_A - image_A) < 1e-6)
    assert np.all(np.abs(saved_image_B - image_B) < 1e-6)


def test_python_polrot_image_function():
    REF_OD_1 = 1.2 
    REF_OD_2 = 3.6
    REF_LINEWIDTH = 3 
    REF_DETUNING_1A = 5
    REF_DETUNING_1B = 7
    REF_DETUNING_2A = 9
    REF_DETUNING_2B = 11 
    REF_INTENSITY_A = 0.7
    REF_INTENSITY_B = 1.6
    REF_INTENSITY_SAT = 2 
    REF_PHASE_SIGN = 1.0

    EXPECTED_RESULT_A = 1.3071046
    EXPECTED_RESULT_B = 1.2745582

    calculated_result_A, calculated_result_B = image_processing_functions.python_polrot_image_function(
                    (REF_OD_1, REF_OD_2), REF_DETUNING_1A, REF_DETUNING_1B, REF_DETUNING_2A, REF_DETUNING_2B, 
                    REF_LINEWIDTH, REF_INTENSITY_A, REF_INTENSITY_B, REF_INTENSITY_SAT, REF_PHASE_SIGN
    )
    assert np.isclose(EXPECTED_RESULT_A, calculated_result_A) 
    assert np.isclose(EXPECTED_RESULT_B, calculated_result_B)

def test_python_and_jit_polrot_image_function_agreement():
    REF_OD_1 = 1.2 
    REF_OD_2 = 3.6
    REF_LINEWIDTH = 3 
    REF_DETUNING_1A = 5
    REF_DETUNING_1B = 7
    REF_DETUNING_2A = 9
    REF_DETUNING_2B = 11 
    REF_INTENSITY_A = 0.7
    REF_INTENSITY_B = 1.6
    REF_INTENSITY_SAT = 2 
    REF_PHASE_SIGN = 1.0

    calculated_result_A_python, calculated_result_B_python = image_processing_functions.python_polrot_image_function(
                    (REF_OD_1, REF_OD_2), REF_DETUNING_1A, REF_DETUNING_1B, REF_DETUNING_2A, REF_DETUNING_2B, 
                    REF_LINEWIDTH, REF_INTENSITY_A, REF_INTENSITY_B, REF_INTENSITY_SAT, REF_PHASE_SIGN
    )

    calculated_result_A_compiled, calculated_result_B_compiled = image_processing_functions._compiled_python_polrot_image_function(
                    (REF_OD_1, REF_OD_2), REF_DETUNING_1A, REF_DETUNING_1B, REF_DETUNING_2A, REF_DETUNING_2B, 
                    REF_LINEWIDTH, REF_INTENSITY_A, REF_INTENSITY_B, REF_INTENSITY_SAT, REF_PHASE_SIGN
    )


def test_get_atom_density_from_polrot_images():
    fake_image_A, fake_image_B = _generate_fake_polrot_images()
    reconstructed_density_1, reconstructed_density_2 = image_processing_functions.get_atom_density_from_polrot_images(fake_image_A, fake_image_B, 
                                                                                                                    POLROT_DETUNING_1A, POLROT_DETUNING_1B,
                                                                                                                    POLROT_DETUNING_2A, POLROT_DETUNING_2B,
                                                                                                                    phase_sign = -1.0)
    saved_density_1 = np.load(os.path.join(RESOURCES_DIRECTORY_PATH, "Fake_Polrot_Atom_Density_1.npy"))
    saved_density_2 = np.load(os.path.join(RESOURCES_DIRECTORY_PATH, "Fake_Polrot_Atom_Density_2.npy"))
    assert np.all(np.abs(saved_density_1 - reconstructed_density_1) < 1e-4)
    assert np.all(np.abs(saved_density_2 - reconstructed_density_2) < 1e-4)
    reconstructed_geo_adjusted_density_1, reconstructed_geo_adjusted_density_2 = image_processing_functions.get_atom_density_from_polrot_images(
                                                                            fake_image_A, fake_image_B, POLROT_DETUNING_1A, POLROT_DETUNING_1B, 
                                                                            POLROT_DETUNING_2A, POLROT_DETUNING_2B, phase_sign = -1.0, 
                                                                            cross_section_imaging_geometry_factor = 0.5)
    assert np.all(np.abs(2 * saved_density_1 - reconstructed_geo_adjusted_density_1) < 1e-4)
    assert np.all(np.abs(2 * saved_density_2 - reconstructed_geo_adjusted_density_2) < 1e-4)  


def test_generate_polrot_lookup_table():
    try:
        image_processing_functions.generate_polrot_lookup_table(POLROT_DETUNING_1A, POLROT_DETUNING_1B, POLROT_DETUNING_2A, POLROT_DETUNING_2B, phase_sign = -1.0,
                                                            num_samps = 100)
        generated_array = np.load("Polrot_Lookup_Table.npy") 
        stored_array = np.load(os.path.join(RESOURCES_DIRECTORY_PATH, "Polrot_Lookup_Table_Small.npy")) 
        print(generated_array - stored_array)
        assert np.all(np.isclose(generated_array, stored_array, rtol = 1e-3, atol = 1e-2))
    finally:
        os.remove("Polrot_Lookup_Table.npy") 
        os.remove("Polrot_Lookup_Table_Params.txt") 


def test_get_polrot_densities_from_lookup_table():
    fake_image_A, fake_image_B = _generate_fake_polrot_images() 
    densities_lookup_1, densities_lookup_2 = np.load(os.path.join(RESOURCES_DIRECTORY_PATH, "Polrot_Lookup_Table_Full.npy"))
    reconstructed_density_1, reconstructed_density_2 = image_processing_functions.get_polrot_densities_from_lookup_table(fake_image_A, fake_image_B, 
                                                                                                                        densities_lookup_1, densities_lookup_2) 
    saved_density_1 = np.load(os.path.join(RESOURCES_DIRECTORY_PATH, "Fake_Polrot_Atom_Density_1.npy"))
    saved_density_2 = np.load(os.path.join(RESOURCES_DIRECTORY_PATH, "Fake_Polrot_Atom_Density_2.npy"))
    assert np.all(np.abs(saved_density_1 - reconstructed_density_1) < 1e-3)
    assert np.all(np.abs(saved_density_2 - reconstructed_density_2) < 1e-3)


def test_rotate_and_crop_hybrid_image():
    X_SIZE = 300 
    Y_SIZE = 300 
    X_CENTER = 110 
    Y_CENTER = 203 
    GAUSSIAN_X_WIDTH = 50
    GAUSSIAN_Y_WIDTH = 20
    center = (X_CENTER, Y_CENTER)
    y_indices, x_indices = np.mgrid[0:Y_SIZE, 0:X_SIZE]
    gaussian_data = data_fitting_functions.two_dimensional_gaussian(x_indices, y_indices, 1.0, X_CENTER, Y_CENTER, GAUSSIAN_X_WIDTH, GAUSSIAN_Y_WIDTH, 0)
    X_CROP_WIDTH = 100 
    Y_CROP_WIDTH = 150
    cropped_rotated_image, cropped_rotated_center = image_processing_functions._rotate_and_crop_hybrid_image(gaussian_data, center, 90, 
                                                                                x_crop_width = X_CROP_WIDTH, y_crop_width = Y_CROP_WIDTH)
    rotated_x_center, rotated_y_center = cropped_rotated_center 
    assert (np.abs(rotated_x_center - X_CROP_WIDTH / 2.0 < 1e-3))
    assert (np.abs(rotated_y_center - Y_CROP_WIDTH / 2.0 < 1e-3))
    EXPECTED_COUNTS_SUM = 5375.8749
    counts_sum = np.sum(cropped_rotated_image)
    assert (np.abs(counts_sum - EXPECTED_COUNTS_SUM) < 1e-3)


def test_get_hybrid_trap_densities_along_harmonic_axis():
    EXPECTED_X_CENTER = 228 
    EXPECTED_Y_CENTER = 398
    SAMPLE_UM_PER_PIXEL = 0.6
    SAMPLE_AXICON_DIAMETER_PIX = 189
    SAMPLE_AXICON_LENGTH_PIX = 250
    SAMPLE_TILT_DEG = 6.3
    SAMPLE_SIDE_TILT_DEG = 10
    SAMPLE_SIDE_ASPECT_RATIO = 1.4
    axicon_diameter_um = SAMPLE_UM_PER_PIXEL * SAMPLE_AXICON_DIAMETER_PIX
    sample_hybrid_trap_data = np.load('resources/Sample_Box_Exp.npy')
    hybrid_trap_harmonic_positions, hybrid_trap_harmonic_data = image_processing_functions.get_hybrid_trap_densities_along_harmonic_axis(sample_hybrid_trap_data, 
                                                                SAMPLE_TILT_DEG, SAMPLE_AXICON_DIAMETER_PIX, SAMPLE_AXICON_LENGTH_PIX,
                                                                SAMPLE_SIDE_TILT_DEG, SAMPLE_SIDE_ASPECT_RATIO, SAMPLE_UM_PER_PIXEL)
    filtered_hybrid_trap_harmonic_data = scipy.signal.savgol_filter(hybrid_trap_harmonic_data, 15, 2)
    max_index = np.argmax(hybrid_trap_harmonic_data) 
    max_value = hybrid_trap_harmonic_data[max_index]
    CENTER_SNIPPET_HALF_WIDTH = 10
    center_snippet = sample_hybrid_trap_data[EXPECTED_Y_CENTER - CENTER_SNIPPET_HALF_WIDTH:EXPECTED_Y_CENTER+CENTER_SNIPPET_HALF_WIDTH, 
                                            EXPECTED_X_CENTER - CENTER_SNIPPET_HALF_WIDTH: EXPECTED_X_CENTER + CENTER_SNIPPET_HALF_WIDTH]
    center_snippet_average_2d_density = np.sum(center_snippet) / center_snippet.size
    center_snippet_average_3d_density = center_snippet_average_2d_density / (SAMPLE_UM_PER_PIXEL * SAMPLE_AXICON_DIAMETER_PIX)
    max_index = np.argmax(filtered_hybrid_trap_harmonic_data)
    max_value = filtered_hybrid_trap_harmonic_data[max_index] 
    max_position = hybrid_trap_harmonic_positions[max_index] 
    assert(np.abs(max_position) < 10) 
    assert(np.abs((center_snippet_average_3d_density - max_value) / center_snippet_average_3d_density < 1e-1))


def test_get_hybrid_cross_section_um():
    SAMPLE_SIDE_ANGLE_DEG = 45
    SAMPLE_SIDE_ASPECT_RATIO = 2 
    SAMPLE_TOP_RADIUS_UM = 100
    #From hand-evaluation of the formula
    EXPECTED_CROSS_SECTION_UM2 = 25132.74123
    cross_section = image_processing_functions.get_hybrid_cross_section_um(SAMPLE_TOP_RADIUS_UM, SAMPLE_SIDE_ANGLE_DEG, SAMPLE_SIDE_ASPECT_RATIO)
    assert np.isclose(cross_section, EXPECTED_CROSS_SECTION_UM2)


def test_bin_and_average_data():
    DATA_LENGTH_1D = 24
    BIN_SIZE_1D_EVEN = 3
    BIN_SIZE_1D_UNEVEN = 5
    def validate_data_1d(data_length, bin_size, rebinned_data):
        expected_rebinned_data = (bin_size - 1) / 2 + bin_size * np.arange(data_length // bin_size)
        assert np.array_equal(rebinned_data, expected_rebinned_data)
    data_to_bin_1d = np.arange(DATA_LENGTH_1D) 
    rebinned_data_even = image_processing_functions.bin_and_average_data(data_to_bin_1d, BIN_SIZE_1D_EVEN)
    validate_data_1d(DATA_LENGTH_1D, BIN_SIZE_1D_EVEN, rebinned_data_even)
    rebinned_data_uneven = image_processing_functions.bin_and_average_data(data_to_bin_1d, BIN_SIZE_1D_UNEVEN)
    validate_data_1d(DATA_LENGTH_1D, BIN_SIZE_1D_UNEVEN, rebinned_data_uneven)
    DATA_LENGTH_2D = 96
    DATA_SHAPE_2D = (8, 12)
    data_to_bin_2d = np.arange(DATA_LENGTH_2D).reshape(DATA_SHAPE_2D)
    data_shape_2d = data_to_bin_2d.shape
    BIN_SIZE_2D_SINGLE_EVEN = 2 
    BIN_SIZE_2D_SINGLE_UNEVEN = 5
    BIN_SIZE_2D_TUPLE_EVEN = (2, 3) 
    BIN_SIZE_2D_TUPLE_UNEVEN = (3, 5)
    def validate_data_2d(data_shape, bin_shape, rebinned_data):
        data_dim_0, data_dim_1 = data_shape 
        bin_dim_0, bin_dim_1 = bin_shape 
        constant_offset = (bin_dim_0 - 1) / 2 * data_dim_1 + (bin_dim_1 - 1)/2 
        base_array = np.expand_dims(bin_dim_0 * data_dim_1 * np.arange(data_dim_0 // bin_dim_0), axis = 1) + bin_dim_1 * np.arange(data_dim_1 // bin_dim_1)
        expected_rebinned_data = constant_offset + base_array
        assert np.array_equal(rebinned_data, expected_rebinned_data)
    rebinned_data_2d_single_even = image_processing_functions.bin_and_average_data(data_to_bin_2d, BIN_SIZE_2D_SINGLE_EVEN)
    validate_data_2d(data_shape_2d, (BIN_SIZE_2D_SINGLE_EVEN, BIN_SIZE_2D_SINGLE_EVEN), rebinned_data_2d_single_even)
    rebinned_data_2d_single_uneven = image_processing_functions.bin_and_average_data(data_to_bin_2d, BIN_SIZE_2D_SINGLE_UNEVEN)
    validate_data_2d(data_shape_2d, (BIN_SIZE_2D_SINGLE_UNEVEN, BIN_SIZE_2D_SINGLE_UNEVEN), rebinned_data_2d_single_uneven)
    rebinned_data_2d_tuple_even = image_processing_functions.bin_and_average_data(data_to_bin_2d, BIN_SIZE_2D_TUPLE_EVEN)
    validate_data_2d(data_shape_2d, BIN_SIZE_2D_TUPLE_EVEN, rebinned_data_2d_tuple_even)
    rebinned_data_2d_tuple_uneven = image_processing_functions.bin_and_average_data(data_to_bin_2d, BIN_SIZE_2D_TUPLE_UNEVEN)
    validate_data_2d(data_shape_2d, BIN_SIZE_2D_TUPLE_UNEVEN, rebinned_data_2d_tuple_uneven) 
    #Check axis omission
    data_to_bin_1d_omissions = np.reshape(np.arange(4), (4, 1)) * np.arange(3)
    rebinned_data_1d_omissions = image_processing_functions.bin_and_average_data(data_to_bin_1d_omissions, 3, omitted_axes = 0)
    expected_rebinned_data_1d_omissions = np.reshape(np.arange(4), (4, 1))
    assert np.array_equal(expected_rebinned_data_1d_omissions, rebinned_data_1d_omissions)
    other_axis_rebinned_data_1d_omissions = image_processing_functions.bin_and_average_data(data_to_bin_1d_omissions, 4, omitted_axes = 1)
    expected_other_axis_rebinned_data_1d_omissions = 1.5 * np.reshape(np.arange(3), (1, 3)) 
    assert np.array_equal(other_axis_rebinned_data_1d_omissions, expected_other_axis_rebinned_data_1d_omissions)
    
def test_get_saturation_counts_from_camera_parameters():
    SAMPLE_CAMERA_COUNTS_TO_PHOTONS_FACTOR = 1.2
    SAMPLE_PIXEL_LENGTH_M = 1e-6 
    SAMPLE_IMAGING_TIME_S = 1e-4
    SAMPLE_LINEWIDTH_Hz = 1e6 
    SAMPLE_RES_CROSS_SECTION_M = 1e-13
    sample_saturation_counts = image_processing_functions.get_saturation_counts_from_camera_parameters(SAMPLE_PIXEL_LENGTH_M, SAMPLE_IMAGING_TIME_S, 
                                                                                                    SAMPLE_CAMERA_COUNTS_TO_PHOTONS_FACTOR, 
                                                                                                    SAMPLE_LINEWIDTH_Hz, SAMPLE_RES_CROSS_SECTION_M)
    #Verified to be the correct result for this set of input parameters
    EXPECTED_SAMPLE_SATURATION_COUNTS = 2618.00
    assert np.isclose(sample_saturation_counts, EXPECTED_SAMPLE_SATURATION_COUNTS)






    

