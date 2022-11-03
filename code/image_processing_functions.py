from multiprocessing import Pool
import os 

import numpy as np
from scipy.optimize import fsolve
from scipy import ndimage

from .c_code._polrot_code import ffi as polrot_image_ffi
from .c_code._polrot_code import lib as polrot_image_lib
from . import data_fitting_functions
from .loading_functions import load_experiment_parameters

EXPERIMENT_PARAMETERS = load_experiment_parameters()


"""
Returns an absorption image, i.e. the ratio of the light counts at a given pixel with and without 
atoms, corrected by the dark image counts. If norm_box_coordinates is specified, uses those coordinates
to normalize the image_with_atoms and the image_without_atoms to have the same counts there.

Note: ROI and norm_box use coordinates in the form [x_min, y_min, x_max, y_max]"""
def get_absorption_image(image_stack, ROI = None, norm_box_coordinates = None, clean_strategy = "default_clipped"):
    with_without_light_ratio = _norm_box_helper(image_stack, norm_box_coordinates = norm_box_coordinates)
    dark_subtracted_image_with_atoms, dark_subtracted_image_without_atoms = _roi_crop_helper(image_stack, ROI = ROI)
    absorption_image = dark_subtracted_image_with_atoms / (dark_subtracted_image_without_atoms * with_without_light_ratio)
    absorption_image = _clean_absorption_image(absorption_image, strategy = clean_strategy)
    return absorption_image

def _roi_crop_helper(image_stack, ROI = None):
    image_with_atoms = image_stack[0] 
    image_without_atoms = image_stack[1] 
    image_dark = image_stack[2]
    if(ROI):
        roi_x_min, roi_y_min, roi_x_max, roi_y_max = ROI 
        image_with_atoms_ROI = image_with_atoms[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        image_without_atoms_ROI = image_without_atoms[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        image_dark_ROI = image_dark[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        return (safe_subtract(image_with_atoms_ROI, image_dark_ROI), safe_subtract(image_without_atoms_ROI, image_dark_ROI))
    else:
        return (safe_subtract(image_with_atoms, image_dark),  safe_subtract(image_without_atoms, image_dark))

"""
Convenience function for safely subtracting two arrays of unsigned type.

Minimum cast: A numpy dtype which represents the 'minimal datatype' to which the 
minuend and subtrahend must be cast. For unsigned type, np.byte is the default, 
enforcing a signed type without data loss. 

Please note: if the differences returned by safe_subtract are later manipulated, 
it may be necessary to use a 'larger' minimum cast for safety. np.byte only guarantees 
that no overflows are obtained when two unsigned integers are subtracted."""

def safe_subtract(x, y, minimum_cast = np.byte):
    newtype = np.result_type(x, y, minimum_cast)
    return x.astype(newtype) - y.astype(newtype)

"""
Helper function for finding the multiplier for the without_atoms image to compensate for light intensity shifts.
"""

def _norm_box_helper(image_stack, norm_box_coordinates = None):
    image_with_atoms = image_stack[0] 
    image_without_atoms = image_stack[1] 
    image_dark = image_stack[2]
    if(norm_box_coordinates):
        norm_x_min, norm_y_min, norm_x_max, norm_y_max = norm_box_coordinates
        norm_with_atoms = image_with_atoms[norm_y_min:norm_y_max, norm_x_min:norm_x_max]
        norm_without_atoms = image_without_atoms[norm_y_min:norm_y_max, norm_x_min:norm_x_max] 
        norm_dark = image_dark[norm_y_min:norm_y_max, norm_x_min:norm_x_max]
        with_atoms_light_sum = np.sum(safe_subtract(norm_with_atoms, norm_dark).astype(float))
        without_atoms_light_sum = np.sum(safe_subtract(norm_without_atoms, norm_dark).astype(float))
        with_without_light_ratio = with_atoms_light_sum / without_atoms_light_sum
        return with_without_light_ratio
    else:
        return 1

"""Cleans an absorption image.

strategies:
'default': Uses numpy's nan_to_num, which changes np.nan to 0 and np.inf to a very large number.
'default_clipped': Uses numpy's nan_to_num, but with infinities (and large finite numbers) clipped at 0 and a hard coded upper value slightly > 1.
'none': Does not clean the image."""
def _clean_absorption_image(abs_image, strategy = 'default'):
    if(strategy == "default"):
        return np.nan_to_num(abs_image)
    elif(strategy == 'default_clipped'):
        ABSORPTION_LIMIT = 2.0
        nan_cleaned_image = np.nan_to_num(abs_image)
        return np.clip(nan_cleaned_image, 0, ABSORPTION_LIMIT)
    elif(strategy == 'none'):
        return abs_image


"""
Returns an od image (i.e. -ln(abs_image)) for a given image stack. Essentially wraps -ln(get_absorption_image) with some extra cleaning."""
def get_absorption_od_image(image_stack, ROI = None, norm_box_coordinates = None, abs_clean_strategy = "default_clipped", od_clean_strategy = 'default_clipped'):
    absorption_image = get_absorption_image(image_stack, ROI = ROI, norm_box_coordinates = norm_box_coordinates,
                                                 clean_strategy = abs_clean_strategy)
    od_image_raw = -np.log(absorption_image)
    return _clean_od_image(od_image_raw, strategy = od_clean_strategy)


def _clean_od_image(od_image, strategy = 'default'):
    if(strategy == 'default'):
        return np.nan_to_num(od_image) 
    elif(strategy == 'default_clipped'):
        OD_UPPER_LIMIT = 5
        nan_cleaned_image = np.nan_to_num(od_image) 
        return np.clip(nan_cleaned_image, -np.inf, OD_UPPER_LIMIT)
    elif(strategy == 'none'):
        return od_image


"""
Perform a naive pixel sum over an image.

sum_region: An iterable [x_min, y_min, x_max, y_max] of the min and max x and y-coordinates. If none, defaults 
to summing over entire image."""
def pixel_sum(image, sum_region = None):
    if(sum_region):
        x_min, y_min, x_max, y_max = sum_region 
        cropped_image = image[y_min:y_max, x_min:x_max]
        return sum(sum(cropped_image)) 
    else:
        return sum(sum(image))


def atom_count_pixel_sum(atom_density_image, pixel_area, sum_region = None):
    atom_counts = atom_density_image * pixel_area 
    return pixel_sum(atom_counts, sum_region = sum_region)


"""
Convert an OD absorption image to an array of 2D atom densities, in units of um^{-2}. Wrapper around individual methods which handle saturation etc.
Warning: All inverse-time units are in units of MHz by convention. All length units are in um."""

def get_atom_density_absorption(image_stack, ROI = None, norm_box_coordinates = None, abs_clean_strategy = 'default_clipped', od_clean_strategy = 'default_clipped',
                                flag = 'beer-lambert', detuning = 0, linewidth = None, res_cross_section = None, species = '6Li', saturation_counts = None):
    if not linewidth:
        linewidth = _get_linewidth_from_species(species)
    if not res_cross_section:
        res_cross_section = _get_res_cross_section_from_species(species)
    od_image = get_absorption_od_image(image_stack, ROI = ROI, norm_box_coordinates=norm_box_coordinates, 
                                        abs_clean_strategy=abs_clean_strategy, od_clean_strategy=od_clean_strategy)
    if(flag == 'beer-lambert'):
        return get_atom_density_from_od_image_beer_lambert(od_image, detuning, linewidth, res_cross_section)
    elif(flag == 'sat_beer-lambert'):
        return get_atom_density_from_stack_sat_beer_lambert(image_stack, od_image, detuning, linewidth, res_cross_section, saturation_counts, 
                                                            ROI = ROI, norm_box_coordinates = norm_box_coordinates)
    else:
        raise ValueError("Flag not recognized. Valid options are 'beer-lambert', 'sat_beer-lambert', and 'doppler_beer-lambert'")



def get_atom_density_from_od_image_beer_lambert(od_image, detuning, linewidth, res_cross_section):
    effective_cross_section = res_cross_section / (1 + np.square(2 * detuning / linewidth)) 
    atom_density_um2 = od_image / effective_cross_section 
    return atom_density_um2


def get_atom_density_from_stack_sat_beer_lambert(image_stack, od_image, detuning, linewidth, res_cross_section,
                                                saturation_counts, ROI = None, norm_box_coordinates = None):
    beer_lambert_term = ((1 + np.square(2 * detuning / linewidth)) / res_cross_section ) * od_image
    with_without_light_ratio = _norm_box_helper(image_stack, norm_box_coordinates=norm_box_coordinates)
    with_atoms_dark_subtracted, without_atoms_dark_subtracted = _roi_crop_helper(image_stack, ROI = ROI)
    without_atoms_dark_subtracted = without_atoms_dark_subtracted * with_without_light_ratio
    saturation_term = (1.0 / res_cross_section) * (without_atoms_dark_subtracted - with_atoms_dark_subtracted) / saturation_counts
    return beer_lambert_term + saturation_term

"""
Warning: Due to the underlying C implementation, this does not support vectorized arguments! OD naught vector must be a 1D vector of length 2, 
containing the optical densities of both species, and all others must be scalars!"""
def wrapped_c_polrot_image_function(od_naught_vector, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth, 
                                    intensity_A, intensity_B, intensity_sat, phase_sign):
    wrapped_od_naught = polrot_image_ffi.new("float[]", list(od_naught_vector))
    output_buffer = polrot_image_ffi.new("double[]", 2)
    status_code = polrot_image_lib.give_polrot_image(wrapped_od_naught, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth,
                                                intensity_A, intensity_B, intensity_sat, phase_sign, output_buffer)
    return np.array([output_buffer[0], output_buffer[1]])


"""
Polrot image function, implemented in python. More readable & accessible, but slower."""
def python_polrot_image_function(od_naughts, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth, intensity_A, intensity_B, intensity_sat, 
                                phase_sign):
    od_naught_1, od_naught_2 = od_naughts 
    od_1A = od_naught_1 * od_lorentzian(detuning_1A, linewidth, intensity_A, intensity_sat) 
    od_1B = od_naught_1 * od_lorentzian(detuning_1B, linewidth, intensity_B, intensity_sat)
    od_2A = od_naught_2 * od_lorentzian(detuning_2A, linewidth, intensity_B, intensity_sat) 
    od_2B = od_naught_2 * od_lorentzian(detuning_2B, linewidth, intensity_B, intensity_sat)
    phi_A = (-od_1A * detuning_1A / linewidth - od_2A * detuning_2A / linewidth ) * phase_sign
    phi_B = (-od_1B * detuning_1B / linewidth - od_2B * detuning_2B / linewidth ) * phase_sign
    abs_A = np.exp(-od_1A / 2.0) * np.exp(-od_2A / 2.0) 
    abs_B = np.exp(-od_1B / 2.0) * np.exp(-od_2B / 2.0) 
    result_A = 0.5 + np.square(abs_A) / 2.0 - abs_A * np.sin(phi_A) 
    result_B = 0.5 + np.square(abs_B) / 2.0 - abs_B * np.sin(phi_B) 
    return np.array([result_A, result_B])


def od_lorentzian(detuning, linewidth, intensity, intensity_sat):
    return 1.0 / (1 + np.square(2 * detuning / linewidth) + np.square(intensity / intensity_sat))

def generate_polrot_lookup_table(detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth = None, res_cross_section = None, phase_sign = 1.0, 
                                species = '6Li', num_samps = 1000, abs_min = 0.0, abs_max = 2.0):
    abs_values = np.linspace(abs_min, abs_max, num = num_samps, endpoint = True)
    my_abs_A_grid = np.zeros((num_samps, num_samps)) 
    my_abs_B_grid = np.zeros((num_samps, num_samps)) 
    for i, abs in enumerate(abs_values):
        my_abs_A_grid[i] = abs * np.ones(num_samps) 
        my_abs_B_grid[:, i] = abs * np.ones(num_samps) 
    my_densities_1_grid, my_densities_2_grid = get_atom_density_from_polrot_images(my_abs_A_grid, my_abs_B_grid, detuning_1A, detuning_1B, detuning_2A, 
                                                detuning_2B, linewidth = linewidth, res_cross_section = res_cross_section, phase_sign = phase_sign,
                                                 species = species)
    np.save("Polrot_Lookup_Table.npy", np.stack((my_densities_1_grid, my_densities_2_grid)))
    with open("Polrot_Lookup_Table_Params.txt", 'w') as f:
        f.write("Detuning 1A: " + str(detuning_1A)) 
        f.write("Detuning 1B: " + str(detuning_1B))
        f.write("Detuning_2A: " + str(detuning_2A)) 
        f.write("Detuning 2B: " + str(detuning_2B))
        f.write("Absorption Min: " + str(abs_min)) 
        f.write("Absorption Max: " + str(abs_max)) 
        f.write("Number samples: " + str(num_samps))


#TODO: Multiprocess this for greater speed
def get_polrot_densities_from_lookup_table(abs_image_A, abs_image_B, densities_1_lookup, densities_2_lookup):
    densities_1_array = np.zeros(abs_image_A.shape)
    densities_2_array = np.zeros(abs_image_B.shape)
    for index, abs_A in np.ndenumerate(abs_image_A):
        abs_B = abs_image_B[index] 
        density_1, density_2 = _lookup_pixel_polrot_densities(densities_1_lookup, densities_2_lookup, abs_A, abs_B) 
        densities_1_array[index] = density_1 
        densities_2_array[index] = density_2 
    return (densities_1_array, densities_2_array)


def _lookup_pixel_polrot_densities(densities_1_lookup, densities_2_lookup, abs_A, abs_B, stride = None, min = None):
    if not stride:
        num_samps = len(densities_1_lookup) 
        stride = 2.0 / (num_samps - 1)
    if not min:
        min = 0.0 
    abs_A_index = int(np.round(abs_A / stride))
    abs_B_index = int(np.round(abs_B / stride))
    densities_1_base_value = densities_1_lookup[abs_A_index, abs_B_index] 
    densities_2_base_value = densities_2_lookup[abs_A_index, abs_B_index]
    #Interpolate between points
    a_diff = abs_A - abs_A_index * stride
    b_diff = abs_B - abs_B_index * stride
    if(a_diff < 0):
        densities_1_A_incremented = densities_1_lookup[abs_A_index - 1, abs_B_index]
        densities_2_A_incremented = densities_2_lookup[abs_A_index - 1, abs_B_index]
    elif(a_diff > 0):
        densities_1_A_incremented = densities_1_lookup[abs_A_index + 1, abs_B_index]
        densities_2_A_incremented = densities_2_lookup[abs_A_index + 1, abs_B_index]
    else:
        densities_1_A_incremented = 0
        densities_2_A_incremented = 0
    if(b_diff < 0):
        densities_1_B_incremented = densities_1_lookup[abs_A_index, abs_B_index - 1]
        densities_2_B_incremented = densities_2_lookup[abs_A_index, abs_B_index - 1]
    elif(b_diff > 0):
        densities_1_B_incremented = densities_1_lookup[abs_A_index, abs_B_index + 1]
        densities_2_B_incremented = densities_2_lookup[abs_A_index, abs_B_index + 1]
    else:
        densities_1_B_incremented = 0
        densities_2_B_incremented = 0
    corrected_density_1 = densities_1_base_value + ((densities_1_A_incremented - densities_1_base_value) * np.abs(a_diff) / stride + 
                                                    (densities_1_B_incremented - densities_1_base_value) * np.abs(b_diff) / stride)
    corrected_density_2 = densities_2_base_value + ((densities_2_A_incremented - densities_2_base_value) * np.abs(a_diff) / stride + 
                                                    (densities_2_B_incremented - densities_2_base_value) * np.abs(b_diff) / stride)
    return (corrected_density_1, corrected_density_2)


def get_atom_density_from_polrot_images(abs_image_A, abs_image_B, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth = None,
                                        res_cross_section = None, intensities_A = None, intensities_B = None, intensities_sat = None, phase_sign = 1.0, 
                                        species = '6Li'):
    if not linewidth:
        linewidth = _get_linewidth_from_species(species)
    if not res_cross_section:
        res_cross_section = _get_res_cross_section_from_species(species)
    if (intensities_A or intensities_B or intensities_sat) and not (intensities_A and intensities_B and intensities_sat):
        raise ValueError("Either specify the intensities and saturation intensity or don't; no mixing.")
    if(np.abs(phase_sign) != 1.0):
        raise ValueError("The phase sign must be +-1.")
    atom_densities_list_1 = []
    atom_densities_list_2 = []
    if not intensities_A:
        intensities_A = np.zeros(abs_image_A.shape)
        intensities_B = np.zeros(abs_image_A.shape)
        intensities_sat = np.inf * np.ones(abs_image_A.shape)
    cpu_count = os.cpu_count()
    cpus_to_use = cpu_count // 2
    map_iterator = zip(abs_image_A.flatten(), abs_image_B.flatten(), generator_factory(detuning_1A), 
                        generator_factory(detuning_1B), generator_factory(detuning_2A), generator_factory(detuning_2B), 
                        generator_factory(linewidth), generator_factory(res_cross_section), intensities_A.flatten(), 
                        intensities_B.flatten(), intensities_sat.flatten(), generator_factory(phase_sign))
    with Pool(cpus_to_use) as p:
        for atom_density_1, atom_density_2 in p.starmap(parallelizable_polrot_density_function, map_iterator):
            atom_densities_list_1.append(atom_density_1)
            atom_densities_list_2.append(atom_density_2)
    atom_densities_array_1 = np.reshape(atom_densities_list_1, abs_image_A.shape)
    atom_densities_array_2 = np.reshape(atom_densities_list_2, abs_image_A.shape)
    return (atom_densities_array_1, atom_densities_array_2)

def generator_factory(value):
    while True:
        yield value

def parallelizable_polrot_density_function(absorption_A, absorption_B, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth, 
                                    res_cross_section, intensity_A, intensity_B, intensity_sat, phase_sign):
        solver_extra_args = (detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth, 
                                intensity_A, intensity_B, intensity_sat, phase_sign)
        def current_function(od_naught_vector, *args):
            return wrapped_c_polrot_image_function(od_naught_vector, *args) - np.array([absorption_A, absorption_B])
        root = fsolve(current_function, [0, 0], args = solver_extra_args)
        od_naught_1, od_naught_2 = root 
        atom_density_1 = od_naught_1 / res_cross_section 
        atom_density_2 = od_naught_2 / res_cross_section 
        return (atom_density_1, atom_density_2) 


#Inverse function for simulating polrot images from atom density. Mostly for testing/debugging purposes.
def get_polrot_images_from_atom_density(densities_1, densities_2, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth = None,
                                        res_cross_section = None, intensities_A = None, intensities_B = None, intensities_sat = None,
                                        phase_sign = 1.0, species = '6Li'):
    if not linewidth:
        linewidth = _get_linewidth_from_species(species)
    if not res_cross_section:
        res_cross_section = _get_res_cross_section_from_species(species)
    if (intensities_A or intensities_B or intensities_sat) and not (intensities_A and intensities_B and intensities_sat):
        raise ValueError("Either specify the intensities and saturation intensity or don't; no mixing.")
    image_A_list = []
    image_B_list = []
    if not intensities_A:
        intensities_A = np.zeros(densities_1.shape)
        intensities_B = np.zeros(densities_2.shape)
        intensities_sat = np.inf * np.ones(densities_1.shape)
    cpu_count = os.cpu_count()
    cpus_to_use = cpu_count // 2
    map_iterator = zip(densities_1.flatten(), densities_2.flatten(), generator_factory(detuning_1A), 
                        generator_factory(detuning_1B), generator_factory(detuning_2A), generator_factory(detuning_2B), 
                        generator_factory(linewidth), generator_factory(res_cross_section), intensities_A.flatten(), 
                        intensities_B.flatten(), intensities_sat.flatten(), generator_factory(phase_sign))
    with Pool(cpus_to_use) as p:
        for absorption_A, absorption_B in p.starmap(parallelizable_density_polrot_function, map_iterator):
            image_A_list.append(absorption_A)
            image_B_list.append(absorption_B)
    image_A_array = np.reshape(image_A_list, densities_1.shape)
    image_B_array = np.reshape(image_B_list, densities_1.shape)
    return (image_A_array, image_B_array)

def parallelizable_density_polrot_function(atom_density_1, atom_density_2, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth, 
                                    res_cross_section, intensity_A, intensity_B, intensity_sat, phase_sign):
        extra_args = (detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth, 
                                intensity_A, intensity_B, intensity_sat, phase_sign)
        od_naught_1 = atom_density_1 * res_cross_section
        od_naught_2 = atom_density_2 * res_cross_section
        od_naught_vector = [od_naught_1, od_naught_2] 
        return wrapped_c_polrot_image_function(od_naught_vector, *extra_args)



def get_hybrid_trap_densities_along_harmonic_axis(hybrid_trap_density_image, center = None, axicon_diameter_pix = None, 
                                                axicon_tilt_deg = None, um_per_pixel = None, rotate_data = True):
    if(center is None):
        center = data_fitting_functions.hybrid_trap_center_finder(hybrid_trap_density_image)
    if(axicon_diameter_pix is None):
        axicon_diameter_pix = EXPERIMENT_PARAMETERS["axicon_diameter_pix"]
    if(axicon_tilt_deg is None):
        axicon_tilt_deg = EXPERIMENT_PARAMETERS["axicon_tilt_deg"]
    if(um_per_pixel is None):
        um_per_pixel = EXPERIMENT_PARAMETERS["top_um_per_pixel"]
    if(rotate_data):
        image_to_use, rotated_center = _rotate_and_crop_hybrid_image(hybrid_trap_density_image, center, axicon_tilt_deg)    
        rotated_x_center, rotated_y_center = rotated_center 
        hybrid_trap_radius_um = um_per_pixel * axicon_diameter_pix / 2.0 
        hybrid_trap_cross_sectional_area_um = np.pi * np.square(hybrid_trap_radius_um) 
        radial_axis_index = 1
        hybrid_trap_radial_integrated_density = um_per_pixel * np.sum(image_to_use, axis = radial_axis_index)
        hybrid_trap_3D_density_harmonic_axis = hybrid_trap_radial_integrated_density / hybrid_trap_cross_sectional_area_um 
        harmonic_axis_positions_um = um_per_pixel * (np.arange(len(image_to_use)) - rotated_y_center)
        return (harmonic_axis_positions_um, hybrid_trap_3D_density_harmonic_axis) 
    else:
        raise NotImplementedError("Hybrid analysis without rotating data not yet implemented")

def _rotate_and_crop_hybrid_image(image, center, rotation_angle_deg, x_crop_width = np.inf, y_crop_width = np.inf):
    x_center, y_center = center 
    image_y_width, image_x_width = image.shape
    image_x_center = (image_x_width - 1) / 2.0 
    image_y_center = (image_y_width - 1) / 2.0 
    x_center_diff = x_center - image_x_center 
    y_center_diff = y_center - image_y_center
    rotation_angle_rad = np.pi / 180 * rotation_angle_deg
    rotated_x_center_diff = np.cos(rotation_angle_rad) * x_center_diff + np.sin(rotation_angle_rad) * y_center_diff
    rotated_y_center_diff = np.cos(rotation_angle_rad) * y_center_diff - np.sin(rotation_angle_rad) * x_center_diff 
    rotated_x_center = image_x_center + rotated_x_center_diff 
    rotated_y_center = image_y_center + rotated_y_center_diff 
    rotated_image = ndimage.rotate(image, rotation_angle_deg, reshape = False)
    cropped_y_max = int(min(image_y_width, np.round(rotated_y_center + y_crop_width / 2.0)))
    cropped_y_min = int(max(0, np.round(rotated_y_center - y_crop_width / 2)))
    cropped_x_max = int(min(image_x_width, np.round(rotated_x_center + x_crop_width / 2.0)))
    cropped_x_min = int(max(0, np.round(rotated_x_center - x_crop_width / 2.0)))
    cropped_rotated_image = rotated_image[cropped_y_min:cropped_y_max, cropped_x_min:cropped_x_max] 
    final_x_center = rotated_x_center - cropped_x_min
    final_y_center = rotated_y_center - cropped_y_min
    return (cropped_rotated_image, (final_x_center, final_y_center)) 

def _get_linewidth_from_species(species):
    LI6_NATURAL_LINEWIDTH_MHZ = 5.87
    NA23_NATURAL_LINEWIDTH_MHZ = 9.79
    if(species == "6Li"):
        linewidth = LI6_NATURAL_LINEWIDTH_MHZ
    elif(species == "23Na"):
        linewidth = NA23_NATURAL_LINEWIDTH_MHZ
    else:
        raise ValueError("Species flag not recognized. Valid options are '6Li' and '23Na'.")
    return linewidth

def _get_res_cross_section_from_species(species):
    LI6_RESONANT_CROSS_SECTION_UM2 = 0.2150
    NA23_RESONANT_CROSS_SECTION_UM2 = 0.1657
    if(species == "6Li"):
        res_cross_section = LI6_RESONANT_CROSS_SECTION_UM2
    elif(species == "23Na"):
        res_cross_section = NA23_RESONANT_CROSS_SECTION_UM2
    else:
        raise ValueError("Species flag not recognized. Valid options are '6Li' and '23Na'.")
    return res_cross_section
