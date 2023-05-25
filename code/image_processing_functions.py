import os 

from numba import jit
import numpy as np
from scipy.optimize import fsolve
from scipy import ndimage

from . import data_fitting_functions



"""
Returns the local variance in a 2d input array.

Given a 2d input array and a square size kernel_size, computes the local variance of the pixels 
within a square centered on each pixel in the array and of size kernel_size."""

def get_pixel_variance(input_image, kernel_size = 3):
    kernel = (1.0 / np.square(kernel_size)) * np.ones((kernel_size, kernel_size))
    locally_averaged_image = ndimage.convolve(input_image, kernel)
    #Technically incorrect procedure, but ad hoc numerically verified to give pretty damn close 
    #to an unbiased answer for uniformly-centered randomly distributed input
    diff_image = input_image - locally_averaged_image 
    diff_squared_image = np.square(diff_image) 
    variance_kernel = 1.0 / (np.square(kernel_size) - 1) * np.ones((kernel_size, kernel_size))
    variance_array = ndimage.convolve(diff_squared_image, variance_kernel) 
    return variance_array

"""
Convenience function for sub-cropping boxes.

Given an array image_array which represents a crop of a larger overall image into coordinates 
specified by overall-box, and a crop_box which specifies a crop of the 
overall image, in the original coordinates, contained entirely within image_array, 
return that crop of the image.

Note: As below, crop_box and overall_box use coordinates in the form [xmin, ymin, xmax, ymax]"""
def subcrop(image_array, crop_box, overall_box):
    overall_xmin, overall_ymin, overall_xmax, overall_ymax = overall_box 
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_box 
    new_crop_xmin = crop_xmin - overall_xmin 
    new_crop_ymin = crop_ymin - overall_ymin 
    new_crop_xmax = crop_xmax - overall_xmin 
    new_crop_ymax = crop_ymax - overall_ymin 
    return image_array[new_crop_ymin:new_crop_ymax, new_crop_xmin:new_crop_xmax]

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
Warning: All inverse-time units are in units of MHz by convention. All length units are in um.

Remark: res_cross_section is _the_ resonant cross section for the transition - i.e. that of a cycling transition driven with the correct polarization, 
i.e. 6 * pi * lambda_bar^2. cross_section_multiplier is provided for accommodating deviations from this thanks to imaging geometry."""

def get_atom_density_absorption(image_stack, ROI = None, norm_box_coordinates = None, abs_clean_strategy = 'default_clipped', od_clean_strategy = 'default_clipped',
                                flag = 'beer-lambert', detuning = 0, linewidth = None, res_cross_section = None, species = '6Li', saturation_counts = None, 
                                cross_section_imaging_geometry_factor = 1.0):
    if not linewidth:
        linewidth = _get_linewidth_from_species(species)
    if not res_cross_section:
        res_cross_section = _get_res_cross_section_from_species(species)
    geometry_adjusted_cross_section = res_cross_section * cross_section_imaging_geometry_factor
    od_image = get_absorption_od_image(image_stack, ROI = ROI, norm_box_coordinates=norm_box_coordinates, 
                                        abs_clean_strategy=abs_clean_strategy, od_clean_strategy=od_clean_strategy)
    if(flag == 'beer-lambert'):
        return get_atom_density_from_od_image_beer_lambert(od_image, detuning, linewidth, geometry_adjusted_cross_section)
    elif(flag == 'sat_beer-lambert'):
        return get_atom_density_from_stack_sat_beer_lambert(image_stack, od_image, detuning, linewidth, geometry_adjusted_cross_section, saturation_counts, 
                                                            ROI = ROI, norm_box_coordinates = norm_box_coordinates)
    else:
        raise ValueError("Flag not recognized. Valid options are 'beer-lambert', 'sat_beer-lambert', and 'doppler_beer-lambert'")



def get_atom_density_from_od_image_beer_lambert(od_image, detuning, linewidth, on_resonance_cross_section):
    effective_cross_section = on_resonance_cross_section / (1 + np.square(2 * detuning / linewidth)) 
    atom_density_um2 = od_image / effective_cross_section 
    return atom_density_um2


def get_atom_density_from_stack_sat_beer_lambert(image_stack, od_image, detuning, linewidth, on_resonance_cross_section,
                                                saturation_counts, ROI = None, norm_box_coordinates = None):
    beer_lambert_term = ((1 + np.square(2 * detuning / linewidth)) / on_resonance_cross_section ) * od_image
    with_without_light_ratio = _norm_box_helper(image_stack, norm_box_coordinates=norm_box_coordinates)
    with_atoms_dark_subtracted, without_atoms_dark_subtracted = _roi_crop_helper(image_stack, ROI = ROI)
    without_atoms_dark_subtracted = without_atoms_dark_subtracted * with_without_light_ratio
    saturation_term = (1.0 / on_resonance_cross_section) * (without_atoms_dark_subtracted - with_atoms_dark_subtracted) / saturation_counts
    return beer_lambert_term + saturation_term


@jit(nopython = True)
def _python_polrot_image_function_with_target_offset(od_naught_vector, abs_A, abs_B, detuning_1A, detuning_1B, detuning_2A, detuning_2B, 
                                        linewidth, intensity_A, intensity_B, intensity_sat, phase_sign):
    fun_val = _compiled_python_polrot_image_function(od_naught_vector, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth,
                                        intensity_A, intensity_B, intensity_sat, phase_sign)
    fun_val[0] -= abs_A 
    fun_val[1] -= abs_B
    return fun_val

"""
Polrot image function, implemented in python. More readable & accessible, but slower."""
@jit(nopython = True)
def _compiled_python_polrot_image_function(od_naughts, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth,
                                             intensity_A, intensity_B, intensity_sat, phase_sign):
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


def python_polrot_image_function(od_naughts, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth,
                                             intensity_A, intensity_B, intensity_sat, phase_sign):
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


@jit(nopython = True)
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
                                        res_cross_section = None, cross_section_imaging_geometry_factor = 1.0, 
                                        intensities_A = None, intensities_B = None, intensities_sat = None, phase_sign = 1.0, 
                                        species = '6Li'):
    if not linewidth:
        linewidth = _get_linewidth_from_species(species)
    if not res_cross_section:
        res_cross_section = _get_res_cross_section_from_species(species)
    geometry_adjusted_cross_section = res_cross_section * cross_section_imaging_geometry_factor
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
    map_iterator = zip(abs_image_A.flatten(), abs_image_B.flatten(), generator_factory(detuning_1A), 
                        generator_factory(detuning_1B), generator_factory(detuning_2A), generator_factory(detuning_2B), 
                        generator_factory(linewidth), generator_factory(geometry_adjusted_cross_section), intensities_A.flatten(), 
                        intensities_B.flatten(), intensities_sat.flatten(), generator_factory(phase_sign))
    for itr_val in map_iterator:
        atom_density_1, atom_density_2 = parallelizable_polrot_density_function(*itr_val)
        atom_densities_list_1.append(atom_density_1)
        atom_densities_list_2.append(atom_density_2)
    atom_densities_array_1 = np.reshape(atom_densities_list_1, abs_image_A.shape)
    atom_densities_array_2 = np.reshape(atom_densities_list_2, abs_image_A.shape)
    return (atom_densities_array_1, atom_densities_array_2)

def generator_factory(value):
    while True:
        yield value

def parallelizable_polrot_density_function(absorption_A, absorption_B, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth, 
                                    on_resonance_cross_section, intensity_A, intensity_B, intensity_sat, phase_sign):
        solver_extra_args = (absorption_A, absorption_B, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth, 
                                intensity_A, intensity_B, intensity_sat, phase_sign)
        root = fsolve(_python_polrot_image_function_with_target_offset, [0, 0], args = solver_extra_args)
        od_naught_1, od_naught_2 = root 
        atom_density_1 = od_naught_1 / on_resonance_cross_section 
        atom_density_2 = od_naught_2 / on_resonance_cross_section 
        return (atom_density_1, atom_density_2) 


#Inverse function for simulating polrot images from atom density. Mostly for testing/debugging purposes.
def get_polrot_images_from_atom_density(densities_1, densities_2, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth = None,
                                        res_cross_section = None, cross_section_imaging_geometry_factor = 1.0,
                                        intensities_A = None, intensities_B = None, intensities_sat = None,
                                        phase_sign = 1.0, species = '6Li'):
    if not linewidth:
        linewidth = _get_linewidth_from_species(species)
    if not res_cross_section:
        res_cross_section = _get_res_cross_section_from_species(species)
    geometry_adjusted_cross_section = res_cross_section * cross_section_imaging_geometry_factor
    if (intensities_A or intensities_B or intensities_sat) and not (intensities_A and intensities_B and intensities_sat):
        raise ValueError("Either specify the intensities and saturation intensity or don't; no mixing.")
    if not intensities_A:
        intensities_A = np.zeros(densities_1.shape)
        intensities_B = np.zeros(densities_2.shape)
        intensities_sat = np.inf * np.ones(densities_1.shape)
    od_naught_1 = densities_1 * geometry_adjusted_cross_section
    od_naught_2 = densities_2 * geometry_adjusted_cross_section
    return python_polrot_image_function(np.array([od_naught_1, od_naught_2]), detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth, 
                            intensities_A, intensities_B, intensities_sat, phase_sign)



def get_hybrid_trap_densities_along_harmonic_axis(hybrid_trap_density_image, axicon_tilt_deg, axicon_diameter_pix, axicon_length_pix,
                                                axicon_side_angle_deg, side_aspect_ratio,
                                                um_per_pixel, center = None, rotate_data = True):
    if(center is None):
        center = data_fitting_functions.hybrid_trap_center_finder(hybrid_trap_density_image, axicon_tilt_deg, axicon_diameter_pix, axicon_length_pix)
    if(rotate_data and axicon_tilt_deg != 0.0):
        image_to_use, new_center = _rotate_and_crop_hybrid_image(hybrid_trap_density_image, center, axicon_tilt_deg)
        center = new_center 
    else:
        image_to_use = hybrid_trap_density_image
    x_center, y_center = center
    hybrid_trap_radius_um = um_per_pixel * axicon_diameter_pix / 2.0
    hybrid_trap_cross_sectional_area_um = get_hybrid_cross_section_um(hybrid_trap_radius_um, axicon_side_angle_deg, side_aspect_ratio)
    radial_axis_index = 1
    hybrid_trap_radial_integrated_density = um_per_pixel * np.sum(image_to_use, axis = radial_axis_index)
    hybrid_trap_3D_density_harmonic_axis = hybrid_trap_radial_integrated_density / hybrid_trap_cross_sectional_area_um 
    harmonic_axis_positions_um = um_per_pixel * (np.arange(len(image_to_use)) - y_center)
    #Fit lorentzian to the 1D-integrated data to improve center-finding fidelity
    results = data_fitting_functions.fit_lorentzian_with_offset(harmonic_axis_positions_um, hybrid_trap_3D_density_harmonic_axis)
    popt, pcov = results
    amp, center, gamma, offset = popt
    refitted_harmonic_axis_positions_um = harmonic_axis_positions_um - center
    return (refitted_harmonic_axis_positions_um, hybrid_trap_3D_density_harmonic_axis)


"""
Convenience function for getting the actual areal cross section of the tilted oval of the hybrid trap.
Note that the tilt angle is the angle made by the semimajor axis of the oval to the plane that the top imaging can see."""
def get_hybrid_cross_section_um(top_radius_um, side_angle_deg, side_aspect_ratio):
    side_angle_rad = side_angle_deg * np.pi / 180 
    theta = side_angle_rad
    semiminor_to_semimajor_ratio = 1.0 / side_aspect_ratio
    s = semiminor_to_semimajor_ratio
    #Slightly nontrivial geometry formula
    seen_radius_to_semimajor_ratio = np.cos(theta) * np.sqrt(1 + np.square(s * np.tan(theta)))
    semimajor_radius_um = top_radius_um / seen_radius_to_semimajor_ratio
    semiminor_radius_um = semiminor_to_semimajor_ratio * semimajor_radius_um 
    cross_section_um = np.pi * semimajor_radius_um * semiminor_radius_um
    return cross_section_um


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
