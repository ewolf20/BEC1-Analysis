import numpy as np
from scipy.optimize import fsolve

from .c_code._polrot_code import ffi as polrot_image_ffi
from .c_code._polrot_code import lib as polrot_image_lib


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
        return (image_with_atoms_ROI - image_dark_ROI, image_without_atoms_ROI - image_dark_ROI)
    else:
        return (image_with_atoms - image_dark, image_without_atoms - image_dark)


#TODO: Figure out issue with roa!
"""
Helper function for finding the multiplier for the without_atoms image to compensate for light intensity shifts.

The norm_strategy flag dictates how normalization is done. One can either have:

aor: Average of ratios - takes the ratio of with atoms to without atoms pixel-by-pixel, averages this ratio, and then 
demands the average be 1

roa: ratio of averages - takes the average pixel brightness of the with and without atoms images, then takes the ratio 
of these averages and demands that this be 1.

A priori we would expect roa to be better, but in the experimental data I've seen so far it's much worse. Need to figure this
out."""

def _norm_box_helper(image_stack, norm_box_coordinates = None, norm_strategy = "aor"):
    image_with_atoms = image_stack[0] 
    image_without_atoms = image_stack[1] 
    image_dark = image_stack[2]
    if(norm_box_coordinates):
        norm_x_min, norm_y_min, norm_x_max, norm_y_max = norm_box_coordinates
        norm_with_atoms = image_with_atoms[norm_y_min:norm_y_max, norm_x_min:norm_x_max]
        norm_without_atoms = image_without_atoms[norm_y_min:norm_y_max, norm_x_min:norm_x_max] 
        norm_dark = image_dark[norm_y_min:norm_y_max, norm_x_min:norm_x_max]
        if(norm_strategy == "roa"):
            with_atoms_light_sum = sum(sum(norm_with_atoms - norm_dark))
            without_atoms_light_sum = sum(sum(norm_without_atoms - norm_dark))
            with_without_light_ratio = with_atoms_light_sum / without_atoms_light_sum
        elif(norm_strategy == "aor"):
            with_without_light_ratio = sum(sum((norm_with_atoms - norm_dark) / (norm_without_atoms - norm_dark))) / norm_with_atoms.size
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
        ABSORPTION_LIMIT = 1.3
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


def atom_count_pixel_sum(atom_density_image, pixel_size, sum_region = None):
    atom_counts = atom_density_image * pixel_size 
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


def get_atom_density_from_polrot_images(abs_image_A, abs_image_B, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth = None,
                                        res_cross_section = None, intensities_A = None, intensities_B = None, intensities_sat = None, species = '6Li'):
    if (intensities_A or intensities_B or intensities_sat) and not (intensities_A and intensities_B and intensities_sat):
        raise ValueError("Either specify the intensities and saturation intensity or don't; no mixing.")
    if not linewidth:
        linewidth = _get_linewidth_from_species(species)
    if not res_cross_section:
        res_cross_section = _get_res_cross_section_from_species(species)
    atom_densities_array_1 = np.zeros(abs_image_A.shape)
    atom_densities_array_2 = np.zeros(abs_image_B.shape)
    if not intensities_A:
        intensity_A = 0
        intensity_B = 0 
        intensity_sat = np.inf
        solver_extra_args = (detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth, intensity_A, intensity_B, intensity_sat)
    for i in range(len(abs_image_A)):
        for j in range(len(abs_image_A[0])):
            if(intensity_A):
                solver_extra_args = (detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth, intensities_A[i][j], 
                                        intensities_B[i][j], intensities_sat[i][j])
            absorption_A = abs_image_A[i][j]
            absorption_B = abs_image_B[i][j]
            def current_function(od_naught_vector, *args):
                return wrapped_polrot_image_function(od_naught_vector, *args) - np.array([absorption_A, absorption_B])
            root = fsolve(current_function, [0, 0], args = solver_extra_args) 
            od_naught_1, od_naught_2 = root
            atom_density_1 = od_naught_1 / res_cross_section 
            atom_density_2 = od_naught_2 / res_cross_section
            atom_densities_array_1[i][j] = atom_density_1 
            atom_densities_array_2[i][j] = atom_density_2
    return (atom_densities_array_1, atom_densities_array_2)

def wrapped_polrot_image_function(od_naught_vector, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth, 
                                    intensity_A, intensity_B, intensity_sat):
    wrapped_od_naught = polrot_image_ffi.new("float[]", list(od_naught_vector))
    output_buffer = polrot_image_ffi.new("double[]", 2)
    status_code = polrot_image_lib.give_polrot_image(wrapped_od_naught, detuning_1A, detuning_1B, detuning_2A, detuning_2B, linewidth,
                                                intensity_A, intensity_B, intensity_sat, output_buffer)
    return np.array([output_buffer[0], output_buffer[1]])

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
