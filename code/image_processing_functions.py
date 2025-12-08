import os 
import warnings

import abel
from numba import jit
import numpy as np
from scipy.optimize import fsolve
from scipy import ndimage


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
def get_absorption_image(image_stack, ROI = None, norm_box_coordinates = None, rebin_pixel_num = None, clean_strategy = "default_clipped"):
    with_without_light_ratio = _norm_box_helper(image_stack, norm_box_coordinates = norm_box_coordinates, roi_coordinates=ROI)
    dark_subtracted_image_with_atoms, dark_subtracted_image_without_atoms = _roi_crop_helper(image_stack, ROI = ROI, rebin_pixel_num = rebin_pixel_num)
    absorption_image = dark_subtracted_image_with_atoms / (dark_subtracted_image_without_atoms * with_without_light_ratio)
    absorption_image = _clean_absorption_image(absorption_image, strategy = clean_strategy)
    return absorption_image


"""

Returns dark-subtracted, norm-box-adjusted counts in the without atoms image, suitable for use as a marker of saturation intensity."""

def get_without_atoms_counts(image_stack, ROI = None, norm_box_coordinates = None, rebin_pixel_num = None):
    with_without_light_ratio = _norm_box_helper(image_stack, norm_box_coordinates = norm_box_coordinates, roi_coordinates=ROI)
    _, dark_subtracted_image_without_atoms = _roi_crop_helper(image_stack, ROI = ROI, rebin_pixel_num = rebin_pixel_num)
    norm_adjusted_dark_subtracted_image_without_atoms = dark_subtracted_image_without_atoms * with_without_light_ratio
    return norm_adjusted_dark_subtracted_image_without_atoms

def _roi_crop_helper(image_stack, ROI = None, rebin_pixel_num = None):
    if not ROI is None:
        roi_x_min, roi_y_min, roi_x_max, roi_y_max = ROI
        image_stack = image_stack[:, roi_y_min:roi_y_max, roi_x_min:roi_x_max]
    if not rebin_pixel_num is None:
        image_stack = bin_and_average_data(image_stack, rebin_pixel_num, omitted_axes = 0)
    image_with_atoms = image_stack[0] 
    image_without_atoms = image_stack[1] 
    image_dark = image_stack[2]
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

def _norm_box_helper(image_stack, norm_box_coordinates = None, roi_coordinates = None):
    image_with_atoms = image_stack[0] 
    image_without_atoms = image_stack[1] 
    image_dark = image_stack[2]
    if not norm_box_coordinates is None:
        norm_x_min, norm_y_min, norm_x_max, norm_y_max = norm_box_coordinates
        norm_with_atoms = image_with_atoms[norm_y_min:norm_y_max, norm_x_min:norm_x_max]
        norm_without_atoms = image_without_atoms[norm_y_min:norm_y_max, norm_x_min:norm_x_max] 
        norm_dark = image_dark[norm_y_min:norm_y_max, norm_x_min:norm_x_max]
        with_atoms_dark_subtracted = safe_subtract(norm_with_atoms, norm_dark, minimum_cast = float)
        without_atoms_dark_subtracted = safe_subtract(norm_without_atoms, norm_dark, minimum_cast = float)
        if not roi_coordinates is None:
            norm_y_width = norm_y_max - norm_y_min 
            norm_x_width = norm_x_max - norm_x_min

            norm_y_indices, norm_x_indices = np.indices((norm_y_width, norm_x_width))

            flattened_norm_x_indices = norm_x_indices.flatten()
            flattened_norm_y_indices = norm_y_indices.flatten()

            flattened_with_atoms_ds = with_atoms_dark_subtracted.flatten()
            flattened_without_atoms_ds = without_atoms_dark_subtracted.flatten()

            roi_x_min, roi_y_min, roi_x_max, roi_y_max = roi_coordinates 

            norm_referenced_roi_xmin = roi_x_min - norm_x_min 
            norm_referenced_roi_xmax = roi_x_max - norm_x_min
            norm_referenced_roi_ymin = roi_y_min - norm_y_min 
            norm_referenced_roi_ymax = roi_y_max - norm_y_min

            norm_and_roi_overlapped_flattened = (
                (flattened_norm_x_indices >= norm_referenced_roi_xmin) &
                (flattened_norm_x_indices < norm_referenced_roi_xmax) &
                (flattened_norm_y_indices >= norm_referenced_roi_ymin) &
                (flattened_norm_y_indices < norm_referenced_roi_ymax)
            )

            if np.all(norm_and_roi_overlapped_flattened):
                raise RuntimeError("Norm box is fully contained in ROI")

            with_atoms_light_sum = np.sum(np.where(
                norm_and_roi_overlapped_flattened, 
                np.zeros(flattened_with_atoms_ds.size), 
                flattened_with_atoms_ds
            ))

            without_atoms_light_sum = np.sum(np.where(
                norm_and_roi_overlapped_flattened, 
                np.zeros(flattened_without_atoms_ds.size),
                flattened_without_atoms_ds
            ))

        else:
            with_atoms_light_sum = np.sum(with_atoms_dark_subtracted) 
            without_atoms_light_sum = np.sum(without_atoms_dark_subtracted)
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
    

"""Rebin a dataset, taking averages over regions of specified size. 

Given a numpy array image_to_bin, take averages over regions of dimension specified by bin dimension, then return 
a smaller array of binned pixels.

Parameters:
data_to_bin: Numpy array to be rebinned
bin_dimensions: Either an int or tuple of ints specifying the size of bins to use. If a tuple of ints, it should be of the same 
    length as image_to_bin.shape; if a single int, this size is used along each bin dimension. If the ith element of bin_dimensions 
    does not evenly divide the ith element of data_to_bin.shape, the data is truncated along this axis.
Omitted axes: If specified, any axis appearing in omitted_axes is not rebinned over. Useful for rebinning only certain axes of an array. 

Note: Where omitted axes is provided, the bin dimensions specified in bin_dimensions run over the non-omitted axes in increasing order.
"""
def bin_and_average_data(data_to_bin, bin_dimensions, omitted_axes = None):

    if omitted_axes is None:
        omitted_axes = ()
    elif isinstance(omitted_axes, int):
        omitted_axes = (omitted_axes,)
    num_data_dimensions = len(data_to_bin.shape)
    num_omitted_axes = len(omitted_axes)
    moved_omitted_axes = tuple((num_data_dimensions - 1) - np.arange(num_omitted_axes))
    moved_axis_array = np.moveaxis(data_to_bin, omitted_axes, moved_omitted_axes)
    num_non_omitted_data_dimensions = num_data_dimensions - num_omitted_axes
    non_omitted_data_dimensions = moved_axis_array.shape[:num_non_omitted_data_dimensions]
    omitted_data_dimensions = moved_axis_array.shape[num_non_omitted_data_dimensions:]
    if isinstance(bin_dimensions, int):
        bin_dimensions = tuple([bin_dimensions] * num_non_omitted_data_dimensions)
    if not len(bin_dimensions) == num_non_omitted_data_dimensions:
        raise ValueError("The length of the bin dimensions does not agree with the data to be binned.")
    #Truncate array to correct size in each non-omitted dimension, then reshape to 2 * num_non_omitted_data_dimensions + omitted_data_dimensions
    slice_list = []
    reshape_dimension_list = []
    for bin_dimension, data_dimension in zip(bin_dimensions, non_omitted_data_dimensions):
        truncated_data_dimension = data_dimension - (data_dimension % bin_dimension)
        slice_list.append(slice(0, truncated_data_dimension))
        reshape_dimension_list.append(truncated_data_dimension // bin_dimension) 
        reshape_dimension_list.append(bin_dimension)
    #The reshape dimension list must be extended to include the omitted dimensions
    reshape_dimension_list.extend(omitted_data_dimensions)
    slice_tuple = tuple(slice_list)
    reshape_dimension_tuple = tuple(reshape_dimension_list)
    truncated_data = moved_axis_array[slice_tuple] 
    reshaped_truncated_data = truncated_data.reshape(reshape_dimension_tuple)
    averaging_axis_tuple = tuple(np.arange(1, 2 * num_non_omitted_data_dimensions, 2)) 
    #Average over the rebinned dimensions
    averaged_data = np.average(reshaped_truncated_data, axis = averaging_axis_tuple)
    #Restore the omitted axes to their original positions 
    restored_position_averaged_data = np.moveaxis(averaged_data, moved_omitted_axes, omitted_axes)
    return restored_position_averaged_data

"""
Returns an od image (i.e. -ln(abs_image)) for a given image stack. Essentially wraps -ln(get_absorption_image) with some extra cleaning."""
def get_absorption_od_image(image_stack, ROI = None, norm_box_coordinates = None, rebin_pixel_num = None,
                            abs_clean_strategy = "default_clipped", od_clean_strategy = 'default_clipped'):
    absorption_image = get_absorption_image(image_stack, ROI = ROI, norm_box_coordinates = norm_box_coordinates,
                                            rebin_pixel_num = rebin_pixel_num, clean_strategy = abs_clean_strategy)
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

def get_atom_density_absorption(image_stack, ROI = None, norm_box_coordinates = None, rebin_pixel_num = None,
                                abs_clean_strategy = 'default_clipped', od_clean_strategy = 'default_clipped',
                                flag = 'beer-lambert', detuning = 0, linewidth = None, res_cross_section = None, species = '6Li', saturation_counts = None, 
                                cross_section_imaging_geometry_factor = 1.0):
    if not linewidth:
        linewidth = _get_linewidth_from_species(species)
    if not res_cross_section:
        res_cross_section = _get_res_cross_section_from_species(species)
    geometry_adjusted_cross_section = res_cross_section * cross_section_imaging_geometry_factor
    od_image = get_absorption_od_image(image_stack, ROI = ROI, norm_box_coordinates=norm_box_coordinates,
                                       rebin_pixel_num = rebin_pixel_num, 
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
    with_without_light_ratio = _norm_box_helper(image_stack, norm_box_coordinates=norm_box_coordinates, roi_coordinates=ROI)
    with_atoms_dark_subtracted, without_atoms_dark_subtracted = _roi_crop_helper(image_stack, ROI = ROI)
    without_atoms_dark_subtracted = without_atoms_dark_subtracted * with_without_light_ratio
    saturation_term = (1.0 / on_resonance_cross_section) * (without_atoms_dark_subtracted - with_atoms_dark_subtracted) / saturation_counts
    return beer_lambert_term + saturation_term



#POLROT IMAGING

@jit(nopython = True)
def _python_polrot_image_function_with_target_offset(od_naught_vector, abs, norm_detunings, 
                                        sat_intensities, phase_sign):
    fun_val = _compiled_python_polrot_image_function(od_naught_vector, norm_detunings,
                                        sat_intensities, phase_sign)
    fun_val -= abs
    return fun_val

"""
Polrot image function, implemented in python. More readable & accessible, but slower."""
@jit(nopython = True)
def _compiled_python_polrot_image_function(od_naughts, norm_detunings,
                                             sat_intensities, phase_sign):
    ods_imaged = od_naughts * od_lorentzian(norm_detunings, sat_intensities)
    phi_values = -phase_sign * np.sum(norm_detunings * ods_imaged, axis = -1)
    abs_values = np.exp(-0.5 * np.sum(ods_imaged, axis = -1))
    results = 0.5 + np.square(abs_values) / 2.0 - abs_values * np.sin(phi_values) 
    return results

def python_polrot_image_function(od_naughts, norm_detunings,
                                             sat_intensities, phase_sign):
    ods_imaged = od_naughts * od_lorentzian(norm_detunings, sat_intensities)
    phi_values = -phase_sign * np.sum(norm_detunings * ods_imaged, axis = -1)
    abs_values = np.exp(-0.5 * np.sum(ods_imaged, axis = -1))
    results = 0.5 + np.square(abs_values) / 2.0 - abs_values * np.sin(phi_values) 
    return results

@jit(nopython = True)
def od_lorentzian(norm_detuning, sat_intensity):
    return 1.0 / (1 + np.square(2 * norm_detuning) + sat_intensity)

def generate_polrot_lookup_table(detunings, linewidth = None, res_cross_section = None, phase_sign = 1.0, 
                                species = '6Li', num_samps = 1000, abs_min = 0.0, abs_max = 2.0):
    abs_values = np.linspace(abs_min, abs_max, num = num_samps, endpoint = True)
    my_abs_A_grid = np.zeros((num_samps, num_samps))
    my_abs_B_grid = np.zeros((num_samps, num_samps))
    for i, abs in enumerate(abs_values):
        my_abs_A_grid[i] = abs * np.ones(num_samps)
        my_abs_B_grid[:, i] = abs * np.ones(num_samps)
    my_abs_grid = np.stack([my_abs_A_grid, my_abs_B_grid])
    my_densities_1_grid, my_densities_2_grid = get_atom_density_from_polrot_images(my_abs_grid, detunings, 
                                                linewidth = linewidth, res_cross_section = res_cross_section, phase_sign = phase_sign,
                                                 species = species)
    np.save("Polrot_Lookup_Table.npy", np.stack((my_densities_1_grid, my_densities_2_grid)))
    with open("Polrot_Lookup_Table_Params.txt", 'w') as f:
        f.write("Detuning A1: " + str(detunings[0][0])) 
        f.write("Detuning B1: " + str(detunings[1][0]))
        f.write("Detuning_A2: " + str(detunings[0][1])) 
        f.write("Detuning B2: " + str(detunings[1][1]))
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


"""
Given a set of absorption images and detunings, reconstruct atom densities from polarization rotation imaging.

Parameters: 

abs_images: An (N, M, P) array of N 2D images, each of dimensions (M, P)
detunings: An (N, N) array of detuning values, with the (i, j) entry denoting the detuning of the jth state 
    from the light used to take the ith image. 
res_cross_section: The resonant cross section used to convert optical densities to atomic densities. Defaults to 6Li cycling D2. 
cross_section_imaging_geometry_factor: A correction to the resonant cross section from imaging geometry.
sat_intensities: an (N, M, P) array of imaging light intensities at the atoms, in units of saturation intensity. Defaults to 0.0.
phase_sign: The appropriate sign for the polrot imaging phase shift, set by experimental config.
"""

def get_atom_density_from_polrot_images(abs_images, detunings, linewidth = None,
                                        res_cross_section = None, cross_section_imaging_geometry_factor = 1.0, 
                                        sat_intensities = None,  phase_sign = 1.0, 
                                        species = '6Li'):
    number_images = abs_images.shape[0]
    image_shape = abs_images.shape[1:]
    if not linewidth:
        linewidth = _get_linewidth_from_species(species)
    norm_detunings = detunings / linewidth
    if not res_cross_section:
        res_cross_section = _get_res_cross_section_from_species(species)
    geometry_adjusted_cross_section = res_cross_section * cross_section_imaging_geometry_factor
    if sat_intensities is None: 
        sat_intensities = np.zeros(abs_images.shape)
    if(np.abs(phase_sign) != 1.0):
        raise ValueError("The phase sign must be +-1.")
    
    #Move the image axis to the end, then flatten the 2D pixel coordinate axes
    abs_images_moved_axis = np.moveaxis(abs_images, 0, -1) 
    iterator_reshaped_abs_images = np.reshape(abs_images_moved_axis, (-1, number_images))
    #Saturation gets an extra axis on the end for correct broadcasting against state index
    sat_intensities_moved_axis = np.moveaxis(sat_intensities, 0, -1)
    iterator_reshaped_sat_intensities = np.reshape(sat_intensities_moved_axis, (-1, number_images, 1))
    
    flattened_atom_densities_list = []

    map_iterator = zip(iterator_reshaped_abs_images, generator_factory(norm_detunings), 
                         generator_factory(geometry_adjusted_cross_section), 
                          iterator_reshaped_sat_intensities,  generator_factory(phase_sign), 
                          generator_factory(np.zeros(number_images)))
    #TODO: Paralellize this. For now, it's just written in a parallelizable form. 
    for itr_val in map_iterator:
        atom_densities = parallelizable_polrot_density_function(*itr_val)
        flattened_atom_densities_list.append(atom_densities)
    flattened_atom_densities_array = np.array(flattened_atom_densities_list) 
    reshaped_atom_densities_array = np.reshape(flattened_atom_densities_array, (*image_shape, number_images))
    final_atom_densities_array = np.moveaxis(reshaped_atom_densities_array, -1, 0)

    return final_atom_densities_array 

def generator_factory(value):
    while True:
        yield value

def parallelizable_polrot_density_function(absorptions, norm_detunings, 
                                    on_resonance_cross_section, sat_intensities, phase_sign, init_guess):
        solver_extra_args = (absorptions, norm_detunings,
                                sat_intensities, phase_sign)
        root = fsolve(_python_polrot_image_function_with_target_offset, init_guess, args = solver_extra_args)
        od_naughts = root
        atom_densities = od_naughts / on_resonance_cross_section
        return atom_densities

#Inverse function for simulating polrot images from atom density. Mostly for testing/debugging purposes.
def get_polrot_images_from_atom_density(densities, detunings, linewidth = None,
                                        res_cross_section = None, cross_section_imaging_geometry_factor = 1.0,
                                        sat_intensities = None, phase_sign = 1.0, species = '6Li'):
    if linewidth is None:
        linewidth = _get_linewidth_from_species(species)
    if res_cross_section is None:
        res_cross_section = _get_res_cross_section_from_species(species)
    geometry_adjusted_cross_section = res_cross_section * cross_section_imaging_geometry_factor
    od_naughts = densities * geometry_adjusted_cross_section
    #Reshape od_naughts and sat_intensities so that the last two axes broadcast against norm_detunings
    od_naughts_moved_axis = np.moveaxis(od_naughts, 0, -1)
    od_naughts_reshaped = np.expand_dims(od_naughts_moved_axis, axis = -2)

    if sat_intensities is None:
        sat_intensities = np.zeros(densities.shape)
    sat_intensities_moved_axis = np.moveaxis(sat_intensities, 0, -1)
    sat_intensities_reshaped = np.expand_dims(sat_intensities_moved_axis, axis = -1)
    norm_detunings = detunings / linewidth
    results =  python_polrot_image_function(od_naughts_reshaped, norm_detunings, 
                            sat_intensities_reshaped, phase_sign)
    results_moved_axis = np.moveaxis(results, -1, 0)
    return results_moved_axis

"""
Given an image, return an angle to rotate it into the xy plane.

Given a 2D numpy array image, determine the principal axes of the image and return an angle to rotate the image so that these 
lie along x and y, respectively. The technique is to compute the correlation matrix
C_ij = <x_i x_j>
and compute the appropriate rotation angle to make this diagonal. 

Parameters: 
    image: A 2D numpy array. If greater than 2D, all but the last two axes will be broadcast over.
    return_com: Boolean, default false. If true, the function returns both the COM of the image as well as the appropriate rotation angle.
    
Returns: 
    angle: The angle, in degrees, by which to rotate the image, under the sign convention of scipy.ndimage.rotate.
    If return_com is true: (angle, coms), with coms = [x_com, y_com] and the same broadcasting convention. 
"""

def get_image_principal_rotation_angle(image, return_com = False):
    image_pixel_covariance = get_image_pixel_covariance(image)

    sigma_y_squared = image_pixel_covariance[0][0]
    sigma_x_squared = image_pixel_covariance[1][1]
    off_diag = image_pixel_covariance[0][1]

    sigma_diff = sigma_y_squared - sigma_x_squared 

    rotation_angle_rad = -0.5 * np.arctan(2 * off_diag / sigma_diff)
    rotation_angle_deg = rotation_angle_rad * 180 / np.pi

    if not return_com:
        return rotation_angle_deg
    else:
        image_coms = get_image_coms(image)
        return (rotation_angle_deg, image_coms)
    

"""Supersample an ndarray image. 

Given an n-dimensional ndarray, return a version which has been supersampled by an integer factor 
along specified axes. 

Parameters:

image: The input ndarray. Assumed to be of numeric type - ragged arrays aren't supported. 

scale_factor: (int or tuple of ints) The multiple by which to supersample. A factor of 2, 
for instance, doubles the number of pixels along a given axis. If an int is given, the same 
supersampling is done along each of the included axes (see below). If a tuple is passed, 
the specified factor is applied to each of the included axes, in order.

included_axes: (int or tuple of ints) If specified, supersampling is done over only
     the specified axes of the input array. If None, all axes are sampled. 
"""

def supersample_image(image, scale_factor, included_axes = None):
    if included_axes is None:
        included_axes = tuple(range(len(image.shape)))
    if isinstance(scale_factor, int):
        scale_factors = scale_factor * np.ones(len(included_axes)) 
    else:
        scale_factors = scale_factor
    if not len(scale_factors) == len(included_axes):
        raise ValueError("scale_factor and included_axes must have the same length.")
    rescaled_array = image 
    for specific_axis, specific_factor in zip(included_axes, scale_factors):
        rescaled_array = np.repeat(rescaled_array, specific_factor, specific_axis)
    return rescaled_array




"""Given an image, return its center of mass. 

Given an two-dimensional numpy array image, return a tuple coms representing the 
center of mass along the x- and y-axis of the image. 

Parameters:

image: An n-dimensional ndarray of pixel values, representing a density map. No check 
is performed that all values are positive. If more than two-dimensional, all axes besides 
the last two are broadcast over.

Returns:

coms: An array [y_com, x_com] Units are pixels: rounding the COM gives the nearest index along that axis 
to the center of mass position. If extra axes are present in image, the array [y_com, x_com] is along the 
first axis of the result."""

def get_image_coms(image):
    normalized_image_weights = image / np.sum(image, axis = (-2, -1), keepdims = True)
    reshaped_normalized_image_weights = np.expand_dims(normalized_image_weights, axis = 0)
    #Only the y and x indices are averaged over
    image_indices_yx = np.indices(image.shape)[-2:]
    integration_axes = (-2, -1)
    weighted_image_indices_yx = reshaped_normalized_image_weights * image_indices_yx
    coms = np.sum(weighted_image_indices_yx, axis = integration_axes)
    return coms


"""Given a 2d image, return the pixel covariance matrix. 

Given a 2d numpy array image, compute the covariance of the pixel coordinates x and y taken over 
the (normalized) density distribution given by the image. In other words, compute 

C_{ij} = 
    [<(y - y_0)^2>          <(x - x_0)(y - y_0)> 
     <(x - x_0)(y - y_0)>    <(x - x_0)^2>      ]

where the expectation value is taken over the density profile of the image, and of course 
the xy are offset to the COM. 

Parameters:

image: A numpy ndarray. Broadcast behavior is as in get_image_coms.

Returns: A numpy array C_ij as described above. If additional axes are present, 
C_ij is along the first two axes of the returned array."""

def get_image_pixel_covariance(image):
    normalized_image_weights = image / np.sum(image, axis = (-2, -1), keepdims = True)
    reshaped_normalized_image_weights = np.expand_dims(normalized_image_weights, axis = (0, 1))
    image_com_yx = get_image_coms(image)
    image_com_yx_expanded = np.expand_dims(image_com_yx, axis = (-2, -1))
    image_indices_yx = np.indices(image.shape)[-2:]
    offset_image_indices_yx = image_indices_yx - image_com_yx_expanded
    unweighted_image_index_products = np.expand_dims(offset_image_indices_yx, axis = 0) * np.expand_dims(offset_image_indices_yx, axis = 1)
    weighted_image_index_products = reshaped_normalized_image_weights * unweighted_image_index_products
    cov = np.sum(weighted_image_index_products, axis = (-2, -1))
    return cov


"""Convolve an image with a gaussian filter. 

Given an input ndarray image, convolve the image with a Gaussian filter along the specified axes. A thin wrapper 
around convolution with scipy.ndimage.convolve; see documentation at 

https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html

Note that one default of the 

Parameters:

image: An n-dimensional numpy array. 

sigma: (float or tuple of floats) The Gaussian width, in pixels, to use for the convolution kernel. If a single float,
the same value is used for each axis. If a tuple, the specified width is used for each dimension.

window_dims: (int or tuple of int) The dimensions of the convolution along the included axes. If not specified, the window 
width defaults to 6x sigma (+- 3 sigma), as a compromise between runtime and fidelity. If any 
dimension is even, it is rounded up to the nearest odd value.

included_axes: (int or tuple of int) The axes over which Gaussian convolution is performed. If None, every axis is included.
Defaults to the final two axes of the image."""
def convolve_gaussian(image, sigma, window_dims = None, included_axes = (-2, -1), mode = "constant", **convolve_kwargs):
    if included_axes is None:
        included_axes = tuple(range(len(image.shape)))
    if isinstance(sigma, (int, float)):
        sigma_values = sigma * np.ones(len(included_axes))
    else:
        sigma_values = np.array(sigma)

    #Obtain excluded axes
    ndims = len(image.shape)
    axes_range = np.arange(ndims)
    included_axes_array = np.array(included_axes) % ndims
    excluded_axes = axes_range[~np.isin(axes_range, included_axes_array)]

    #Construct window dimensions
    WINDOW_SIGMA_MULTIPLIER = 6
    image_shape_array = np.array(image.shape)
    if window_dims is None:
        window_dims = np.round(sigma_values * WINDOW_SIGMA_MULTIPLIER).astype(int)
    else:
        window_dims = np.array(window_dims)
    window_dims = window_dims + (1 - window_dims % 2)


    #Get gaussian values along included axes
    bare_indices = np.indices(window_dims)
    bare_indices_shuffled = np.moveaxis(bare_indices, 0, -1)
    centered_indices_shuffled = bare_indices_shuffled - (window_dims - 1) / 2

    def multidimensional_gaussian(indices_shuffled, sigmas):
        return np.exp(-1.0 * np.sum(np.square(indices_shuffled / sigmas) / 2.0, axis = -1))
    
    gaussian_values = multidimensional_gaussian(centered_indices_shuffled, sigma_values)
    normalized_gaussian_values = gaussian_values / np.sum(gaussian_values) 
    normalized_gaussian_values_extended = np.expand_dims(normalized_gaussian_values, axis = tuple(excluded_axes))
    return ndimage.convolve(image, normalized_gaussian_values_extended, mode = mode, **convolve_kwargs)


"""Inverse Abel transform of a profile. 

Given a (potentially multi-dimensional) profile, compute the inverse Abel transform along the specified axis, 
leaving others untouched. profile is assumed to be well-conditioned - see parameters. 

This method is a thin wrapper around the PyAbel package - 

For basic notes, see e.g. https://en.wikipedia.org/wiki/Abel_transform. For implementation details, 
see https://pyabel.readthedocs.io/en/latest/readme_link.html

Parameters:

profile: A 1D or 2D numpy array. If the image is 2D, the symmetry axis of the image is taken to be along the 0th 
axis (the "z-axis"), so that this axis is preserved while the other (the "y" axis, by convention of the Abel transform, 
but in our convention the "x"-axis) is integrated over for the transform.

Kwargs as documented on https://pyabel.readthedocs.io/en/latest/abel.html
(Defaults are appropriate for basic use)

"""
def inverse_abel(profile, **kwargs):
    return abel.Transform(profile, **kwargs).transform


def get_saturation_counts_from_camera_parameters(pixel_length_at_atoms_m, imaging_time_s, camera_count_to_photon_factor, linewidth_Hz, 
                                                res_cross_section_m, saturation_multiplier = 1.0):
    atomic_pixel_area = np.square(pixel_length_at_atoms_m)
    photon_current_to_saturation_conversion_factor_mks = 2 * res_cross_section_m / (2 * np.pi * linewidth_Hz)
    fudged_photon_current_to_saturation_conversion_mks = photon_current_to_saturation_conversion_factor_mks * saturation_multiplier
    counts_to_photon_current_conversion_factor_mks = (1.0 / atomic_pixel_area) * (1.0 / imaging_time_s) * camera_count_to_photon_factor
    saturation_counts = 1.0 / (counts_to_photon_current_conversion_factor_mks * fudged_photon_current_to_saturation_conversion_mks)
    return saturation_counts



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
