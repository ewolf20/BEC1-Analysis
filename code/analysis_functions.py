import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid 
from scipy import ndimage
from scipy.signal import savgol_filter

from . import data_fitting_functions, image_processing_functions, science_functions
from .measurement import Run


#RAW IMAGES (Convenience functions for getting the raw pixel data from shots, cropped within an ROI)

#WARNING: For efficiency reasons related to memory mapping, combined with code simplicity,
#these functions violate the contract of returning multiple results (here, the different frames), 
#as a tuple. Instead, they return numpy arrays. This does not interfere with the 'contract' of measurement.py, 
#which currently only requires an iterable.

#WARNING: For similar efficiency reasons, the parameter

def get_raw_pixels_na_catch(my_measurement, my_run, crop_roi = False, memmap = False):
    my_run_image_array = my_run.get_image("Catch", memmap = memmap) 
    if crop_roi:
        roi = my_measurement.measurement_parameters["ROI"] 
        x_min, y_min, x_max, y_max = roi
        my_run_image_array_cropped = my_run_image_array[:, y_min:y_max, x_min:x_max] 
        return my_run_image_array_cropped
    else:
        return my_run_image_array

def get_raw_pixels_side(my_measurement, my_run, crop_roi = False, memmap = False):
    my_run_image_array = my_run.get_image("Side", memmap = memmap)
    if crop_roi:
        roi = my_measurement.measurement_parameters["ROI"] 
        x_min, y_min, x_max, y_max = roi 
        my_run_image_array_cropped = my_run_image_array[:, y_min:y_max, x_min:x_max]
        return my_run_image_array_cropped
    else:
        return my_run_image_array
    

def get_raw_pixels_top_A(my_measurement, my_run, crop_roi = False, memmap = False):
    my_run_image_array = my_run.get_image("TopA", memmap = memmap)
    if crop_roi:
        roi = my_measurement.measurement_parameters["ROI"] 
        x_min, y_min, x_max, y_max = roi 
        my_run_image_array_cropped = my_run_image_array[:, y_min:y_max, x_min:x_max]
        return my_run_image_array_cropped
    else:
        return my_run_image_array
    

def get_raw_pixels_top_B(my_measurement, my_run, crop_roi = False, memmap = False):
    my_run_image_array = my_run.get_image("TopB", memmap = memmap)
    if crop_roi:
        roi = my_measurement.measurement_parameters["ROI"] 
        x_min, y_min, x_max, y_max = roi 
        my_run_image_array_cropped = my_run_image_array[:, y_min:y_max, x_min:x_max]
        return my_run_image_array_cropped
    else:
        return my_run_image_array


#ABS IMAGES (Sometimes called 'Fake OD')

def get_abs_image_na_catch(my_measurement, my_run, rebin_pixel_num = None):
    my_run_image_array = get_raw_pixels_na_catch(my_measurement, my_run, memmap = True)
    my_run_abs_image = image_processing_functions.get_absorption_image(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"], 
                                                rebin_pixel_num = rebin_pixel_num)
    return my_run_abs_image

def get_abs_image_side(my_measurement, my_run, rebin_pixel_num = None):
    my_run_image_array = get_raw_pixels_side(my_measurement, my_run, memmap = True)
    my_run_abs_image = image_processing_functions.get_absorption_image(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"], 
                                                rebin_pixel_num = rebin_pixel_num)
    return my_run_abs_image


def get_abs_image_top_A(my_measurement, my_run, rebin_pixel_num = None):
    my_run_image_array_A = get_raw_pixels_top_A(my_measurement, my_run, memmap = True)
    my_run_abs_image = image_processing_functions.get_absorption_image(my_run_image_array_A, ROI = my_measurement.measurement_parameters["ROI"], 
                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"], 
                                                rebin_pixel_num = rebin_pixel_num)
    return my_run_abs_image

def get_abs_image_top_B(my_measurement, my_run, rebin_pixel_num = None):
    my_run_image_array_B = get_raw_pixels_top_B(my_measurement, my_run, memmap = True)
    my_run_abs_image = image_processing_functions.get_absorption_image(my_run_image_array_B, ROI = my_measurement.measurement_parameters["ROI"], 
                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"], 
                                                rebin_pixel_num = rebin_pixel_num)
    return my_run_abs_image
    
def get_abs_images_top_double(my_measurement, my_run, rebin_pixel_num = None):
    return (get_abs_image_top_A(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num),
            get_abs_image_top_B(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num))

#OD IMAGES

def get_od_image_na_catch(my_measurement, my_run, rebin_pixel_num = None):
    my_run_image_array = get_raw_pixels_na_catch(my_measurement, my_run, memmap = True)
    my_run_od_image = image_processing_functions.get_absorption_od_image(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"], 
                                                                rebin_pixel_num = rebin_pixel_num)
    return my_run_od_image

def get_od_image_side(my_measurement, my_run, rebin_pixel_num = None):
    my_run_image_array = get_raw_pixels_side(my_measurement, my_run, memmap = True)
    my_run_od_image = image_processing_functions.get_absorption_od_image(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"], 
                                                                rebin_pixel_num = rebin_pixel_num)
    return my_run_od_image

def get_od_image_top_A(my_measurement, my_run, rebin_pixel_num = None):
    my_run_image_array_A = get_raw_pixels_top_A(my_measurement, my_run, memmap = True)
    my_run_od_image_A = image_processing_functions.get_absorption_od_image(my_run_image_array_A, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"], 
                                                                rebin_pixel_num = rebin_pixel_num)
    return my_run_od_image_A


def get_od_image_top_B(my_measurement, my_run, rebin_pixel_num = None):
    my_run_image_array_B = get_raw_pixels_top_B(my_measurement, my_run, memmap = True)
    my_run_od_image_B = image_processing_functions.get_absorption_od_image(my_run_image_array_B, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"],
                                                                rebin_pixel_num = rebin_pixel_num)
    return my_run_od_image_B

def get_od_images_top_double(my_measurement, my_run, rebin_pixel_num = None):
    od_image_A = get_od_image_top_A(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num) 
    od_image_B = get_od_image_top_B(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num) 
    return (od_image_A, od_image_B)

#PIXEL SUMS

def get_od_pixel_sum_na_catch(my_measurement, my_run, rebin_pixel_num = None):
    my_run_abs_image = get_od_image_na_catch(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num)
    pixel_sum = image_processing_functions.pixel_sum(my_run_abs_image)
    return pixel_sum

def get_od_pixel_sum_side(my_measurement, my_run, rebin_pixel_num = None):
    my_run_abs_image = get_od_image_side(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num)
    pixel_sum = image_processing_functions.pixel_sum(my_run_abs_image)
    return pixel_sum


def get_od_pixel_sum_top_A(my_measurement, my_run, rebin_pixel_num = None):
    my_run_abs_image_A = get_od_image_top_A(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num)
    pixel_sum_A = image_processing_functions.pixel_sum(my_run_abs_image_A)
    return pixel_sum_A

def get_od_pixel_sum_top_B(my_measurement, my_run, rebin_pixel_num = None):
    my_run_abs_image_B = get_od_image_top_B(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num)
    pixel_sum_B = image_processing_functions.pixel_sum(my_run_abs_image_B)
    return pixel_sum_B

def get_od_pixel_sums_top_double(my_measurement, my_run, rebin_pixel_num = None):
    return (get_od_pixel_sum_top_A(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num), 
            get_od_pixel_sum_top_B(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num))

#ATOM DENSITIES

def get_atom_density_side_li_lf(my_measurement, my_run, rebin_pixel_num = None):
    my_run_image_array = get_raw_pixels_side(my_measurement, my_run, memmap = True)
    frequency_multiplier = my_measurement.experiment_parameters["li_lf_freq_multiplier"]
    nominal_resonance_frequency = my_measurement.experiment_parameters["li_lf_res_freq"]
    nominal_frequency = my_run.parameters["LFImgFreq"]
    side_cross_section_geometry_factor = my_measurement.experiment_parameters["li_side_sigma_multiplier"]
    detuning = frequency_multiplier * (nominal_frequency - nominal_resonance_frequency)
    atom_density_image = image_processing_functions.get_atom_density_absorption(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                            norm_box_coordinates=my_measurement.measurement_parameters["norm_box"],
                                                            rebin_pixel_num = rebin_pixel_num,
                                                            detuning = detuning, 
                                                            cross_section_imaging_geometry_factor=side_cross_section_geometry_factor)
    return atom_density_image

def get_atom_density_side_li_hf(my_measurement, my_run, state_index = None, b_field_condition = "unitarity", rebin_pixel_num = None):
    if state_index is None:
        raise RuntimeError("The state of the imaging must be specified.")
    
    my_run_image_array = get_raw_pixels_side(my_measurement, my_run, memmap = True) 
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    side_cross_section_geometry_factor = my_measurement.experiment_parameters["li_side_sigma_multiplier"]
    nominal_resonance_frequency = _get_resonance_frequency_from_state_index(my_measurement, state_index)
    hf_lock_frequency_adjustment = _get_hf_lock_frequency_adjustment_from_b_field_condition(my_measurement, b_field_condition)
    nominal_frequency = my_run.parameters["ImagFreq0"]
    detuning = frequency_multiplier * (nominal_frequency - nominal_resonance_frequency) + hf_lock_frequency_adjustment
    atom_density_image = image_processing_functions.get_atom_density_absorption(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                            norm_box_coordinates=my_measurement.measurement_parameters["norm_box"],
                                                            rebin_pixel_num = rebin_pixel_num, 
                                                            detuning = detuning, cross_section_imaging_geometry_factor=side_cross_section_geometry_factor)
    return atom_density_image

def get_atom_density_top_A_abs(my_measurement, my_run, state_index = 1, b_field_condition = "unitarity", rebin_pixel_num = None):
    #Find the true detuning from the resonance in absolute frequency space,
    #taking into account shifts in AOM frequency and hf frequency offset lock setpoint
    nominal_resonance_frequency = _get_resonance_frequency_from_state_index(my_measurement, state_index)
    nominal_frequency = my_run.parameters["ImagFreq1"]
    hf_lock_frequency_adjustment = _get_hf_lock_frequency_adjustment_from_b_field_condition(my_measurement, b_field_condition)
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    detuning = frequency_multiplier * (nominal_frequency - nominal_resonance_frequency) + hf_lock_frequency_adjustment

    #Adjust for imaging geometry-dependent cross section
    top_cross_section_geometry_factor = my_measurement.experiment_parameters["li_top_sigma_multiplier"]
    #Process
    my_run_image_array = get_raw_pixels_top_A(my_measurement, my_run, memmap = True)
    atom_density_image = image_processing_functions.get_atom_density_absorption(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                            norm_box_coordinates=my_measurement.measurement_parameters["norm_box"], 
                                                            rebin_pixel_num = rebin_pixel_num,
                                                            detuning = detuning, cross_section_imaging_geometry_factor=top_cross_section_geometry_factor)
    return atom_density_image

def get_atom_density_top_B_abs(my_measurement, my_run, state_index = 3, b_field_condition = "unitarity", rebin_pixel_num = None):
    nominal_resonance_frequency = _get_resonance_frequency_from_state_index(my_measurement, state_index)
    nominal_frequency = my_run.parameters["ImagFreq2"]
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    hf_lock_frequency_adjustment = _get_hf_lock_frequency_adjustment_from_b_field_condition(my_measurement, b_field_condition)
    top_cross_section_geometry_factor = my_measurement.experiment_parameters["li_top_sigma_multiplier"]
    detuning = frequency_multiplier * (nominal_frequency - nominal_resonance_frequency) + hf_lock_frequency_adjustment
    my_run_image_array = get_raw_pixels_top_B(my_measurement, my_run, memmap = True)
    atom_density_image = image_processing_functions.get_atom_density_absorption(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                            norm_box_coordinates=my_measurement.measurement_parameters["norm_box"],
                                                            rebin_pixel_num = rebin_pixel_num, detuning = detuning, 
                                                            cross_section_imaging_geometry_factor=top_cross_section_geometry_factor)
    return atom_density_image


def get_atom_densities_top_abs(my_measurement, my_run, first_state_index = 1, second_state_index = 3, b_field_condition = "unitarity", 
                               rebin_pixel_num = None):
    return (get_atom_density_top_A_abs(my_measurement, my_run, state_index = first_state_index, b_field_condition=b_field_condition, rebin_pixel_num = rebin_pixel_num),
     get_atom_density_top_B_abs(my_measurement, my_run, state_index = second_state_index, b_field_condition = b_field_condition, rebin_pixel_num = rebin_pixel_num))



def get_atom_densities_top_polrot(my_measurement, my_run, first_state_index = 1, second_state_index = 3, b_field_condition = "unitarity", 
                                  rebin_pixel_num = None, first_state_fudge = 1.0, second_state_fudge = 1.0, use_saturation = True, 
                                  average_saturation = False):
    first_state_resonance_frequency = _get_resonance_frequency_from_state_index(my_measurement, first_state_index)
    second_state_resonance_frequency = _get_resonance_frequency_from_state_index(my_measurement, second_state_index)
    nominal_frequency_A = my_run.parameters["ImagFreq1"]
    nominal_frequency_B = my_run.parameters["ImagFreq2"]
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    hf_lock_frequency_adjustment = _get_hf_lock_frequency_adjustment_from_b_field_condition(my_measurement, b_field_condition)
    detuning_1A = frequency_multiplier * (nominal_frequency_A - first_state_resonance_frequency) + hf_lock_frequency_adjustment
    detuning_1B = frequency_multiplier * (nominal_frequency_B - first_state_resonance_frequency) + hf_lock_frequency_adjustment
    detuning_2A = frequency_multiplier * (nominal_frequency_A - second_state_resonance_frequency) + hf_lock_frequency_adjustment
    detuning_2B = frequency_multiplier * (nominal_frequency_B - second_state_resonance_frequency) + hf_lock_frequency_adjustment
    top_cross_section_geometry_factor = my_measurement.experiment_parameters["li_top_sigma_multiplier"]
    polrot_phase_sign = my_measurement.experiment_parameters["polrot_phase_sign"]
    image_array_A = get_raw_pixels_top_A(my_measurement, my_run, memmap = True)
    image_array_B = get_raw_pixels_top_B(my_measurement, my_run, memmap = True)
    abs_image_A = image_processing_functions.get_absorption_image(image_array_A, 
                                                                ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates=my_measurement.measurement_parameters["norm_box"], 
                                                                rebin_pixel_num = rebin_pixel_num)
    abs_image_B = image_processing_functions.get_absorption_image(image_array_B, ROI = my_measurement.measurement_parameters["ROI"], 
                                                    norm_box_coordinates=my_measurement.measurement_parameters["norm_box"], 
                                                    rebin_pixel_num = rebin_pixel_num)
    if not use_saturation:        
        intensities_A = None 
        intensities_B = None 
        intensities_sat = None
    else:
        intensities_sat = get_saturation_counts_top(my_measurement, my_run)
        intensities_A = image_processing_functions.get_without_atoms_counts(image_array_A, 
                                                                ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"], 
                                                                rebin_pixel_num = rebin_pixel_num)
        intensities_B = image_processing_functions.get_without_atoms_counts(image_array_B, 
                                                                ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"], 
                                                                rebin_pixel_num = rebin_pixel_num)
        if average_saturation:
            intensities_A = np.average(intensities_A)
            intensities_B = np.average(intensities_B)
    atom_density_first, atom_density_second = image_processing_functions.get_atom_density_from_polrot_images(abs_image_A, abs_image_B,
                                                                detuning_1A, detuning_1B, detuning_2A, detuning_2B, phase_sign = polrot_phase_sign, 
                                                                cross_section_imaging_geometry_factor = top_cross_section_geometry_factor,
                                                                intensities_A = intensities_A, intensities_B = intensities_B,  
                                                                intensities_sat = intensities_sat)
    atom_density_first = atom_density_first * first_state_fudge
    atom_density_second = atom_density_second * second_state_fudge
    return (atom_density_first, atom_density_second)

def get_atom_densities_box_autocut(my_measurement, my_run, vert_crop_point = 0.5, horiz_crop_point = 0.00, widths_free = False, density_to_use = 2,
                                   first_stored_density_name = None, second_stored_density_name = None, imaging_mode = "polrot",
                                   **get_density_kwargs):
    density_1, density_2 = _load_densities_top_double(my_measurement, my_run, first_stored_density_name, second_stored_density_name, 
                                                    imaging_mode, **get_density_kwargs)
    if density_to_use == 1:
        crop_density = density_1 
    elif density_to_use == 2:
        crop_density = density_2 
    else:
        raise ValueError("Density_to_use must be 1 or 2.")
    box_crop = box_autocut(my_measurement, crop_density, vert_crop_point = vert_crop_point, 
                            horiz_crop_point = horiz_crop_point, widths_free = widths_free)
    x_min, y_min, x_max, y_max = box_crop 
    density_1_cropped = density_1[y_min:y_max, x_min:x_max] 
    density_2_cropped = density_2[y_min:y_max, x_min:x_max]
    return (density_1_cropped, density_2_cropped)

def get_top_atom_densities_COM_centered(my_measurement, my_run, first_stored_density_name = None, second_stored_density_name = None, 
                                    imaging_mode = "polrot", density_to_use = 2, crop_not_pad = True, **get_density_kwargs):
    density_1, density_2 = _load_densities_top_double(my_measurement, my_run, first_stored_density_name, second_stored_density_name, 
                                                    imaging_mode, **get_density_kwargs)
    if density_to_use == 1:
        com_density = density_1
    elif density_to_use == 2:
        com_density = density_2
    else:
        raise ValueError("Density_to_use must be 1 or 2.")
    y_com, x_com = image_processing_functions.get_image_coms(com_density)
    original_x_center = (com_density.shape[1] - 1) / 2
    original_y_center = (com_density.shape[0] - 1) / 2
    x_com_center_difference = x_com - original_x_center
    y_com_center_difference = y_com - original_y_center
    return _com_center_helper(crop_not_pad, density_1, density_2, x_com_center_difference, y_com_center_difference)

def _com_center_helper(crop_not_pad, density_1, density_2, x_com_center_difference, y_com_center_difference):
    #if crop_not_pad is true, then shift the pixel center by cropping the image 
    center_x_pix_shift = int(np.rint(2 * x_com_center_difference))
    center_y_pix_shift = int(np.rint(2 * y_com_center_difference))
    if crop_not_pad:
        if center_y_pix_shift >= 0:
            y_cropped_density_1 = density_1[center_y_pix_shift:]
            y_cropped_density_2 = density_2[center_y_pix_shift:]
        else:
            y_cropped_density_1 = density_1[:center_y_pix_shift] 
            y_cropped_density_2 = density_2[:center_y_pix_shift]
        if center_x_pix_shift >= 0:
            returned_density_1 = y_cropped_density_1[:, center_x_pix_shift:] 
            returned_density_2 = y_cropped_density_2[:, center_x_pix_shift:]
        else:
            returned_density_1 = y_cropped_density_1[:, :center_x_pix_shift] 
            returned_density_2 = y_cropped_density_2[:, :center_x_pix_shift]
    else:
        if center_y_pix_shift >= 0:
            y_pad_tuple = (0, center_y_pix_shift) 
        else:
            y_pad_tuple = (-center_y_pix_shift, 0)
        if center_x_pix_shift >= 0:
            x_pad_tuple = (0, center_x_pix_shift)
        else:
            x_pad_tuple = (-center_x_pix_shift, 0)
        returned_density_1 = np.pad(density_1, (y_pad_tuple, x_pad_tuple))
        returned_density_2 = np.pad(density_2, (y_pad_tuple, x_pad_tuple))
    return (returned_density_1, returned_density_2)


def get_x_integrated_atom_densities_top_double(my_measurement, my_run, first_stored_density_name = None, second_stored_density_name = None, 
                                    imaging_mode = "polrot", **get_density_kwargs):
    density_1, density_2 = _load_densities_top_double(my_measurement, my_run, first_stored_density_name, second_stored_density_name, 
                                                    imaging_mode, **get_density_kwargs)
    density_1_x_integrated = np.sum(density_1, axis = 1) * my_measurement.experiment_parameters["top_um_per_pixel"]
    density_2_x_integrated = np.sum(density_2, axis = 1) * my_measurement.experiment_parameters["top_um_per_pixel"]
    return (density_1_x_integrated, density_2_x_integrated)


def get_y_integrated_atom_densities_top_double(my_measurement, my_run, first_stored_density_name = None, second_stored_density_name = None, 
                                        imaging_mode = "polrot", **get_density_kwargs):
    density_1, density_2 = _load_densities_top_double(my_measurement, my_run, first_stored_density_name, second_stored_density_name, 
                                                    imaging_mode, **get_density_kwargs)
    density_1_y_integrated = np.sum(density_1, axis = 0) * my_measurement.experiment_parameters["top_um_per_pixel"]
    density_2_y_integrated = np.sum(density_2, axis = 0) * my_measurement.experiment_parameters["top_um_per_pixel"]
    return (density_1_y_integrated, density_2_y_integrated)



def get_xy_atom_density_pixel_coms_top_double(my_measurement, my_run, first_stored_density_name = None, second_stored_density_name = None, 
                                        imaging_mode = "polrot", **get_density_kwargs):
    
    density_1, density_2 = _load_densities_top_double(my_measurement, my_run, first_stored_density_name, second_stored_density_name, 
                                                      imaging_mode, **get_density_kwargs)
    y_com_1, x_com_1 = image_processing_functions.get_image_coms(density_1)
    y_com_2, x_com_2 = image_processing_functions.get_image_coms(density_2)
    return (x_com_1, y_com_1, x_com_2, y_com_2)

#ATOM COUNTS

"""
For these and subsequent functions, there is a common keyword parameter: stored_density_name. 
If passed, the functions assume that the atom density is stored as an analysis result within the run 
object, under the name stored_density_name. This is to allow patterns where multiple analyses which rely on 
the atom number density can be run without re-calculating it (typically the slowest part of any analysis)."""

def _get_atom_count_from_density(pixel_length_um, atom_density):
    pixel_area_um = np.square(pixel_length_um)
    return image_processing_functions.atom_count_pixel_sum(atom_density, pixel_area_um)


def get_atom_count_side_li_lf(my_measurement, my_run, stored_density_name = None):
    atom_density = _load_density_side_li_lf(my_measurement, my_run, stored_density_name)
    pixel_length_um = my_measurement.experiment_parameters["side_low_mag_um_per_pixel"]
    return _get_atom_count_from_density(pixel_length_um, atom_density)


def get_atom_count_side_li_hf(my_measurement, my_run, stored_density_name = None, **get_density_kwargs):
    atom_density = _load_density_side_li_hf(my_measurement, my_run, stored_density_name, **get_density_kwargs)
    pixel_length_um = my_measurement.experiment_parameters["side_high_mag_um_per_pixel"]
    return _get_atom_count_from_density(pixel_length_um, atom_density)


def get_atom_count_top_A_abs(my_measurement, my_run, stored_density_name = None, **get_density_kwargs):
    atom_density = _load_density_top_A_abs(my_measurement, my_run, stored_density_name, **get_density_kwargs)
    pixel_length_um = my_measurement.experiment_parameters["top_um_per_pixel"]
    return _get_atom_count_from_density(pixel_length_um, atom_density)

def get_atom_count_top_B_abs(my_measurement, my_run, stored_density_name = None, **get_density_kwargs):
    atom_density = _load_density_top_B_abs(my_measurement, my_run, stored_density_name, **get_density_kwargs)
    pixel_length_um = my_measurement.experiment_parameters["top_um_per_pixel"]
    return _get_atom_count_from_density(pixel_length_um, atom_density)

def get_atom_counts_top_AB_abs(my_measurement, my_run, first_stored_density_name = None, second_stored_density_name = None, 
                               **get_density_kwargs):
    atom_density_A, atom_density_B = _load_densities_top_AB_abs(my_measurement, my_run, first_stored_density_name, second_stored_density_name, 
                                                                    **get_density_kwargs)
    pixel_length_um = my_measurement.experiment_parameters["top_um_per_pixel"] 
    counts_A = _get_atom_count_from_density(pixel_length_um, atom_density_A) 
    counts_B = _get_atom_count_from_density(pixel_length_um, atom_density_B)
    return (counts_A, counts_B)

def get_atom_counts_top_polrot(my_measurement, my_run, first_stored_density_name = None, second_stored_density_name = None, 
                               **get_density_kwargs):
    atom_density_first, atom_density_second = _load_densities_polrot(my_measurement, my_run, 
                                                first_stored_density_name, second_stored_density_name, **get_density_kwargs)
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    atom_count_first = image_processing_functions.atom_count_pixel_sum(atom_density_first, pixel_area)
    atom_count_second = image_processing_functions.atom_count_pixel_sum(atom_density_second, pixel_area)
    return (atom_count_first, atom_count_second)

#TRAP-SPECIFIC_ANALYSES

#HYBRID TRAP - BOX EXP

def get_hybrid_trap_densities_along_harmonic_axis(my_measurement, my_run, autocut = False, 
                                                  first_stored_density_name = None, second_stored_density_name = None, 
                                                  imaging_mode = "polrot", return_positions = True, return_potentials = False, 
                                                    **get_density_kwargs):
    atom_density_first, atom_density_second = _load_densities_top_double(
                                                my_measurement, my_run, first_stored_density_name, second_stored_density_name, 
                                                imaging_mode, **get_density_kwargs)

    positions_first, densities_first = _hybrid_trap_density_helper(my_measurement, atom_density_first)
    positions_second, densities_second = _hybrid_trap_density_helper(my_measurement, atom_density_second)
    if autocut:
        first_start_index, first_stop_index = science_functions.hybrid_trap_autocut(densities_first)
        positions_first = positions_first[first_start_index:first_stop_index]
        densities_first = densities_first[first_start_index:first_stop_index]
        second_start_index, second_stop_index = science_functions.hybrid_trap_autocut(densities_second) 
        positions_second = positions_second[second_start_index:second_stop_index] 
        densities_second = densities_second[second_start_index:second_stop_index]

    axial_trap_freq = my_measurement.experiment_parameters["axial_trap_frequency_hz"]
    potentials_first = science_functions.get_li_energy_hz_in_1D_trap(positions_first * 1e-6, axial_trap_freq)
    potentials_second = science_functions.get_li_energy_hz_in_1D_trap(positions_second * 1e-6, axial_trap_freq)

    first_return_list = []
    second_return_list = []
    if return_positions:
        first_return_list.append(positions_first)
        second_return_list.append(positions_second)
    if return_potentials:
        first_return_list.append(potentials_first)
        second_return_list.append(potentials_second)
    first_return_list.append(densities_first)
    second_return_list.append(densities_second)
    return (*first_return_list, *second_return_list)

##START

def _hybrid_trap_density_helper(my_measurement, hybrid_trap_density_image):
    axicon_tilt_deg = my_measurement.experiment_parameters["axicon_tilt_deg"]
    axicon_diameter_pix = my_measurement.experiment_parameters["axicon_diameter_pix"]
    axicon_length_pix = my_measurement.experiment_parameters["hybrid_trap_typical_length_pix"]
    um_per_pixel = my_measurement.experiment_parameters["top_um_per_pixel"]
    center = data_fitting_functions.hybrid_trap_center_finder(hybrid_trap_density_image, axicon_tilt_deg, axicon_diameter_pix, axicon_length_pix)
    if(axicon_tilt_deg != 0.0):
        image_to_use, new_center = _rotate_and_crop_hybrid_image(hybrid_trap_density_image, center, axicon_tilt_deg)
        center = new_center 
    else:
        image_to_use = hybrid_trap_density_image
    x_center, y_center = center
    hybrid_trap_radius_um = um_per_pixel * axicon_diameter_pix / 2.0
    hybrid_trap_cross_sectional_area_um = get_hybrid_cross_section_um(my_measurement)
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


##END  

def get_hybrid_trap_average_energy(my_measurement, my_run, autocut = True, imaging_mode = "polrot", return_sub_energies = False,
                                    first_stored_density_name = None, second_stored_density_name = None, **get_density_kwargs):
    positions_first, densities_first, positions_second, densities_second = get_hybrid_trap_densities_along_harmonic_axis( 
                                                                    my_measurement, my_run, autocut = autocut, 
                                                                    imaging_mode = imaging_mode,
                                                                    first_stored_density_name = first_stored_density_name, 
                                                                    second_stored_density_name = second_stored_density_name, 
                                                                    **get_density_kwargs)
    trap_cross_section_um = get_hybrid_cross_section_um(my_measurement)
    trap_freq = my_measurement.experiment_parameters["axial_trap_frequency_hz"]
    average_energy_first = science_functions.get_hybrid_trap_average_energy(positions_first, densities_first, trap_cross_section_um,
                                                                            trap_freq) 
    counts_first = trapezoid(trap_cross_section_um * densities_first, x = positions_first)
    average_energy_second = science_functions.get_hybrid_trap_average_energy(positions_second, densities_second, trap_cross_section_um,
                                                                            trap_freq) 
    counts_second = trapezoid(trap_cross_section_um * densities_second, x = positions_second)
    overall_average_energy = (average_energy_first * counts_first + average_energy_second * counts_second) / (counts_first + counts_second)
    if return_sub_energies:
        return (overall_average_energy, average_energy_first, average_energy_second) 
    else:
        return overall_average_energy
    

def get_hybrid_trap_compressibilities(my_measurement, my_run, autocut = False, imaging_mode = "polrot", return_errors = False, 
                                    return_positions = False, return_potentials = True, window_size = 21, first_stored_density_name = None,
                                      second_stored_density_name = None, **get_density_kwargs):
    positions_1, potentials_1, densities_1, positions_2, potentials_2, densities_2 = get_hybrid_trap_densities_along_harmonic_axis( 
                                                                    my_measurement, my_run, autocut = autocut, 
                                                                    imaging_mode = imaging_mode,
                                                                    return_potentials = True, return_positions = True,
                                                                    first_stored_density_name = first_stored_density_name, 
                                                                    second_stored_density_name = second_stored_density_name, 
                                                                    **get_density_kwargs)
    return _compressibility_helper(positions_1, potentials_1, densities_1, positions_2, potentials_2, densities_2, 
                                   return_positions, return_potentials, return_errors, window_size)
  

def _compressibility_helper(positions_1, potentials_1, densities_1, positions_2, potentials_2, densities_2, 
                            return_positions, return_potentials, return_errors, window_size):
    index_breakpoints_1 = np.arange(0, len(potentials_1), window_size)
    energy_midpoints_1 = (potentials_1[index_breakpoints_1][:-1] + potentials_1[index_breakpoints_1 - 1][1:]) / 2.0
    position_midpoints_1 = (positions_1[index_breakpoints_1][:-1] + positions_1[index_breakpoints_1 - 1][1:]) / 2.0

    index_breakpoints_2 = np.arange(0, len(potentials_2), window_size)
    energy_midpoints_2 = (potentials_2[index_breakpoints_2][:-1] + potentials_2[index_breakpoints_2 - 1][1:]) / 2.0
    position_midpoints_2 = (positions_2[index_breakpoints_2][:-1] + positions_2[index_breakpoints_2 - 1][1:]) / 2.0
    compressibility_result_1 = science_functions.get_hybrid_trap_compressibilities_window_fit(
        potentials_1, densities_1, index_breakpoints_1, return_errors = return_errors
    )

    compressibility_result_2 = science_functions.get_hybrid_trap_compressibilities_window_fit(
        potentials_2, densities_2, index_breakpoints_2, return_errors = return_errors
    )

    return_list_1 = [] 
    return_list_2 = []

    if return_positions:
        return_list_1.append(position_midpoints_1)
        return_list_2.append(position_midpoints_2)
    if return_potentials:
        return_list_1.append(energy_midpoints_1)
        return_list_2.append(energy_midpoints_2)
    if return_errors:
        compressibility_1, error_1 = compressibility_result_1
        return_list_1.append(compressibility_1)
        return_list_1.append(error_1) 
        compressibility_2, error_2 = compressibility_result_2
        return_list_2.append(compressibility_2)
        return_list_2.append(error_2)
    else:
        compressibility_1 = compressibility_result_1
        return_list_1.append(compressibility_1)
        compressibility_2 = compressibility_result_2 
        return_list_2.append(compressibility_2)
    return (*return_list_1, *return_list_2)    


#AXIAL SQUISH


def get_axial_squish_densities_along_harmonic_axis(my_measurement, my_run, autocut = False, 
                                                  first_stored_density_name = None, second_stored_density_name = None, 
                                                  imaging_mode = "polrot", return_positions = False, return_potentials = True,
                                                  majority_autocut_upper_buffer = 0,**get_density_kwargs):
    
    densities_first, densities_second = get_hybrid_trap_densities_along_harmonic_axis(my_measurement, my_run, autocut = False, 
                                        first_stored_density_name = first_stored_density_name, second_stored_density_name = second_stored_density_name, 
                                        imaging_mode = imaging_mode, return_positions = False, return_potentials = False, 
                                        **get_density_kwargs)
    index_positions = np.arange(len(densities_first)) 
    absolute_hybrid_center_index = my_measurement.experiment_parameters["hybrid_trap_center_pix_polrot"]
    _, roi_ymin, *_ = my_measurement.measurement_parameters["ROI"]
    relative_hybrid_center_index = absolute_hybrid_center_index - roi_ymin
    center_referenced_index_positions = index_positions - relative_hybrid_center_index
    center_referenced_positions_um = center_referenced_index_positions * my_measurement.experiment_parameters["top_um_per_pixel"]
    trap_freq = my_measurement.experiment_parameters["axial_trap_frequency_hz"]
    harmonic_potential = science_functions.get_li_energy_hz_in_1D_trap(center_referenced_positions_um * 1e-6, trap_freq)
    gradient_voltage = my_run.parameters["Axial_Squish_Imaging_Grad_V"]
    gradient_voltage_to_Hz_um = my_measurement.experiment_parameters["axial_gradient_Hz_per_um_V"]
    gradient_Hz_um = gradient_voltage * gradient_voltage_to_Hz_um 
    gradient_potential = center_referenced_positions_um * gradient_Hz_um 
    overall_potential = gradient_potential + harmonic_potential
    if autocut:
        _, upper_cut_index_1 = science_functions.hybrid_trap_autocut(densities_first)
        _, upper_cut_index_2 = science_functions.hybrid_trap_autocut(densities_second)
        #Lower cut is done at majority cloud max density, upper cut as for the hybrid trap 
        if np.sum(densities_first) > np.sum(densities_second):
            majority_densities = densities_first 
            upper_cut_index_1 += majority_autocut_upper_buffer
        else:
            majority_densities = densities_second 
            upper_cut_index_2 += majority_autocut_upper_buffer
        LOWER_CUT_BUFFER = 5
        lower_cut_index = np.argmax(majority_densities) + LOWER_CUT_BUFFER
        densities_first = densities_first[lower_cut_index:upper_cut_index_1] 
        densities_second = densities_second[lower_cut_index:upper_cut_index_2] 
        potentials_first = overall_potential[lower_cut_index:upper_cut_index_1] 
        potentials_second = overall_potential[lower_cut_index:upper_cut_index_2] 
        positions_first = center_referenced_positions_um[lower_cut_index:upper_cut_index_1] 
        positions_second = center_referenced_positions_um[lower_cut_index:upper_cut_index_2]
    else:
        potentials_first = overall_potential
        potentials_second = overall_potential
        positions_first = center_referenced_positions_um 
        positions_second = center_referenced_positions_um
    return_list_1 = [] 
    return_list_2 = [] 
    if return_positions:
        return_list_1.append(positions_first) 
        return_list_2.append(positions_second)
    if return_potentials:
        return_list_1.append(potentials_first) 
        return_list_2.append(potentials_second)
    return_list_1.append(densities_first)
    return_list_2.append(densities_second)
    return (*return_list_1, *return_list_2)


def get_axial_squish_absolute_pressures(my_measurement, my_run, autocut = True, 
                                                  first_stored_density_name = None, second_stored_density_name = None, 
                                                  imaging_mode = "polrot", return_positions = False, return_potentials = True,
                                                  return_densities = False, **get_density_kwargs):
    positions_1, potentials_1, densities_1, positions_2, potentials_2, densities_2 = get_axial_squish_densities_along_harmonic_axis(
        my_measurement, my_run, autocut = autocut, first_stored_density_name = first_stored_density_name, 
        second_stored_density_name = second_stored_density_name, imaging_mode = imaging_mode, return_positions = True, 
        return_potentials = True, **get_density_kwargs
    )
    pressures_1_absolute = science_functions.get_absolute_pressures(potentials_1, densities_1)
    pressures_2_absolute = science_functions.get_absolute_pressures(potentials_2, densities_2)
    return_list_1 = []
    return_list_2 = []
    if return_positions:
        return_list_1.append(positions_1)
        return_list_2.append(positions_2)
    if return_potentials:
        return_list_1.append(potentials_1)
        return_list_2.append(potentials_2)
    if return_densities:
        return_list_1.append(densities_1)
        return_list_2.append(densities_2)
    return_list_1.append(pressures_1_absolute)
    return_list_2.append(pressures_2_absolute)
    return (*return_list_1, *return_list_2)


def get_axial_squish_normalized_pressures(my_measurement, my_run, autocut = True, normalized_pressure_cut = True, 
                                          normalized_pressure_relative_cut_point = 0.1, first_stored_density_name = None, 
                                          second_stored_density_name = None, imaging_mode = "polrot", return_positions = False,
                                            return_potentials = True, return_densities = False, **get_density_kwargs):
    (positions_1, potentials_1, densities_1, pressures_1_absolute, 
     positions_2, potentials_2, densities_2, pressures_2_absolute) = get_axial_squish_absolute_pressures(
         my_measurement, my_run, autocut = autocut, first_stored_density_name=first_stored_density_name, 
         second_stored_density_name = second_stored_density_name, imaging_mode = imaging_mode, 
         return_positions = True, return_potentials = True, return_densities = True, **get_density_kwargs)
    fermi_pressures_1 = science_functions.get_ideal_fermi_pressure_hz_um_from_density(densities_1 * 1e18) 
    fermi_pressures_2 = science_functions.get_ideal_fermi_pressure_hz_um_from_density(densities_2 * 1e18)
    normalized_pressures_1 = pressures_1_absolute / fermi_pressures_1 
    normalized_pressures_2 = pressures_2_absolute / fermi_pressures_2
    if normalized_pressure_cut:
        density_cut_point_1 = normalized_pressure_relative_cut_point * np.max(densities_1)
        _, pressure_cut_index_1 = science_functions.hybrid_trap_autocut(densities_1, cut_value = density_cut_point_1)
        density_cut_point_2 = normalized_pressure_relative_cut_point * np.max(densities_2)
        _, pressure_cut_index_2 = science_functions.hybrid_trap_autocut(densities_2, cut_value = density_cut_point_2)

        normalized_pressures_1 = normalized_pressures_1[:pressure_cut_index_1] 
        normalized_pressures_2 = normalized_pressures_2[:pressure_cut_index_2]

        positions_1 = positions_1[:pressure_cut_index_1] 
        positions_2 = positions_2[:pressure_cut_index_2]

        potentials_1 = potentials_1[:pressure_cut_index_1] 
        potentials_2 = potentials_2[:pressure_cut_index_2]

        densities_1 = densities_1[:pressure_cut_index_1] 
        densities_2 = densities_2[:pressure_cut_index_2] 
    return_list_1 = [] 
    return_list_2 = [] 
    if return_positions:
        return_list_1.append(positions_1) 
        return_list_2.append(positions_2) 
    if return_potentials:
        return_list_1.append(potentials_1) 
        return_list_2.append(potentials_2) 
    if return_densities:
        return_list_1.append(densities_1) 
        return_list_2.append(densities_2) 

    return_list_1.append(normalized_pressures_1) 
    return_list_2.append(normalized_pressures_2)

    return (*return_list_1, *return_list_2) 


def get_axial_squish_compressibilities(my_measurement, my_run, autocut = False, imaging_mode = "polrot", return_errors = False, 
                                    return_positions = False, return_potentials = True, return_densities = False, 
                                      window_size = 21, first_stored_density_name = None,
                                      second_stored_density_name = None, **get_density_kwargs):
    positions_1, potentials_1, densities_1, positions_2, potentials_2, densities_2 = get_axial_squish_densities_along_harmonic_axis(
        my_measurement, my_run, autocut = autocut, first_stored_density_name=first_stored_density_name, 
        second_stored_density_name = second_stored_density_name, imaging_mode = imaging_mode, return_positions = True, 
        return_potentials = True, **get_density_kwargs)
    return _compressibility_helper(positions_1, potentials_1, densities_1, 
                    positions_2, potentials_2, densities_2, return_positions, return_potentials, return_errors, 
                    window_size)


def get_axial_squish_compressibilities_vs_pressure(my_measurement, my_run, autocut = True, imaging_mode = "polrot", return_errors = False, 
                                    return_imbalances = False, window_size = 21, normalized_pressure_cut = True,
                                    normalized_pressure_relative_cut_point = 0.1, first_stored_density_name = None,
                                      second_stored_density_name = None, **get_density_kwargs):
    compressibility_return_values = get_axial_squish_compressibilities(
            my_measurement, my_run, autocut = autocut, imaging_mode = imaging_mode, return_errors = return_errors, 
            return_positions = True, return_potentials = False, return_densities = False, window_size = window_size, 
            first_stored_density_name = first_stored_density_name, second_stored_density_name = second_stored_density_name, 
            **get_density_kwargs)
    if return_errors:
        position_midpoints_1, compressibilities_1, errors_1, position_midpoints_2, compressibilities_2, errors_2 = compressibility_return_values
    else:
        position_midpoints_1, compressibilities_1, position_midpoints_2, compressibilities_2 = compressibility_return_values
    positions_1, densities_1, normalized_pressures_1, positions_2, densities_2, normalized_pressures_2 = get_axial_squish_normalized_pressures(
        my_measurement, my_run, autocut = autocut, normalized_pressure_cut = normalized_pressure_cut, 
        normalized_pressure_relative_cut_point = normalized_pressure_relative_cut_point, 
        first_stored_density_name = first_stored_density_name, 
        second_stored_density_name = second_stored_density_name, 
        imaging_mode = imaging_mode, return_positions = True, return_potentials = False, return_densities = True, 
         **get_density_kwargs)
    
    position_range_1_lower = np.min(positions_1)
    position_range_1_upper = np.max(positions_1)

    position_range_2_lower = np.min(positions_2)
    position_range_2_upper = np.max(positions_2)

    included_indices_1 = np.logical_and(position_midpoints_1 > position_range_1_lower, position_midpoints_1 < position_range_1_upper)
    included_indices_2 = np.logical_and(position_midpoints_2 > position_range_2_lower, position_midpoints_2 < position_range_2_upper)

    included_position_midpoints_1 = position_midpoints_1[included_indices_1] 
    included_compressibilities_1 = compressibilities_1[included_indices_1]

    included_position_midpoints_2 = position_midpoints_2[included_indices_2]
    included_compressibilities_2 = compressibilities_2[included_indices_2]

    if return_errors: 
        included_errors_1 = errors_1[included_indices_1] 
        included_errors_2 = errors_2[included_indices_2] 

    smoothed_normalized_pressures_1 = savgol_filter(normalized_pressures_1, window_size, 2)
    smoothed_normalized_pressures_2 = savgol_filter(normalized_pressures_2, window_size, 2) 

    #Positions are monotonically increasing
    compressibility_interpolated_pressures_1 = np.interp(included_position_midpoints_1, positions_1, smoothed_normalized_pressures_1)
    compressibility_interpolated_pressures_2 = np.interp(included_position_midpoints_2, positions_2, smoothed_normalized_pressures_2)

    smoothed_densities_1 = savgol_filter(densities_1, window_size, 2)
    smoothed_densities_2 = savgol_filter(densities_2, window_size, 2)

    def imbalance_interpolant_function(position):
        interp_densities_1 = np.interp(position, positions_1, smoothed_densities_1) 
        interp_densities_2 = np.interp(position, positions_2, smoothed_densities_2)
        return (interp_densities_2 - interp_densities_1) / (interp_densities_2 + interp_densities_1)

    imbalances_1 = np.where(
        np.logical_and(included_position_midpoints_1 > position_range_2_lower, included_position_midpoints_1 < position_range_2_upper), 
        -1.0 * imbalance_interpolant_function(included_position_midpoints_1), 
        -1.0
    )

    imbalances_2 = np.where(
        np.logical_and(included_position_midpoints_2 > position_range_1_lower, included_position_midpoints_2 < position_range_1_upper), 
        imbalance_interpolant_function(included_position_midpoints_2), 
        1.0
    )

    return_list_1 = [] 
    return_list_2 = [] 

    return_list_1.append(compressibility_interpolated_pressures_1)
    return_list_2.append(compressibility_interpolated_pressures_2)

    if return_imbalances:
        return_list_1.append(imbalances_1) 
        return_list_2.append(imbalances_2)

    return_list_1.append(included_compressibilities_1) 
    return_list_2.append(included_compressibilities_2) 

    if return_errors:
        return_list_1.append(included_errors_1)
        return_list_2.append(included_errors_2)
    
    return (*return_list_1, *return_list_2)

def get_balanced_axial_squish_fitted_mu_and_T(my_measurement, my_run, autocut = True, 
                                                  first_stored_density_name = None, second_stored_density_name = None, 
                                                  imaging_mode = "polrot", fit_prefactor = False, return_errors = False,
                                                  show_plots = False, save_plots = False, save_pathname = ".",
                                                    **get_density_kwargs):
    #Only one species is important, since the gas is by assumption balanced
    potentials_1, densities_1, *_ = get_axial_squish_densities_along_harmonic_axis(
        my_measurement, my_run, autocut = autocut, first_stored_density_name = first_stored_density_name, 
        second_stored_density_name = second_stored_density_name, imaging_mode = imaging_mode, 
        return_positions = False, return_potentials = True, **get_density_kwargs)
    if fit_prefactor:
        fit_results = data_fitting_functions.fit_li6_balanced_density_with_prefactor(potentials_1, densities_1)
    else:
        fit_results = data_fitting_functions.fit_li6_balanced_density(potentials_1, densities_1)
    fit_popt, fit_pcov = fit_results 
    if show_plots or save_plots:
        run_id = my_run.parameters["id"]
        plt.plot(potentials_1, densities_1, label = "Data")
        if not fit_prefactor:
            mu, T = fit_popt 
            mu_kHz = mu / 1000 
            T_kHz = T / 1000
            plt.plot(potentials_1, data_fitting_functions.li6_balanced_density(potentials_1, *fit_popt), 
                label = "Fit, $\mu$ = {0:.1f} kHz, $T$ = {1:.1f} kHz".format(mu_kHz, T_kHz))
        else:
            mu, T, prefactor = fit_popt 
            mu_kHz = mu / 1000 
            T_kHz = T / 1000
            plt.plot(potentials_1, data_fitting_functions.li6_balanced_density_with_prefactor(potentials_1, *fit_popt), 
                label = "Fit, $\mu$ = {0:.1f} kHz, $T$ = {1:.1f} kHz, Prefactor = {2:.2f}".format(mu_kHz, T_kHz, prefactor))
        plt.legend()
        plt.xlabel("Potential (Hz)") 
        plt.ylabel("Density $\left(\mu m^{-3}\\right)$")
        plt.suptitle("Balanced Density Fitting, Run ID = {0:d}".format(run_id))
        if save_plots:
            figure_name = "{0:d}_Balanced_Squish_Fit.png".format(run_id)
            full_save_name = os.path.join(save_pathname, figure_name)
            plt.savefig(full_save_name, bbox_inches = "tight")
        if show_plots:
            plt.show()
        else:
            plt.cla()
    if return_errors:
        errors = np.sqrt(np.diag(fit_pcov))
        return (*fit_popt, *errors)
    else:
        return (*fit_popt,) 
    

def get_imbalanced_axial_squish_fitted_mu_and_T(my_measurement, my_run, autocut = True, majority_autocut_upper_buffer = 50,
                                                  first_stored_density_name = None, second_stored_density_name = None, 
                                                  imaging_mode = "polrot", fit_prefactor = False, return_errors = False,
                                                  show_plots = False, save_plots = False, save_pathname = ".",
                                                    **get_density_kwargs):
    potentials_1, densities_1, potentials_2, densities_2 = get_axial_squish_densities_along_harmonic_axis(
        my_measurement, my_run, autocut = autocut, first_stored_density_name = first_stored_density_name, 
        second_stored_density_name = second_stored_density_name, imaging_mode = imaging_mode, 
        return_positions = False, return_potentials = True, majority_autocut_upper_buffer=majority_autocut_upper_buffer,
        **get_density_kwargs)
    
    if np.sum(densities_1) > np.sum(densities_2):
        majority_densities = densities_1
        majority_potentials = potentials_1
        minority_potentials = potentials_2
        minority_densities = densities_2
    else:
        majority_densities = densities_2
        majority_potentials = potentials_2
        minority_potentials = potentials_1
        minority_densities = densities_1

    #If the autocutting hasn't been done yet, it MUST be done on the minority species on the right hand side
    if not autocut:
        _, spin_polarized_clip_index = science_functions.hybrid_trap_autocut(minority_densities)
    else:
        spin_polarized_clip_index = len(minority_potentials)

    clipped_majority_densities = majority_densities[spin_polarized_clip_index:]
    clipped_majority_potentials = majority_potentials[spin_polarized_clip_index:]

    if fit_prefactor:
        fit_results = data_fitting_functions.fit_li6_ideal_fermi_density_with_prefactor(clipped_majority_potentials, clipped_majority_densities)
    else:
        fit_results = data_fitting_functions.fit_li6_ideal_fermi_density(clipped_majority_potentials, clipped_majority_densities)
    fit_popt, fit_pcov = fit_results 
    fit_sigmas = np.sqrt(np.diag(fit_pcov))

    if show_plots or save_plots:
        run_id = my_run.parameters["id"]
        plt.plot(majority_potentials, majority_densities, label = "Majority")
        plt.plot(minority_potentials, minority_densities, label = "Minority")
        plt.vlines(min(clipped_majority_potentials), min(majority_densities), max(majority_densities), 
                   color = "black", linestyles = "dashed")
        plt.legend()
        plt.xlabel("Potential (Hz)")
        plt.ylabel("Atom Density $\left(\mu m^{-1}\\right)$")
        if save_plots:
            figure_save_name = "{0}_Imbalanced_Squish_Densities.png".format(str(run_id))
            full_save_path = os.path.join(save_pathname, figure_save_name)
            plt.savefig(full_save_path, bbox_inches = "tight")
        if show_plots:
            plt.show()
        else:
            plt.cla()
        #Then plot majority fitting 
        plt.plot(clipped_majority_potentials, clipped_majority_densities, label = "Data")
        if not fit_prefactor:
            mu, T = fit_popt 
            mu_err, T_err = fit_sigmas
            mu_kHz = mu / 1000 
            mu_err_kHz = mu_err / 1000
            T_kHz = T / 1000
            T_err_kHz = T_err / 1000
            plt.plot(clipped_majority_potentials, data_fitting_functions.li6_ideal_fermi_density(clipped_majority_potentials, *fit_popt), 
                label = "Fit, $\mu$ = {0:.1f} +- {1:.1f} kHz, $T$ = {2:.1f} +- {3:.1f} kHz".format(mu_kHz, mu_err_kHz, T_kHz, T_err_kHz))
        else:
            mu, T, prefactor = fit_popt 
            mu_err, T_err, prefactor_err = fit_sigmas
            mu_kHz = mu / 1000 
            T_kHz = T / 1000
            mu_err_kHz = mu_err / 1000 
            T_err_kHz = T_err / 1000

            plt.plot(clipped_majority_potentials, data_fitting_functions.li6_ideal_fermi_density_with_prefactor(clipped_majority_potentials, *fit_popt), 
                label = "Fit, $\mu$ = {0:.1f} +- {1:.1f} kHz, $T$ = {2:.1f} +- {3:.2f} kHz, Prefactor = {4:.3f} +- {5:.3f}".format(
                    mu_kHz, mu_err_kHz, T_kHz, T_err_kHz, prefactor, prefactor_err))
        plt.legend()
        plt.xlabel("Potential (Hz)") 
        plt.ylabel("Density $\left(\mu m^{-3}\\right)$")
        plt.suptitle("Ideal Fermi Density Fitting, Run ID = {0:d}".format(run_id))
        if save_plots:
            figure_name = "{0}_Imbalanced_Squish_Fit.png".format(str(run_id))
            full_save_name = os.path.join(save_pathname, figure_name)
            plt.savefig(full_save_name, bbox_inches = "tight")
        if show_plots:
            plt.show()
        else:
            plt.cla()


    if return_errors:
        errors = np.sqrt(np.diag(fit_pcov))
        return (*fit_popt, *errors)
    else:
        return (*fit_popt,)

    

#BOX TRAP

"""
Get the fourier component of designated order from box shake data. 

Given a measurement and run, extract (1d) fourier components from the atom densities, integrated along the 
non-shaken direction of the box. 

Params:

Most of the standard ones, plus, 

order: The order of the fourier component to extract. If not passed, the dominant (nonzero) fourier amplitude will be 
returned - though be warned that this may differ between runs!!!

no_shake_density_name_(first, second): If not None, the function will assume that the atom density of the 
(first/second) state with no box shaking is stored in measurement_analysis_results under the given name. 
If None, the analysis will run without background subtraction.

#NOTE: The analysis does not autorun the get_no_shake_average_profiles function because, as currently structured, 
this would involve a new call for every run to be analyzed. This could be worked around, but I consider it better 
to explicitly evaluate the density names first"""
def get_box_shake_fourier_amplitudes(my_measurement, my_run, return_phases = False, autocut = False, autocut_density_to_use = 2, 
                                     autocut_widths_free = False, autocut_vert_crop_point = 0.5, autocut_horiz_crop_point = 0.00, 
                                     no_shake_density_name_first = None, order = None, no_shake_density_name_second = None, 
                                     first_stored_density_name = None, second_stored_density_name = None, imaging_mode = "polrot", 
                                     **get_density_kwargs):
    if no_shake_density_name_first is None:
        no_shake_density_first = 0.0
    else:
        no_shake_density_first = my_measurement.measurement_analysis_results[no_shake_density_name_first]
        if autocut:
            no_shake_density_first = box_autocut(my_measurement, no_shake_density_first, vert_crop_point = autocut_vert_crop_point, 
                                                 horiz_crop_point = autocut_horiz_crop_point, widths_free = autocut_widths_free)
    if no_shake_density_name_second is None:
        no_shake_density_second = 0.0 
    else:
        no_shake_density_second = my_measurement.measurement_analysis_results[no_shake_density_name_second] 
        if autocut:
            no_shake_density_second = box_autocut(my_measurement, no_shake_density_second, vert_crop_point = autocut_vert_crop_point, 
                                                 horiz_crop_point = autocut_horiz_crop_point, widths_free = autocut_widths_free)    
    if autocut:
        atom_density_first, atom_density_second = get_atom_densities_box_autocut(
            my_measurement, my_run, vert_crop_point = autocut_vert_crop_point, horiz_crop_point = autocut_horiz_crop_point,
            widths_free = autocut_widths_free, density_to_use = autocut_density_to_use,
            first_stored_density_name = first_stored_density_name, second_stored_density_name = second_stored_density_name,
            imaging_mode = imaging_mode, **get_density_kwargs)
    else:
        atom_density_first, atom_density_second = _load_densities_top_double(my_measurement, my_run, 
                                                    first_stored_density_name, second_stored_density_name, imaging_mode, 
                                                    **get_density_kwargs)
    bs_density_first = atom_density_first - no_shake_density_first 
    bs_density_second = atom_density_second - no_shake_density_second
    #Current convention has the integration direction as the last index, i.e. the x-axis. 
    integrated_density_first = np.sum(bs_density_first, axis = -1) * my_measurement.experiment_parameters["top_um_per_pixel"]
    integrated_density_second = np.sum(bs_density_second, axis = -1) * my_measurement.experiment_parameters["top_um_per_pixel"]
    x_delta = my_measurement.experiment_parameters["top_um_per_pixel"]
    fft_results_first = data_fitting_functions.get_fft_peak(x_delta, integrated_density_first, order = order)
    frequency_first, amp_first, phase_first = fft_results_first 
    fft_results_second = data_fitting_functions.get_fft_peak(x_delta, integrated_density_second, order = order)
    frequency_second, amp_second, phase_second = fft_results_second
    if not return_phases:
        return (amp_first, amp_second)
    else:
        return (amp_first, phase_first, amp_second, phase_second)



def get_box_in_situ_fermi_energies_from_counts(my_measurement, my_run, first_stored_density_name = None, second_stored_density_name = None, 
                                               imaging_mode = "polrot", **get_density_kwargs):
    if imaging_mode == "polrot":
        counts_first, counts_second = get_atom_counts_top_polrot(my_measurement, my_run, first_stored_density_name = first_stored_density_name, 
                                                                 second_stored_density_name = second_stored_density_name, 
                                                                 **get_density_kwargs) 
    elif imaging_mode == "abs":
        counts_first, counts_second = get_atom_counts_top_AB_abs(my_measurement, my_run, first_stored_density_name = first_stored_density_name, 
                                                                 second_stored_density_name = second_stored_density_name, 
                                                                 **get_density_kwargs)
    box_length_pix = my_measurement.experiment_parameters["box_length_pix"]
    um_per_pixel = my_measurement.experiment_parameters["top_um_per_pixel"]
    box_length_um = box_length_pix * um_per_pixel
    cross_section_um = get_hybrid_cross_section_um(my_measurement)
    first_fermi_energy_hz = science_functions.get_box_fermi_energy_from_counts(counts_first, cross_section_um, box_length_um)
    second_fermi_energy_hz = science_functions.get_box_fermi_energy_from_counts(counts_second, cross_section_um, box_length_um)
    return (first_fermi_energy_hz, second_fermi_energy_hz)



#RAPID RAMP

"""
Get the integrated density along the rapid ramp harmonic axis."""
def get_rapid_ramp_densities_along_harmonic_axis(my_measurement, my_run, first_stored_density_name = None, second_stored_density_name = None,
                                                  imaging_mode = "abs", **get_density_kwargs):
    RR_B_FIELD_CONDITION = "rapid_ramp"
    if not "b_field_condition" in get_density_kwargs:
        get_density_kwargs["b_field_condition"] = RR_B_FIELD_CONDITION
    atom_density_first, atom_density_second = _load_densities_top_double(my_measurement, my_run,
                                                first_stored_density_name, second_stored_density_name, imaging_mode, 
                                                **get_density_kwargs)
    #Rotate images 
    rr_angle = my_measurement.experiment_parameters["rr_tilt_deg"]
    atom_density_first = ndimage.rotate(atom_density_first, rr_angle, reshape = False)
    atom_density_second = ndimage.rotate(atom_density_second, rr_angle, reshape = False)
    #The non-harmonic axis is the x axis in our experiments...
    integrated_atom_density_first = np.sum(atom_density_first, axis = -1) * my_measurement.experiment_parameters["top_um_per_pixel"]
    integrated_atom_density_second = np.sum(atom_density_second, axis = -1) * my_measurement.experiment_parameters["top_um_per_pixel"]
    return (integrated_atom_density_first, integrated_atom_density_second)


"""
Get the rapid ramp condensate fraction via a "correct", fit based approach that fits the condensate and 
thermals in a multi-step manner akin to that described in https://doi.org/10.1063/1.3125051"""

def get_rr_condensate_fractions_fit(my_measurement, my_run, first_stored_density_name = None, second_stored_density_name = None, 
                                    imaging_mode = "abs", **get_density_kwargs):
    integrated_atom_density_first, integrated_atom_density_second = get_rapid_ramp_densities_along_harmonic_axis(
        my_measurement, my_run, first_stored_density_name = first_stored_density_name, 
        second_stored_density_name = second_stored_density_name, imaging_mode = imaging_mode, **get_density_kwargs)
    condensate_results_first, thermal_results_first = data_fitting_functions.fit_one_dimensional_condensate(integrated_atom_density_first)
    condensate_results_second, thermal_results_second = data_fitting_functions.fit_one_dimensional_condensate(integrated_atom_density_second)
    condensate_popt_first, _ = condensate_results_first 
    condensate_popt_second, _ = condensate_results_second 
    thermal_popt_first, _ = thermal_results_first 
    thermal_popt_second, _ = thermal_results_second 
    condensate_counts_first = data_fitting_functions.one_d_condensate_integral(*condensate_popt_first)
    thermal_counts_first = data_fitting_functions.thermal_bose_integral(*thermal_popt_first)
    condensate_counts_second = data_fitting_functions.one_d_condensate_integral(*condensate_popt_second)
    thermal_counts_second = data_fitting_functions.thermal_bose_integral(*thermal_popt_second)
    condensate_fraction_first = condensate_counts_first / (condensate_counts_first + thermal_counts_first)
    condensate_fraction_second = condensate_counts_second / (condensate_counts_second + thermal_counts_second)
    return (condensate_fraction_first, condensate_fraction_second)


"""
Get the condensate fraction via a 'kludge': Define a box inside of which the condensate is found, subtract the average density of a region 
just outside that box, sum up the atom counts inside, and then"""
def get_rr_condensate_fractions_box(my_measurement, my_run, first_stored_density_name = None, second_stored_density_name = None, 
                                    imaging_mode = "abs", **get_density_kwargs):
    RR_B_FIELD_CONDITION = "rapid_ramp"
    if not "b_field_condition" in get_density_kwargs:
        get_density_kwargs["b_field_condition"] = "rapid_ramp"
    atom_density_first, atom_density_second = _load_densities_top_double(my_measurement, my_run, first_stored_density_name, 
                                                                        second_stored_density_name, imaging_mode, **get_density_kwargs)
    #Rotate images
    rr_angle = my_measurement.experiment_parameters["rr_tilt_deg"]
    atom_density_first = ndimage.rotate(atom_density_first, rr_angle, reshape = False)
    atom_density_second = ndimage.rotate(atom_density_second, rr_angle, reshape = False)
    #Naive summing approach with background subtraction outside of the box
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    total_counts_first = image_processing_functions.atom_count_pixel_sum(atom_density_first, pixel_area)
    total_counts_second = image_processing_functions.atom_count_pixel_sum(atom_density_second, pixel_area)
    rr_roi = my_measurement.measurement_parameters["rr_condensate_roi"]
    rr_xmin, rr_ymin, rr_xmax, rr_ymax = rr_roi 
    rr_density_first = image_processing_functions.subcrop(atom_density_first, rr_roi, my_measurement.measurement_parameters["ROI"])
    rr_density_second = image_processing_functions.subcrop(atom_density_second, rr_roi, my_measurement.measurement_parameters["ROI"])
    #Create a new box immediately adjacent to, but below, the rr_roi box
    subtract_xmin = rr_xmin
    subtract_xmax = rr_xmax
    subtract_ymax = rr_ymin
    subtract_ymin = rr_ymin - (rr_ymax - rr_ymin) 
    subtract_box = (subtract_xmin, subtract_ymin, subtract_xmax, subtract_ymax)
    subtract_density_first = image_processing_functions.subcrop(atom_density_first, subtract_box, my_measurement.measurement_parameters["ROI"])
    subtract_density_second = image_processing_functions.subcrop(atom_density_second, subtract_box, my_measurement.measurement_parameters["ROI"])
    subtract_average_first = np.average(subtract_density_first) 
    subtract_average_second = np.average(subtract_density_second)
    bs_rr_density_first = rr_density_first - subtract_average_first
    bs_rr_density_second = rr_density_second - subtract_average_second
    rr_counts_first = image_processing_functions.atom_count_pixel_sum(bs_rr_density_first, pixel_area) 
    rr_counts_second = image_processing_functions.atom_count_pixel_sum(bs_rr_density_second, pixel_area)
    rr_fraction_first = rr_counts_first / total_counts_first 
    rr_fraction_second = rr_counts_second / total_counts_second 
    return (rr_fraction_first, rr_fraction_second)



#MISC DENSITY MANIPULATION

"""
Normalize densities to uniform shape. 

Given an arbitrary variable name stored_density_name which for each run corresponds to an ndarray, 
crop or pad these potentially different-sized ndarrays into a uniform shape.

WARNING: As currently written, function silently adds a variable '{0}_uniform_reshape_dimensions'.'format(stored_density_name) to
my_measurement.measurement_analysis_results. May need to change this behavior if it's detrimental in the wild."""
def get_uniform_reshaped_density(my_measurement, my_run, stored_density_name = None, crop_not_pad = True, crop_or_pad_position = "sym"):
    stored_density = _load_mandatory_stored_density(my_measurement, my_run, stored_density_name)
    stored_density_reshape_dimensions_name = stored_density_name + "_uniform_reshape_dimensions"
    if not stored_density_reshape_dimensions_name in my_measurement.measurement_analysis_results:
        reshape_dimensions = get_uniform_density_reshape_dimensions(my_measurement, stored_density_name, crop_not_pad)
        my_measurement.measurement_analysis_results[stored_density_reshape_dimensions_name] = reshape_dimensions
    else:
        reshape_dimensions = my_measurement.measurement_analysis_results[stored_density_reshape_dimensions_name]
    reshape_dimensions_array = reshape_dimensions
    reshaped_density = _reshape_crop_pad_helper(stored_density, reshape_dimensions_array, crop_not_pad, crop_or_pad_position)
    return reshaped_density


def _reshape_crop_pad_helper(stored_density, reshape_dimensions_array, crop_not_pad, crop_or_pad_position):
    density_shape_array = np.array(stored_density.shape)
    if crop_not_pad:
        amounts_to_resize = density_shape_array - reshape_dimensions_array
    else:
        amounts_to_resize = reshape_dimensions_array - density_shape_array 

    if crop_or_pad_position == "sym":
        lower_increments = amounts_to_resize // 2 
        upper_increments = amounts_to_resize - lower_increments  
    elif crop_or_pad_position == "lower":
        lower_increments = amounts_to_resize
        upper_increments = np.zeros(amounts_to_resize.shape).astype(int)
    elif crop_or_pad_position == "upper":
        lower_increments = np.zeros(amounts_to_resize.shape).astype(int)
        upper_increments = amounts_to_resize
    else:
        raise ValueError("Supported values for crop_or_pad_position are 'sym', 'lower', 'upper'")
    
    if crop_not_pad:
        slice_list = [] 
        for lower_increment, upper_increment, density_length in zip(lower_increments, upper_increments, density_shape_array):
            lower_slice_val = lower_increment
            upper_slice_val = density_length - upper_increment
            slice_list.append(slice(lower_slice_val, upper_slice_val))
        slice_tuple = tuple(slice_list)
        return stored_density[slice_tuple]
    else:
        pad_tuples_list = []
        for lower_increment, upper_increment in zip(lower_increments, upper_increments):
            pad_tuple = (lower_increment, upper_increment)
            pad_tuples_list.append(pad_tuple)
        overall_pad_tuple = tuple(pad_tuples_list)
        return np.pad(stored_density, overall_pad_tuple)
    

def get_stored_density_rotation_angle(my_measurement, my_run, stored_density_name = None, mode = "variance"):
    stored_density = _load_mandatory_stored_density(my_measurement, my_run, stored_density_name)
    if mode == "variance":
        extracted_angle = image_processing_functions.get_image_principal_rotation_angle(stored_density)
        return extracted_angle
    else:
        raise ValueError("Supported extraction methods are: 'variance'")
    


#UTILITY NON-RUN ANALYSIS FUNCTIONS, POSSIBLY FOR EXTERNAL CALLING

def box_autocut(my_measurement, atom_density_to_fit, vert_crop_point = 0.5, horiz_crop_point = 0.00, widths_free = False):
    if not widths_free:
        horiz_radius = my_measurement.experiment_parameters["axicon_diameter_pix"] / 2
        vert_width = my_measurement.experiment_parameters["box_length_pix"]
        box_crop = data_fitting_functions.crop_box(atom_density_to_fit, 
                            vert_crop_point = vert_crop_point, horiz_crop_point = horiz_crop_point, 
                            horiz_radius = horiz_radius, vert_width = vert_width)
    else:
        box_crop = data_fitting_functions.crop_box(atom_density_to_fit,
                            vert_crop_point = vert_crop_point, horiz_crop_point = horiz_crop_point)
    return box_crop


def get_uniform_density_reshape_dimensions(my_measurement, stored_density_name, crop_not_pad):
    stored_densities = my_measurement.get_analysis_value_from_runs(stored_density_name, ignore_absent = True, ignore_errors = True, numpyfy = False)
    stored_density_dimensions_list = [stored_density.shape for stored_density in stored_densities]
    stored_density_dimensions_array = np.array(stored_density_dimensions_list)
    #NOTE: Weird syntax below is deliberate, to avoid a known issue where JSON can't serialize np.int64
    if crop_not_pad:
        return tuple([int(a) for a in np.min(stored_density_dimensions_array, axis = 0)])
    else:
        return tuple([int(a) for a in np.max(stored_density_dimensions_array, axis = 0)])


"""
Convenience function for obtaining saturation camera counts for a run; dividing the without atoms image by this number gives saturation parameter. Pulls 
from both measurement and run to put together all the necessary pieces, including optionally an ad-hoc fudge based on Ramsey-method calibration 
of saturation intensity.

REMARK: There are in principle two ingredients missing from the calculation. Different imaging methods may in principle have different fractions 
of the total light intensity at the location of the atoms actually interact with them - in polrot imaging, half of the light on the atoms is the 
"wrong polarization" and highly off-resonant, for instance. Likewise, different imaging methods may have different amounts of the light intensity
present at the atoms actually reach the camera - in polrot imaging, when there are no atoms, only half of the light intensity at the atoms reaches the 
camera, thanks to a right circular polarizer. 

These two factors are omitted because, for all of the imaging schemes we use so far, they cancel out - they are both 1 in absorption, and both 0.5 in 
polarization rotation. They should, however, in general be taken into account.

In a similar vein, it is generically not appropriate to speak of _the_ saturation intensity for images where multiple atomic transitions contribute to the 
produced image - the cross sections, hence saturations, will generally be different. However, for our case, all transitions are sufficiently close 
to cycling as to make this unnecessary to consider."""

def get_saturation_counts_top(my_measurement, my_run, apply_ramsey_fudge = True):
    lithium_linewidth_MHz = image_processing_functions._get_linewidth_from_species("6Li")
    lithium_linewidth_Hz = lithium_linewidth_MHz * 1e6
    lithium_bare_res_cross_section_um = image_processing_functions._get_res_cross_section_from_species("6Li")
    lithium_bare_res_cross_section_m = lithium_bare_res_cross_section_um * 1e-12
    top_cross_section_geometry_factor = my_measurement.experiment_parameters["li_top_sigma_multiplier"]
    lithium_top_geo_adjusted_res_cross_section_m = lithium_bare_res_cross_section_m * top_cross_section_geometry_factor
    top_um_per_pixel = my_measurement.experiment_parameters["top_um_per_pixel"]
    top_m_per_pixel = top_um_per_pixel * 1e-6
    camera_quantum_efficiency = my_measurement.experiment_parameters["top_camera_quantum_efficiency"] 
    camera_counts_per_photoelectron = my_measurement.experiment_parameters["top_camera_counts_per_photoelectron"]
    camera_count_to_photon_factor = 1.0 / (camera_counts_per_photoelectron * camera_quantum_efficiency) 
    camera_post_atom_photon_transmission = my_measurement.experiment_parameters["top_camera_post_atom_photon_transmission"]
    if apply_ramsey_fudge:
        camera_saturation_fudge = (1.0 / camera_post_atom_photon_transmission) * my_measurement.experiment_parameters["top_camera_saturation_ramsey_fudge"]
    else:
        camera_saturation_fudge = (1.0 / camera_post_atom_photon_transmission)
    imaging_time_us = my_run.parameters["ImageTime"]
    imaging_time_s = imaging_time_us * 1e-6 
    saturation_counts = image_processing_functions.get_saturation_counts_from_camera_parameters(top_m_per_pixel, imaging_time_s, camera_count_to_photon_factor, 
                                                                            lithium_linewidth_Hz, lithium_top_geo_adjusted_res_cross_section_m, 
                                                                            saturation_multiplier = camera_saturation_fudge)
    return saturation_counts


"""
Convenience function for getting the actual areal cross section of the tilted oval of the hybrid trap.
Note that the tilt angle is the angle made by the semimajor axis of the oval to the plane that the top imaging can see."""
def get_hybrid_cross_section_um(my_measurement):
    axicon_diameter_pix = my_measurement.experiment_parameters["axicon_diameter_pix"]
    um_per_pixel = my_measurement.experiment_parameters["top_um_per_pixel"]
    axicon_side_angle_deg = my_measurement.experiment_parameters["axicon_side_angle_deg"]
    axicon_side_aspect_ratio = my_measurement.experiment_parameters["axicon_side_aspect_ratio"]
    box_radius_um = um_per_pixel * axicon_diameter_pix / 2
    top_radius_um = box_radius_um
    cross_section_um = _hybrid_cross_section_geometry_formula(box_radius_um, axicon_side_angle_deg, axicon_side_aspect_ratio)
    return cross_section_um

def _hybrid_cross_section_geometry_formula(top_radius_um, side_angle_deg, side_aspect_ratio):
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




#RUN COMBINING FUNCTIONS

#List of parameters which are always unique per run, or specifically reserved for iteration tracking
EXCLUDED_RUN_PARAMETERS_LIST = ["id", "runtime", "IterationNum", "Iteration", "IterationCount"] 


"""Return a hash of run experiment parameters. 

Run hash function for use in measurement.combine_runs. Return a hash of the dictionary my_run.parameters, where values which either guaranteed 
to be unique per run (id, runtime) or are specifically reserved as dummy variables (Iteration) are excluded. """
def identical_parameters_run_hash_function(my_run):
    run_parameters_sans_exclusions = {key:my_run.parameters[key] for key in my_run.parameters if not key in EXCLUDED_RUN_PARAMETERS_LIST}
    return hash(json.dumps(run_parameters_sans_exclusions))

def all_equal_run_hash_function(my_run):
    return 0


"""Function factory for run_averaging. 

Returns a run combining function for use in measurement.combine_runs. Given a string or tuple of strings result_names and an averaging function 
result_average_function, apply result_average_function to the list [run_1.analysis_results[result_name_1], run_2.analysis_results[result_name_1], ...]
and then store the result in the analysis_results of the returned run. Any names not specifically provided are excluded."""
def average_results_identical_runs_run_combine_function_factory(result_names, result_average_function):
    if not isinstance(result_names, tuple):
        result_names = (result_names,)
    def run_combine_function(run_list):
        first_run = run_list[0]
        first_run_non_unique_parameters = {key:first_run.parameters[key] for key in first_run.parameters if not key in EXCLUDED_RUN_PARAMETERS_LIST}
        combined_string_run_ids = ",".join([str(run.parameters["id"]) for run in run_list])
        combined_run_parameters = first_run_non_unique_parameters
        combined_run_parameters["id"] = combined_string_run_ids
        combined_analysis_results = {}
        for result_name in result_names:
            averaged_result = result_average_function([run.analysis_results[result_name] for run in run_list])
            combined_analysis_results[result_name] = averaged_result
        combined_run = Run(combined_string_run_ids, None, combined_run_parameters, analysis_results = combined_analysis_results, 
                           connected_mode = False)
        return combined_run
    return run_combine_function

#Convenience function for most common use case
average_densities_13_run_combine_function = average_results_identical_runs_run_combine_function_factory(
                                        ("densities_1", "densities_3"),
                                         lambda x: np.average(x, axis = 0))










#UTILITY, NOT INTENDED FOR EXTERNAL CALLING
def _get_resonance_frequency_from_state_index(my_measurement, state_index):
    STATE_1_PARAMETER_NAME = "state_1_unitarity_res_freq_MHz"
    STATE_2_PARAMETER_NAME = "state_2_unitarity_res_freq_MHz" 
    STATE_3_PARAMETER_NAME = "state_3_unitarity_res_freq_MHz"
    if(state_index == 1):
        return my_measurement.experiment_parameters[STATE_1_PARAMETER_NAME]
    elif(state_index == 2):
        return my_measurement.experiment_parameters[STATE_2_PARAMETER_NAME]
    elif(state_index == 3):
        return my_measurement.experiment_parameters[STATE_3_PARAMETER_NAME]
    else:
        raise ValueError("Invalid state index")

def _get_hf_lock_frequency_adjustment_from_b_field_condition(my_measurement, b_field_condition):
    if b_field_condition == "unitarity":
        lock_value_for_nominal_resonance = my_measurement.experiment_parameters["hf_lock_unitarity_resonance_value"]
    elif b_field_condition == "rapid_ramp":
        lock_value_for_nominal_resonance = my_measurement.experiment_parameters["hf_lock_rr_resonance_value"]
    elif b_field_condition == "zero_crossing":
        lock_value_for_nominal_resonance = my_measurement.experiment_parameters["hf_lock_zero_crossing_resonance_value"]
    lock_frequency_multiplier = my_measurement.experiment_parameters["hf_lock_frequency_multiplier"]
    lock_setpoint = my_measurement.experiment_parameters["hf_lock_setpoint"]
    return lock_frequency_multiplier * (lock_setpoint - lock_value_for_nominal_resonance)

def _load_densities_polrot(my_measurement, my_run, first_stored_density_name, second_stored_density_name, **get_density_kwargs):
    if first_stored_density_name is None or second_stored_density_name is None:
        atom_density_first, atom_density_second = get_atom_densities_top_polrot(my_measurement, my_run, **get_density_kwargs)
    else:
        atom_density_first = my_run.analysis_results[first_stored_density_name]
        atom_density_second = my_run.analysis_results[second_stored_density_name]
    return (atom_density_first, atom_density_second)

"""
Convenience function for loading a pair of densities from top double imaging.

Given a measurement and run, loads the atom densities from top double imaging, wrapping the cases of different imaging modes 
as well as the possibility that the densities have been precomputed and stored.

Parameters:

my_measurement, my_run: The Measurement and Run objects, respectively, for which the computation is done 

first/second_stored_density_name: keys under which the precomputed atom density has been stored in my_run.analysis_results. 
If None, the density is computed from scratch (see Remark, below)

imaging_mode: The type of imaging used to generate the images from which atomic density is computed. Currently accepts 'polrot' 
and 'abs'.

**get_density_kwargs: All of the kwargs which are to be passed to the function which computes the density from an image.


Remark: As currently set up, there is unnecessary computation done in the corner case where, for absorption imaging, one 
set of atomic densities has been precomputed while the other has not. This is an unusual situation, and the computation is deemend
an acceptable trade-off to simplify the code - otherwise, the density kwargs need to be massaged to work for both top double and top A and B 
separately."""
def _load_densities_top_double(my_measurement, my_run, first_stored_density_name, second_stored_density_name, imaging_mode, 
                               **get_density_kwargs):
    if imaging_mode == "polrot":
        return _load_densities_polrot(my_measurement, my_run, first_stored_density_name, 
                            second_stored_density_name, **get_density_kwargs)
    elif imaging_mode == "abs":
        return _load_densities_top_AB_abs(my_measurement, my_run, first_stored_density_name, second_stored_density_name, **get_density_kwargs)

def _load_densities_top_AB_abs(my_measurement, my_run, first_stored_density_name, second_stored_density_name, **get_density_kwargs):
    if first_stored_density_name is None or second_stored_density_name is None:
        density_A, density_B = get_atom_densities_top_abs(my_measurement, my_run, **get_density_kwargs)
    else:
        density_A = my_run.analysis_results[first_stored_density_name] 
        density_B = my_run.analysis_results[second_stored_density_name]
    return (density_A, density_B)

def _load_density_top_A_abs(my_measurement, my_run, stored_density_name, **get_density_kwargs):
    if stored_density_name is None:
        atom_density = get_atom_density_top_A_abs(my_measurement, my_run, **get_density_kwargs)
    else:
        atom_density = my_run.analysis_results[stored_density_name]
    return atom_density

def _load_density_top_B_abs(my_measurement, my_run, stored_density_name, **get_density_kwargs):
    if stored_density_name is None:
        atom_density = get_atom_density_top_B_abs(my_measurement, my_run, **get_density_kwargs)
    else:
        atom_density = my_run.analysis_results[stored_density_name]
    return atom_density


def _load_density_side_li_hf(my_measurement, my_run, stored_density_name, **get_density_kwargs):
    if stored_density_name is None:
        atom_density = get_atom_density_side_li_hf(my_measurement, my_run, **get_density_kwargs)
    else:
        atom_density = my_run.analysis_results[stored_density_name]
    return atom_density


def _load_density_side_li_lf(my_measurement, my_run, stored_density_name):
    if stored_density_name is None:
        atom_density = get_atom_density_side_li_lf(my_measurement, my_run)
    else:
        atom_density = my_run.analysis_results[stored_density_name]
    return atom_density

def _load_mandatory_stored_density(my_measurement, my_run, stored_density_name):
    if stored_density_name is None:
        raise ValueError("The stored density name must be specified")
    else:
        return my_run.analysis_results[stored_density_name]


