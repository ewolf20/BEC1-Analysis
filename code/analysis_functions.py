import numpy as np
from scipy.integrate import trapezoid 
from scipy import ndimage

from . import data_fitting_functions, image_processing_functions, science_functions


#RAW IMAGES (Convenience functions for getting the raw pixel data from shots, cropped within an ROI)

def get_raw_pixels_side(my_measurement, my_run, crop_roi = False):
    my_run_image_array = my_run.get_image("Side", memmap = False)
    if crop_roi:
        roi = my_measurement.measurement_parameters["ROI"] 
        x_min, y_min, x_max, y_max = roi 
        my_run_image_array_cropped = my_run_image_array[:, y_min:y_max, x_min:x_max]
        return (*my_run_image_array_cropped,)
    else:
        return (*my_run_image_array,)
    

def get_raw_pixels_top_A(my_measurement, my_run, crop_roi = False):
    my_run_image_array = my_run.get_image("TopA", memmap = False)
    if crop_roi:
        roi = my_measurement.measurement_parameters["ROI"] 
        x_min, y_min, x_max, y_max = roi 
        my_run_image_array_cropped = my_run_image_array[:, y_min:y_max, x_min:x_max]
        return (*my_run_image_array_cropped,)
    else:
        return (*my_run_image_array,)
    

def get_raw_pixels_top_B(my_measurement, my_run, crop_roi = False):
    my_run_image_array = my_run.get_image("TopB", memmap = False)
    if crop_roi:
        roi = my_measurement.measurement_parameters["ROI"] 
        x_min, y_min, x_max, y_max = roi 
        my_run_image_array_cropped = my_run_image_array[:, y_min:y_max, x_min:x_max]
        return (*my_run_image_array_cropped,)
    else:
        return (*my_run_image_array,)


#ABS IMAGES (Sometimes called 'Fake OD')

def get_abs_image_side(my_measurement, my_run):
    my_run_image_array = my_run.get_image('Side', memmap = True) 
    my_run_abs_image = image_processing_functions.get_absorption_image(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"])
    return my_run_abs_image


def get_abs_image_top_A(my_measurement, my_run):
    my_run_image_array_A = my_run.get_image('TopA', memmap = True) 
    my_run_abs_image = image_processing_functions.get_absorption_image(my_run_image_array_A, ROI = my_measurement.measurement_parameters["ROI"], 
                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"])
    return my_run_abs_image

def get_abs_image_top_B(my_measurement, my_run):
    my_run_image_array_B = my_run.get_image('TopB', memmap = True) 
    my_run_abs_image = image_processing_functions.get_absorption_image(my_run_image_array_B, ROI = my_measurement.measurement_parameters["ROI"], 
                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"])
    return my_run_abs_image
    
def get_abs_images_top_double(my_measurement, my_run):
    return (get_abs_image_top_A(my_measurement, my_run), get_abs_image_top_B(my_measurement, my_run))

#OD IMAGES

def get_od_image_side(my_measurement, my_run):
    my_run_image_array = my_run.get_image('Side', memmap = True) 
    my_run_od_image = image_processing_functions.get_absorption_od_image(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"])
    return my_run_od_image

def get_od_image_top_A(my_measurement, my_run):
    my_run_image_array_A = my_run.get_image('TopA', memmap = True) 
    my_run_od_image_A = image_processing_functions.get_absorption_od_image(my_run_image_array_A, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"])
    return my_run_od_image_A


def get_od_image_top_B(my_measurement, my_run):
    my_run_image_array_B = my_run.get_image('TopB', memmap = True) 
    my_run_od_image_B = image_processing_functions.get_absorption_od_image(my_run_image_array_B, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"])
    return my_run_od_image_B

def get_od_images_top_double(my_measurement, my_run):
    od_image_A = get_od_image_top_A(my_measurement, my_run) 
    od_image_B = get_od_image_top_B(my_measurement, my_run) 
    return (od_image_A, od_image_B)

#PIXEL SUMS

def get_od_pixel_sum_side(my_measurement, my_run):
    my_run_abs_image = get_od_image_side(my_measurement, my_run)
    pixel_sum = image_processing_functions.pixel_sum(my_run_abs_image)
    return pixel_sum


def get_od_pixel_sum_top_A(my_measurement, my_run):
    my_run_abs_image_A = get_od_image_top_A(my_measurement, my_run)
    pixel_sum_A = image_processing_functions.pixel_sum(my_run_abs_image_A)
    return pixel_sum_A

def get_od_pixel_sum_top_B(my_measurement, my_run):
    my_run_abs_image_B = get_od_image_top_B(my_measurement, my_run)
    pixel_sum_B = image_processing_functions.pixel_sum(my_run_abs_image_B)
    return pixel_sum_B

def get_od_pixel_sums_top_double(my_measurement, my_run):
    return (get_od_pixel_sum_top_A(my_measurement, my_run), get_od_pixel_sum_top_B(my_measurement, my_run))


#ATOM DENSITIES

def get_atom_density_side_li_lf(my_measurement, my_run):
    my_run_image_array = my_run.get_image('Side', memmap = True) 
    frequency_multiplier = my_measurement.experiment_parameters["li_lf_freq_multiplier"]
    nominal_resonance_frequency = my_measurement.experiment_parameters["li_lf_res_freq"]
    nominal_frequency = my_run.parameters["LFImgFreq"]
    side_cross_section_geometry_factor = my_measurement.experiment_parameters["li_side_sigma_multiplier"]
    detuning = frequency_multiplier * (nominal_frequency - nominal_resonance_frequency)
    atom_density_image = image_processing_functions.get_atom_density_absorption(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                            norm_box_coordinates=my_measurement.measurement_parameters["norm_box"], detuning = detuning, 
                                                            cross_section_imaging_geometry_factor=side_cross_section_geometry_factor)
    return atom_density_image

def get_atom_density_side_li_hf(my_measurement, my_run, state_index = None, b_field_condition = "unitarity"):
    if state_index is None:
        raise RuntimeError("The state of the imaging must be specified.")
    
    my_run_image_array = my_run.get_image('Side', memmap = True) 
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    side_cross_section_geometry_factor = my_measurement.experiment_parameters["li_side_sigma_multiplier"]
    nominal_resonance_frequency = _get_resonance_frequency_from_state_index(my_measurement, state_index)
    hf_lock_frequency_adjustment = _get_hf_lock_frequency_adjustment_from_b_field_condition(my_measurement, b_field_condition)
    nominal_frequency = my_run.parameters["ImagFreq0"]
    detuning = frequency_multiplier * (nominal_frequency - nominal_resonance_frequency) + hf_lock_frequency_adjustment
    atom_density_image = image_processing_functions.get_atom_density_absorption(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                            norm_box_coordinates=my_measurement.measurement_parameters["norm_box"], detuning = detuning,
                                                            cross_section_imaging_geometry_factor=side_cross_section_geometry_factor)
    return atom_density_image

def get_atom_density_top_A_abs(my_measurement, my_run, state_index = 1, b_field_condition = "unitarity"):
    #Find the true detuning from the resonance in absolute frequency space,
    #taking into account shifts in AOM frequency and hf frequency offset lock setpoint
    nominal_resonance_frequencies_list = [my_measurement.experiment_parameters["state_1_unitarity_res_freq_MHz"], 
                                        my_measurement.experiment_parameters["state_2_unitarity_res_freq_MHz"], 
                                        my_measurement.experiment_parameters["state_3_unitarity_res_freq_MHz"]] 
    nominal_resonance_frequency = nominal_resonance_frequencies_list[state_index - 1]
    nominal_frequency = my_run.parameters["ImagFreq1"]
    hf_lock_frequency_adjustment = _get_hf_lock_frequency_adjustment_from_b_field_condition(my_measurement, b_field_condition)
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    detuning = frequency_multiplier * (nominal_frequency - nominal_resonance_frequency) + hf_lock_frequency_adjustment


    #Adjust for imaging geometry-dependent cross section
    top_cross_section_geometry_factor = my_measurement.experiment_parameters["li_top_sigma_multiplier"]


    #Process
    my_run_image_array = my_run.get_image('TopA', memmap = True) 
    atom_density_image = image_processing_functions.get_atom_density_absorption(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                            norm_box_coordinates=my_measurement.measurement_parameters["norm_box"], detuning = detuning, 
                                                            cross_section_imaging_geometry_factor=top_cross_section_geometry_factor)
    return atom_density_image

def get_atom_density_top_B_abs(my_measurement, my_run, state_index = 3, b_field_condition = "unitarity"):
    nominal_resonance_frequencies_list = [my_measurement.experiment_parameters["state_1_unitarity_res_freq_MHz"], 
                                        my_measurement.experiment_parameters["state_2_unitarity_res_freq_MHz"], 
                                        my_measurement.experiment_parameters["state_3_unitarity_res_freq_MHz"]] 
    nominal_resonance_frequency = nominal_resonance_frequencies_list[state_index - 1]
    nominal_frequency = my_run.parameters["ImagFreq2"]
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    hf_lock_frequency_adjustment = _get_hf_lock_frequency_adjustment_from_b_field_condition(my_measurement, b_field_condition)
    top_cross_section_geometry_factor = my_measurement.experiment_parameters["li_top_sigma_multiplier"]
    detuning = frequency_multiplier * (nominal_frequency - nominal_resonance_frequency) + hf_lock_frequency_adjustment
    my_run_image_array = my_run.get_image('TopB', memmap = True) 
    atom_density_image = image_processing_functions.get_atom_density_absorption(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                            norm_box_coordinates=my_measurement.measurement_parameters["norm_box"], detuning = detuning, 
                                                            cross_section_imaging_geometry_factor=top_cross_section_geometry_factor)
    return atom_density_image


def get_atom_densities_top_abs(my_measurement, my_run, state_index_A = 1, state_index_B = 3, b_field_condition = "unitarity"):
    return (get_atom_density_top_A_abs(my_measurement, my_run, state_index = state_index_A, b_field_condition=b_field_condition),
     get_atom_density_top_B_abs(my_measurement, my_run, state_index = state_index_B, b_field_condition = b_field_condition))



def get_atom_densities_top_polrot(my_measurement, my_run, first_state_index = 1, second_state_index = 3, b_field_condition = "unitarity", 
                                first_state_fudge = 1.0, second_state_fudge = 1.0):
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
    image_array_A = my_run.get_image('TopA', memmap = True)
    image_array_B = my_run.get_image('TopB', memmap = True)
    abs_image_A = image_processing_functions.get_absorption_image(image_array_A, 
                                                                ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates=my_measurement.measurement_parameters["norm_box"])
    abs_image_B = image_processing_functions.get_absorption_image(image_array_B, ROI = my_measurement.measurement_parameters["ROI"], 
                                                    norm_box_coordinates=my_measurement.measurement_parameters["norm_box"])
    atom_density_first, atom_density_second = image_processing_functions.get_atom_density_from_polrot_images(abs_image_A, abs_image_B,
                                                                detuning_1A, detuning_1B, detuning_2A, detuning_2B, phase_sign = polrot_phase_sign, 
                                                                cross_section_imaging_geometry_factor = top_cross_section_geometry_factor)
    atom_density_first = atom_density_first * first_state_fudge
    atom_density_second = atom_density_second * second_state_fudge
    return (atom_density_first, atom_density_second)

def get_atom_densities_box_autocut(my_measurement, my_run, vert_crop_point = 0.5, horiz_crop_point = 0.00, widths_free = False, density_to_use = 2,
                                   first_stored_density_name = None, second_stored_density_name = None, imaging_mode = "polrot",
                                   **get_density_kwargs):
    density_1, density_2 = _load_densities_top_double(my_measurement, my_run, first_stored_density_name, second_stored_density_name, 
                                                    imaging_mode, get_density_kwargs)
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


def get_x_integrated_atom_densities_top_double(my_measurement, my_run, first_stored_density_name = None, second_stored_density_name = None, 
                                    imaging_mode = "polrot", **get_density_kwargs):
    density_1, density_2 = _load_densities_top_double(my_measurement, my_run, first_stored_density_name, second_stored_density_name, 
                                                    imaging_mode, get_density_kwargs)
    density_1_x_integrated = np.sum(density_1, axis = 1) * my_measurement.experiment_parameters["top_um_per_pixel"]
    density_2_x_integrated = np.sum(density_2, axis = 1) * my_measurement.experiment_parameters["top_um_per_pixel"]
    return (density_1_x_integrated, density_2_x_integrated)


def get_y_integrated_atom_densities_top_double(my_measurement, my_run, first_stored_density_name = None, second_stored_density_name = None, 
                                        imaging_mode = "polrot", **get_density_kwargs):
    density_1, density_2 = _load_densities_top_double(my_measurement, my_run, first_stored_density_name, second_stored_density_name, 
                                                    imaging_mode, get_density_kwargs)
    density_1_x_integrated = np.sum(density_1, axis = 1) * my_measurement.experiment_parameters["top_um_per_pixel"]
    density_2_x_integrated = np.sum(density_2, axis = 1) * my_measurement.experiment_parameters["top_um_per_pixel"]
    return (density_1_x_integrated, density_2_x_integrated)
#ATOM COUNTS

"""
For these and subsequent functions, there is a common keyword parameter: stored_density_name. 
If passed, the functions assume that the atom density is stored as an analysis result within the run 
object, under the name stored_density_name. This is to allow patterns where multiple analyses which rely on 
the atom number density can be run without re-calculating it (typically the slowest part of any analysis)."""

def get_atom_count_side_li_lf(my_measurement, my_run, stored_density_name = None):
    atom_density = _load_density_side_li_lf(my_measurement, my_run, stored_density_name)
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"]) 
    return image_processing_functions.atom_count_pixel_sum(atom_density, pixel_area)


def get_atom_count_side_li_hf(my_measurement, my_run, state_index = 1, stored_density_name = None, b_field_condition = "unitarity"):
    atom_density = _load_density_side_li_hf(my_measurement, my_run, state_index, stored_density_name, b_field_condition)
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    return image_processing_functions.atom_count_pixel_sum(atom_density, pixel_area)


def get_atom_count_top_A_abs(my_measurement, my_run, state_index = 1, stored_density_name = None, b_field_condition = "unitarity"):
    atom_density = _load_density_top_A_abs(my_measurement, my_run, state_index, stored_density_name, b_field_condition)
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    return image_processing_functions.atom_count_pixel_sum(atom_density, pixel_area) 

def get_atom_count_top_B_abs(my_measurement, my_run, state_index = 3, stored_density_name = None, b_field_condition = "unitarity"):
    atom_density = _load_density_top_B_abs(my_measurement, my_run, state_index, stored_density_name, b_field_condition)
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    return image_processing_functions.atom_count_pixel_sum(atom_density, pixel_area)

def get_atom_counts_top_AB_abs(my_measurement, my_run, first_state_index = 1, second_state_index = 3, 
                                first_stored_density_name = None, second_stored_density_name = None, b_field_condition = "unitarity"):
    return (get_atom_count_top_A_abs(my_measurement, my_run, state_index = first_state_index, stored_density_name=first_stored_density_name, 
                                    b_field_condition = b_field_condition), 
            get_atom_count_top_B_abs(my_measurement, my_run, state_index = second_state_index, stored_density_name=second_stored_density_name,
                                    b_field_condition = b_field_condition))

def get_atom_counts_top_polrot(my_measurement, my_run, first_state_index = 1, second_state_index = 3, first_stored_density_name = None, 
                                second_stored_density_name = None, b_field_condition = "unitarity"):
    atom_density_first, atom_density_second = _load_densities_polrot(my_measurement, my_run, first_state_index, second_state_index, 
                                                first_stored_density_name, second_stored_density_name, b_field_condition)
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    atom_count_first = image_processing_functions.atom_count_pixel_sum(atom_density_first, pixel_area)
    atom_count_second = image_processing_functions.atom_count_pixel_sum(atom_density_second, pixel_area)
    return (atom_count_first, atom_count_second)

#TRAP-SPECIFIC_ANALYSES

#HYBRID TRAP - BOX EXP

def get_hybrid_trap_densities_along_harmonic_axis(my_measurement, my_run, autocut = True, 
                                                  first_stored_density_name = None, second_stored_density_name = None, 
                                                  imaging_mode = "polrot", **get_density_kwargs):
    HYBRID_TRAP_B_FIELD_CONDITION = "unitarity"
    atom_density_first, atom_density_second = _load_densities_top_double(
                                                my_measurement, my_run, first_stored_density_name, second_stored_density_name, 
                                                imaging_mode, get_density_kwargs)
    axicon_tilt_deg = my_measurement.experiment_parameters["axicon_tilt_deg"]
    axicon_diameter_pix = my_measurement.experiment_parameters["axicon_diameter_pix"]
    axicon_length_pix = my_measurement.experiment_parameters["hybrid_trap_typical_length_pix"]
    axicon_side_angle_deg = my_measurement.experiment_parameters["axicon_side_angle_deg"]
    axicon_side_aspect_ratio = my_measurement.experiment_parameters["axicon_side_aspect_ratio"]
    um_per_pixel = my_measurement.experiment_parameters["top_um_per_pixel"]
    positions_first, densities_first = image_processing_functions.get_hybrid_trap_densities_along_harmonic_axis(atom_density_first, axicon_tilt_deg, 
                                        axicon_diameter_pix, axicon_length_pix, axicon_side_angle_deg, axicon_side_aspect_ratio, um_per_pixel)
    positions_second, densities_second = image_processing_functions.get_hybrid_trap_densities_along_harmonic_axis(atom_density_second, axicon_tilt_deg, 
                                        axicon_diameter_pix, axicon_length_pix, axicon_side_angle_deg, axicon_side_aspect_ratio, um_per_pixel)
    if autocut:
        first_start_index, first_stop_index = science_functions.hybrid_trap_autocut(densities_first)
        positions_first = positions_first[first_start_index:first_stop_index]
        densities_first = densities_first[first_start_index:first_stop_index]
        second_start_index, second_stop_index = science_functions.hybrid_trap_autocut(densities_second) 
        positions_second = positions_second[second_start_index:second_stop_index] 
        densities_second = densities_second[second_start_index:second_stop_index]
    return (positions_first, densities_first, positions_second, densities_second)


def get_hybrid_trap_average_energy(my_measurement, my_run, first_state_index = 1, second_state_index = 3, 
                                    autocut = True, imaging_mode = "polrot", return_sub_energies = False,
                                    first_stored_density_name = None, second_stored_density_name = None):
    positions_first, densities_first, positions_second, densities_second = get_hybrid_trap_densities_along_harmonic_axis( 
                                                                    my_measurement, my_run, first_state_index = first_state_index, 
                                                                    second_state_index = second_state_index, autocut = autocut, 
                                                                    imaging_mode = imaging_mode,
                                                                    first_stored_density_name = first_stored_density_name, 
                                                                    second_stored_density_name = second_stored_density_name)
    axicon_diameter_pix = my_measurement.experiment_parameters["axicon_diameter_pix"]
    um_per_pixel = my_measurement.experiment_parameters["top_um_per_pixel"]
    axicon_side_angle_deg = my_measurement.experiment_parameters["axicon_side_angle_deg"]
    axicon_side_aspect_ratio = my_measurement.experiment_parameters["axicon_side_aspect_ratio"]
    top_radius_um = um_per_pixel * axicon_diameter_pix / 2
    trap_cross_section_um = image_processing_functions.get_hybrid_cross_section_um(top_radius_um, axicon_side_angle_deg, axicon_side_aspect_ratio)
    trap_freq = my_measurement.experiment_parameters["axial_trap_frequency_hz"]
    #Autocut False because it's already been done...
    average_energy_first = science_functions.get_hybrid_trap_average_energy(positions_first, densities_first, trap_cross_section_um,
                                                                            trap_freq, autocut = False) 
    counts_first = trapezoid(trap_cross_section_um * densities_first, x = positions_first)
    average_energy_second = science_functions.get_hybrid_trap_average_energy(positions_second, densities_second, trap_cross_section_um,
                                                                            trap_freq, autocut = False) 
    counts_second = trapezoid(trap_cross_section_um * densities_second, x = positions_second)
    overall_average_energy = (average_energy_first * counts_first + average_energy_second * counts_second) / (counts_first + counts_second)
    if return_sub_energies:
        return (overall_average_energy, average_energy_first, average_energy_second) 
    else:
        return overall_average_energy


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
                                     no_shake_density_name_first = None, no_shake_density_name_second = None, 
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
                                                    get_density_kwargs)
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



def get_box_in_situ_fermi_energies_from_counts(my_measurement, my_run, first_state_index = 1, second_state_index = 3, imaging_mode = "polrot", 
                                b_field_condition = "unitarity", first_stored_density_name = None, second_stored_density_name = None):
    if imaging_mode == "polrot":
        counts_first, counts_second = get_atom_counts_top_polrot(my_measurement, my_run, first_state_index=first_state_index, 
                                                    second_state_index = second_state_index, first_stored_density_name=first_stored_density_name, 
                                                    second_stored_density_name=second_stored_density_name, b_field_condition=b_field_condition) 
    elif imaging_mode == "abs":
        counts_first, counts_second = get_atom_counts_top_AB_abs(my_measurement, my_run, first_state_index = first_state_index, 
                                                    second_state_index = second_state_index, first_stored_density_name=first_stored_density_name, 
                                                    second_stored_density_name = second_stored_density_name, b_field_condition = b_field_condition)
    axicon_diameter_pix = my_measurement.experiment_parameters["axicon_diameter_pix"]
    box_length_pix = my_measurement.experiment_parameters["box_length_pix"]
    um_per_pixel = my_measurement.experiment_parameters["top_um_per_pixel"]
    box_length_um = box_length_pix * um_per_pixel
    axicon_side_angle_deg = my_measurement.experiment_parameters["axicon_side_angle_deg"]
    axicon_side_aspect_ratio = my_measurement.experiment_parameters["axicon_side_aspect_ratio"]
    box_radius_um = um_per_pixel * axicon_diameter_pix / 2 
    cross_section_um = image_processing_functions.get_hybrid_cross_section_um(box_radius_um, axicon_side_angle_deg, axicon_side_aspect_ratio)
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
                                                get_density_kwargs)
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
                                                                        second_stored_density_name, imaging_mode, get_density_kwargs)
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
    subtract_ymax = rr_ymin - 1 
    subtract_ymin = rr_ymin - 1 - (rr_ymax - rr_ymin) 
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


#UTILITY, POSSIBLY FOR EXTERNAL CALLING

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





def _load_densities_top_double(my_measurement, my_run, first_stored_density_name, second_stored_density_name, imaging_mode, 
                               get_density_kwargs):
    if imaging_mode == "polrot":
        return _load_densities_polrot(my_measurement, my_run, first_stored_density_name, 
                            second_stored_density_name, **get_density_kwargs)
    elif imaging_mode == "abs":
        density_1 = _load_density_top_A_abs(my_measurement, my_run, first_stored_density_name, **get_density_kwargs)
        density_2 = _load_density_top_B_abs(my_measurement, my_run, second_stored_density_name, **get_density_kwargs)
        return (density_1, density_2)

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



