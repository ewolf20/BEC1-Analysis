import numpy as np
from scipy.integrate import trapezoid 
from scipy import ndimage

from . import data_fitting_functions, image_processing_functions, science_functions


#PIXEL SUMS

def get_od_pixel_sum_side(my_measurement, my_run):
    my_run_image_array = my_run.get_image('Side', memmap = True) 
    my_run_abs_image = image_processing_functions.get_absorption_od_image(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"])
    pixel_sum = image_processing_functions.pixel_sum(my_run_abs_image)
    return pixel_sum


def get_od_pixel_sum_top_A(my_measurement, my_run):
    my_run_image_array_A = my_run.get_image('TopA', memmap = True) 
    my_run_abs_image_A = image_processing_functions.get_absorption_od_image(my_run_image_array_A, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"])
    pixel_sum_A = image_processing_functions.pixel_sum(my_run_abs_image_A)
    return pixel_sum_A

def get_od_pixel_sum_top_B(my_measurement, my_run):
    my_run_image_array_B = my_run.get_image('TopB', memmap = True) 
    my_run_abs_image_B = image_processing_functions.get_absorption_od_image(my_run_image_array_B, ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates = my_measurement.measurement_parameters["norm_box"])
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
    detuning = frequency_multiplier * (nominal_frequency - nominal_resonance_frequency)
    atom_density_image = image_processing_functions.get_atom_density_absorption(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                            norm_box_coordinates=my_measurement.measurement_parameters["norm_box"], detuning = detuning)
    return atom_density_image

def get_atom_density_side_li_hf(my_measurement, my_run, state_index = None):
    if state_index is None:
        raise RuntimeError("The state of the imaging must be specified.")
    
    my_run_image_array = my_run.get_image('Side', memmap = True) 
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    if state_index == 1:
        nominal_resonance_frequency = my_measurement.experiment_parameters["state_1_unitarity_res_freq_MHz"]
    elif state_index == 2:
        nominal_resonance_frequency = my_measurement.experiment_parameters["state_2_unitarity_res_freq_MHz"]
    elif state_index == 3:
        nominal_resonance_frequency = my_measurement.experiment_parameters["state_3_unitarity_res_freq_MHz"]
    nominal_frequency = my_run.parameters["ImagFreq0"]
    detuning = frequency_multiplier * (nominal_frequency - nominal_resonance_frequency)
    atom_density_image = image_processing_functions.get_atom_density_absorption(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                            norm_box_coordinates=my_measurement.measurement_parameters["norm_box"], detuning = detuning)
    return atom_density_image

def get_atom_density_top_A_abs(my_measurement, my_run, state_index = 1):
    nominal_resonance_frequencies_list = [my_measurement.experiment_parameters["state_1_unitarity_res_freq_MHz"], 
                                        my_measurement.experiment_parameters["state_2_unitarity_res_freq_MHz"], 
                                        my_measurement.experiment_parameters["state_3_unitarity_res_freq_MHz"]] 
    nominal_resonance_frequency = nominal_resonance_frequencies_list[state_index - 1]
    nominal_frequency = my_run.parameters["ImagFreq1"]
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    detuning = frequency_multiplier * (nominal_frequency - nominal_resonance_frequency)
    my_run_image_array = my_run.get_image('TopA', memmap = True) 
    atom_density_image = image_processing_functions.get_atom_density_absorption(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                            norm_box_coordinates=my_measurement.measurement_parameters["norm_box"], detuning = detuning)
    return atom_density_image

def get_atom_density_top_B_abs(my_measurement, my_run, state_index = 3):
    nominal_resonance_frequencies_list = [my_measurement.experiment_parameters["state_1_unitarity_res_freq_MHz"], 
                                        my_measurement.experiment_parameters["state_2_unitarity_res_freq_MHz"], 
                                        my_measurement.experiment_parameters["state_3_unitarity_res_freq_MHz"]] 
    nominal_resonance_frequency = nominal_resonance_frequencies_list[state_index - 1]
    nominal_frequency = my_run.parameters["ImagFreq2"]
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    detuning = frequency_multiplier * (nominal_frequency - nominal_resonance_frequency)
    my_run_image_array = my_run.get_image('TopB', memmap = True) 
    atom_density_image = image_processing_functions.get_atom_density_absorption(my_run_image_array, ROI = my_measurement.measurement_parameters["ROI"], 
                                                            norm_box_coordinates=my_measurement.measurement_parameters["norm_box"], detuning = detuning)
    return atom_density_image


def get_atom_densities_top_abs(my_measurement, my_run, state_index_A = 1, state_index_B = 3):
    return (get_atom_density_top_A_abs(my_measurement, my_run, state_index = state_index_A), get_atom_density_top_B_abs(my_measurement, my_run, state_index = state_index_B))



def get_atom_densities_top_polrot(my_measurement, my_run, first_state_index = 1, second_state_index = 3):
    first_state_resonance_frequency = _get_resonance_frequency_from_state_index(my_measurement, first_state_index)
    second_state_resonance_frequency = _get_resonance_frequency_from_state_index(my_measurement, second_state_index)
    nominal_frequency_A = my_run.parameters["ImagFreq1"]
    nominal_frequency_B = my_run.parameters["ImagFreq2"]
    frequency_multiplier = my_measurement.experiment_parameters["li_hf_freq_multiplier"]
    detuning_1A = frequency_multiplier * (nominal_frequency_A - first_state_resonance_frequency)
    detuning_1B = frequency_multiplier * (nominal_frequency_B - first_state_resonance_frequency)
    detuning_2A = frequency_multiplier * (nominal_frequency_A - second_state_resonance_frequency) 
    detuning_2B = frequency_multiplier * (nominal_frequency_B - second_state_resonance_frequency)
    polrot_phase_sign = my_measurement.experiment_parameters["polrot_phase_sign"]
    image_array_A = my_run.get_image('TopA', memmap = True) 
    image_array_B = my_run.get_image('TopB', memmap = True)
    abs_image_A = image_processing_functions.get_absorption_image(image_array_A, 
                                                                ROI = my_measurement.measurement_parameters["ROI"], 
                                                                norm_box_coordinates=my_measurement.measurement_parameters["norm_box"])
    abs_image_B = image_processing_functions.get_absorption_image(image_array_B, ROI = my_measurement.measurement_parameters["ROI"], 
                                                    norm_box_coordinates=my_measurement.measurement_parameters["norm_box"])
    atom_density_first, atom_density_second = image_processing_functions.get_atom_density_from_polrot_images(abs_image_A, abs_image_B,
                                                                detuning_1A, detuning_1B, detuning_2A, detuning_2B, phase_sign = polrot_phase_sign)
    return (atom_density_first, atom_density_second)




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


def get_atom_count_side_li_hf(my_measurement, my_run, state_index = 1, stored_density_name = None):
    atom_density = _load_density_side_li_hf(my_measurement, my_run, state_index, stored_density_name)
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    return image_processing_functions.atom_count_pixel_sum(atom_density, pixel_area)


def get_atom_count_top_A_abs(my_measurement, my_run, state_index = 1, stored_density_name = None):
    atom_density = _load_density_top_A_abs(my_measurement, my_run, state_index, stored_density_name)
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    return image_processing_functions.atom_count_pixel_sum(atom_density, pixel_area) 

def get_atom_count_top_B_abs(my_measurement, my_run, state_index = 3, stored_density_name = None):
    atom_density = _load_density_top_B_abs(my_measurement, my_run, state_index, stored_density_name)
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    return image_processing_functions.atom_count_pixel_sum(atom_density, pixel_area)

def get_atom_counts_top_AB_abs(my_measurement, my_run, first_state_index = 1, second_state_index = 3, 
                                first_stored_density_name = None, second_stored_density_name = None):
    return (get_atom_count_top_A_abs(my_measurement, my_run, state_index = first_state_index, stored_density_name=first_stored_density_name), 
            get_atom_count_top_B_abs(my_measurement, my_run, state_index = second_state_index, stored_density_name=second_stored_density_name))

def get_atom_counts_top_polrot(my_measurement, my_run, first_state_index = 1, second_state_index = 3, first_stored_density_name = None, 
                                second_stored_density_name = None):
    atom_density_first, atom_density_second = _load_densities_polrot(my_measurement, my_run, first_state_index, second_state_index, 
                                                first_stored_density_name, second_stored_density_name)
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    atom_count_first = image_processing_functions.atom_count_pixel_sum(atom_density_first, pixel_area)
    atom_count_second = image_processing_functions.atom_count_pixel_sum(atom_density_second, pixel_area)
    return (atom_count_first, atom_count_second)

#TRAP-SPECIFIC_ANALYSES

#HYBRID TRAP - BOX EXP

def get_hybrid_trap_densities_along_harmonic_axis(my_measurement, my_run, first_state_index = 1, second_state_index = 3, 
                                                    autocut = True, imaging_mode = "polrot",
                                                    first_stored_density_name = None, second_stored_density_name = None):
    if imaging_mode == "polrot":
        atom_density_first, atom_density_second = _load_densities_polrot(my_measurement, my_run, first_state_index, second_state_index, 
                                                    first_stored_density_name, second_stored_density_name)
    elif imaging_mode == "abs":
        atom_density_first = _load_density_top_A_abs(my_measurement, my_run, first_state_index, first_stored_density_name)
        atom_density_second = _load_density_top_B_abs(my_measurement, my_run, second_state_index, second_stored_density_name)
    axicon_tilt_deg = my_measurement.experiment_parameters["axicon_tilt_deg"]
    axicon_diameter_pix = my_measurement.experiment_parameters["axicon_diameter_pix"]
    axicon_length_pix = my_measurement.experiment_parameters["hybrid_trap_typical_length_pix"]
    um_per_pixel = my_measurement.experiment_parameters["top_um_per_pixel"]
    positions_first, densities_first = image_processing_functions.get_hybrid_trap_densities_along_harmonic_axis(atom_density_first, axicon_tilt_deg, 
                                        axicon_diameter_pix, axicon_length_pix, um_per_pixel)
    positions_second, densities_second = image_processing_functions.get_hybrid_trap_densities_along_harmonic_axis(atom_density_second, axicon_tilt_deg, 
                                        axicon_diameter_pix, axicon_length_pix, um_per_pixel)
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
    trap_cross_section_um = np.pi * np.square(um_per_pixel * axicon_diameter_pix / 2)
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
def get_box_shake_fourier_amplitudes_polrot(my_measurement, my_run, first_state_index = 1, second_state_index = 3, 
                                        order = None, no_shake_density_name_first = None, 
                                        no_shake_density_name_second = None,
                                        imaging_mode = "polrot",
                                        first_stored_density_name = None, second_stored_density_name = None):
    if no_shake_density_name_first is None:
        no_shake_density_first = 0.0
    else:
        no_shake_density_first = my_measurement.measurement_analysis_results[no_shake_density_name_first]
    if no_shake_density_name_second is None:
        no_shake_density_second = 0.0 
    else:
        no_shake_density_second = my_measurement.measurement_analysis_results[no_shake_density_name_second] 
    if imaging_mode == "polrot":
        atom_density_first, atom_density_second = _load_densities_polrot(my_measurement, my_run, first_state_index, 
                                                    second_state_index, first_stored_density_name, second_stored_density_name)
    elif imaging_mode == "abs":
        atom_density_first = _load_density_top_A_abs(my_measurement, my_run, first_state_index, first_stored_density_name)
        atom_density_second = _load_density_top_B_abs(my_measurement, my_run, second_state_index, second_stored_density_name)
    bs_density_first = atom_density_first - no_shake_density_first 
    bs_density_second = atom_density_second - no_shake_density_second
    #Current convention has the integration direction as the last index, i.e. the x-axis. 
    integrated_density_first = np.sum(bs_density_first, axis = -1)
    integrated_density_second = np.sum(bs_density_second, axis = -1)
    x_delta = my_measurement.experiment_parameters["top_um_per_pixel"]
    fft_results_first = data_fitting_functions.get_fft_peak(x_delta, integrated_density_first, order = order)
    frequency_first, amp_first, phase_first = fft_results_first 
    fft_results_second = data_fitting_functions.get_fft_peak(x_delta, integrated_density_second, order = order)
    frequency_second, amp_second, phase_second = fft_results_second 
    return (amp_first, amp_second)



#RAPID RAMP

"""
Get the rapid ramp condensate fraction via a "correct", fit based approach that fits the condensate and 
thermals in a multi-step manner akin to that described in https://doi.org/10.1063/1.3125051"""

def get_rr_condensate_fractions_fit(my_measurement, my_run, imaging_mode = "abs", first_state_index = 1, second_state_index = 3, 
                                    first_stored_density_name = None, second_stored_density_name = None):
    if imaging_mode == "polrot":
        atom_density_first, atom_density_second = _load_densities_polrot(my_measurement, my_run, first_state_index, second_state_index, 
                                                    first_stored_density_name, second_stored_density_name)
    elif imaging_mode == "abs":
        atom_density_first = _load_density_top_A_abs(my_measurement, my_run, first_state_index, first_stored_density_name)
        atom_density_second = _load_density_top_B_abs(my_measurement, my_run, second_state_index, second_stored_density_name)
    #Rotate images 
    rr_angle = my_measurement.experiment_parameters["rr_tilt_deg"]
    atom_density_first = ndimage.rotate(atom_density_first, rr_angle, reshape = False)
    atom_density_second = ndimage.rotate(atom_density_second, rr_angle, reshape = False)
    #Atom densities are integrated along x axis by default... 
    integrated_atom_density_first = np.sum(atom_density_first, axis = -1)
    integrated_atom_density_second = np.sum(atom_density_second, axis = -1)
    condensate_results_first, thermal_results_first = data_fitting_functions.fit_one_dimensional_condensate(integrated_atom_density_first)
    condensate_results_second, thermal_results_second = data_fitting_functions.fit_one_dimensional_condensate(integrated_atom_density_second)
    condensate_counts_first = data_fitting_functions.one_d_condensate_integral(*condensate_results_first)
    thermal_counts_first = data_fitting_functions.thermal_bose_integral(*thermal_results_first)
    condensate_counts_second = data_fitting_functions.one_d_condensate_integral(*condensate_results_second)
    thermal_counts_second = data_fitting_functions.thermal_bose_integral(*thermal_results_second)
    condensate_fraction_first = condensate_counts_first / (condensate_counts_first + thermal_counts_first)
    condensate_fraction_second = condensate_counts_second / (condensate_counts_second + thermal_counts_second)
    return (condensate_fraction_first, condensate_fraction_second)


"""
Get the condensate fraction via a 'kludge': Define a box inside of which the condensate is found, subtract the average density of a region 
just outside that box, sum up the atom counts inside, and then"""
def get_rr_condensate_fractions_kludge(my_measurement, my_run, imaging_mode = "abs", first_state_index = 1, second_state_index = 3, 
                                    first_stored_density_name = None, second_stored_density_name = None):
    if imaging_mode == "polrot":
        atom_density_first, atom_density_second = _load_densities_polrot(my_measurement, my_run, first_state_index, second_state_index, 
                                                    first_stored_density_name, second_stored_density_name)
    elif imaging_mode == "abs":
        atom_density_first = _load_density_top_A_abs(my_measurement, my_run, first_state_index, first_stored_density_name)
        atom_density_second = _load_density_top_B_abs(my_measurement, my_run, second_state_index, second_stored_density_name)
    #Rotate images 
    rr_angle = my_measurement.experiment_parameters["rr_tilt_deg"]
    atom_density_first = ndimage.rotate(atom_density_first, rr_angle, reshape = False)
    atom_density_second = ndimage.rotate(atom_density_second, rr_angle, reshape = False)
    #Naive summing approach with background subtraction outside of the box
    pixel_area = np.square(my_measurement.experiment_parameters["top_um_per_pixel"])
    total_counts_first = image_processing_functions.atom_count_pixel_sum(atom_density_first, pixel_area)
    total_counts_second = image_processing_functions.atom_count_pixel_sum(atom_density_second, pixel_area)
    rr_roi = my_measurement.experiment_parameters["rr_roi"]
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


#MEASUREMENT-WIDE FUNCTIONS
"""
Certain analyses must be run on an entire measurement, and are sufficiently common as to warrant 
inclusion here - for instance, establishing a no_shake background for box shots. These analyses have a different 
calling signature, being called on fun(my_measurement, **kwargs)

NOTE: It is _not_ appropriate to include functions here which involve only taking an average over run analyses; this 
is better done by performing the analysis on all runs, then averaging over the results returned by get_analysis_value_from_runs."""


def get_no_shake_average_profiles(my_measurement, first_state_index = 1, second_state_index = 3,
                                    imaging_mode = "polrot",
                                        first_stored_density_name = None, second_stored_density_name = None):
    no_shake_sum_first = 0.0 
    no_shake_sum_second = 0.0 
    counter = 0 
    for run_id in my_measurement.runs_dict:
        current_run = my_measurement.runs_dict[run_id] 
        if not current_run.is_badshot and current_run.parameters["ShakingCycles"] == 0:
            if imaging_mode == "polrot":
                density_first, density_second = _load_densities_polrot(my_measurement, current_run, first_state_index, 
                                                        second_state_index, first_stored_density_name, second_stored_density_name)
            elif imaging_mode == "abs":
                density_first = _load_density_top_A_abs(my_measurement, current_run, first_state_index, first_stored_density_name)
                density_second = _load_density_top_B_abs(my_measurement, current_run, second_state_index, second_stored_density_name)
            no_shake_sum_first += density_first 
            no_shake_sum_second += density_second
            counter += 1 
    no_shake_average_first = no_shake_sum_first / counter 
    no_shake_average_second = no_shake_sum_second / counter
    return (no_shake_average_first, no_shake_average_second)

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

def _load_densities_polrot(my_measurement, my_run, first_state_index, second_state_index, first_stored_density_name, 
                            second_stored_density_name):
    if first_stored_density_name is None or second_stored_density_name is None:
        atom_density_first, atom_density_second = get_atom_densities_top_polrot(my_measurement, my_run, first_state_index=first_state_index, 
                                                    second_state_index=second_state_index)
    else:
        atom_density_first = my_run.analysis_results[first_stored_density_name]
        atom_density_second = my_run.analysis_results[second_stored_density_name]
    return (atom_density_first, atom_density_second)

def _load_density_top_A_abs(my_measurement, my_run, state_index, stored_density_name):
    if stored_density_name is None:
        atom_density = get_atom_density_top_A_abs(my_measurement, my_run, state_index = state_index)
    else:
        atom_density = my_run.analysis_results[stored_density_name]
    return atom_density

def _load_density_top_B_abs(my_measurement, my_run, state_index, stored_density_name):
    if stored_density_name is None:
        atom_density = get_atom_density_top_B_abs(my_measurement, my_run, state_index = state_index)
    else:
        atom_density = my_run.analysis_results[stored_density_name]
    return atom_density


def _load_density_side_li_hf(my_measurement, my_run, state_index, stored_density_name):
    if stored_density_name is None:
        atom_density = get_atom_density_side_li_hf(my_measurement, my_run, state_index=state_index)
    else:
        atom_density = my_run.analysis_results[stored_density_name]
    return atom_density


def _load_density_side_li_lf(my_measurement, my_run, stored_density_name):
    if stored_density_name is None:
        atom_density = get_atom_density_side_li_lf(my_measurement, my_run)
    else:
        atom_density = my_run.analysis_results[stored_density_name]
    return atom_density



