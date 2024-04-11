import warnings

import numpy as np 
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
from scipy.special import erf, erfinv
from scipy.signal import argrelextrema

from . import science_functions, statistics_functions, eos_functions

def fit_lorentzian(x_vals, y_vals, errors = None, amp_guess = None, center_guess = None, gamma_guess = None,
                                    filter_outliers = False, report_inliers = False):
    x_vals, y_vals, errors = _numpy_condition_data(x_vals, y_vals, errors = errors)
    param_guesses = _peak_guess_helper(x_vals, y_vals, amp_guess = amp_guess, center_guess = center_guess, 
                                      width_guess = gamma_guess, fit_offset = False)
    results = curve_fit(lorentzian, x_vals, y_vals, p0 = param_guesses, sigma = errors)
    return _filter_outliers_helper(x_vals, y_vals, lorentzian, results, errors = errors, report_inliers = report_inliers, 
                                   filter_outliers = filter_outliers)


def fit_lorentzian_with_offset(x_vals, y_vals, errors = None, amp_guess = None, center_guess = None, gamma_guess = None, offset_guess = None,
                                    filter_outliers = False, report_inliers = False):
    x_vals, y_vals, errors = _numpy_condition_data(x_vals, y_vals, errors = errors)
    param_guesses = _peak_guess_helper(x_vals, y_vals, amp_guess = amp_guess, center_guess = center_guess, 
                                      width_guess = gamma_guess, offset_guess = offset_guess, fit_offset = True)
    results = curve_fit(lorentzian_with_offset, x_vals, y_vals, p0 = param_guesses, sigma = errors)
    conditioned_results = _condition_peak_results(results)
    return _filter_outliers_helper(x_vals, y_vals, lorentzian_with_offset, conditioned_results, errors = errors, report_inliers = report_inliers, 
                                   filter_outliers = filter_outliers)

def fit_lorentzian_times_freq(x_vals, y_vals, errors = None, amp_guess = None, center_guess = None, gamma_guess = None,
                                    filter_outliers = False, report_inliers = False):
    x_vals, y_vals, errors = _numpy_condition_data(x_vals, y_vals, errors = errors)
    naive_param_guesses = _peak_guess_helper(x_vals, y_vals, amp_guess = amp_guess, center_guess = center_guess, 
                                      width_guess = gamma_guess, fit_offset = False)
    naive_amp_guess, center_guess, gamma_guess = naive_param_guesses 
    amp_guess = naive_amp_guess / center_guess
    param_guesses = [amp_guess, center_guess, gamma_guess] 
    results = curve_fit(lorentzian_times_freq, x_vals, y_vals, p0 = param_guesses, sigma = errors)
    conditioned_results = _condition_peak_results(results)
    return _filter_outliers_helper(x_vals, y_vals, lorentzian_times_freq, conditioned_results, errors = errors, report_inliers = report_inliers, 
                                   filter_outliers = filter_outliers)

def lorentzian(x, amp, center, gamma):
    return amp * 1.0 / (np.square(2.0 * (x - center) / gamma) + 1)

def lorentzian_with_offset(x, amp, center, gamma, offset):
    return lorentzian(x, amp, center, gamma) + offset

def lorentzian_times_freq(freq, amp, center, gamma):
    return amp * freq / (1 + np.square(2.0 * (freq - center) / gamma))

def get_lorentzian_times_freq_amp_to_peak_conversion_factor(center, gamma):
    peak_freq = np.sqrt(np.square(center) + np.square(gamma / 2))
    lorentzian_times_freq_on_peak_value = lorentzian_times_freq(peak_freq, 1.0, center, gamma)
    return lorentzian_times_freq_on_peak_value



def fit_gaussian(x_vals, y_vals, errors = None, amp_guess = None, center_guess = None, sigma_guess = None,
                                    filter_outliers = False, report_inliers = False):
    x_vals, y_vals, errors = _numpy_condition_data(x_vals, y_vals, errors = errors)
    param_guesses = _peak_guess_helper(x_vals, y_vals, amp_guess = amp_guess, center_guess = center_guess, 
                                      width_guess = sigma_guess, fit_offset = False)
    results = curve_fit(gaussian, x_vals, y_vals, p0 = param_guesses, sigma = errors)
    conditioned_results = _condition_peak_results(results)
    return _filter_outliers_helper(x_vals, y_vals, gaussian, conditioned_results, errors = errors, report_inliers = report_inliers, 
                                   filter_outliers = filter_outliers)


def fit_gaussian_with_offset(x_vals, y_vals, errors = None, amp_guess = None, center_guess = None, sigma_guess = None, offset_guess = None,
                                    filter_outliers = False, report_inliers = False):
    x_vals, y_vals, errors = _numpy_condition_data(x_vals, y_vals, errors = errors)
    param_guesses = _peak_guess_helper(x_vals, y_vals, amp_guess = amp_guess, center_guess = center_guess, 
                                      width_guess = sigma_guess, offset_guess = offset_guess, fit_offset = True)
    results = curve_fit(gaussian_with_offset, x_vals, y_vals, p0 = param_guesses, sigma = errors)
    conditioned_results = _condition_peak_results(results)
    return _filter_outliers_helper(x_vals, y_vals, gaussian_with_offset, conditioned_results, errors = errors, report_inliers = report_inliers, 
                                   filter_outliers = filter_outliers)

def gaussian(x, amp, center, sigma):
    return amp * np.exp(-np.square(x - center) / (2 * np.square(sigma)))

def gaussian_with_offset(x, amp, center, sigma, offset):
    return gaussian(x, amp, center, sigma) + offset


def _peak_guess_helper(x_vals, y_vals, amp_guess = None, center_guess = None, width_guess = None, offset_guess = None, fit_offset = False):
    data_average = np.average(y_vals)
    peak_index = np.argmax(np.abs(y_vals))
    peak_height = y_vals[peak_index] 
    peak_x = x_vals[peak_index]
    x_range = max(x_vals) - min(x_vals)
    if amp_guess is None:
        amp_guess = peak_height - data_average
    if center_guess is None:
        center_guess = peak_x
    if width_guess is None:
        x_range = max(x_vals) - min(x_vals) 
        width_guess = x_range / 2
    if offset_guess is None and fit_offset:
        offset_guess = data_average - amp_guess * width_guess / x_range
    if fit_offset:
        return (amp_guess, center_guess, width_guess, offset_guess)
    else:
        return (amp_guess, center_guess, width_guess)


def _numpy_condition_data(x_vals, y_vals, errors = None):
    x_vals = np.array(x_vals) 
    y_vals = np.array(y_vals) 
    if not errors is None:
        errors = np.array(errors) 
    return (x_vals, y_vals, errors)


def _filter_outliers_helper(x_vals, y_vals, fit_fun, results, errors = None, report_inliers = False, filter_outliers = False):
    if not filter_outliers:
        return results
    popt, pcov = results
    inlier_indices = _filter_1d_outliers(x_vals, y_vals, fit_fun, popt)
    filtered_x_vals = x_vals[inlier_indices] 
    filtered_y_vals = y_vals[inlier_indices] 
    if not errors is None:
        errors = errors[inlier_indices]
    results = curve_fit(fit_fun, filtered_x_vals, filtered_y_vals, p0=popt, sigma = errors)
    if report_inliers:
        return (results, inlier_indices) 
    else:
        return results
    
#The above "peak" functions have a width which may be negative; this makes it positive
def _condition_peak_results(results):
    popt, pcov = results 
    popt[2] = np.abs(popt[2])
    return (popt, pcov)


def fit_two_dimensional_gaussian(image_to_fit, center = None, amp = None, width_sigmas = None, offset = None, errors = None):
    image_indices_grid = np.indices(image_to_fit.shape)
    flattened_image_indices = image_indices_grid.reshape((2, image_indices_grid[0].size)) 
    flattened_image = image_to_fit.flatten()
    if(errors):
        errors = errors.flatten()
    number_y_pixels = image_to_fit.shape[0] 
    number_x_pixels = image_to_fit.shape[1] 
    if((not amp) or (not offset)):
        image_average = sum(sum(image_to_fit)) / image_to_fit.size 
    if(amp):
        amp_guess = amp 
    else:
        if(np.amax(image_to_fit) - image_average > image_average - np.amin(image_to_fit)):
            amp_guess = 1.0 
        else:
            amp_guess = -1.0 
    if(center):
        x_center_guess, y_center_guess = center 
    else:
        x_center_guess = number_x_pixels // 2 
        y_center_guess = number_y_pixels // 2 
    if(width_sigmas):
        sigma_x_guess, sigma_y_guess = width_sigmas 
    else:
        sigma_x_guess = number_x_pixels // 2 
        sigma_y_guess = number_y_pixels // 2
    if(offset):
        offset_guess = offset 
    else:
        offset_guess = image_average
    params = np.array([amp_guess, x_center_guess, y_center_guess, sigma_x_guess, sigma_y_guess, offset_guess])
    results = curve_fit(wrapped_two_dimensional_gaussian, flattened_image_indices, flattened_image, p0 = params, sigma = errors)
    return results

    

def wrapped_two_dimensional_gaussian(coordinates, amp, x_center, y_center, sigma_x, sigma_y, offset):
    x, y = coordinates 
    return two_dimensional_gaussian(x, y, amp, x_center, y_center, sigma_x, sigma_y, offset)


def two_dimensional_gaussian(x, y, amp, x_center, y_center, sigma_x, sigma_y, offset):
    return amp * np.exp(-np.square(x - x_center) / (2 * np.square(sigma_x)) - np.square(y - y_center) / (2.0 * np.square(sigma_y))) + offset
    

def two_dimensional_rotated_gaussian(x, y, amp, x_center, y_center, sigma_x_prime, sigma_y_prime, tilt_angle, offset):
    x_diff = x - x_center 
    y_diff = y - y_center 
    x_prime_diff = np.cos(tilt_angle) * x_diff + np.sin(tilt_angle) * y_diff 
    y_prime_diff = np.cos(tilt_angle) * y_diff - np.sin(tilt_angle) * x_diff 
    return amp * np.exp(-np.square(x_prime_diff) / (2 * np.square(sigma_x_prime)) - np.square(y_prime_diff) / (2 * np.square(sigma_y_prime))) + offset

def fit_one_dimensional_cosine(x_values, data, freq = None, amp = None, phase = None, offset = None, errors = None):
    if(not np.all([freq, amp, phase, offset])):
        freq_helper_guess, amp_helper_guess, phase_helper_guess, offset_helper_guess = _cosine_guess_helper(x_values, data)
    if(not freq):
        freq_guess = freq_helper_guess 
    else:
        freq_gues = freq 
    if(not amp):
        amp_guess = amp_helper_guess 
    else:
        amp_guess = amp 
    if(not phase):
        phase_guess = phase_helper_guess 
    else:
        phase_guess = phase 
    if(not offset):
        offset_guess = offset_helper_guess 
    else:
        offset_guess = offset
    params = np.array([freq_guess, amp_guess, phase_guess, offset_guess])
    results = curve_fit(one_dimensional_cosine, x_values, data, p0 = params, sigma = errors)
    return results

def _cosine_guess_helper(x_values, data):
    SPACING_REL_TOLERANCE = 1e-3
    sorted_x_values, sorted_data = _sort_and_deduplicate_xy_data(x_values, data)
    x_values_delta = sorted_x_values[1] - sorted_x_values[0]
    values_evenly_spaced = True
    for diff in np.diff(sorted_x_values):
        if(np.abs(diff - x_values_delta) / x_values_delta > SPACING_REL_TOLERANCE):
            values_evenly_spaced = False 
            break
    if(not values_evenly_spaced):
        MAXIMUM_INTERPOLATION_NUMBER = 1000
        x_width = sorted_x_values[-1] - sorted_x_values[0] 
        minimum_spacing = max(min(np.diff(sorted_x_values)), x_width / MAXIMUM_INTERPOLATION_NUMBER)
        num_samps = int(np.floor(x_width / minimum_spacing)) + 1
        interpolated_x_values = np.linspace(sorted_x_values[0], sorted_x_values[-1], num_samps) 
        interpolation_function = interp1d(sorted_x_values, sorted_data, kind = "cubic")
        interpolated_data = interpolation_function(interpolated_x_values)
        x_values_delta = minimum_spacing 
        x_values = interpolated_x_values 
        data = interpolated_data
    else:
        x_values = sorted_x_values 
        data = sorted_data  
    data_length = len(data)
    guessed_offset = sum(data) / len(data)
    centered_data = data - guessed_offset
    guessed_frequency, guessed_amp, guessed_phase = get_fft_peak(x_values_delta, centered_data, order = None)
    return (guessed_frequency, guessed_amp, guessed_phase, guessed_offset)

"""
Convenience function for the fast fourier transform of data.

Given an x, y dataset in correct order for FFT (sorted, and with equally spaced datapoints), 
return the frequency, amplitude and phase associated with a peak in the fft spectrum.

X_delta: The spacing between points in the independent variable. Required so that the corresponding frequency can be returned. 

Y_data: The data to be transformed. 

order: The order of the peak to return. If None, the largest non-zero order is returned.

Axis: Specifies the axis along which to (1D) Fourier transform for greater than 1D data. 
"""
def get_fft_peak(x_delta, y_data, order = None, axis = -1):
    #Put the axis to fft on at the end
    original_axis_position = axis 
    new_axis_position = -1
    moved_axis_ydata = np.moveaxis(y_data, original_axis_position, new_axis_position)
    data_length = moved_axis_ydata.shape[new_axis_position]
    centered_data_fft = np.fft.fft(moved_axis_ydata, axis = new_axis_position)
    fft_real_frequencies = np.fft.fftfreq(data_length) * 1.0 / x_delta 
    positive_fft_cutoff = int(np.floor(data_length / 2))
    positive_fft_values = centered_data_fft[..., 1:positive_fft_cutoff]
    positive_fft_frequencies = fft_real_frequencies[1:positive_fft_cutoff]
    if(order is None):
        fft_peak_indices = np.argmax(np.abs(positive_fft_values), axis = new_axis_position, keepdims = True)
        fft_peak_values = np.squeeze(np.take_along_axis(positive_fft_values, fft_peak_indices, axis = new_axis_position), axis = new_axis_position)
        fft_frequency = positive_fft_frequencies[np.squeeze(fft_peak_indices, axis = new_axis_position)]
    else:
        fft_peak_index = order - 1
        fft_peak_values = positive_fft_values[..., fft_peak_index]
        fft_frequency = positive_fft_frequencies[fft_peak_index] * np.ones(fft_peak_values.shape)
    fft_phase = np.angle(fft_peak_values)
    fft_amp = np.abs(fft_peak_values) * 2.0 / data_length
    return (fft_frequency, fft_amp, fft_phase)


def one_dimensional_cosine(x_values, freq, amp, phase, offset):
    return amp * np.cos(2 * np.pi * x_values * freq + phase) + offset


#By convention, frequencies are in kHz, and times are in ms. 
def fit_rf_spect_detuning_scan(rf_freqs, transfers, tau, center = None, rabi_freq = None, errors = None,
                            filter_outliers = False, report_inliers = False):
    rf_freqs = np.array(rf_freqs)
    transfers = np.array(transfers) 
    if(errors):
        errors = np.array(errors)
    if(center is None):
        center_guess = _find_center_helper(rf_freqs, transfers)
    else:
        center_guess = center 
    if(rabi_freq is None):
        rabi_freq_guess = 1.0 
    else:
        rabi_freq_guess = rabi_freq
    def wrapped_rf_spect_function_factory(tau):
        def rf_spect_function(rf_freqs, center, rabi_freq):
            return rf_spect_detuning_scan(rf_freqs, tau, center, rabi_freq)
        return rf_spect_function
    tau_wrapped_function = wrapped_rf_spect_function_factory(tau)
    params = np.array([center_guess, rabi_freq_guess])
    results = curve_fit(tau_wrapped_function, rf_freqs, transfers, p0 = params, sigma = errors)
    popt, pcov = results
    if(filter_outliers):
        inlier_indices = _filter_1d_outliers(rf_freqs, transfers, tau_wrapped_function, popt)
        filtered_rf_freqs = rf_freqs[inlier_indices] 
        filtered_transfers = transfers[inlier_indices] 
        if(errors):
            errors = errors[inlier_indices]
        refitted_results = curve_fit(tau_wrapped_function, filtered_rf_freqs, filtered_transfers, p0 = popt, sigma = errors)
        if(report_inliers):
            return (refitted_results, inlier_indices)
        else:
            return refitted_results
    else:
        return results


def rf_spect_detuning_scan(rf_freqs, tau, center, rabi_freq):
    omega_rabi = 2 * np.pi * rabi_freq
    detunings = 2 * np.pi * (rf_freqs - center)
    populations_excited = science_functions.two_level_system_population_rabi(tau, omega_rabi, detunings)[1]
    return populations_excited


def hybrid_trap_center_finder(image_to_fit, tilt_deg, hybrid_pixel_width, hybrid_pixel_length, center_guess = None):
    width_sigma = hybrid_pixel_width / 4
    length_sigma = hybrid_pixel_length / 4
    tilt_rad = tilt_deg * np.pi / 180
    estimated_image_amplitude = np.sum(image_to_fit) / (hybrid_pixel_length * hybrid_pixel_width)
    def wrapped_hybrid_fitting_gaussian(coordinate, x_center, y_center):
        y, x = coordinate 
        return two_dimensional_rotated_gaussian(x, y, estimated_image_amplitude, x_center, y_center, width_sigma, 
                                            length_sigma, tilt_rad, 0)
    #Numpy puts vertical (y) index first by default
    image_indices_grid = np.indices(image_to_fit.shape)
    flattened_image_indices = image_indices_grid.reshape((2, image_indices_grid[0].size)) 
    flattened_image = image_to_fit.flatten()
    if(center_guess is None):
        image_y_length = image_to_fit.shape[0] 
        y_center_guess = int(np.floor(image_y_length / 2)) 
        image_x_length = image_to_fit.shape[1] 
        x_center_guess = int(np.floor(image_x_length / 2))
    else:
        x_center_guess, y_center_guess = center_guess
    params = np.array([x_center_guess, y_center_guess])
    results = curve_fit(wrapped_hybrid_fitting_gaussian, flattened_image_indices, flattened_image, p0 = params)
    popt, pcov = results 
    return popt
    

"""Helper function for finding the center of data with an even symmetry point.

Given a dataset x, y, uses least-squares optimization to try to find the maximally evenly symmetric 
point in the data, i.e. the point for which |f(x_0 + a) - f(x_0 - a)| is minimized in a least squares sense 
for all the available data."""

def _find_center_helper(x_data, y_data):
    INTERPOLATION_PRECISION = 1000
    sorted_deduplicated_x_values, sorted_deduplicated_y_values = _sort_and_deduplicate_xy_data(x_data, y_data)
    y_interpolation = interp1d(sorted_deduplicated_x_values, sorted_deduplicated_y_values, kind = "cubic")
    minimum_x = min(x_data)
    maximum_x = max(x_data)
    x_data_center = 0.5 * (maximum_x + minimum_x)
    x_data_width = maximum_x - minimum_x
    interpolation_step = (maximum_x - minimum_x) / INTERPOLATION_PRECISION
    def symmetry_test_function(center_guess):
        difference_metric = 0.0
        if(center_guess > maximum_x or center_guess < minimum_x):
            pass
        else:
            if (center_guess - minimum_x < maximum_x - center_guess):
                range = center_guess - minimum_x 
                test_window_half_width = int(np.floor(range / interpolation_step))
                test_window_center_index = test_window_half_width
                test_window_number_points = 2 * test_window_half_width + 1
                test_window_frequencies = np.linspace(minimum_x, center_guess + range, test_window_number_points)
            else:
                range = maximum_x - center_guess 
                test_window_half_width = int(np.floor(range / interpolation_step))
                test_window_center_index = test_window_half_width
                test_window_number_points = 2 * test_window_half_width + 1
                test_window_frequencies = np.linspace(center_guess - range, maximum_x, test_window_number_points) 
            normed_squared_difference_sum = 0.0
            for i in (np.arange(test_window_half_width) + 1):
                left_point = test_window_center_index - i 
                right_point = test_window_center_index + i 
                left_value = y_interpolation(test_window_frequencies[left_point])
                right_value = y_interpolation(test_window_frequencies[right_point])
                squared_difference = np.square(left_value - right_value)
                normed_squared_difference = squared_difference / np.sqrt(np.square(left_value) + np.square(right_value))
                normed_squared_difference_sum += normed_squared_difference
            difference_metric = normed_squared_difference_sum / test_window_number_points
        #Ad hoc penalty for a center close to the edge
        #Added to keep the code from railing to the edges...
        data_abs_sum = sum(np.abs(y_data))
        normalized_distance_from_center = (center_guess - x_data_center) / (x_data_width / 2)
        AD_HOC_X_POWER = 26
        ad_hoc_penalty = data_abs_sum * np.power(normalized_distance_from_center, AD_HOC_X_POWER)
        return difference_metric + ad_hoc_penalty
    relative_maxima = argrelextrema(sorted_deduplicated_y_values, np.greater, order = 2)
    relative_minima = argrelextrema(sorted_deduplicated_y_values, np.less, order = 2)
    relative_extrema = np.append(relative_maxima, relative_minima) 
    extrema_x_values = sorted_deduplicated_x_values[relative_extrema]
    minimal_function_value = np.inf
    best_center_guess = None
    for initial_center_guess in extrema_x_values:
        optimization_results = minimize(symmetry_test_function, [initial_center_guess])
        optimized_center_guess = optimization_results.x[0] 
        optimized_function_value = optimization_results.fun
        if optimized_function_value < minimal_function_value:
            best_center_guess = optimized_center_guess
            minimal_function_value = optimized_function_value
    return best_center_guess


"""
Fit a rapid ramp condensate using the multi-step approach described in https://doi.org/10.1063/1.3125051
"""

def fit_one_dimensional_condensate(one_dimensional_data, initial_center_guess = None, initial_condensate_width_guess = None, 
                                initial_gaussian_width_guess = None, initial_condensate_amp_guess = None, initial_thermal_amp_guess = None):
    positions = np.arange(len(one_dimensional_data))
    (initial_center_guess, initial_condensate_width_guess, initial_gaussian_width_guess, 
    initial_condensate_amp_guess, initial_thermal_amp_guess
    ) = _fit_rr_condensate_populate_guesses(
                                        one_dimensional_data, initial_center_guess, initial_condensate_width_guess, initial_gaussian_width_guess, 
                                        initial_condensate_amp_guess, initial_thermal_amp_guess
                                                                                                    )
    p_init_bimodal = [
        initial_center_guess, 
        initial_condensate_width_guess, 
        initial_condensate_amp_guess,
        initial_gaussian_width_guess, 
        initial_thermal_amp_guess
    ]
    bimodal_popt, bimodal_pcov = curve_fit(condensate_bimodal_function, positions, one_dimensional_data, p0 = p_init_bimodal)
    bimodal_center, bimodal_cwidth, bimodal_camp, bimodal_gwidth, bimodal_tamp = bimodal_popt 
    bimodal_center_index = int(np.round(bimodal_center))
    bimodal_cwidth = np.abs(bimodal_cwidth) 
    SAFETY_MARGIN = 1.2
    condensate_window_width = int(np.round(SAFETY_MARGIN * bimodal_cwidth))
    condensate_position_slice = positions[bimodal_center_index - condensate_window_width:bimodal_center_index + condensate_window_width]
    condensate_excluded_position_slice = positions[~np.isin(positions, condensate_position_slice)]
    condensate_excluded_data_slice = one_dimensional_data[condensate_excluded_position_slice]
    p_init_thermal = [
        bimodal_center, 
        bimodal_gwidth, 
        bimodal_tamp
    ]
    thermal_popt, thermal_pcov = curve_fit(thermal_bose_function, condensate_excluded_position_slice, condensate_excluded_data_slice, p0 = p_init_thermal)
    p_init_condensate = [
        bimodal_center, 
        bimodal_cwidth, 
        bimodal_camp
    ]
    condensate_thermal_subtracted_data_slice = one_dimensional_data[condensate_position_slice] - thermal_bose_function(condensate_position_slice, *thermal_popt)
    condensate_popt, condensate_pcov = curve_fit(one_d_condensate_function, condensate_position_slice, 
                                    condensate_thermal_subtracted_data_slice, p0 = p_init_condensate)
    return ((condensate_popt, condensate_pcov), (thermal_popt, thermal_pcov))

def _fit_rr_condensate_populate_guesses(one_dimensional_data, initial_center_guess, initial_condensate_width_guess, initial_gaussian_width_guess, 
        initial_condensate_amp_guess, initial_thermal_amp_guess):
    data_width = len(one_dimensional_data)
    if initial_center_guess is None:
        initial_center_guess = np.argmax(one_dimensional_data)
    if initial_condensate_width_guess is None:
        #Hard coded best attempt...
        initial_condensate_width_guess = data_width // 10
    if initial_gaussian_width_guess is None:
        #Again, hard coded guess
        initial_gaussian_width_guess = data_width // 3
    if initial_thermal_amp_guess is None:
        initial_thermal_amp_guess = one_dimensional_data[initial_center_guess + initial_condensate_width_guess]
    if initial_condensate_amp_guess is None:
        initial_condensate_amp_guess = max(one_dimensional_data) - initial_thermal_amp_guess
    return (initial_center_guess, initial_condensate_width_guess, initial_gaussian_width_guess, 
            initial_condensate_amp_guess, initial_thermal_amp_guess)





def one_d_condensate_function(x, center, condensate_width, condensate_amp):
    return condensate_amp * np.square(np.maximum(0, 1 - np.square((x - center) / condensate_width)))

def one_d_condensate_integral(center, condensate_width, condensate_amp):
    return 16 / 15 * condensate_amp * condensate_width

def thermal_bose_function(x, center, gaussian_width, gaussian_amp):
    gaussian_factor = np.exp(-np.square(x - center) / (2 * np.square(gaussian_width)))
    return gaussian_amp / (1 + 1/4 + 1/9) * (gaussian_factor + 1/4 * np.square(gaussian_factor) + 1/9 * np.power(gaussian_factor, 3))

#Ditto the above for the thermal bose function
def thermal_bose_integral(center, gaussian_width, gaussian_amp):
    return np.sqrt(2 * np.pi) * gaussian_amp * gaussian_width * (1 + np.power(2, -2.5) + np.power(3, -2.5))

def condensate_bimodal_function(x, center, condensate_width, condensate_amp, gaussian_width, gaussian_amp):
    return one_d_condensate_function(x, center, condensate_width, condensate_amp) + thermal_bose_function(x, center, gaussian_width, gaussian_amp)



#Box fitting

def fit_error_function_rectangle(positions, values,
                        amp_guess = None, width_guess = None, center_guess = None, edge_width_guess = None,
                        fit_width = True):
    max_value = np.max(values) 
    if amp_guess is None:
        amp_guess = max_value 
    positions_above_half_max = positions[values > 0.5 * max_value] 
    positions_above_half_max_len = len(positions_above_half_max)
    if center_guess is None:
        center_guess = positions_above_half_max[positions_above_half_max_len // 2]
    if edge_width_guess is None:
        positions_delta = np.diff(positions)[0] 
        edge_width_guess = positions_delta
    if fit_width:
        if width_guess is None:
            width_guess = positions_above_half_max[-1] - positions_above_half_max[0] 
        p_init = [
            amp_guess, 
            width_guess, 
            center_guess, 
            edge_width_guess
        ]
        return curve_fit(error_function_rectangle, positions, values, p0 = p_init)
    else:
        if width_guess is None:
            raise ValueError("If not fitting the width, it must be specified.")
        def error_function_rectangle_fixed_width(x, amp, center, edge_width):
            return error_function_rectangle(x, amp, width_guess, center, edge_width)
        p_init = [
            amp_guess, 
            center_guess,
            edge_width_guess
        ]
        return curve_fit(error_function_rectangle_fixed_width, positions, values, p0 = p_init)

def error_function_rectangle(x, amp, width, center, edge_width):
    rect_on = center - width / 2.0 
    rect_off = center + width / 2.0 
    return amp / 2.0 * (erf((x - rect_on) / edge_width) - erf((x - rect_off) / edge_width))




def _error_function_rectangle_crop_helper(width, center, edge_width, crop_point = 0.5):
    #Calculates the crop point assuming the other erf is identically 1; usually a good assumption
    erf_value = 2.0 * (crop_point - 0.5)
    erf_contribution = edge_width * erfinv(erf_value)
    crop_center_shift = width / 2.0 - erf_contribution
    return (center - crop_center_shift, center + crop_center_shift)




def fit_semicircle(positions, values, 
                    amp_guess = None, center_guess = None, radius_guess = None, 
                    fit_radius = True):
    max_value = np.max(values)
    if amp_guess is None:
        amp_guess = max_value
    positions_above_05 = positions[values > 0.05 * max_value]
    len_positions_above_05 = len(positions_above_05)
    if center_guess is None:
        center_guess = positions_above_05[len_positions_above_05 // 2]
    if fit_radius:
        if radius_guess is None:
            radius_guess = 0.5 * (positions_above_05[-1] - positions_above_05[0])
        p_init = [
            amp_guess, 
            center_guess, 
            radius_guess
        ]
        return curve_fit(semicircle, positions, values, p0 = p_init)
    else:
        if radius_guess is None:
            raise ValueError("If the radius is not fitted, it must be specified.")
        def semicircle_fixed_radius(positions, amp, center):
            return semicircle(positions, amp, center, radius_guess)
        p_init = [
            amp_guess, 
            center_guess
        ]
        return curve_fit(semicircle_fixed_radius, positions, values, p0 = p_init)

def semicircle(positions, amp, center, radius):
    past_radius_indices = np.abs((positions - center) / radius) >= 1
    within_radius_indices = np.logical_not(past_radius_indices)
    return_array = np.zeros(len(positions))
    return_array[past_radius_indices] = 0.0 
    return_array[within_radius_indices] = amp * np.sqrt(1 - np.square((positions[within_radius_indices] - center) / radius))
    return return_array


def _semicircle_crop_helper(center, radius, crop_point = 0.0):
    if crop_point < 0.0:
        raise ValueError("Crop point can't be negative.")
    crop_center_shift = np.sqrt(1 - np.square(crop_point)) * radius 
    return (center - crop_center_shift, center + crop_center_shift)



def crop_box(atom_densities, vert_crop_point = 0.5, horiz_crop_point = 0.01, horiz_radius = None, vert_width = None):
    x_integrated_atom_densities = np.sum(atom_densities, axis = -1) 
    x_int_len = len(x_integrated_atom_densities)
    x_integrated_indices = np.arange(x_int_len)
    y_integrated_atom_densities = np.sum(atom_densities, axis = 0)
    y_int_len = len(y_integrated_atom_densities)
    y_integrated_indices = np.arange(y_int_len)
    if horiz_radius is None:
        horiz_fit_results = fit_semicircle(y_integrated_indices, y_integrated_atom_densities)
        horiz_popt, horiz_pcov = horiz_fit_results 
        h_amp, h_center, h_radius = horiz_popt 
    else:
        horiz_fit_results = fit_semicircle(y_integrated_indices, y_integrated_atom_densities, 
                                                            fit_radius = False, radius_guess = horiz_radius)
        horiz_popt, horiz_pcov = horiz_fit_results 
        h_amp, h_center = horiz_popt 
        h_radius = horiz_radius
    h_crop_min, h_crop_max = _semicircle_crop_helper(h_center, h_radius, crop_point = horiz_crop_point)
    h_crop_diff = np.rint(h_crop_max - h_crop_min).astype(int)
    h_crop_min = np.rint(h_crop_min).astype(int)
    if h_crop_min < 0:
        warnings.warn("The horizontal crop minimum is outside the image window.")
    h_crop_min = np.max((0, h_crop_min))
    #Force the rounding to be insensitive to the relative positions of h_crop_min and h_crop_max
    h_crop_max = h_crop_min + h_crop_diff
    if h_crop_max > y_int_len - 1:
        warnings.warn("The horizontal crop maximum is outside the image window.")
    h_crop_max = np.min((y_int_len - 1, h_crop_max))
    if vert_width is None:
        vert_fit_results = fit_error_function_rectangle(x_integrated_indices, x_integrated_atom_densities)
        vert_popt, vert_pcov = vert_fit_results 
        v_amp, v_width, v_center, v_edge_width = vert_popt 
    else:
        vert_fit_results = fit_error_function_rectangle(x_integrated_indices, x_integrated_atom_densities, fit_width = False, 
                                                width_guess = vert_width)
        vert_popt, vert_pcov = vert_fit_results 
        v_amp, v_center, v_edge_width = vert_popt 
        v_width = vert_width
    v_crop_min, v_crop_max = _error_function_rectangle_crop_helper(v_width, v_center, v_edge_width, crop_point = vert_crop_point)
    v_crop_diff = np.rint(v_crop_max - v_crop_min).astype(int)
    v_crop_min = np.rint(v_crop_min).astype(int)
    if v_crop_min < 0:
        warnings.warn("The vertical crop minimum is outside the image window.")
    v_crop_min = np.max((0, v_crop_min))
    #Force the difference between the two to be insensitive to the relative position of v_crop_min and v_crop_max
    v_crop_max = v_crop_min + v_crop_diff
    if v_crop_max > x_int_len - 1:
        warnings.warn("The vertical crop maximum is outside the image window.")
    v_crop_max = np.min((x_int_len - 1, v_crop_max))
    return (h_crop_min, v_crop_min, h_crop_max, v_crop_max)


#Fitting ideal Fermi density data

#Fit the (3d) fermi density of ideal lithium-6 vs potential to extract the global chemical potential mu_0 and kBT in Hz. 
#Note that the chemical potential is returned in absolute terms; if an additive constant is added to the potentials, the same 
#constant will be added to the chemical potential.
def fit_li6_ideal_fermi_density(potentials_Hz, densities_um, errors = None, absolute_mu_0_Hz_guess = None, kBT_Hz_guess = None):
    return _fit_li6_density_helper(
        li6_ideal_fermi_density, potentials_Hz, densities_um, errors = errors, absolute_mu_0_Hz_guess = absolute_mu_0_Hz_guess, 
        kBT_Hz_guess = kBT_Hz_guess)

#Fit the (3d) Fermi density of ideal lithium-6 as above, but with a free initial multiplicative prefactor
#representing a possible miscalibration of density. Here, fitting the chemical potential requires exploring 
#regions of sufficiently high fugacity to go beyond the ideal gas limit. 
def fit_li6_ideal_fermi_density_with_prefactor(potentials_Hz, densities_um, errors = None, absolute_mu_0_Hz_guess = None, kBT_Hz_guess = None, 
                                               prefactor_guess = 1.0):
    return _fit_li6_density_helper(
        li6_ideal_fermi_density_with_prefactor, potentials_Hz, densities_um, errors = errors, absolute_mu_0_Hz_guess= absolute_mu_0_Hz_guess, 
        kBT_Hz_guess=kBT_Hz_guess, prefactor_guess=prefactor_guess
    )


#Ibid above, but with the balanced EOS replacing the spin-polarized EOS
def fit_li6_balanced_density(potentials_Hz, densities_um, errors = None, absolute_mu_0_Hz_guess = None, kBT_Hz_guess = None):
    return _fit_li6_density_helper(
        li6_balanced_density, potentials_Hz, densities_um, errors = errors, absolute_mu_0_Hz_guess = absolute_mu_0_Hz_guess, 
        kBT_Hz_guess = kBT_Hz_guess)


def fit_li6_balanced_density_with_prefactor(potentials_Hz, densities_um, errors = None, absolute_mu_0_Hz_guess = None, kBT_Hz_guess = None, 
                                               prefactor_guess = 1.0):
    return _fit_li6_density_helper(
        li6_balanced_density_with_prefactor, potentials_Hz, densities_um, errors = errors, absolute_mu_0_Hz_guess= absolute_mu_0_Hz_guess, 
        kBT_Hz_guess=kBT_Hz_guess, prefactor_guess=prefactor_guess
    )


def _fit_li6_density_helper(function, potentials_Hz, densities_um, errors = None, absolute_mu_0_Hz_guess = None, 
                                        kBT_Hz_guess = None, prefactor_guess = None):
    potentials_Hz, densities_um, errors = _numpy_condition_data(potentials_Hz, densities_um, errors = errors)
    potential_minimum = np.min(potentials_Hz)
    if absolute_mu_0_Hz_guess is None:
        #Assume zero temperature fermi gas at maximum density point
        peak_density_um = np.max(densities_um)
        relative_mu_0_Hz_guess = science_functions.get_fermi_energy_hz_from_density(peak_density_um * 1e18)
        absolute_mu_0_Hz_guess = relative_mu_0_Hz_guess + potential_minimum
    if kBT_Hz_guess is None: 
        #Assume T/T_F value of 0.5. Inconsistent with above... but useful for getting order of magnitude
        kBT_Hz_guess = 0.5 * (absolute_mu_0_Hz_guess - potential_minimum)
    #None means we don't fit with prefactor
    if prefactor_guess is None:
        param_guesses = [absolute_mu_0_Hz_guess, kBT_Hz_guess]
    else:
        param_guesses = [absolute_mu_0_Hz_guess, kBT_Hz_guess, prefactor_guess]
    results = curve_fit(function, potentials_Hz, densities_um, p0 = param_guesses, sigma = errors)
    return results

def li6_ideal_fermi_density(potentials_Hz, absolute_mu_0_Hz, kBT_Hz):
    local_mu_values = absolute_mu_0_Hz - potentials_Hz
    local_betamu_values = local_mu_values / kBT_Hz 
    return eos_functions.ideal_fermi_density_um(local_betamu_values, kBT_Hz, species = "6Li")

def li6_ideal_fermi_density_with_prefactor(potentials_Hz, absolute_mu_0_Hz, kBT_Hz, prefactor):
    return prefactor * li6_ideal_fermi_density(potentials_Hz, absolute_mu_0_Hz, kBT_Hz)


def li6_balanced_density(potentials_Hz, absolute_mu_0_Hz, kBT_Hz):
    local_mu_values = absolute_mu_0_Hz - potentials_Hz
    local_betamu_values = local_mu_values / kBT_Hz
    return eos_functions.balanced_density_um(local_betamu_values, kBT_Hz, species = "6Li")

def li6_balanced_density_with_prefactor(potentials_Hz, absolute_mu_0_Hz, kBT_Hz, prefactor):
    return prefactor * li6_balanced_density(potentials_Hz, absolute_mu_0_Hz, kBT_Hz)


def _sort_and_deduplicate_xy_data(x_values, y_values):
    sorted_x_values, sorted_y_values = zip(*sorted(zip(x_values, y_values), key = lambda f: f[0]))
    sorted_x_values = np.array(sorted_x_values) 
    sorted_y_values = np.array(sorted_y_values)
    deduplicated_x_values = [] 
    deduplicated_y_values = [] 
    most_recent_x_value = -np.inf 
    most_recent_y_sum = 0.0
    recurrence_counter = 0 
    for x_value, y_value in zip(sorted_x_values, sorted_y_values):
        if(x_value == most_recent_x_value):
            recurrence_counter += 1
            most_recent_y_sum += y_value 
        else:
            if(recurrence_counter > 0):
                deduplicated_x_values.append(most_recent_x_value)
                most_recent_y_average = most_recent_y_sum / recurrence_counter 
                deduplicated_y_values.append(most_recent_y_average)
            most_recent_x_value = x_value 
            most_recent_y_sum = y_value
            recurrence_counter = 1
    deduplicated_x_values.append(most_recent_x_value)
    deduplicated_y_values.append(most_recent_y_sum / recurrence_counter)
    return (np.array(deduplicated_x_values), np.array(deduplicated_y_values))



def _group_like_x_xy_data(x_data, y_data, rtol = 1e-5, atol = 1e-8):
    unique_x_data_values_list = [] 
    like_x_y_data_list = []
    for x_val in x_data:
        for present_x_val in unique_x_data_values_list:
            if np.isclose(x_val, present_x_val, rtol = rtol, atol = atol):
                break 
        else:
            unique_x_data_values_list.append(x_val)
    unique_x_data = np.array(unique_x_data_values_list) 
    for unique_x_val in unique_x_data:
        associated_y_vals = y_data[np.isclose(x_data, unique_x_val, rtol = rtol, atol = atol)]
        like_x_y_data_list.append(associated_y_vals)
    return (unique_x_data, like_x_y_data_list)


def bootstrap_fit_covariance(fit_function, x_data, y_data, popt, n_resamples = 500, 
                    return_full_bootstrap_result = False, ignore_errors = False, x_rtol = 1e-5, x_atol = 1e-8, 
                    rng_seed = None):
    unique_x_data, like_x_y_data_list = _group_like_x_xy_data(x_data, y_data, rtol = x_rtol, atol = x_atol)
    def fit_statistic(*y_data_list):
        x_data_array = np.array([]) 
        y_data_array = np.array([])
        for unique_x_val, y_data in zip(unique_x_data, y_data_list):
            num_y_vals = len(y_data)
            x_data_array = np.concatenate((x_data_array, unique_x_val * np.ones(num_y_vals)))
            y_data_array = np.concatenate((y_data_array, y_data))
        results = curve_fit(fit_function, x_data_array, y_data_array, p0 = popt)
        statistic_popt, _ = results
        return statistic_popt 
    bootstrap_result = statistics_functions.generalized_bootstrap(like_x_y_data_list, fit_statistic, n_resamples = n_resamples, vectorized = False, 
                        ignore_errors = ignore_errors, rng_seed = rng_seed)
    if return_full_bootstrap_result:
        return bootstrap_result 
    else:
        return bootstrap_result.covariance_matrix
    

"""
Given a fitting function & parameter values and a set of x-y data (as np arrays)
they purport to fit, filter outliers using Student's t-test at the specified confidence level.

Returns the indices of the x-y data which are _INLIERS_, i.e. the complement of outliers,
points that can be identified as having a chance of less than alpha to occur."""
def _filter_1d_outliers(x_values, y_values, fitting_func, popt, alpha = 1e-4):
    num_params = len(popt)
    fit_values = fitting_func(x_values, *popt)
    residuals = y_values - fit_values
    return statistics_functions.filter_1d_residuals(residuals, num_params)



"""
Helper function for using a Monte Carlo analysis to get the covariance matrix of 
the best-fit parameters for a function fun to a dataset x_data, y_data."""
def monte_carlo_fit_covariance(fit_function, x_data, y_data, y_errors, popt, num_samples = 100, rng_seed = None):
    popt_list = []
    rng = np.random.default_rng(seed = rng_seed)
    for i in range(num_samples):
        simulated_y_data = y_data + rng.normal(loc = 0.0, scale = y_errors, size = y_errors.shape)
        results = curve_fit(fit_function, x_data, simulated_y_data, p0 = popt)
        simulated_popt, _ = results 
        popt_list.append(simulated_popt) 
    #Make sure the monte carlo sample axis is -1 and the parameter axis is 0
    popt_array = np.transpose(np.array(popt_list))
    popt_average = np.average(popt_array, axis = -1, keepdims = True) 
    popt_deviations_array = popt_array - popt_average
    pcov = np.matmul(popt_deviations_array, np.transpose(popt_deviations_array)) / np.size(popt_deviations_array, axis = -1)
    return pcov
    

"""
Convenience function for getting a pretty_printable fit report from scipy.optimize.curve_fit"""
def fit_report(model_function, fit_results, precision = 3):
    popt, pcov = fit_results
    report_string = ''
    report_string = report_string + "Model function: " + model_function.__name__ + "\n \n"
    varnames_tuple = get_varnames_from_function(model_function) 
    varnames_list = list(varnames_tuple) 
    #Some base functions have parameters that shouldn't be fitted, e.g. for rf spect
    #By convention, these names will come first in the parameters, after the independent variables
    #Thus, take only the last n names from varnames list, with n the length of popt
    varnames_to_skip = len(varnames_list) - len(popt)
    fitted_varnames = varnames_list[varnames_to_skip:]
    my_sigmas = np.sqrt(np.diag(pcov))
    for varname, value, sigma in zip(fitted_varnames, popt, my_sigmas):
        report_string = report_string + "Parameter: {0}\tValue: {1:.{4}e} ± {2:.2e} \t({3:.2%}) \n".format(varname, value, sigma, np.abs(sigma / value), precision)
    report_string = report_string + "\n"
    report_string = report_string + "Correlations (unreported are <0.1): \n"
    for i in range(len(popt)):
        for j in range(i + 1, len(popt)):
            covariance = pcov[i][j]
            correlation = covariance / (my_sigmas[i] * my_sigmas[j]) 
            if(np.abs(correlation) > 0.1):
                report_string = report_string + "{0} and {1}: \t {2:.2f}\n".format(fitted_varnames[i], fitted_varnames[j], correlation)
    return report_string

def get_varnames_from_function(my_func):
    arg_names = my_func.__code__.co_varnames[:my_func.__code__.co_argcount]
    DEFAULT_INDEPENDENT_VARNAMES = ['t', 'x', 'y', 'x_values', 'rf_freqs', 'potentials_Hz']
    arg_names = [f for f in arg_names if (not f in DEFAULT_INDEPENDENT_VARNAMES)]
    return arg_names