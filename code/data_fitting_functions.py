import warnings

import numpy as np 
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
from scipy.special import betainc
from scipy.signal import argrelextrema

from .science_functions import two_level_system_population_rabi

def fit_imaging_resonance_lorentzian(frequencies, counts, errors = None, linewidth = None, center = None, offset = None,
                                    filter_outliers = False, report_inliers = False, monte_carlo_cov = False, monte_carlo_samples = 1000):
    #Cast to guarantee we can use array syntax
    frequencies = np.array(frequencies) 
    counts = np.array(counts)
    if(not errors is None):
        errors = np.array(errors)
    #Rough magnitude expected for gamma in any imaging resonance lorentzian we would plot
    INITIAL_GAMMA_GUESS = 5.0
    data_average = sum(counts) / len(counts)
    frequency_average = sum(frequencies) / len(frequencies)
    frequency_range = max(frequencies) - min(frequencies) 
    center_guess = frequencies[np.argmax(np.abs(counts))]
    gamma_guess = INITIAL_GAMMA_GUESS
    offset_guess = data_average
    if(max(counts) - data_average > data_average - min(counts)):
        amp_guess = max(counts) - data_average 
    else:
        amp_guess = data_average - min(counts)
    params = np.ones(4)
    params[0] = amp_guess 
    if(center):
        params[1] = center 
    else:
        params[1] = center_guess 
    if(linewidth):
        params[2] = linewidth 
    else:
        params[2] = gamma_guess
    if(offset):
        params[3] = offset 
    else:
        params[3] = offset_guess
    results = curve_fit(imaging_resonance_lorentzian, frequencies, counts, p0 = params, sigma = errors)
    if(filter_outliers):
        popt, pcov = results
        inlier_indices = _filter_1d_outliers(frequencies, counts, imaging_resonance_lorentzian, 
                                                            popt)
        frequencies = frequencies[inlier_indices] 
        counts = counts[inlier_indices] 
        if(errors):
            errors = errors[inlier_indices]
        results = curve_fit(imaging_resonance_lorentzian, frequencies, counts, p0 = popt, sigma = errors)
    if(monte_carlo_cov):
        popt, _ = results 
        pcov = _monte_carlo_covariance_helper(imaging_resonance_lorentzian, frequencies, counts, errors, popt, num_samples = monte_carlo_samples)
        results = (popt, pcov) 
    if(report_inliers):
        return (results, inlier_indices) 
    else:
        return results

def imaging_resonance_lorentzian(imaging_freq, amp, center, gamma, offset):
    return amp * 1.0 / (np.square(2.0 * (imaging_freq - center) / gamma) + 1) + offset



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
    populations_excited = two_level_system_population_rabi(tau, omega_rabi, detunings)[1]
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

"""
Given a fitting function & parameter values and a set of x-y data (as np arrays)
they purport to fit, filter outliers using Student's t-test at the specified confidence level.

Returns the indices of the x-y data which are _INLIERS_, i.e. the complement of outliers,
points that can be identified as having a chance of less than alpha to occur."""
def _filter_1d_outliers(x_values, y_values, fitting_func, popt, alpha = 1e-4):
    num_params = len(popt)
    num_samples = len(y_values)
    fit_values = fitting_func(x_values, *popt)
    residuals = y_values - fit_values
    sigma_sum = np.sum(np.square(residuals))
    studentized_residuals = np.zeros(residuals.shape)
    for i, residual in enumerate(residuals):
        sigma_sum_sans_current = sigma_sum - np.square(residual)
        sigma_squared_sans_current = (1.0 / (num_samples - num_params - 1)) * sigma_sum_sans_current 
        sigma_sans_current = np.sqrt(sigma_squared_sans_current)
        studentized_residual = residual / sigma_sans_current 
        studentized_residuals[i] = studentized_residual
    is_inlier_array = _studentized_residual_test(studentized_residuals, num_samples - num_params - 1, alpha)
    inlier_indices = np.nonzero(is_inlier_array)[0]
    return inlier_indices

#Source for approach: https://en.wikipedia.org/wiki/Studentized_residual
def _studentized_residual_test(t, degrees_of_freedom, alpha):
    nu = degrees_of_freedom
    abs_t = np.abs(t)
    x = nu / (np.square(t) + nu)
    #Formula source: https://en.wikipedia.org/wiki/Student%27s_t-distribution
    #Scipy betainc is the _regularized_ incomplete beta function
    probability_of_occurrence = 0.5 * betainc(nu / 2, 0.5, x)
    return probability_of_occurrence > alpha

def _dynamic_np_slice(m, axis, start = None, stop = None):
    if start is None:
        start = 0 
    if stop is None:
        stop = m.shape[axis] 
    slc = [slice(None)] * len(m.shape)
    slc[axis] = slice(start, stop) 
    slc = tuple(slc) 
    return m[slc]

"""
Helper function for using a Monte Carlo analysis to get the covariance matrix of 
the best-fit parameters for a function fun to a dataset x_data, y_data."""
def _monte_carlo_covariance_helper(fun, x_data, y_data, y_errors, popt, num_samples = 100):
    if(y_errors is None):
        raise RuntimeError("Monte Carlo covariance analysis not supported for non-specified errors.")
    popt_list = []
    for i in range(num_samples):
        simulated_y_data = y_data + np.random.normal(loc = 0.0, scale = y_errors, size = y_errors.shape)
        results = curve_fit(fun, x_data, simulated_y_data, p0 = popt)
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
    DEFAULT_INDEPENDENT_VARNAMES = ['t', 'x', 'y', 'imaging_freq', 'x_values', 'rf_freqs']
    arg_names = [f for f in arg_names if (not f in DEFAULT_INDEPENDENT_VARNAMES)]
    return arg_names