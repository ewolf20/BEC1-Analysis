import numpy as np 
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from .science_functions import two_level_system_population_rabi


def fit_imaging_resonance_lorentzian(frequencies, counts, errors = None, linewidth = None, center = None, offset = None):
    #Rough magnitude expected for gamma in any imaging resonance lorentzian we would plot
    INITIAL_GAMMA_GUESS = 5.0
    data_average = sum(counts) / len(counts)
    frequency_average = sum(frequencies) / len(frequencies)
    frequency_range = max(frequencies) - min(frequencies) 
    center_guess = frequency_average
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
    sorted_x_values, sorted_data = _sort_xy_data(x_values, data)
    x_values_delta = sorted_x_values[1] - sorted_x_values[0]
    values_evenly_spaced = True
    for diff in np.diff(sorted_x_values):
        if(np.abs(diff - x_values_delta) / x_values_delta > SPACING_REL_TOLERANCE):
            values_evenly_spaced = False 
            break
    if(not values_evenly_spaced):
        minimum_spacing = min(sorted_x_values[1:] - sorted_x_values[:-1])
        x_width = sorted_x_values[-1] - sorted_x_values[0] 
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
    centered_data_fft = np.fft.fft(centered_data) 
    fft_real_frequencies = np.fft.fftfreq(data_length) * 1.0 / x_values_delta
    positive_fft_cutoff = int(np.floor(data_length / 2)) 
    positive_fft_values = centered_data_fft[1:positive_fft_cutoff]
    positive_fft_frequencies = fft_real_frequencies[1:positive_fft_cutoff] 
    fft_peak_position = np.argmax(np.abs(positive_fft_values)) 
    guessed_frequency = positive_fft_frequencies[fft_peak_position] 
    guessed_phase = np.angle(positive_fft_values[fft_peak_position])
    guessed_amp = np.abs(positive_fft_values[fft_peak_position]) * 2.0 / data_length
    return (guessed_frequency, guessed_amp, guessed_phase, guessed_offset)


def one_dimensional_cosine(x_values, freq, amp, phase, offset):
    return amp * np.cos(2 * np.pi * x_values * freq + phase) + offset


#By convention, frequencies are in kHz, and times are in ms. 
def fit_rf_spect_detuning_scan(rf_freqs, transfers, tau, center = None, rabi_freq = None, errors = None):
    if(center is None):
        center_guess = sum(rf_freqs) / len(rf_freqs) 
    else:
        center_guess = center 
    if(rabi_freq is None):
        rabi_freq_guess = 1.0 
    else:
        rabi_freq_guess = rabi_freq
    def wrapped_rf_spect_function_factory(tau):
        def rf_spect_function(rf_freqs, center, rabi_freq):
            return rf_spect_detuning_scan(rf_freqs, center, tau, rabi_freq)
        return rf_spect_function
    tau_wrapped_function = wrapped_rf_spect_function_factory(tau)
    params = np.array([center_guess, rabi_freq_guess])
    results = curve_fit(tau_wrapped_function, rf_freqs, transfers, p0 = params, sigma = errors)
    return results


def rf_spect_detuning_scan(rf_freqs, center, tau, rabi_freq):
    detunings = rf_freqs - center 
    populations_excited = two_level_system_population_rabi(tau, rabi_freq, detunings)[1]
    return populations_excited


"""Helper function for finding the center of data with an even symmetry point.

Given a dataset x, y, uses least-squares optimization to try to find the maximally evenly symmetric 
point in the data, i.e. the point for which |f(x_0 + a) - f(x_0 - a)| is minimized in a least squares sense 
for all the available data."""

def _find_center_helper(x_data, y_data, tau):
    INTERPOLATION_PRECISION = 1000
    transfer_interpolation = interp1d(x_data, y_data, kind = "cubic")
    minimum_x = min(x_data) 
    maximum_x = max(x_data)
    interpolation_step = (maximum_x - minimum_x) / INTERPOLATION_PRECISION
    def symmetry_test_function(center_guess):
        if(center_guess - minimum_x < maximum_x - center_guess):
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
        squared_difference = 0.0
        for i in (np.arange(test_window_half_width) + 1):
            left_point = test_window_center_index - i 
            right_point = test_window_center_index + i 
            left_value = transfer_interpolation(test_window_frequencies[left_point])
            right_value = transfer_interpolation(test_window_frequencies[right_point])
            squared_difference += np.square(left_value - right_value)
        mean_squared_difference = squared_difference / test_window_number_points
        #Ad hoc penalty for a center close to the edge
        data_mean_abs = sum(np.abs(y_data))
        return mean_squared_difference
    
    







def _sort_xy_data(x_values, y_values):
    sorted_x_values, sorted_y_values = zip(*sorted(zip(x_values, y_values), key = lambda f: f[0]))
    sorted_x_values = np.array(sorted_x_values) 
    sorted_y_values = np.array(sorted_y_values)
    return (sorted_x_values, sorted_y_values)
"""
Convenience function for getting a pretty_printable fit report from scipy.optimize.curve_fit"""
def fit_report(model_function, fit_results):
    popt, pcov = fit_results
    report_string = ''
    report_string = report_string + "Model function: " + model_function.__name__ + "\n \n"
    varnames_tuple = get_varnames_from_function(model_function) 
    my_sigmas = np.sqrt(np.diag(pcov))
    for varname, value, sigma in zip(varnames_tuple, popt, my_sigmas):
        report_string = report_string + "Parameter: {0}\tValue: {1:.3e} ± {2:.2e} \t({3:.2%}) \n".format(varname, value, sigma, np.abs(sigma / value))
    report_string = report_string + "\n"
    report_string = report_string + "Correlations (unreported are <0.1): \n" 
    for i in range(len(popt)):
        for j in range(i + 1, len(popt)):
            covariance = pcov[i][j] 
            correlation = covariance / (my_sigmas[i] * my_sigmas[j]) 
            if(np.abs(correlation) > 0.1):
                report_string = report_string + "{0} and {1}: \t {2:.2f}\n".format(varnames_tuple[i], varnames_tuple[j], correlation)
    return report_string

def get_varnames_from_function(my_func):
    arg_names = my_func.__code__.co_varnames[:my_func.__code__.co_argcount]
    DEFAULT_INDEPENDENT_VARNAMES = ['t', 'x', 'y', 'imaging_freq', 'x_values', 'rf_freqs']
    arg_names = [f for f in arg_names if (not f in DEFAULT_INDEPENDENT_VARNAMES)]
    return arg_names