import numpy as np 
from scipy.optimize import curve_fit


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

def imaging_resonance_lorentzian(freq, amp, center, gamma, offset):
    return amp * 1.0 / (np.square(2.0 * (freq - center) / gamma) + 1) + offset



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
    DEFAULT_INDEPENDENT_VARNAMES = ['t', 'x', 'y', 'freq']
    arg_names = [f for f in arg_names if (not f in DEFAULT_INDEPENDENT_VARNAMES)]
    return arg_names