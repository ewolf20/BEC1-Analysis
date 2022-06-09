import numpy as np 
from scipy.optimize import curve_fit


def fit_imaging_resonance_lorentzian(frequencies, counts, errors = None, linewidth = None, center = None, offset = None):
    data_average = sum(counts) / len(counts)
    frequency_average = sum(frequencies) / len(frequencies)
    frequency_range = max(frequencies) - min(frequencies) 
    center_guess = frequency_average
    gamma_guess = frequency_range
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
    results = curve_fit(imaging_lorentzian_function, frequencies, counts, p0 = params, sigma = errors)
    return results


def imaging_lorentzian_function(freq, amp, center, gamma, offset):
    return amp * 1.0 / (np.square(freq - center) + np.square(gamma) / 4) + offset
    

"""
Convenience function for getting a pretty_printable fit report from scipy.optimize.curve_fit"""
def fit_report(model_function, popt, pcov):
    report_string = ''
    report_string = report_string + "Model function: " + model_function.__name__ + "\n \n"
    varnames_tuple = get_varnames_from_function(model_function) 
    my_sigmas = np.sqrt(np.diag(pcov))
    for varname, value, sigma in zip(varnames_tuple, popt, my_sigmas):
        report_string = report_string + "Parameter: " + varname + "\tValue: " + str(value) + " ± " + str(sigma) + "\t(" + str(100 * sigma / value) + "%)" + "\n"
    report_string = report_string + "\n"
    report_string = report_string + "Correlations (unreported are <0.1): \n" 
    for i in range(len(popt)):
        for j in range(i + 1, len(popt)):
            covariance = pcov[i][j] 
            correlation = covariance / (my_sigmas[i] * my_sigmas[j]) 
            if(np.abs(correlation) > 0.1):
                report_string = report_string + varnames_tuple[i] + " and " + varnames_tuple[j] + " :\t" + str(correlation) + '\n' 
    return report_string

def get_varnames_from_function(my_func):
    arg_names = my_func.__code__.co_varnames[1:my_func.__code__.co_argcount]
    return arg_names