import numpy as np 
from lmfit import Model, Parameters



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
    fit_params = Parameters() 
    fit_params.add('amp', value = amp_guess, vary = True)
    if(linewidth):
        fit_params.add('gamma', value = linewidth, vary = False) 
    else:
        fit_params.add('gamma', value = gamma_guess, vary = True)
    if(center):
        fit_params.add('center', value = center, vary = False)
    else:
        fit_params.add('center', value = center_guess, vary = True)
    if(offset):
        fit_params.add('offset', value = offset, vary = False) 
    else:
        fit_params.add('offset', value = offset_guess, vary = True)
    if(errors):
        weights = 1.0 / errors 
    else:
        weights = None
    lorentzian_model = Model(lorentzian_function)
    lorentzian_fit = lorentzian_model.fit(counts, freq = frequencies, params = fit_params, weights = weights)
    return lorentzian_fit


def lorentzian_function(freq, amp, center, gamma, offset):
    return amp * 1.0 / (np.square(freq - center) + np.square(gamma) / 4) + offset
    