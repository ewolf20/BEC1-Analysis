from collections import namedtuple
from dataclasses import make_dataclass 

import numpy as np 


#Redefining classes used in scipy.stat.bootstrap in case they move in future releases
ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])



#Use "standard_error" to match convention for scipy.stats.bootstrap, but what is returned 
#is the STANDARD DEVIATION
fields = ['confidence_interval', 'bootstrap_distribution', 'standard_error']
BootstrapResult = make_dataclass("BootstrapResult", fields)

"""A generalized version of bootstrapping suitable for statistics on samples whose datapoints are not scalar. 

Given a sequence data of datasets on which bootstrapping should be conducted, estimate 
the confidence interval on a statistic which takes as input len(data) distinct iterables data_i, plus optionally extra 
parameters specified in *args, and returns a scalar or ndarray value. 

Importantly, if vectorized is False, no assumption is made on the nature of the data points within a sample data_i; these can be 
scalars, vectors, or even general objects. It would be perfectly valid for data_i to take the form

[PlayingCard("AceOfSpades"), PlayingCard("EightOfSpades"), PlayingCard("AceOfClubs"), PlayingCard("EightOfClubs")]

for which form a possible resampling would be

[PlayingCard("AceOfSpades"), PlayingCard("AceOfClubs"), PlayingCard("AceOfClubs"), PlayingCard("EightOfSpades")]

Note, however, that if vectorized is False and the samples in data_i are mutable objects, care must be taken that statistic does not mutate them. Likewise, 
if generators are used for data_i, they must not be mutated by iteration. 

Parameters:

data: A sequence (data_1, data_2, ... data_n) of samples used to obtain a value of the statistic and on which bootstrapping should be performed. The samples
data_i may be any indexable iterable if vectorized is False; otherwise, data_i should be ndarrays or array-like.

statistic: A function with signature f(data_1, data_2, ... data_n, *args). If vectorized is True, the function should treat data_i as ndarrays 
and also accept kwargs resampling_axis and additional_axes, as specified below.

vectorized: An argument indicating whether the function statistic supports vectorized input. If False, it should be assumed that 
    only the assumptions given above for statistic hold. 

    If True, it is assumed that the function statistic can handle ndarrays data_i of arbitrary shape and supports two keyword arguments: resampling_axis 
    and additional_axes. Resampling axis is, as stipulated below, the axis of the ndarray data_i along which values should be resampled;
    a typical use case would be for statistic to average over this axis. Additional_axes 
    may be an int or a tuple of ints; statistic may use additional_axes in essentially any way. 
    
    It is further assumed that:
        a) The function statistic collapses resampling_axis and additional_axes (e.g. integrates, averages, etc over them), 
        b) If statistic is a vector-valued function of the sample distribution, the axes associated with this vector value appear first in any returned array, and 
        c) Except for any shift in axis index from a) or b), any axis of data_i not appearing in resampling_axis or additional_axes is preserved untouched. 

Resampling axis (int): The axis along which resampling is to be performed. Default is 0; only used when vectorized is True. Otherwise, resampling is assumed 
to be performed along the first (and possibly only) axis of data_i.

Additional_axes (tuple of ints): A tuple of additional axes passed to statistic for vectorized input.
                                Beyond those stipulated above, no assumptions are made on how statistic uses these additional axes. 

n_resamples: The number of bootstrap resamples to perform

confidence_level: The returned confidence interval will be returned under the assumption that a fraction confidence_level of the bootstrap statistic values 
lie within the confidence interval. 

batch_size: Only used if vectorized is True. If specified, the vectorized resampled data is passed to statistic in batches of batch_size resamplings, thereby 
allowing control over the space used in memory. If not specified and vectorized is True, the entire set of n_resamples resamplings is passed at once. 


Remark: This code takes heavy inspiration from scipy.stat.bootstrap, but is designed to offer more permissive bootstrapping (supporting e.g. samples whose
    "data points" are ndarrays, objects, etc.). Correspondingly, it should be expected to be slower in almost all cases, possibly significantly; where 
    this extra flexibility is not necessary, using scipy.stat.bootstrap should be preferred. 
"""
def generalized_bootstrap(data, statistic, *args, resampling_axis = 0, additional_axes = (),
                        method = "basic", n_resamples = 100, confidence_level = 0.95, vectorized = False, 
                        batch_size = None):
    if(not batch_size):
        batch_size = n_resamples
    rng = np.random.default_rng()
    if not vectorized:
        bootstrapped_statistic_values_list = []
        for i in range(n_resamples):
            new_input_list = []
            for data_sample in data:
                n_dat = len(data_sample) 
                new_sample_list = []
                random_indices = rng.integers(low = 0, high = n_dat, size = n_dat)
                for random_index in random_indices:
                    new_sample_list.append(data_sample[random_index])
                new_input_list.append(new_sample_list)
            statistic_value = statistic(*new_input_list, *args)
            bootstrapped_statistic_values_list.append(statistic_value)
        bootstrapped_statistic_values = np.stack(bootstrapped_statistic_values_list, axis = -1)
    else:
        vectorized_bootstrapped_statistic_values_list = [] 
        for k in range(0, n_resamples, batch_size):
            actual_batch = min(batch_size, n_resamples - k)
            resampled_data_sample_list = []
            for data_sample in data:
                #new_additional_axes is constant 
                vectorized_resampled_array, new_additional_axes = _reshape_data_array_for_vectorized_bootstrap(rng, 
                                                            data_sample, resampling_axis, additional_axes, actual_batch)
                resampled_data_sample_list.append(vectorized_resampled_array)
            new_resampling_axis = -1
            vectorized_bootstrapped_statistic_values = statistic(*resampled_data_sample_list, *args, resampling_axis = new_resampling_axis, 
                                                        additional_axes = new_additional_axes)
            vectorized_bootstrapped_statistic_values_list.append(vectorized_bootstrapped_statistic_values)
        bootstrapped_statistic_values = np.concatenate(vectorized_bootstrapped_statistic_values_list, axis = -1)
    alpha = (1 - confidence_level) / 2
    percentile_interval = (alpha, 1 - alpha) 
    c_lower = np.percentile(bootstrapped_statistic_values, percentile_interval[0] * 100, axis = -1) 
    c_upper = np.percentile(bootstrapped_statistic_values, percentile_interval[1] * 100, axis = -1)
    if method == "basic":
        if vectorized:
            initial_statistic_value = statistic(*data, *args, resampling_axis = resampling_axis, additional_axes = additional_axes)
        else:
            initial_statistic_value = statistic(*data, *args) 
        confidence_interval_lower = 2 * initial_statistic_value - c_upper 
        confidence_interval_upper = 2 * initial_statistic_value - c_lower
    elif method == "percentile":
        confidence_interval_lower = c_lower 
        confidence_interval_upper = c_upper
    standard_deviation = np.std(bootstrapped_statistic_values, axis = -1, ddof = 1)
    bootstrap_distribution = bootstrapped_statistic_values
    return BootstrapResult(confidence_interval = ConfidenceInterval(confidence_interval_lower, confidence_interval_upper), 
                            bootstrap_distribution = bootstrap_distribution, standard_error = standard_deviation)


def _reshape_data_array_for_vectorized_bootstrap(rng, data_array, resampling_axis, additional_axes, n_resamples):
    data_array_shape = data_array.shape
    number_of_axes = len(data_array_shape)
    length_along_resampling_axis = data_array_shape[resampling_axis]
    #Permute axis labels by putting the resampling axis to the end
    axis_labels = np.arange(number_of_axes)
    axis_labels[resampling_axis:-1] = axis_labels[resampling_axis+1:]
    axis_labels[-1] = resampling_axis
    #Translate the old additional axes to their new values in the new, permuted array
    new_additional_axes_list = [] 
    for additional_axis in additional_axes:
        new_additional_axis = np.nonzero(axis_labels == additional_axis)[0][0] 
        new_additional_axes_list.append(new_additional_axis) 
    new_additional_axes = tuple(new_additional_axes_list) 
    #Permute the array
    transposed_data_array = np.transpose(data_array, axes = axis_labels)
    resampling_indices = rng.integers(0, high = length_along_resampling_axis, size = (n_resamples, length_along_resampling_axis))
    #Create a vectorized resampled array; now the last axis indexes the elements of a particular resample of the data, 
    #and the penultimate axis indexes the different resamples used for bootstrapping.
    vectorized_resampled_array = transposed_data_array[..., resampling_indices] 
    return (vectorized_resampled_array, new_additional_axes)
