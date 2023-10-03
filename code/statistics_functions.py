from collections import namedtuple
from dataclasses import make_dataclass 

import warnings

import numpy as np 
from scipy.special import betainc


#Redefining classes used in scipy.stat.bootstrap in case they move in future releases
ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])



#Use "standard_error" to match convention for scipy.stats.bootstrap, but what is returned 
#is the STANDARD DEVIATION
fields = ['confidence_interval', 'bootstrap_distribution', 'standard_error', 'covariance_matrix']
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
    a typical use case would be for statistic to average over this axis. Additional_axes is a (potentially empty) tuple of ints; 
    statistic may use additional_axes in essentially any way. 
    
    It is further assumed that:
        a) The function statistic collapses resampling_axis and additional_axes (i.e. these axes do not appear in the return value), 
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

ignore_errors: Only used if vectorized is False. If True, errors thrown while a statistic is being evaluated on a dataset are handled and raised as warnings, 
with the corresponding statistic value omitted from the bootstrapped distribution. Useful for statistics given by e.g. fit algorithms which are likely, 
but not guaranteed to converge. 


Remark: This code takes heavy inspiration from scipy.stat.bootstrap, but is designed to offer more permissive bootstrapping (supporting e.g. samples whose
    "data points" are ndarrays, objects, etc.). Correspondingly, it should be expected to be slower in almost all cases, possibly significantly; where 
    this extra flexibility is not necessary, using scipy.stat.bootstrap should be preferred. 
"""
def generalized_bootstrap(data, statistic, *args, resampling_axis = 0, additional_axes = (),
                        method = "basic", n_resamples = 100, confidence_level = 0.95, vectorized = False, 
                        batch_size = None, ignore_errors = False, rng_seed = None):
    if(not batch_size):
        batch_size = n_resamples
    rng = np.random.default_rng(seed = rng_seed)
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
            try:
                statistic_value = statistic(*new_input_list, *args)
            except Exception as e:
                if ignore_errors:
                    warnings.warn(str(e))
                else:
                    raise e
            else:
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
    bootstrap_statistic_averages = np.average(bootstrapped_statistic_values, axis = -1) 
    bootstrap_statistic_averages_reshaped = np.expand_dims(bootstrap_statistic_averages, axis = -1)
    bootstrap_distribution_length = bootstrapped_statistic_values.shape[-1]
    bootstrap_distribution_deviations = bootstrapped_statistic_values - bootstrap_statistic_averages_reshaped
    covariance_matrix = np.matmul(bootstrap_distribution_deviations, np.transpose(bootstrap_distribution_deviations)) / (bootstrap_distribution_length - 1)
    bootstrap_distribution = bootstrapped_statistic_values
    return BootstrapResult(confidence_interval = ConfidenceInterval(confidence_interval_lower, confidence_interval_upper), 
                            bootstrap_distribution = bootstrap_distribution, standard_error = standard_deviation, 
                            covariance_matrix = covariance_matrix)

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


"""
Convenience function for normally distributed error propagation through arbitrary functions via Monte Carlo.

Given a function fun of scalar arguments params with independent, normally distributed errors of standard deviation 
param_sigmas, calculate the variance of the function value.

Parameters:
fun: The function whose output is to be evaluated. Must have signature f(*params) = a. Whether a is scalar or a 1D 
for scalar values of the parameters will influence the return type.

params: The center values of the parameters being used to evaluate the function. 

param_sigmas: The sigma values for the (assumed normally distributed) errors on the parameters

vectorized: Whether the function supports vector input; if so, calculations will be done in vectorized fashion for greater speed. 
    Note: If vectorized is true and a is a 1D vector for scalar parameter values, then it is assumed that the index associated with 
    a comes first in the array returned by the function.  

"""
def monte_carlo_error_propagation(fun, params, param_sigmas, vectorized = False, monte_carlo_samples = 1000):
    randomized_params_array = np.zeros((len(params), monte_carlo_samples))
    for i, param, param_sigma in zip(range(len(params)), params, param_sigmas):
        monte_carlo_param_values = np.random.normal(loc = param, scale = param_sigma, size = monte_carlo_samples)
        randomized_params_array[i] = monte_carlo_param_values
    if(not vectorized):
        function_values = []
        for j in range(monte_carlo_samples):
            function_values.append(fun(*randomized_params_array[:, j]))
        #Now, if present, the index of a comes first, followed by the monte carlo index
        function_values = np.transpose(np.array(function_values))
    else:
        function_values = fun(*randomized_params_array)
    function_value_averages = np.average(function_values, axis = -1, keepdims = True) 
    function_value_deviations = function_values - function_value_averages 
    function_covariance_matrix = np.matmul(function_value_deviations, np.transpose(function_value_deviations)) / np.size(function_values, axis = -1)
    return np.squeeze(function_covariance_matrix)



"""
Given a data sample data, returns a boolean representing whether the mean of the data is greater than mean_value with
confidence specified by confidence_level

Axis is provided to support vectorized input."""
def mean_location_test(data, mean_test_value, confidence_level = 0.95, axis = -1):
    number_samples = np.size(data, axis = axis)
    sample_mean = np.average(data, axis = axis, keepdims = True) 
    deviations = data - sample_mean 
    sample_mean = np.squeeze(sample_mean)
    student_sigma = np.sqrt(np.sum(np.square(deviations), axis = axis) / (number_samples - 1))
    studentized_mean_difference = (sample_mean - mean_test_value) / (student_sigma / np.sqrt(number_samples))
    t = studentized_mean_difference
    #Degrees of freedom
    nu = number_samples - 1
    x = nu / (np.square(t) + nu)
    alpha = 1.0 - confidence_level
    #Fraction of the t distribution lying at above the studentized mean difference
    probability_of_t_occurrence = 0.5 * betainc(nu / 2, 0.5, x)
    return np.where(sample_mean < mean_test_value, False, probability_of_t_occurrence < alpha)



"""
Given a fitting function & parameter values and a set of 1D x-y data (as np arrays)
they purport to fit, filter outliers using Student's t-test at the specified confidence level.

Returns the indices of the x-y data which are _INLIERS_, i.e. the complement of outliers,
points that can be identified as having a chance of less than alpha to occur.

If iterative is false, runs through the data only once to check for outliers. If true, 
iteratively prunes out detected outliers and returns to the data, implementing 
Grubbs' test"""

def filter_1d_residuals(residuals, degs_of_freedom, alpha = 1e-4, iterative = False):
    num_samples = len(residuals)
    current_mask = np.ma.nomask
    masked_residuals = np.ma.array(residuals)
    while True:
        masked_residuals.mask = current_mask
        sigma_sum = np.sum(np.square(masked_residuals))
        sigma_sum_sans_one_array = sigma_sum - np.square(masked_residuals)
        sigma_squared_sans_one_array = sigma_sum_sans_one_array * (1.0 / (num_samples - degs_of_freedom - 1))
        sigma_sans_one_array = np.sqrt(sigma_squared_sans_one_array)
        studentized_residuals = masked_residuals / sigma_sans_one_array
        is_outlier_array = np.logical_not(_studentized_residual_test(studentized_residuals, num_samples - degs_of_freedom - 1, alpha))
        current_mask = np.logical_or(current_mask, np.ma.filled(is_outlier_array, fill_value = True))
        if not iterative or not np.any(is_outlier_array):
            break
    inlier_indices = np.nonzero(np.logical_not(current_mask))[0]
    return inlier_indices


"""
Given a set of purportedly normally distributed values with unknown mean and standard deviation, 
apply a Student t-test to prune out outlier points, i.e. those with probability less than alpha of 
occurring.

If iterative is True, repeatedly iterate through the data after discovering outliers, implementing 
Grubbs' test.

The difference between this and filter_residuals, above, is that it also strips outliers from the _mean_ 
of the data, recalculating this every time an outlier is removed. """
def filter_mean_outliers(values, alpha = 1e-4, iterative = False):
    DEGREES_OF_FREEDOM = 1
    num_samples = len(values) 
    current_mask = np.ma.nomask 
    masked_values = np.ma.array(values)
    while True:
        masked_values.mask = current_mask 
        masked_mean = np.average(masked_values) 
        masked_deviations = masked_values - masked_mean 
        sigma_sum = np.sum(np.square(masked_deviations)) 
        sigma_sum_sans_one_array = sigma_sum - np.square(masked_deviations)
        sigma_squared_sans_one_array = sigma_sum_sans_one_array * (1.0 / (num_samples - DEGREES_OF_FREEDOM - 1))
        sigma_sans_one_array = np.sqrt(sigma_squared_sans_one_array) 
        studentized_deviations = masked_deviations / sigma_sans_one_array 
        is_outlier_array = np.logical_not(_studentized_residual_test(studentized_deviations, num_samples - DEGREES_OF_FREEDOM - 1, alpha))
        current_mask = np.logical_or(current_mask, np.ma.filled(is_outlier_array, fill_value = True))
        if not iterative or not np.any(is_outlier_array):
            break
    inlier_indices = np.nonzero(np.logical_not(current_mask))[0] 
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


"""
Given corresponding sets of one-dimensional x_data and y_data, average y data for points with identical x data.

If return_deviations is True, returns a set of deviations associated with each sample at a given x value. 
If return_error_of_mean is True, returns the errors of the mean associated with each sample, assuming normally-distributed data. 

Returns: A tuple (unique_x_data, y_averages, [y_deviations], [y_errors_of_mean]), with bracketed arguments optional depending on kwargs"""
def average_over_like_x(x_data, y_data, return_deviations = True, return_error_of_mean = False):
    unique_x_data = np.unique(x_data) 
    unique_y_data = np.array([np.average(y_data[x_data == x_val]) for x_val in unique_x_data])
    return_list = [unique_x_data, unique_y_data]
    if return_deviations:
        deviations = np.array([np.std(y_data[x_data == x_val], ddof = 1) for x_val in unique_x_data]) 
        return_list.append(deviations) 
    if return_error_of_mean:
        errors_of_mean = np.array([np.std(y_data[x_data == x_val]) / np.sqrt(np.count_nonzero(x_data == x_val)) for x_val in unique_x_data])
        return_list.append(errors_of_mean) 
    return tuple(return_list)