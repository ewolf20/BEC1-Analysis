import os
from sre_constants import RANGE
import sys 

import matplotlib.pyplot as plt
import numpy as np 

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

from BEC1_Analysis.code import statistics_functions

class NumberClass():
    def __init__(self, val):
        self.val = val
    
    def give_val(self):
        return self.val


def test_generalized_bootstrap():
    my_random_normals = np.load("resources/Sample_Normal_Randoms.npy")
    num_bootstraps = 10000
    EXPECTED_BOOTSTRAP_STANDARD_DEVIATION = 0.0876
    num_samples = len(my_random_normals)
    my_random_object_list = [] 
    for random_number in my_random_normals:
        my_random_object_list.append(NumberClass(random_number))
    def test_object_statistic(data, *args):
        mean_sum = 0.0
        for number_object in data:
            mean_sum += number_object.give_val()
        mean = mean_sum / len(data)
        return mean
    bootstrap_object_result = statistics_functions.generalized_bootstrap((my_random_object_list,), test_object_statistic, n_resamples = num_bootstraps) 
    object_standard_error = bootstrap_object_result.standard_error 
    assert(np.abs((object_standard_error - EXPECTED_BOOTSTRAP_STANDARD_DEVIATION) / EXPECTED_BOOTSTRAP_STANDARD_DEVIATION) < 2e-1)
    def test_array_statistic(data, *args, resampling_axis = 0, additional_axes = ()):
        overall_axes = [resampling_axis] 
        overall_axes.extend(additional_axes)
        return np.sum(data, axis = tuple(overall_axes)) / np.size(data, resampling_axis)
    ARRAY_EXTRA_AXIS_LENGTH = 10
    my_random_array = np.matmul(my_random_normals.reshape(len(my_random_normals), 1), np.ones(shape = (1, ARRAY_EXTRA_AXIS_LENGTH)))
    bootstrap_array_no_additional_result = statistics_functions.generalized_bootstrap((my_random_array,), test_array_statistic, vectorized = True, 
                                                            additional_axes = (), n_resamples = num_bootstraps, batch_size = 100)
    bootstrap_array_no_additional_standard_deviations = bootstrap_array_no_additional_result.standard_error
    assert len(bootstrap_array_no_additional_standard_deviations == ARRAY_EXTRA_AXIS_LENGTH)
    assert np.all(np.abs(bootstrap_array_no_additional_standard_deviations - EXPECTED_BOOTSTRAP_STANDARD_DEVIATION) < 2e-1)
    bootstrap_array_no_additional_covariance_matrix = bootstrap_array_no_additional_result.covariance_matrix 
    bootstrap_array_no_additional_correlation_matrix = np.matmul(np.diag(1.0 / bootstrap_array_no_additional_standard_deviations), np.matmul(
        bootstrap_array_no_additional_covariance_matrix, np.diag(1.0 / bootstrap_array_no_additional_standard_deviations)
    ))
    assert np.all(np.isclose(bootstrap_array_no_additional_correlation_matrix, np.ones(bootstrap_array_no_additional_correlation_matrix.shape)))
    bootstrap_array_result = statistics_functions.generalized_bootstrap((my_random_array,), test_array_statistic, vectorized = True, 
                                                            additional_axes = (1,), n_resamples = num_bootstraps, batch_size = 100)
    bootstrap_array_standard_deviation = bootstrap_array_result.standard_error
    expected_array_bootstrap_standard_deviation = ARRAY_EXTRA_AXIS_LENGTH * EXPECTED_BOOTSTRAP_STANDARD_DEVIATION
    assert (np.abs((bootstrap_array_standard_deviation - expected_array_bootstrap_standard_deviation) / expected_array_bootstrap_standard_deviation) < 2e-1)

def test_monte_carlo_error_propagation():
    a = 2.0 
    b = 3.0 
    a_error = 0.5 
    b_error = 0.2
    NUM_MONTE_CARLO_SAMPLES = 10000
    EXPECTED_VARIANCE = (np.square(a) + np.square(a_error)) * (np.square(b) + np.square(b_error)) - np.square(a) * np.square(b) 
    monte_carlo_variance = statistics_functions.monte_carlo_error_propagation(np.multiply, (a, b), (a_error, b_error), vectorized = True, 
                                                    monte_carlo_samples = NUM_MONTE_CARLO_SAMPLES)
    monte_carlo_unvectorized_variance = statistics_functions.monte_carlo_error_propagation(np.multiply, (a, b), (a_error, b_error), vectorized = False, 
                                                    monte_carlo_samples = NUM_MONTE_CARLO_SAMPLES)
    assert(np.isclose(monte_carlo_variance, EXPECTED_VARIANCE, rtol = 1e-1))
    assert(np.isclose(monte_carlo_variance, monte_carlo_unvectorized_variance, rtol = 1e-1))

def test_mean_location_test():
    random_normals = np.load("resources/Sample_Normal_Randoms.npy")
    random_normals_length = len(random_normals)
    assert not statistics_functions.mean_location_test(random_normals, 0)
    assert statistics_functions.mean_location_test(random_normals + 1.0, 0) 
    assert not statistics_functions.mean_location_test(random_normals + 0.3, 0.3) 
    assert statistics_functions.mean_location_test(random_normals + 0.6, 0.3) 
    assert statistics_functions.mean_location_test(random_normals, -0.3)
    #Test vectorization
    ARRAY_DIMENSION_LENGTH = 10
    random_normals_array = np.matmul(random_normals.reshape(random_normals_length, 1), np.linspace(0, 1, ARRAY_DIMENSION_LENGTH).reshape(1, ARRAY_DIMENSION_LENGTH))
    assert np.all(statistics_functions.mean_location_test(random_normals_array + 1.0, 0, axis = 0)) 



def test_linear_center_location_test():
    random_normals = np.load("resources/Sample_Normal_Randoms.npy")[:-1]
    random_normals_length = len(random_normals)
    assert random_normals_length % 2 == 1
    x_values = np.arange(random_normals_length) - (random_normals_length - 1) / 2
    ADDED_SLOPE = 5.00
    random_normals_plus_slope = random_normals + ADDED_SLOPE * x_values
    assert statistics_functions.linear_center_location_test(x_values, random_normals_plus_slope, -0.25, confidence_level = 0.975) 
    assert statistics_functions.linear_center_location_test(x_values, random_normals_plus_slope, -0.5, confidence_level = 0.999)
    assert not statistics_functions.linear_center_location_test(x_values, random_normals_plus_slope, 0.1, confidence_level = 0.95)

    ARRAY_DIMENSION_LENGTH = 10
    random_normals_array = np.expand_dims(random_normals_plus_slope, 0) * np.expand_dims(np.ones(ARRAY_DIMENSION_LENGTH), 1)
    broadcast_x_values = np.broadcast_to(x_values, random_normals_array.shape)
    assert np.all(statistics_functions.linear_center_location_test(broadcast_x_values, random_normals_array, -0.25, axis = -1)) 



def test_filter_1d_residuals():
    randoms = np.load(os.path.join("resources", "Sample_Normal_Randoms.npy"))
    OUTLIER_INDEX = 46
    OFFSET = 5
    DEGS_FREEDOM = 1
    randoms[OUTLIER_INDEX] = 5
    inlier_indices = statistics_functions.filter_1d_residuals(randoms, DEGS_FREEDOM)
    assert len(inlier_indices) == (len(randoms) - 1)
    assert not OUTLIER_INDEX in inlier_indices
    # Test iterative filtering 
    SUPER_OUTLIER_INDEX = 45 
    randoms[SUPER_OUTLIER_INDEX] = 100
    inlier_indices_noniterative = statistics_functions.filter_1d_residuals(randoms, DEGS_FREEDOM)
    assert len(inlier_indices_noniterative) == (len(randoms) - 1)
    assert not SUPER_OUTLIER_INDEX in inlier_indices_noniterative
    inlier_indices_iterative = statistics_functions.filter_1d_residuals(randoms, DEGS_FREEDOM, iterative = True)
    assert len(inlier_indices_iterative) == (len(randoms) - 2)
    assert not SUPER_OUTLIER_INDEX in inlier_indices_iterative
    assert not OUTLIER_INDEX in inlier_indices_iterative


def test_filter_mean_outliers():
    randoms = np.load(os.path.join("resources", "Sample_Normal_Randoms.npy"))
    OUTLIER_INDEX = 46
    OUTLIER = 5
    SUPER_OUTLIER_INDEX = 47
    SUPER_OUTLIER = 10000
    #super outlier is so big that the other outlier wouldn't be detected if super outlier is included in the mean
    randoms[OUTLIER_INDEX] = OUTLIER 
    randoms[SUPER_OUTLIER_INDEX] = SUPER_OUTLIER
    inlier_indices_noniterative = statistics_functions.filter_mean_outliers(randoms, iterative = False) 
    assert len(inlier_indices_noniterative) == len(randoms) - 1
    assert not SUPER_OUTLIER_INDEX in inlier_indices_noniterative
    inlier_indices_iterative = statistics_functions.filter_mean_outliers(randoms, iterative = True) 
    assert len(inlier_indices_iterative) == len(randoms) - 2 
    assert not SUPER_OUTLIER_INDEX in inlier_indices_iterative
    assert not OUTLIER_INDEX in inlier_indices_iterative



def test_average_over_like_x():
    randoms = _load_random_normals()
    RANGE_LENGTH = 10
    assert len(randoms) % RANGE_LENGTH == 0
    x_vals = np.repeat(np.arange(RANGE_LENGTH), len(randoms) // RANGE_LENGTH)
    offset_randoms = randoms + 3 * x_vals
    unique_x_vals, average_y_vals = statistics_functions.average_over_like_x(x_vals, offset_randoms, return_deviations = False, return_error_of_mean = False)
    assert np.array_equal(unique_x_vals, np.arange(RANGE_LENGTH))
    EXPECTED_Y_AVERAGE_VALS = np.array([-6.59214014e-03, 2.52218476e+00, 6.16827372e+00, 9.11775992e+00,
                                    1.20192817e+01,  1.50737530e+01,  1.78039561e+01,  2.09544976e+01,
                                    2.40639429e+01,  2.68503367e+01])
    assert np.all(np.isclose(average_y_vals, EXPECTED_Y_AVERAGE_VALS))
    *_, y_deviations = statistics_functions.average_over_like_x(x_vals, offset_randoms, return_deviations = True, return_error_of_mean = False)
    EXPECTED_Y_DEVIATION_VALS = np.array([1.17118523, 0.97547752, 0.92421303, 0.90790253, 0.68763213, 1.0387802,
                                            0.60376971, 1.08057771, 0.81931116, 0.76054765])
    assert np.all(np.isclose(y_deviations, EXPECTED_Y_DEVIATION_VALS))
    *_, y_errors_of_mean = statistics_functions.average_over_like_x(x_vals, offset_randoms, return_deviations = False, return_error_of_mean = True)
    EXPECTED_Y_ERROR_OF_MEAN_VALS = EXPECTED_Y_DEVIATION_VALS / np.sqrt(len(randoms) // RANGE_LENGTH)
    assert np.all(np.isclose(y_errors_of_mean, EXPECTED_Y_ERROR_OF_MEAN_VALS))
    X_REPEATS_2D = 5
    x_vals_2d = np.repeat(np.arange(RANGE_LENGTH), X_REPEATS_2D)
    randoms_reshaped = randoms.reshape(X_REPEATS_2D * RANGE_LENGTH, len(randoms) // (X_REPEATS_2D * RANGE_LENGTH))
    unique_x_vals_2d, average_y_vals_2d, deviations_2d, errors_of_mean_2d = statistics_functions.average_over_like_x(
                                                                            x_vals_2d, randoms_reshaped, return_deviations = True,
                                                                              return_error_of_mean = True)
    assert np.array_equal(unique_x_vals_2d, np.arange(RANGE_LENGTH))
    EXPECTED_2D_Y_AVERAGES = np.array([
        [ 0.65580764, -0.66899192],
        [-0.71966297, -0.23596751],
        [ 0.2738427,   0.06270474],
        [-0.25239047,  0.4879103 ],
        [-0.30428683,  0.34285026],
        [ 0.04340743,  0.10409856],
        [-0.20038237, -0.1917054 ],
        [ 0.54197035, -0.63297514],
        [-0.38303987,  0.51092559],
        [-0.34759999,  0.04827336]])
    assert np.all(np.isclose(average_y_vals_2d, EXPECTED_2D_Y_AVERAGES))
    EXPECTED_2D_DEVIATIONS = np.array([
        [0.94532353, 1.04675603],
        [1.25175811, 0.6541236 ],
        [1.01851094, 0.92555676],
        [0.54399093, 1.10281067],
        [0.74028922, 0.50409893],
        [0.45206284, 1.49037973],
        [0.74483831, 0.51514953],
        [0.94499783, 0.93347575],
        [0.73847547, 0.6822971 ],
        [0.72190149, 0.82606571]
    ])
    assert np.all(np.isclose(deviations_2d, EXPECTED_2D_DEVIATIONS))
    EXPECTED_2D_ERRORS_OF_MEAN = EXPECTED_2D_DEVIATIONS / np.sqrt(X_REPEATS_2D)
    assert np.all(np.isclose(errors_of_mean_2d, EXPECTED_2D_ERRORS_OF_MEAN))

    
    


#Random normals contains 100 normal deviates of standard deviation 1
def _load_random_normals():
    random_normal_path = os.path.join("resources", "Sample_Normal_Randoms.npy") 
    randoms = np.load(random_normal_path)
    assert len(randoms) == 100
    return np.load(random_normal_path)
    
