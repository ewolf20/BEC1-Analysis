import os
import shutil
import sys

import matplotlib.pyplot as plt
import mpmath 
import numpy as np 


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

RESOURCES_DIRECTORY_PATH = "./resources"

from BEC1_Analysis.code import numerical_functions, loading_functions


def test_polylog_taylor_series_coefficient_generator():
    result = numerical_functions.polylog_taylor_series_coefficient_generator(10)
    result_sum = np.sum(result)
    #Sum rule has every row after the first two sum to zero...
    assert np.isclose(result_sum, 2.0)


def test_polylog_specific_taylor_series_generator():
    EXPECTED_RESULT = np.array([-2.16270071-8.67361738e-19j, -1.67908973+8.67361738e-19j,
                                0.3040215 -4.87890978e-19j])
    result = numerical_functions.polylog_specific_taylor_series_generator(-3, 2, 5/2)
    assert np.all(np.isclose(result, EXPECTED_RESULT))


vectorized_mpmath_polylog = np.vectorize(mpmath.fp.polylog, otypes = [complex])

def test_polylog_taylor_series():
    SERIES_CENTER = -100
    SERIES_ORDER = 5
    SERIES_S = 5/2
    polylog_specific_taylor_series_coefficients = numerical_functions.polylog_specific_taylor_series_generator(SERIES_CENTER, SERIES_ORDER, SERIES_S)
    z_values = np.linspace(-110, -90, 10000) 
    taylor_series_polylog_values = numerical_functions.polylog_taylor_series(z_values, SERIES_CENTER, polylog_specific_taylor_series_coefficients)
    mpmath_polylog_values = vectorized_mpmath_polylog(5/2, z_values)
    assert np.all(np.isclose(taylor_series_polylog_values, mpmath_polylog_values, rtol = 1e-7, atol = 0.0))


def test_generate_and_save_taylor_series_coefficients():
    COEFFS_SAVE_FILENAME_3_2 = "Polylog_Taylor_Coefficients_3_2.npy" 
    COEFFS_SAVE_FILENAME_5_2 = "Polylog_Taylor_Coefficients_5_2.npy" 
    CENTERS_SAVE_FILENAME = "Polylog_Taylor_Centers.npy" 
    TEMP_SAVE_DIRECTORY = "polylog_temp"
    save_directory_pathname = os.path.join("resources", TEMP_SAVE_DIRECTORY)
    try:
        os.mkdir(save_directory_pathname)
        centers_save_pathname = os.path.join(save_directory_pathname, CENTERS_SAVE_FILENAME)
        coeffs_save_pathname_3_2 = os.path.join(save_directory_pathname, COEFFS_SAVE_FILENAME_3_2)
        coeffs_save_pathname_5_2 = os.path.join(save_directory_pathname, COEFFS_SAVE_FILENAME_5_2)
        numerical_functions.generate_and_save_taylor_series_coefficients(5/2, coeffs_save_pathname_5_2, centers_save_pathname)
        numerical_functions.generate_and_save_taylor_series_coefficients(3/2, coeffs_save_pathname_3_2, centers_save_pathname)
        just_generated_centers = np.load(centers_save_pathname)
        just_generated_coeffs_3_2 = np.load(coeffs_save_pathname_3_2)
        just_generated_coeffs_5_2 = np.load(coeffs_save_pathname_5_2)
        stored_centers, stored_coeffs_3_2, stored_coeffs_5_2 = loading_functions.load_polylog_analytic_continuation_parameters()
        assert np.all(np.isclose(just_generated_centers, stored_centers))
        assert np.all(np.isclose(just_generated_coeffs_3_2, stored_coeffs_3_2))
        assert np.all(np.isclose(just_generated_coeffs_5_2, stored_coeffs_5_2))
    finally:
        shutil.rmtree(save_directory_pathname)


def test_stored_coeffs_polylog_taylor_series():
    centers, coeffs_3_2, coeffs_5_2 = loading_functions.load_polylog_analytic_continuation_parameters()
    z_values = -np.logspace(-0.1, 10.0, num = 1000, base = np.e)
    mp_math_values_3_2 = vectorized_mpmath_polylog(3/2, z_values)
    mp_math_values_5_2 = vectorized_mpmath_polylog(5/2, z_values)
    homebrew_values_3_2 = numerical_functions.stored_coeffs_polylog_taylor_series(z_values, centers, coeffs_3_2)
    homebrew_values_5_2 = numerical_functions.stored_coeffs_polylog_taylor_series(z_values, centers, coeffs_5_2)
    assert np.all(np.isclose(homebrew_values_3_2, mp_math_values_3_2, rtol = 1e-10, atol = 0.0))
    assert np.all(np.isclose(homebrew_values_5_2, mp_math_values_5_2, rtol = 1e-10, atol = 0.0))
