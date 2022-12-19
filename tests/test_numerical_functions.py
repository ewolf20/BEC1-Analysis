import os
import sys
import time

import matplotlib.pyplot as plt
import mpmath 
import numpy as np 


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

RESOURCES_DIRECTORY_PATH = "./resources"

from BEC1_Analysis.code import numerical_functions


def test_polylog_taylor_series_coefficient_generator():
    result = numerical_functions.polylog_taylor_series_coefficient_generator(10)
    result_sum = np.sum(result)
    assert np.isclose(result_sum, 2.0)


def test_polylog_specific_taylor_series_generator():
    result = numerical_functions.polylog_specific_taylor_series_generator(-3, 5, 5/2)
    print(result)


vectorized_mpmath_polylog = np.vectorize(mpmath.fp.polylog, otypes = [complex])

def test_polylog_taylor_series():
    SERIES_CENTER = -10000
    SERIES_ORDER = 5
    SERIES_S = 5/2
    polylog_specific_taylor_series_coefficients = numerical_functions.polylog_specific_taylor_series_generator(SERIES_CENTER, SERIES_ORDER, SERIES_S)
    z_values = np.linspace(-1000, -15000, 10000) 
    taylor_series_polylog_values = numerical_functions.polylog_taylor_series(z_values, SERIES_CENTER, polylog_specific_taylor_series_coefficients)
    mpmath_polylog_values = vectorized_mpmath_polylog(5/2, z_values)
    plt.plot(z_values, (taylor_series_polylog_values - mpmath_polylog_values) / mpmath_polylog_values)
    plt.show()


def test_generate_and_save_taylor_series_coefficients():
    coeffs_save_path_3_2 = "../resources/Polylog_Taylor_Coefficients_3_2.npy" 
    coeffs_save_path_5_2 = "../resources/Polylog_Taylor_Coefficients_5_2.npy" 
    centers_save_path = "../resources/Polylog_Taylor_Centers.npy" 
    numerical_functions.generate_and_save_taylor_series_coefficients(5/2, coeffs_save_path_5_2, centers_save_path)


def test_stored_coeffs_polylog_taylor_series():
    centers_path = "../resources/Polylog_Taylor_Centers.npy" 
    coeffs_3_2_path = "../resources/Polylog_Taylor_Coefficients_3_2.npy"
    coeffs_5_2_path = "../resources/Polylog_Taylor_Coefficients_5_2.npy"
    centers = np.load(centers_path) 
    coeffs_3_2 = np.load(coeffs_3_2_path) 
    coeffs_5_2 = np.load(coeffs_5_2_path)
    z_values = -np.logspace(-0.1, 10.0, num = 1000, base = np.e)
    mp_math_values_3_2 = vectorized_mpmath_polylog(3/2, z_values)
    mp_math_values_5_2 = vectorized_mpmath_polylog(5/2, z_values)
    homebrew_values_3_2 = numerical_functions.stored_coeffs_polylog_taylor_series(z_values, centers, coeffs_3_2)
    homebrew_values_5_2 = numerical_functions.stored_coeffs_polylog_taylor_series(z_values, centers, coeffs_5_2)
    plt.plot(z_values, (homebrew_values_3_2 - mp_math_values_3_2) / mp_math_values_3_2)
    plt.plot(z_values, (homebrew_values_5_2 - mp_math_values_5_2) / mp_math_values_5_2)
    plt.show()
