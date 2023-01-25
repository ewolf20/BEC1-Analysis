import os
from random import sample 
import sys 

import matplotlib.pyplot as plt
import numpy as np 

from scipy.optimize import curve_fit

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

TEST_DATA_DIRECTORY_PATH = "./resources/test_data"

from BEC1_Analysis.code import data_fitting_functions

#TODO: Make this test work properly
def test_fit_imaging_resonance_lorentzian():
    EXPECTED_AMP = 30.20
    EXPECTED_CENTER = 313.99
    EXPECTED_GAMMA = 4.92
    EXPECTED_OFFSET = 10.05
    LORENTZIAN_TEST_DATA_PATH = os.path.join(TEST_DATA_DIRECTORY_PATH, 'Imaging_Lorentzian_Test_Data.npy') 
    test_frequencies, test_values = np.load(LORENTZIAN_TEST_DATA_PATH)
    lorentzian_results = data_fitting_functions.fit_imaging_resonance_lorentzian(test_frequencies, test_values)
    fit_report = data_fitting_functions.fit_report(data_fitting_functions.imaging_resonance_lorentzian, lorentzian_results)
    popt, pcov = lorentzian_results
    amp, center, gamma, offset = popt 
    assert np.abs(amp - EXPECTED_AMP) < 0.01
    assert np.abs(center - EXPECTED_CENTER) < 0.01
    assert np.abs(gamma - EXPECTED_GAMMA) < 0.01 
    assert np.abs(offset - EXPECTED_OFFSET) < 0.01
    OUTLIER_EXPECTED_AMP = 30.12
    OUTLIER_EXPECTED_CENTER = 314.0
    OUTLIER_EXPECTED_GAMMA = 4.937
    OUTLIER_EXPECTED_OFFSET = 10.05
    OUTLIER_PLACING_POSITION = 29
    test_values_with_outlier = np.copy(test_values)
    test_values_with_outlier[OUTLIER_PLACING_POSITION] = 0.0
    outlier_results, inlier_indices = data_fitting_functions.fit_imaging_resonance_lorentzian(test_frequencies, test_values_with_outlier, filter_outliers = True, 
                                                                                            report_inliers = True)
    outlier_popt, outlier_pcov = outlier_results 
    outlier_stripped_test_values = test_values_with_outlier[inlier_indices]
    outlier_stripped_frequencies = test_frequencies[inlier_indices] 
    outlier_fit_report = data_fitting_functions.fit_report(data_fitting_functions.imaging_resonance_lorentzian, outlier_results)
    outlier_amp, outlier_center, outlier_gamma, outlier_offset = outlier_popt 
    assert np.abs(outlier_amp - OUTLIER_EXPECTED_AMP) < 1e-2 
    assert np.abs(outlier_center - OUTLIER_EXPECTED_CENTER) < 1e-1
    assert np.abs(outlier_gamma - OUTLIER_EXPECTED_GAMMA) < 1e-3
    assert not np.any(np.isin(OUTLIER_PLACING_POSITION, inlier_indices))
    assert len(inlier_indices) == len(test_frequencies) - 1



GAUSSIAN_X_PIXEL_NUM = 490 
GAUSSIAN_Y_PIXEL_NUM = 500     
GAUSSIAN_SIMULATED_X_CENTER = 130 
GAUSSIAN_SIMULATED_Y_CENTER = 420
GAUSSIAN_SIMULATED_X_WIDTH = 45
GAUSSIAN_SIMULATED_Y_WIDTH = 30
GAUSSIAN_SIMULATED_AMP = 3.14
GAUSSIAN_SIMULATED_OFFSET = 10

def test_fit_two_dimensional_gaussian():
    noisy_image = simulate_2D_gaussian_image()
    fit_results = data_fitting_functions.fit_two_dimensional_gaussian(noisy_image)
    fit_report = data_fitting_functions.fit_report(data_fitting_functions.two_dimensional_gaussian, fit_results)
    popt, pcov = fit_results 
    amp, x_center, y_center, x_width, y_width, offset = popt 
    assert np.abs(amp - GAUSSIAN_SIMULATED_AMP < 0.1) 
    assert np.abs(x_center - GAUSSIAN_SIMULATED_X_CENTER < 1)
    assert np.abs(y_center - GAUSSIAN_SIMULATED_Y_CENTER < 1)
    assert np.abs(x_width - GAUSSIAN_SIMULATED_X_WIDTH < 1) 
    assert np.abs(y_width - GAUSSIAN_SIMULATED_Y_WIDTH < 1)
    assert np.abs(offset - GAUSSIAN_SIMULATED_OFFSET < 0.1)


def simulate_2D_gaussian_image():
    y_coordinates = np.arange(GAUSSIAN_Y_PIXEL_NUM)
    x_coordinates = np.arange(GAUSSIAN_X_PIXEL_NUM)
    y_grid, x_grid = np.meshgrid(y_coordinates, x_coordinates)
    simulated_noiseless_image = data_fitting_functions.two_dimensional_gaussian(x_grid, y_grid, GAUSSIAN_SIMULATED_AMP, GAUSSIAN_SIMULATED_X_CENTER, 
                                                        GAUSSIAN_SIMULATED_Y_CENTER, GAUSSIAN_SIMULATED_X_WIDTH, GAUSSIAN_SIMULATED_Y_WIDTH, 
                                                        GAUSSIAN_SIMULATED_OFFSET)
    NOISE_MAGNITUDE = 0.5
    noisy_image = simulated_noiseless_image + np.random.normal(loc = 0.0, scale = NOISE_MAGNITUDE, size = simulated_noiseless_image.shape)
    return noisy_image


def test_fit_one_dimensional_cosine():
    SAMPLE_FREQ = 1.3
    SAMPLE_AMP = 1.0 
    SAMPLE_PHASE = 2.1 
    SAMPLE_OFFSET = 12.4
    X_ENDPOINT = 10 
    NUM_SAMPS = 100
    NOISE_AMP = 0.1
    sequential_x = np.linspace(0, X_ENDPOINT, NUM_SAMPS) 
    noiseless_sequential_y = data_fitting_functions.one_dimensional_cosine(sequential_x, SAMPLE_FREQ, SAMPLE_AMP, SAMPLE_PHASE, SAMPLE_OFFSET)
    noisy_sequential_y = np.load(os.path.join(TEST_DATA_DIRECTORY_PATH, "Sample_Cosine_Data.npy"))
    fit_results_sequential = data_fitting_functions.fit_one_dimensional_cosine(sequential_x, noisy_sequential_y)
    popt_s, pcov_s = fit_results_sequential 
    freq_s, amp_s, phase_s, offset_s = popt_s
    assert((freq_s - SAMPLE_FREQ) / (SAMPLE_FREQ) < 5e-2)
    assert((amp_s - SAMPLE_AMP) / (SAMPLE_AMP) < 5e-2)
    assert((phase_s - SAMPLE_PHASE) / (SAMPLE_PHASE) < 5e-2)
    assert((offset_s - SAMPLE_OFFSET) / (SAMPLE_OFFSET) < 5e-2)
    POLLUTION_AMP = 0.2
    POLLUTION_FREQUENCY = 2.4
    polluted_sequential_y = noisy_sequential_y + data_fitting_functions.one_dimensional_cosine(sequential_x, POLLUTION_FREQUENCY, POLLUTION_AMP, 0, 0)
    fit_results_polluted = data_fitting_functions.fit_one_dimensional_cosine(sequential_x, polluted_sequential_y) 
    popt_p, pcov_p = fit_results_polluted
    freq_p, amp_p, phase_p, offset_p = popt_p
    assert((freq_p - SAMPLE_FREQ) / (SAMPLE_FREQ) < 5e-2)
    assert((amp_p - SAMPLE_AMP) / (SAMPLE_AMP) < 5e-2)
    assert((phase_p - SAMPLE_PHASE) / (SAMPLE_PHASE) < 5e-2)
    assert((offset_p - SAMPLE_OFFSET) / (SAMPLE_OFFSET) < 5e-2)
    NON_SEQUENTIAL_INDICES = [71, 37, 15, 46, 28, 95, 60, 39, 53, 17, 96, 87, 75, 52, 24, 97, 76,
     1, 31, 42, 14, 61, 89, 58, 41, 74, 64, 27, 40, 84, 43, 98, 20, 22, 66,
      6, 30, 57, 8, 91, 78, 38, 10, 90, 82, 63, 94, 35, 4, 2]
    non_sequential_x = sequential_x[NON_SEQUENTIAL_INDICES] 
    non_sequential_noisy_y = noisy_sequential_y[NON_SEQUENTIAL_INDICES] 
    fit_results_non_sequential = data_fitting_functions.fit_one_dimensional_cosine(non_sequential_x, non_sequential_noisy_y) 
    popt_n, pcov_n = fit_results_non_sequential
    freq_n, amp_n, phase_n, offset_n = popt_n 
    assert((freq_n - SAMPLE_FREQ) / (SAMPLE_FREQ) < 5e-2)
    assert((amp_n - SAMPLE_AMP) / (SAMPLE_AMP) < 5e-2)
    assert((phase_n - SAMPLE_PHASE) / (SAMPLE_PHASE) < 5e-2)
    assert((offset_n - SAMPLE_OFFSET) / (SAMPLE_OFFSET) < 5e-2)


def test_get_fft_peak():
    NUM_ANGLE_POINTS = 100
    X_DELTA = 1.0 / NUM_ANGLE_POINTS
    COSINE_FREQUENCY = 16
    COSINE_PHASE = 1.0
    COSINE_AMPLITUDE = 2.3
    angles = np.linspace(0, 1, NUM_ANGLE_POINTS, endpoint = False)
    cosine_with_phase = COSINE_AMPLITUDE * np.cos(2 * np.pi * COSINE_FREQUENCY * angles + COSINE_PHASE)
    peak_freq, peak_amp, peak_phase = data_fitting_functions.get_fft_peak(X_DELTA, cosine_with_phase)
    assert np.isclose(COSINE_FREQUENCY, peak_freq)
    assert np.isclose(COSINE_AMPLITUDE, peak_amp)
    assert np.isclose(COSINE_PHASE, peak_phase)
    order_freq, order_amp, order_phase = data_fitting_functions.get_fft_peak(X_DELTA, cosine_with_phase, order = 16)
    assert np.isclose(COSINE_FREQUENCY, order_freq)
    assert np.isclose(COSINE_AMPLITUDE, order_amp)
    assert np.isclose(COSINE_PHASE, order_phase)
    NUM_ONES_INDICES = 10
    cosine_with_phase_array = np.matmul(cosine_with_phase.reshape((NUM_ANGLE_POINTS, 1)), np.ones((1, NUM_ONES_INDICES)))
    cosine_with_phase_array_transposed = np.transpose(cosine_with_phase_array) 
    peak_array_freq, peak_array_amp, peak_array_phase = data_fitting_functions.get_fft_peak(X_DELTA, cosine_with_phase_array, axis = 0)
    assert len(peak_array_freq) == NUM_ONES_INDICES
    assert len(peak_array_amp) == NUM_ONES_INDICES
    assert len(peak_array_phase) == NUM_ONES_INDICES
    assert np.allclose(peak_array_freq, COSINE_FREQUENCY)
    assert np.allclose(peak_array_amp, COSINE_AMPLITUDE)
    assert np.allclose(peak_array_phase, COSINE_PHASE)
    peak_array_transpose_freq, peak_array_transpose_amp, peak_array_transpose_phase = data_fitting_functions.get_fft_peak(X_DELTA,
                                                                                     cosine_with_phase_array_transposed, axis = 1)
    assert len(peak_array_transpose_freq) == NUM_ONES_INDICES
    assert len(peak_array_transpose_amp) == NUM_ONES_INDICES
    assert len(peak_array_transpose_phase) == NUM_ONES_INDICES
    assert np.allclose(peak_array_transpose_freq, COSINE_FREQUENCY)
    assert np.allclose(peak_array_transpose_amp, COSINE_AMPLITUDE)
    assert np.allclose(peak_array_transpose_phase, COSINE_PHASE)
    order_array_freq, order_array_amp, order_array_phase = data_fitting_functions.get_fft_peak(X_DELTA, cosine_with_phase_array, axis = 0, order = 16)
    assert len(order_array_freq) == NUM_ONES_INDICES
    assert len(order_array_amp) == NUM_ONES_INDICES
    assert len(order_array_phase) == NUM_ONES_INDICES
    assert np.allclose(order_array_freq, COSINE_FREQUENCY)
    assert np.allclose(order_array_amp, COSINE_AMPLITUDE)
    assert np.allclose(order_array_phase, COSINE_PHASE)



def test_sort_and_deduplicate_xy_data():
    TARGET_X_ARRAY = np.array([0, 1, 2, 3, 4]) 
    TARGET_Y_ARRAY = np.array([0, 2, 3, 6, 8]) 
    initial_x_array = np.array([4, 2, 1, 2, 3, 0]) 
    initial_y_array = np.array([8, 2, 2, 4, 6, 0]) 
    final_x_array, final_y_array = data_fitting_functions._sort_and_deduplicate_xy_data(initial_x_array, initial_y_array)
    assert (np.all(np.abs(final_x_array - TARGET_X_ARRAY) < 1e-5))
    assert (np.all(np.abs(final_y_array - TARGET_Y_ARRAY) < 1e-5))


def test_fit_rf_spect_detuning_scan():
    SAMPLE_CENTER = 22
    SAMPLE_RABI = 1.47
    SAMPLE_TAU = 0.2
    sample_frequencies = np.linspace(0, 50, 100)
    sample_transfers = data_fitting_functions.rf_spect_detuning_scan(sample_frequencies, SAMPLE_TAU, SAMPLE_CENTER, SAMPLE_RABI)
    sample_noisy_transfers = np.load(os.path.join(TEST_DATA_DIRECTORY_PATH, "Sample_RF_Transfers.npy"))
    fit_results = data_fitting_functions.fit_rf_spect_detuning_scan(sample_frequencies, sample_noisy_transfers, SAMPLE_TAU)
    popt, pcov = fit_results 
    center, rabi_freq = popt
    assert (np.abs((center - SAMPLE_CENTER) / SAMPLE_CENTER) < 3e-2) 
    assert (np.abs((rabi_freq - SAMPLE_RABI) / SAMPLE_RABI) < 3e-2)
    OUTLIER_FREQUENCY = 12
    OUTLIER_VALUE = 0.7
    sample_frequencies_with_outlier = np.append(sample_frequencies, OUTLIER_FREQUENCY)
    sample_noisy_transfers_with_outlier = np.append(sample_noisy_transfers, OUTLIER_VALUE)
    outlier_fit_results, inlier_indices = data_fitting_functions.fit_rf_spect_detuning_scan(sample_frequencies_with_outlier, sample_noisy_transfers_with_outlier,
                                                                 SAMPLE_TAU, filter_outliers = True, report_inliers = True)
    popt_o, pcov_o = outlier_fit_results
    overall_indices = np.arange(len(sample_frequencies_with_outlier))
    outlier_indices = overall_indices[~np.isin(overall_indices, inlier_indices)]
    assert len(outlier_indices) == 1 
    assert outlier_indices[0] == len(sample_frequencies)
    center_o, rabi_freq_o = popt_o
    assert (np.abs((center_o - SAMPLE_CENTER) / SAMPLE_CENTER) < 3e-2) 
    assert (np.abs((rabi_freq_o - SAMPLE_RABI) / SAMPLE_RABI) < 3e-2)


def test_hybrid_trap_center_finder():
    EXPECTED_X_CENTER = 228
    EXPECTED_Y_CENTER = 398
    SAMPLE_WIDTH = 189
    SAMPLE_LENGTH = 250
    SAMPLE_TILT_DEG = 6.3
    sample_hybrid_trap_data = np.load('resources/Sample_Box_Exp.npy')
    center_guess = data_fitting_functions.hybrid_trap_center_finder(sample_hybrid_trap_data, SAMPLE_TILT_DEG, SAMPLE_WIDTH, SAMPLE_LENGTH) 
    x_center_guess, y_center_guess = center_guess
    assert (np.abs(x_center_guess - EXPECTED_X_CENTER) < 5) 
    assert (np.abs(y_center_guess - EXPECTED_Y_CENTER) < 5)


def test_monte_carlo_covariance_helper():
    NUM_SAMPLES = 10000
    DATA_LENGTH = 100
    SAMPLE_SLOPE = 2.3 
    SAMPLE_INTERCEPT = -3.1
    EXPECTED_COVARIANCE_MATRIX = np.array([[ 0.11762376, -0.05881188],
                                            [-0.05881188,  0.03940594]])
    def my_fitting_function(x, a, b):
        return a*x + b 
    x_values = np.linspace(0, 1, 100) 
    normal_randoms = np.load(os.path.join("resources", "Sample_Normal_Randoms.npy"))
    y_values = SAMPLE_SLOPE * x_values + SAMPLE_INTERCEPT + normal_randoms[:len(x_values)]
    errors = np.ones(len(x_values)) 
    results = curve_fit(my_fitting_function, x_values, y_values, sigma = errors, absolute_sigma = True)
    popt, pcov = results
    pcov_monte = data_fitting_functions._monte_carlo_covariance_helper(my_fitting_function, x_values, y_values, errors, popt, 
                                                                num_samples = NUM_SAMPLES)
    assert np.all(np.abs((pcov_monte - pcov) / pcov) < 2e-1)