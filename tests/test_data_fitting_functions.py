import os
from random import sample 
import sys 

import matplotlib.pyplot as plt
import numpy as np 

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
    noisy_sequential_y = noiseless_sequential_y + np.random.normal(loc = 0.0, scale = NOISE_AMP, size = len(noiseless_sequential_y))
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

def test_filter_1d_outliers():
    SAMPLE_CENTER = 23
    SAMPLE_AMP = 5.0 
    SAMPLE_OFFSET = 10 
    SAMPLE_GAMMA = 6.0 
    sample_frequencies = np.linspace(0, 50, 100) 
    sample_noiseless_lorentzian_data = data_fitting_functions.imaging_resonance_lorentzian(sample_frequencies, SAMPLE_AMP, SAMPLE_CENTER, 
                                                                SAMPLE_GAMMA, SAMPLE_OFFSET)
    OUTLIER_INDEX = 46
    sample_noiseless_lorentzian_data[OUTLIER_INDEX] = SAMPLE_OFFSET
    outlier_freq = sample_frequencies[OUTLIER_INDEX]  
    noisy_lorentzian_data = sample_noiseless_lorentzian_data + np.random.normal(loc = 0.0, scale = 0.2, size = len(sample_frequencies))  
    fit_results_untrimmed = data_fitting_functions.fit_imaging_resonance_lorentzian(sample_frequencies, noisy_lorentzian_data)
    popt, pcov = fit_results_untrimmed
    fit_residuals = noisy_lorentzian_data - data_fitting_functions.imaging_resonance_lorentzian(sample_frequencies, *popt)
    inlier_indices = data_fitting_functions._filter_1d_outliers(sample_frequencies, noisy_lorentzian_data, 
                                            data_fitting_functions.imaging_resonance_lorentzian, popt)
    print(inlier_indices)
    assert not np.any(np.isin(OUTLIER_INDEX, inlier_indices))
    assert len(inlier_indices) == (len(sample_frequencies) - 1)



def test_fit_rf_spect_detuning_scan():
    SAMPLE_CENTER = 22
    SAMPLE_RABI = 1.47
    SAMPLE_TAU = 0.2
    sample_frequencies = np.linspace(0, 50, 100)
    sample_transfers = data_fitting_functions.rf_spect_detuning_scan(sample_frequencies, SAMPLE_TAU, SAMPLE_CENTER, SAMPLE_RABI)
    sample_noisy_transfers = sample_transfers + np.random.normal(loc = 0.0, scale = 0.005, size = len(sample_transfers))
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
