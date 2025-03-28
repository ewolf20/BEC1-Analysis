import os
import sys 

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

TEST_DATA_DIRECTORY_PATH = "./resources/test_data"

from BEC1_Analysis.code import data_fitting_functions, eos_functions

def test_fit_lorentzian():
    normal_randoms = np.load(os.path.join("resources", "Sample_Normal_Randoms.npy"))
    normal_randoms_rescaled = normal_randoms / 5
    x_vals = np.linspace(0, 1, 100)
    CENTER = 0.42
    AMP = 3.14 
    GAMMA = 0.2718
    OFFSET = 0.56
    y_vals_no_offset = data_fitting_functions.lorentzian(x_vals, AMP, CENTER, GAMMA) + normal_randoms_rescaled
    y_vals_with_offset = y_vals_no_offset + OFFSET 
    no_offset_fit_results = data_fitting_functions.fit_lorentzian(x_vals, y_vals_no_offset)
    offset_fit_results = data_fitting_functions.fit_lorentzian_with_offset(x_vals, y_vals_with_offset)
    no_offset_popt, no_offset_pcov = no_offset_fit_results 
    offset_popt, offset_pcov = offset_fit_results 
    EXPECTED_NO_OFFSET_POPT = np.array([3.1615846,  0.41945715, 0.26892573])
    EXPECTED_OFFSET_POPT = np.array([3.18529419, 0.4195095,  0.27649611, 0.52010561])
    assert np.all(np.isclose(offset_popt, EXPECTED_OFFSET_POPT))
    assert np.all(np.isclose(no_offset_popt, EXPECTED_NO_OFFSET_POPT))


def test_fit_lorentzian_times_freq():
    normal_randoms = np.load(os.path.join("resources", "Sample_Normal_Randoms.npy"))
    normal_randoms_rescaled = normal_randoms / 5 
    x_vals = np.linspace(0, 5, 100) 
    CENTER = 0.5
    GAMMA = 1.07
    AMP = 3.5
    y_vals = data_fitting_functions.lorentzian_times_freq(x_vals, AMP, CENTER, GAMMA) + normal_randoms_rescaled
    fit_results = data_fitting_functions.fit_lorentzian_times_freq(x_vals, y_vals) 
    popt, pcov = fit_results 
    EXPECTED_POPT = np.array([3.48214247, 0.47819892, 1.095833])
    assert np.all(np.isclose(popt, EXPECTED_POPT))
    amp, center, gamma = popt 
    amp_to_peak_conversion = data_fitting_functions.get_lorentzian_times_freq_amp_to_peak_conversion_factor(center, gamma)
    EXPECTED_AMP_TO_PEAK_CONVERSION = 0.60272303
    assert np.isclose(amp_to_peak_conversion, EXPECTED_AMP_TO_PEAK_CONVERSION)


def test_fit_gaussian():
    normal_randoms = np.load(os.path.join("resources", "Sample_Normal_Randoms.npy"))
    normal_randoms_rescaled = normal_randoms / 5
    x_vals = np.linspace(0, 1, 100)
    CENTER = 0.42
    AMP = 3.14 
    SIGMA = 0.162
    OFFSET = 0.56
    y_vals_no_offset = data_fitting_functions.gaussian(x_vals, AMP, CENTER, SIGMA) + normal_randoms_rescaled
    y_vals_with_offset = y_vals_no_offset + OFFSET 
    no_offset_fit_results = data_fitting_functions.fit_gaussian(x_vals, y_vals_no_offset)
    offset_fit_results = data_fitting_functions.fit_gaussian_with_offset(x_vals, y_vals_with_offset)
    no_offset_popt, no_offset_pcov = no_offset_fit_results 
    offset_popt, offset_pcov = offset_fit_results
    EXPECTED_NO_OFFSET_POPT = np.array([3.1659692, 0.4197707, 0.16001888])
    EXPECTED_OFFSET_POPT = np.array([ 3.18150399, 0.41981671, 0.16140804, 0.53924337])
    assert np.all(np.isclose(offset_popt, EXPECTED_OFFSET_POPT))
    assert np.all(np.isclose(no_offset_popt, EXPECTED_NO_OFFSET_POPT))

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
    assert np.isclose(amp, GAUSSIAN_SIMULATED_AMP, rtol = 1e-2)
    assert np.isclose(x_center, GAUSSIAN_SIMULATED_X_CENTER, atol = 1)
    assert np.isclose(y_center, GAUSSIAN_SIMULATED_Y_CENTER, atol = 1)
    assert np.isclose(x_width, GAUSSIAN_SIMULATED_X_WIDTH, rtol = 1e-2) 
    assert np.isclose(y_width, GAUSSIAN_SIMULATED_Y_WIDTH, rtol = 1e-2)
    assert np.isclose(offset, GAUSSIAN_SIMULATED_OFFSET, rtol = 1e-2)


def simulate_2D_gaussian_image():
    y_coordinates, x_coordinates = np.indices((GAUSSIAN_Y_PIXEL_NUM, GAUSSIAN_X_PIXEL_NUM))
    simulated_noiseless_image = data_fitting_functions.two_dimensional_gaussian(x_coordinates, y_coordinates, GAUSSIAN_SIMULATED_AMP, GAUSSIAN_SIMULATED_X_CENTER, 
                                                        GAUSSIAN_SIMULATED_Y_CENTER, GAUSSIAN_SIMULATED_X_WIDTH, GAUSSIAN_SIMULATED_Y_WIDTH, 
                                                        GAUSSIAN_SIMULATED_OFFSET)
    NOISE_MAGNITUDE = 0.5
    rng = np.random.default_rng(1337)
    noisy_image = simulated_noiseless_image + rng.normal(loc = 0.0, scale = NOISE_MAGNITUDE, size = simulated_noiseless_image.shape)
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



def test_fit_one_dimensional_condensate():
    EXPECTED_POPT = np.array([521.5539, 33.3197, 8637.0955, 136.6156, 2470.3321])
    positions, integrated_data = np.load(os.path.join("resources", "RR_Integrated_Data.npy"))
    fit_results_condensate, fit_results_thermal = data_fitting_functions.fit_one_dimensional_condensate(integrated_data)
    condensate_popt, condensate_pcov = fit_results_condensate 
    thermal_popt, thermal_pcov = fit_results_thermal 
    center, condensate_width, condensate_amp = condensate_popt 
    _, thermal_width, thermal_amp = thermal_popt 
    popt = np.array([center, condensate_width, condensate_amp, thermal_width, thermal_amp])
    assert np.all(np.isclose(popt, EXPECTED_POPT))

#TODO: Modify bootstrap and monte carlo to take in an external random object!!!

def test_monte_carlo_fit_covariance():
    RNG_SEED = 1337
    NUM_SAMPLES = 10000
    DATA_LENGTH = 100
    SAMPLE_SLOPE = 2.3 
    SAMPLE_INTERCEPT = -3.1
    EXPECTED_COVARIANCE_MATRIX = np.array([[0.11568357, -0.05788674],
                                            [-0.05788674, 0.03921054]])
    def my_fitting_function(x, a, b):
        return a*x + b 
    x_values = np.linspace(0, 1, 100) 
    normal_randoms = np.load(os.path.join("resources", "Sample_Normal_Randoms.npy"))
    y_values = SAMPLE_SLOPE * x_values + SAMPLE_INTERCEPT + normal_randoms[:len(x_values)]
    errors = np.ones(len(x_values)) 
    results = curve_fit(my_fitting_function, x_values, y_values, sigma = errors, absolute_sigma = True)
    popt, pcov = results
    pcov_monte = data_fitting_functions.monte_carlo_fit_covariance(my_fitting_function, x_values, y_values, errors, popt, 
                                                                num_samples = NUM_SAMPLES, rng_seed = RNG_SEED)
    assert np.all(np.abs((pcov_monte - pcov) / pcov) < 1e-1)
    assert np.all(np.isclose(pcov_monte, EXPECTED_COVARIANCE_MATRIX))


def test_fit_error_function_rectangle():
    normal_randoms = np.load(os.path.join("resources", "Sample_Normal_Randoms.npy"))
    indices = np.arange(len(normal_randoms)) 
    RECTANGLE_CENTER = 42
    RECTANGLE_WIDTH = 50
    RECTANGLE_AMP = 1.0
    noiseless_rectangle = np.where(
        np.abs(indices - RECTANGLE_CENTER) < RECTANGLE_WIDTH / 2, 
        RECTANGLE_AMP, 
        0.0 
    )
    smoothed_noiseless_rectangle = savgol_filter(noiseless_rectangle, 11, 3)
    noisy_rectangle = smoothed_noiseless_rectangle + normal_randoms * 0.1
    fit_results = data_fitting_functions.fit_error_function_rectangle(indices, noisy_rectangle)
    popt, pcov = fit_results
    amp, width, center, edge_width = popt
    EXPECTED_AMP = 1.013
    EXPECTED_CENTER = 42.06 
    EXPECTED_WIDTH = 48.56
    EXPECTED_EDGE_WIDTH = 1.924
    assert np.isclose(amp, EXPECTED_AMP, rtol = 1e-3)
    assert np.isclose(center, EXPECTED_CENTER, rtol = 1e-3)
    assert np.isclose(width, EXPECTED_WIDTH, rtol = 1e-3)
    assert np.isclose(edge_width, EXPECTED_EDGE_WIDTH, rtol = 1e-3)
    #Without fitting width...
    


def test_fit_semicircle():
    normal_randoms = np.load(os.path.join("resources", "Sample_Normal_Randoms.npy"))
    indices = np.arange(len(normal_randoms)) 
    CIRCLE_CENTER = 42
    CIRCLE_RADIUS = 22
    CIRCLE_AMP = 1.0 
    noiseless_circle = np.where(
        np.abs(indices - CIRCLE_CENTER) < CIRCLE_RADIUS, 
        CIRCLE_AMP * np.sqrt(1 - np.square((indices - CIRCLE_CENTER) / CIRCLE_RADIUS)),
        0.0
    )
    noisy_circle = noiseless_circle + normal_randoms * 0.1 
    fit_results = data_fitting_functions.fit_semicircle(indices, noisy_circle) 
    popt, pcov = fit_results
    amp, center, radius = popt
    EXPECTED_AMP = 1.005 
    EXPECTED_CENTER = 41.99 
    EXPECTED_RADIUS = 22.08
    assert np.isclose(amp, EXPECTED_AMP, rtol = 1e-3)
    assert np.isclose(center, EXPECTED_CENTER, rtol = 1e-3)
    assert np.isclose(radius, EXPECTED_RADIUS, rtol = 1e-3)


def test_crop_box():
    sample_box_data = np.load(os.path.join("resources", "Sample_Box.npy")) 
    EXPECTED_BOX_CROP = (34, 50, 214, 174)
    box_crop = data_fitting_functions.crop_box(sample_box_data)
    assert EXPECTED_BOX_CROP == box_crop
    EXPECTED_BOX_CROP_SPECIFIED_POINTS = (70, 39, 178, 186)
    box_crop_specified_points = data_fitting_functions.crop_box(sample_box_data, horiz_crop_point = 0.8, vert_crop_point = 0.05)
    assert box_crop_specified_points == EXPECTED_BOX_CROP_SPECIFIED_POINTS
    FIXED_VERT_WIDTH = 90 
    FIXED_HORIZ_RADIUS = 120
    EXPECTED_BOX_CROP_FIXED_WIDTHS = (4, 68, 244, 158)
    box_crop_fixed_widths = data_fitting_functions.crop_box(sample_box_data, horiz_radius = FIXED_HORIZ_RADIUS, vert_width = FIXED_VERT_WIDTH)
    assert EXPECTED_BOX_CROP_FIXED_WIDTHS == box_crop_fixed_widths


def test_fit_li6_ideal_fermi_density():
    normal_randoms = np.load(os.path.join("resources", "Sample_Normal_Randoms.npy"))
    potential_values = np.linspace(0, 1500, num = len(normal_randoms)) 
    MU_0_VALUE = 800
    kBT_VALUE = 200
    local_mu_values = MU_0_VALUE - potential_values 
    local_betamu_values = local_mu_values / kBT_VALUE
    noise_free_densities = eos_functions.ideal_fermi_density_um(local_betamu_values, kBT_VALUE)
    noisy_densities = noise_free_densities * (1.0 + 0.05 * normal_randoms) 
    fit_results = data_fitting_functions.fit_li6_ideal_fermi_density(potential_values, noisy_densities)
    popt, pcov = fit_results
    EXPECTED_POPT = np.array([793.61463, 204.85814])
    assert np.all(np.isclose(popt, EXPECTED_POPT))
    POTENTIAL_OFFSET = 3141
    offset_potential_values = potential_values + POTENTIAL_OFFSET
    fit_results_offset = data_fitting_functions.fit_li6_ideal_fermi_density(offset_potential_values, noisy_densities)
    offset_popt, offset_pcov = fit_results_offset 
    expected_offset_popt = EXPECTED_POPT + np.array([POTENTIAL_OFFSET, 0.0])
    assert np.all(np.isclose(offset_popt, expected_offset_popt))


def test_fit_li6_ideal_fermi_density_with_prefactor():
    normal_randoms = np.load(os.path.join("resources", "Sample_Normal_Randoms.npy"))
    potential_values = np.linspace(0, 1500, num = len(normal_randoms)) 
    MU_0_VALUE = 1000
    kBT_VALUE = 200
    PREFACTOR_VALUE = 1.5
    local_mu_values = MU_0_VALUE - potential_values 
    local_betamu_values = local_mu_values / kBT_VALUE
    noise_free_densities = PREFACTOR_VALUE * eos_functions.ideal_fermi_density_um(local_betamu_values, kBT_VALUE)
    noisy_densities = noise_free_densities * (1.0 + 0.05 * normal_randoms) 
    fit_results = data_fitting_functions.fit_li6_ideal_fermi_density_with_prefactor(potential_values, noisy_densities)
    popt, pcov = fit_results
    EXPECTED_POPT = np.array([1024.22259, 192.358369, 1.43997535])
    assert np.all(np.isclose(popt, EXPECTED_POPT))
    POTENTIAL_OFFSET = 3141
    offset_potential_values = potential_values + POTENTIAL_OFFSET
    fit_results_offset = data_fitting_functions.fit_li6_ideal_fermi_density_with_prefactor(offset_potential_values, noisy_densities)
    offset_popt, offset_pcov = fit_results_offset 
    expected_offset_popt = EXPECTED_POPT + np.array([POTENTIAL_OFFSET, 0.0, 0.0])
    assert np.all(np.isclose(offset_popt, expected_offset_popt))



def test_fit_li6_balanced_density():
    normal_randoms = np.load(os.path.join("resources", "Sample_Normal_Randoms.npy"))
    potential_values = np.linspace(0, 1500, num = len(normal_randoms)) 
    MU_0_VALUE = 800
    kBT_VALUE = 200
    local_mu_values = MU_0_VALUE - potential_values 
    local_betamu_values = local_mu_values / kBT_VALUE
    noise_free_densities = eos_functions.balanced_density_um(local_betamu_values, kBT_VALUE)
    noisy_densities = noise_free_densities * (1.0 + 0.05 * normal_randoms) 
    fit_results = data_fitting_functions.fit_li6_balanced_density(potential_values, noisy_densities)
    popt, pcov = fit_results
    EXPECTED_POPT = np.array([798.2285, 202.1856])
    assert np.all(np.isclose(popt, EXPECTED_POPT))
    POTENTIAL_OFFSET = 3141
    offset_potential_values = potential_values + POTENTIAL_OFFSET
    fit_results_offset = data_fitting_functions.fit_li6_balanced_density(offset_potential_values, noisy_densities)
    offset_popt, offset_pcov = fit_results_offset 
    expected_offset_popt = EXPECTED_POPT + np.array([POTENTIAL_OFFSET, 0.0])
    assert np.all(np.isclose(offset_popt, expected_offset_popt))


def test_fit_li6_polaron_eos_densities():
    RNG_SEED = 1337 
    rng = np.random.default_rng(seed = RNG_SEED)

    SAMPLE_MU_UP_BARE_HZ = 3000
    SAMPLE_MU_DOWN_BARE_HZ = -1000
    SAMPLE_T_HZ = 1000 
    POTENTIAL_MIN_HZ = 1000 
    POTENTIAL_RANGE_MAJ_HZ = 3000 
    POTENTIAL_RANGE_MIN_HZ = 2000 
    NUM_SAMPLES_MAJ = 600
    NUM_SAMPLES_MIN = 400
    potentials_majority = POTENTIAL_MIN_HZ + np.linspace(0, POTENTIAL_RANGE_MAJ_HZ, NUM_SAMPLES_MAJ)
    potentials_minority = POTENTIAL_MIN_HZ + np.linspace(0, POTENTIAL_RANGE_MIN_HZ, NUM_SAMPLES_MIN)
    local_mu_up_maj = SAMPLE_MU_UP_BARE_HZ - potentials_majority 
    local_mu_down_maj = SAMPLE_MU_DOWN_BARE_HZ - potentials_majority 
    local_mu_up_min = SAMPLE_MU_UP_BARE_HZ - potentials_minority 
    local_mu_down_min = SAMPLE_MU_DOWN_BARE_HZ - potentials_minority 

    sample_densities_majority_um = eos_functions.polaron_eos_majority_density_um(local_mu_up_maj, local_mu_down_maj, SAMPLE_T_HZ)
    sample_densities_minority_um = eos_functions.polaron_eos_minority_density_um(local_mu_up_min, local_mu_down_min, SAMPLE_T_HZ)

    NOISE_SCALE_UM = 0.003
    density_noise_maj = rng.normal(0.0, NOISE_SCALE_UM, NUM_SAMPLES_MAJ)
    density_noise_min = rng.normal(0.0, NOISE_SCALE_UM, NUM_SAMPLES_MIN)
    noisy_densities_majority_um = sample_densities_majority_um + density_noise_maj 
    noisy_densities_minority_um = sample_densities_minority_um + density_noise_min
    fit_results = data_fitting_functions.fit_li6_polaron_eos_densities(potentials_majority, potentials_minority, 
                                                                      noisy_densities_majority_um, noisy_densities_minority_um)
    popt, pcov = fit_results 
    sigmas = np.sqrt(np.diag(pcov))

    EXPECTED_POPT = np.array([3015.25932, -961.2358, 985.85386])
    assert np.allclose(EXPECTED_POPT, popt)
    




def test_fit_li6_balanced_density_with_prefactor():
    normal_randoms = np.load(os.path.join("resources", "Sample_Normal_Randoms.npy"))
    potential_values = np.linspace(0, 1500, num = len(normal_randoms)) 
    MU_0_VALUE = 1000
    kBT_VALUE = 250
    PREFACTOR_VALUE = 1.5
    local_mu_values = MU_0_VALUE - potential_values 
    local_betamu_values = local_mu_values / kBT_VALUE
    noise_free_densities = PREFACTOR_VALUE * eos_functions.balanced_density_um(local_betamu_values, kBT_VALUE)
    noisy_densities = noise_free_densities * (1.0 + 0.01 * normal_randoms) 
    fit_results = data_fitting_functions.fit_li6_balanced_density_with_prefactor(potential_values, noisy_densities)
    popt, pcov = fit_results
    EXPECTED_POPT = np.array([1001.573632,  250.070182, 1.493098])
    assert np.all(np.isclose(popt, EXPECTED_POPT))
    POTENTIAL_OFFSET = 3141
    offset_potential_values = potential_values + POTENTIAL_OFFSET
    fit_results_offset = data_fitting_functions.fit_li6_balanced_density_with_prefactor(offset_potential_values, noisy_densities)
    offset_popt, offset_pcov = fit_results_offset 
    expected_offset_popt = EXPECTED_POPT + np.array([POTENTIAL_OFFSET, 0.0, 0.0])
    assert np.all(np.isclose(offset_popt, expected_offset_popt))


def test_bootstrap_fit_covariance():
    RNG_SEED = 1337
    NUM_SAMPLES = 500
    DATA_LENGTH = 100 
    SAMPLE_SLOPE = 2.3 
    SAMPLE_INTERCEPT = -3.1
    EXPECTED_BOOTSTRAP_LOWER_SLOPE = 1.735
    EXPECTED_BOOTSTRAP_UPPER_SLOPE = 2.850
    EXPECTED_BOOTSTRAP_LOWER_INTERCEPT = -3.479
    EXPECTED_BOOTSTRAP_UPPER_INTERCEPT = -2.803
    def my_fitting_function(x, a, b):
        return a * x + b 
    x_values = np.repeat(np.linspace(0, 1, 25), 4)
    normal_randoms = np.load(os.path.join("resources", "Sample_Normal_Randoms.npy")) 
    y_values = SAMPLE_SLOPE * x_values + SAMPLE_INTERCEPT + normal_randoms[:len(x_values)]
    errors = np.ones(len(x_values))
    results = curve_fit(my_fitting_function, x_values, y_values, sigma = errors, absolute_sigma = True)
    popt, _ = results 
    bootstrap_results = data_fitting_functions.bootstrap_fit_covariance(my_fitting_function, x_values, 
                            y_values, popt, n_resamples = NUM_SAMPLES, return_full_bootstrap_result = True, 
                            rng_seed = RNG_SEED)
    slope_low, intercept_low = bootstrap_results.confidence_interval[0]
    slope_high, intercept_high = bootstrap_results.confidence_interval[1]
    assert np.isclose(slope_low, EXPECTED_BOOTSTRAP_LOWER_SLOPE, rtol = 1e-3)
    assert np.isclose(slope_high, EXPECTED_BOOTSTRAP_UPPER_SLOPE, rtol = 1e-3)
    assert np.isclose(intercept_low, EXPECTED_BOOTSTRAP_LOWER_INTERCEPT, rtol = 1e-3)
    assert np.isclose(intercept_high, EXPECTED_BOOTSTRAP_UPPER_INTERCEPT, rtol = 1e-3)
