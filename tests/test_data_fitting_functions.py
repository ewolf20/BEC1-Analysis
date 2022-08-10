import os 
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