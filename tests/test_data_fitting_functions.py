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


def test_fit_two_dimensional_gaussian():
    noisy_image = simulate_2D_gaussian_image()
    plt.imshow(noisy_image) 
    plt.show()
    fit_results = data_fitting_functions.fit_two_dimensional_gaussian(noisy_image)
    assert True


def simulate_2D_gaussian_image():
    X_PIXEL_NUM = 490 
    Y_PIXEL_NUM = 500 
    y_coordinates = np.arange(X_PIXEL_NUM)
    x_coordinates = np.arange(Y_PIXEL_NUM)
    SIMULATED_X_CENTER = 130 
    SIMULATED_Y_CENTER = 420 
    SIMULATED_X_WIDTH = 45
    SIMULATED_Y_WIDTH = 30
    SIMULATED_AMP = 3.14
    SIMULATED_OFFSET = 10
    x_grid, y_grid = np.meshgrid(y_coordinates, x_coordinates)
    simulated_noiseless_image = data_fitting_functions.two_dimensional_gaussian(x_grid, y_grid, SIMULATED_AMP, SIMULATED_X_CENTER, 
                                                        SIMULATED_Y_CENTER, SIMULATED_X_WIDTH, SIMULATED_Y_WIDTH, SIMULATED_OFFSET)
    NOISE_MAGNITUDE = 0.314
    noisy_image = simulated_noiseless_image + np.random.normal(loc = 0.0, scale = NOISE_MAGNITUDE, size = simulated_noiseless_image.shape)
    return noisy_image