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
    EXPECTED_AMP = 4.78
    EXPECTED_CENTER = 119.94
    EXPECTED_GAMMA = 2.37 
    EXPECTED_OFFSET = 10.18
    LORENTZIAN_TEST_DATA_PATH = os.path.join(TEST_DATA_DIRECTORY_PATH, 'Imaging_Lorentzian_Test_Data.npy') 
    test_frequencies, test_values = np.load(LORENTZIAN_TEST_DATA_PATH)
    lorentzian_results = data_fitting_functions.fit_imaging_resonance_lorentzian(test_frequencies, test_values)
    popt, pcov = lorentzian_results 
    amp, center, gamma, offset = popt 
    assert np.abs(amp - EXPECTED_AMP) < 0.01
    assert np.abs(center - EXPECTED_CENTER) < 0.01
    assert np.abs(gamma - EXPECTED_GAMMA) < 0.01 
    assert np.abs(offset - EXPECTED_OFFSET) < 0.01