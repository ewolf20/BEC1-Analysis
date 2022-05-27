import os 
import sys 

import matplotlib.pyplot as plt
import numpy as np 

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

TEST_DATA_DIRECTORY_PATH = "./resources/test_data"

from BEC1_Analysis.code import data_fitting_functions


def test_fit_imaging_resonance_lorentzian():
    LORENTZIAN_TEST_DATA_PATH = os.path.join(TEST_DATA_DIRECTORY_PATH, 'Imaging_Lorentzian_Test_Data.npy') 
    test_frequencies, test_values = np.load(LORENTZIAN_TEST_DATA_PATH)
    lorentzian_fit = data_fitting_functions.fit_imaging_resonance_lorentzian(test_frequencies, test_values)
    plt.plot(test_frequencies, test_values, 'o')
    plt.plot(test_frequencies, lorentzian_fit.best_fit, 'x')
    plt.show()
    print(lorentzian_fit.fit_report())
