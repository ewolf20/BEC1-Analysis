import os
import sys

import matplotlib.pyplot as plt
import numpy as np 


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

RESOURCES_DIRECTORY_PATH = "./resources"

from BEC1_Analysis.code import science_functions

def test_two_level_system_population_rabi():
    TEST_DETUNING_A = 1
    TEST_OMEGA_A = 1
    TEST_TAU_A = np.pi / np.sqrt(2) 
    two_level_populations_A = science_functions.two_level_system_population_rabi(TEST_TAU_A, TEST_OMEGA_A, TEST_DETUNING_A)
    two_level_population_1A, two_level_population_2A = two_level_populations_A
    assert np.abs(two_level_population_1A - 0.5) < 1e-4
    assert np.abs(two_level_population_2A - 0.5) < 1e-4
    TEST_DETUNING_B = 0
    TEST_OMEGA_B = 1
    TEST_TAU_B = np.pi
    two_level_populations_B = science_functions.two_level_system_population_rabi(TEST_TAU_B, TEST_OMEGA_B, TEST_DETUNING_B)
    two_level_population_1B, two_level_population_2B = two_level_populations_B
    assert np.abs(two_level_population_1B- 0.0) < 1e-4 
    assert np.abs(two_level_population_2B - 1.0) < 1e-4

def test_get_fermi_energy_hz_from_density():
    SAMPLE_DENSITY = 0.314
    #Checked calculation for this value manually & cross-checked with another student
    EXPECTED_FERMI_ENERGY = 5.8969e-09
    energy = science_functions.get_fermi_energy_hz_from_density(SAMPLE_DENSITY) 
    assert (np.abs((energy - EXPECTED_FERMI_ENERGY) / EXPECTED_FERMI_ENERGY) < 1e-4)


def test_get_li_energy_hz_in_1D_trap():
    SAMPLE_DISPLACEMENT = 3.27
    SAMPLE_FREQUENCY = 12.5
    #Checked calculation manually for these values
    EXPECTED_ENERGY = 12.5928e9
    energy = science_functions.get_li_energy_hz_in_1D_trap(SAMPLE_DISPLACEMENT, SAMPLE_FREQUENCY)
    assert (np.abs((energy - EXPECTED_ENERGY) / EXPECTED_ENERGY) < 1e-4)

def test_get_box_fermi_energy_from_counts():
    SAMPLE_RADIUS_UM = 37 
    SAMPLE_LENGTH_UM = 52 
    SAMPLE_COUNTS = 1337 
    #Calculated manually to agree with this
    EXPECTED_ENERGY = 420.459
    energy = science_functions.get_box_fermi_energy_from_counts(SAMPLE_COUNTS, SAMPLE_RADIUS_UM, SAMPLE_LENGTH_UM)
    assert (np.abs((energy - EXPECTED_ENERGY) / EXPECTED_ENERGY) < 1e-4)