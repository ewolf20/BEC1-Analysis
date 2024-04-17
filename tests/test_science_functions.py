import os
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np 


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

RESOURCES_DIRECTORY_PATH = "./resources"

from BEC1_Analysis.code import science_functions, eos_functions


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


def test_get_ideal_fermi_pressure_hz_um_from_density():
    SAMPLE_DENSITY = 2.71818
    #Checked calculation for this value manually 
    EXPECTED_PRESSURE_HZ_UM = 2.703112e-26
    pressure_hz_um = science_functions.get_ideal_fermi_pressure_hz_um_from_density(SAMPLE_DENSITY)
    assert np.isclose(EXPECTED_PRESSURE_HZ_UM, pressure_hz_um, atol = 0.0, rtol = 1e-6)


def test_get_li_energy_hz_in_1D_trap():
    SAMPLE_DISPLACEMENT = 3.27
    SAMPLE_FREQUENCY = 12.5
    #Checked calculation manually for these values
    EXPECTED_ENERGY = 4.97144e11
    energy = science_functions.get_li_energy_hz_in_1D_trap(SAMPLE_DISPLACEMENT, SAMPLE_FREQUENCY)
    assert (np.abs((energy - EXPECTED_ENERGY) / EXPECTED_ENERGY) < 1e-4)

def test_get_box_fermi_energy_from_counts():
    SAMPLE_RADIUS_UM = 37 
    SAMPLE_LENGTH_UM = 52 
    SAMPLE_COUNTS = 1337 
    sample_cross_section = np.square(SAMPLE_RADIUS_UM) * np.pi
    #Calculated manually to agree with this
    EXPECTED_ENERGY = 420.459
    energy = science_functions.get_box_fermi_energy_from_counts(SAMPLE_COUNTS, sample_cross_section, SAMPLE_LENGTH_UM)
    assert (np.abs((energy - EXPECTED_ENERGY) / EXPECTED_ENERGY) < 1e-4)

def test_get_hybrid_trap_average_energy():
    EXPECTED_AVERAGE_ENERGY_MANUAL_CUT = 5182.678774053225
    SAMPLE_RADIUS_UM = 70 
    SAMPLE_TRAP_FREQ = 23
    trap_cross_section_um = np.pi * np.square(SAMPLE_RADIUS_UM)
    sample_hybrid_trap_cut_data = np.load("resources/Sample_Box_Exp_Cut.npy")
    harmonic_positions, densities = sample_hybrid_trap_cut_data 
    cut_harmonic_positions = harmonic_positions[100:800]
    cut_densities = densities[100:800]
    average_particle_energy_manual_cut = science_functions.get_hybrid_trap_average_energy(cut_harmonic_positions, cut_densities,
                                                                 trap_cross_section_um, SAMPLE_TRAP_FREQ)
    assert (np.isclose(average_particle_energy_manual_cut, EXPECTED_AVERAGE_ENERGY_MANUAL_CUT))


def test_hybrid_trap_autocut():
    EXPECTED_STAT_START = 116
    EXPECTED_STAT_STOP = 716
    EXPECTED_SAVGOL_START = 110 
    EXPECTED_SAVGOL_STOP = 725
    sample_hybrid_trap_cut_data = np.load("resources/Sample_Box_Exp_Cut.npy")
    harmonic_positions, densities = sample_hybrid_trap_cut_data 
    statistics_start_index, statistics_stop_index = science_functions.hybrid_trap_autocut(densities, mode = "statistics")
    savgol_start_index, savgol_stop_index = science_functions.hybrid_trap_autocut(densities, mode = "savgol")
    assert statistics_start_index == EXPECTED_STAT_START
    assert statistics_stop_index == EXPECTED_STAT_STOP
    assert savgol_start_index == EXPECTED_SAVGOL_START
    assert savgol_stop_index == EXPECTED_SAVGOL_STOP


def _generate_sample_compressibilities_and_densities(positions_um, trap_freq):
    #Use a simulated ideal fermi distribution
    sample_potentials = science_functions.get_li_energy_hz_in_1D_trap(positions_um * 1e-6, trap_freq)
    sample_potential_max = np.max(sample_potentials)
    SAMPLE_CHEMICAL_POTENTIAL = 0.5 * sample_potential_max
    SAMPLE_TEMPERATURE = 0.200 * sample_potential_max
    local_chemical_potentials = SAMPLE_CHEMICAL_POTENTIAL - sample_potentials 
    local_betamus = local_chemical_potentials / SAMPLE_TEMPERATURE
    sample_densities = eos_functions.ideal_fermi_density_um(local_betamus, SAMPLE_TEMPERATURE)
    compressibility_vs_betamu_function = eos_functions.get_ideal_eos_functions(key = "kappa_over_kappa0")
    expected_compressibilities = compressibility_vs_betamu_function(local_betamus)
    return (expected_compressibilities, sample_densities)



def test_get_hybrid_trap_compressibilities_savgol():
    SAMPLE_TRAP_FREQ = 12.3
    num_samples = 500 
    sample_positions_um = np.linspace(0, 300, num = num_samples)
    expected_compressibilities, sample_densities = _generate_sample_compressibilities_and_densities(sample_positions_um, SAMPLE_TRAP_FREQ)
    SAVGOL_WINDOW = 15
    SAVGOL_POLYORDER = 2
    extracted_compressibilities = science_functions.get_hybrid_trap_compressibilities_savgol(sample_positions_um, 
                                                                                sample_densities, SAMPLE_TRAP_FREQ, 
                                                                                savgol_window_length = SAVGOL_WINDOW, 
                                                                                savgol_polyorder = SAVGOL_POLYORDER)
    #Savitzky-Golay method must necessarily have edge effects; remove high edge where issues occur
    high_clip_index = SAVGOL_WINDOW
    clipped_extracted_compressibilities = extracted_compressibilities[high_clip_index:]
    clipped_expected_compressibilities = expected_compressibilities[high_clip_index:]
    assert np.all(np.isclose(clipped_expected_compressibilities, clipped_extracted_compressibilities, rtol = 5e-2))


def test_get_hybrid_trap_compressibilities_window_fit():
    SAMPLE_TRAP_FREQ = 12.3
    num_samples = 500
    sample_positions_um = np.linspace(0, 300, num = num_samples)
    sample_potentials_hz = science_functions.get_li_energy_hz_in_1D_trap(sample_positions_um * 1e-6, SAMPLE_TRAP_FREQ)
    expected_compressibilities, sample_densities = _generate_sample_compressibilities_and_densities(sample_positions_um, SAMPLE_TRAP_FREQ)
    #Use a simulated ideal fermi distribution
    breakpoint_indices = np.arange(0, sample_potentials_hz.size, 20)
    energy_midpoints = (sample_potentials_hz[breakpoint_indices - 1][1:] + sample_potentials_hz[breakpoint_indices][:-1]) / 2.0
    POLYORDER = 2
    extracted_compressibilities = science_functions.get_hybrid_trap_compressibilities_window_fit(sample_potentials_hz, 
                                                                                sample_densities, breakpoint_indices, 
                                                                                polyorder = POLYORDER)

    interpolated_expected_compressibilities = np.interp(energy_midpoints, sample_potentials_hz, expected_compressibilities)
    assert np.all(np.isclose(interpolated_expected_compressibilities, extracted_compressibilities, rtol = 1e-3))

def test_get_li6_br_energy_MHz():
    sample_field_G = 690
    EXPECTED_STATE_1_ENERGY = -1049.0933
    EXPECTED_STATE_2_ENERGY = -973.0597
    state_1_energy = science_functions.get_li6_br_energy_MHz(sample_field_G, 1)
    state_2_energy = science_functions.get_li6_br_energy_MHz(sample_field_G, 2)
    assert (np.abs((EXPECTED_STATE_1_ENERGY - state_1_energy) / EXPECTED_STATE_1_ENERGY))
    assert (np.abs((EXPECTED_STATE_2_ENERGY - state_2_energy) / EXPECTED_STATE_2_ENERGY))

def test_get_field_from_li6_resonance():
    SAMPLE_RESONANCE_FREQ = 75.9156
    EXPECTED_B_FIELD = 635.01
    SAMPLE_INDICES = (1, 2)
    extracted_field = science_functions.get_field_from_li6_resonance(SAMPLE_RESONANCE_FREQ, SAMPLE_INDICES)
    assert(np.abs((extracted_field - EXPECTED_B_FIELD) / EXPECTED_B_FIELD) < 1e-5)