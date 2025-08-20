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


def test_get_polaron_eos_mus_and_T_from_box_energy():
    SAMPLE_BOX_VOLUME_UM = 100
    SAMPLE_MU_UP = 2000 
    SAMPLE_MU_DOWN = -500
    SAMPLE_T = 1000 
    #First get the pressure and densities from the forward functions 
    sample_minority_density_um = eos_functions.polaron_eos_minority_density_um(SAMPLE_MU_UP, SAMPLE_MU_DOWN, SAMPLE_T)
    sample_majority_density_um = eos_functions.polaron_eos_majority_density_um(SAMPLE_MU_UP, SAMPLE_MU_DOWN, SAMPLE_T)
    sample_pressure_Hz_um = eos_functions.polaron_eos_pressure_Hz_um(SAMPLE_MU_UP, SAMPLE_MU_DOWN, SAMPLE_T) 
    #Convert to experimental units 
    sample_majority_counts = sample_majority_density_um * SAMPLE_BOX_VOLUME_UM
    sample_minority_counts = sample_minority_density_um * SAMPLE_BOX_VOLUME_UM
    sample_energy_Hz = 3/2 * sample_pressure_Hz_um * SAMPLE_BOX_VOLUME_UM
    #Then solve for the values 
    extracted_mu_up, extracted_mu_down, extracted_T = science_functions.get_polaron_eos_mus_and_T_from_box_counts_and_energy(
        sample_majority_counts, sample_minority_counts, sample_energy_Hz, SAMPLE_BOX_VOLUME_UM
    )
    assert np.isclose(extracted_mu_up, SAMPLE_MU_UP) 
    assert np.isclose(extracted_mu_down, SAMPLE_MU_DOWN) 
    assert np.isclose(extracted_T, SAMPLE_T)



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
    sample_cross_section_um = np.square(SAMPLE_RADIUS_UM) * np.pi
    box_volume_um = sample_cross_section_um * SAMPLE_LENGTH_UM
    #Calculated manually to agree with this
    EXPECTED_ENERGY = 420.459
    energy = science_functions.get_box_fermi_energy_from_counts(SAMPLE_COUNTS, box_volume_um)
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
    EXPECTED_STAT_START = 118
    EXPECTED_STAT_STOP = 707
    EXPECTED_SAVGOL_START = 118
    EXPECTED_SAVGOL_STOP = 718
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


def test_get_absolute_pressures():
    sample_potentials = np.e * np.linspace(0, 1, 100)
    sample_densities_1d = np.pi * np.linspace(0, 1, 100)
    expected_pressures = np.array([np.trapz(sample_densities_1d[i:], x = sample_potentials[i:]) for i in range(len(sample_densities_1d))])
    extracted_pressures = science_functions.get_absolute_pressures(sample_potentials, sample_densities_1d)
    assert np.allclose(extracted_pressures, expected_pressures)
    sample_densities_superfluous_dim = np.expand_dims(sample_densities_1d, 0)
    expected_pressures_superfluous_dim = np.expand_dims(expected_pressures, 0)
    extracted_pressures_superfluous_dim = science_functions.get_absolute_pressures(sample_potentials, sample_densities_superfluous_dim)
    assert np.allclose(expected_pressures_superfluous_dim, extracted_pressures_superfluous_dim)
    sample_densities_2d = np.expand_dims(sample_densities_1d, 0) * np.arange(4).reshape((4, 1))
    expected_pressures_2d = np.expand_dims(expected_pressures, 0) * np.arange(4).reshape((4, 1)) 
    extracted_pressures_2d = science_functions.get_absolute_pressures(sample_potentials, sample_densities_2d) 
    assert np.allclose(extracted_pressures_2d, expected_pressures_2d)


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


def test_get_scattering_length_feshbach_a0():
    sample_field_vals = np.linspace(600, 610, 100) 
    #Manually feed values in from science functions into Feshbach function; no test of correctness of values
    indices = science_functions.SCATTERING_LENGTH_INDICES_LIST 
    centers = science_functions.SCATTERING_LENGTH_CENTERS_LIST_G
    widths = science_functions.SCATTERING_LENGTH_WIDTHS_LIST_G
    background_lengths = science_functions.SCATTERING_LENGTH_BACKGROUND_LENGTHS_LIST_A0 

    for index, center, width, background_length in zip(indices, centers, widths, background_lengths):
        expected_values = science_functions.feshbach_function(sample_field_vals, center, width, background_length)
        extracted_values = science_functions.get_scattering_length_feshbach_a0(index, sample_field_vals)
        assert np.allclose(expected_values, extracted_values)


def test_feshbach_function():
    SAMPLE_WIDTH = 100 
    SAMPLE_CENTER = 750 
    SAMPLE_BACKGROUND = -300 
    SAMPLE_FIELD = 900
    #MANUALLY CALCULATED VALUE
    EXPECTED_VALUE = -100 
    extracted_value = science_functions.feshbach_function(SAMPLE_FIELD, SAMPLE_CENTER, SAMPLE_WIDTH, SAMPLE_BACKGROUND)
    assert np.isclose(EXPECTED_VALUE, extracted_value)


def test_get_scattering_length_tabulated_a0():
    SAMPLE_FIELD_G = 300.5 
    #Values from manual inspection of tabulated data
    EXPECTED_SCATTERING_LENGTH_12 = -288.2 
    EXPECTED_SCATTERING_LENGTH_13 = -888.15
    EXPECTED_SCATTERING_LENGTH_23 = -451.65
    extracted_scattering_length_12 = science_functions.get_scattering_length_tabulated_a0((2, 1), SAMPLE_FIELD_G)
    extracted_scattering_length_13 = science_functions.get_scattering_length_tabulated_a0((1, 3), SAMPLE_FIELD_G)
    extracted_scattering_length_23 = science_functions.get_scattering_length_tabulated_a0((2, 3), SAMPLE_FIELD_G) 
    assert np.isclose(EXPECTED_SCATTERING_LENGTH_12, extracted_scattering_length_12)
    assert np.isclose(EXPECTED_SCATTERING_LENGTH_13, extracted_scattering_length_13)
    assert np.isclose(EXPECTED_SCATTERING_LENGTH_23, extracted_scattering_length_23)


def test_get_mean_field_shift_Hz():
    SAMPLE_DENSITY_UM = 3.14
    SAMPLE_SCATTERING_LENGTH_A0 = 1337
    #From manual evaluation of relevant formula with given values
    EXPECTED_SHIFT_HZ = 4691.103628941158
    extracted_shift_Hz = science_functions.get_mean_field_shift_Hz(SAMPLE_DENSITY_UM, SAMPLE_SCATTERING_LENGTH_A0)
    assert np.isclose(extracted_shift_Hz, EXPECTED_SHIFT_HZ)


def test_get_density_um_from_clock_shift_Hz():
    SAMPLE_INITIAL_STATE = 1 
    SAMPLE_FINAL_STATE = 2 
    SAMPLE_SPECTATOR_STATE = 3
    SAMPLE_SPECTATOR_DENSITY_UM = 3.14
    SAMPLE_FIELD_G = 300

    initial_spectator_index_pair = (SAMPLE_INITIAL_STATE, SAMPLE_SPECTATOR_STATE)
    final_spectator_index_pair = (SAMPLE_FINAL_STATE, SAMPLE_SPECTATOR_STATE)

    feshbach_scattering_length_initial_spectator = science_functions.get_scattering_length_feshbach_a0(initial_spectator_index_pair, 
                                                                                                       SAMPLE_FIELD_G)
    feshbach_scattering_length_final_spectator = science_functions.get_scattering_length_feshbach_a0(final_spectator_index_pair, 
                                                                                                     SAMPLE_FIELD_G)
    feshbach_shift_initial = science_functions.get_mean_field_shift_Hz(SAMPLE_SPECTATOR_DENSITY_UM, feshbach_scattering_length_initial_spectator)
    feshbach_shift_final = science_functions.get_mean_field_shift_Hz(SAMPLE_SPECTATOR_DENSITY_UM, feshbach_scattering_length_final_spectator)
    feshbach_mean_field_shift = feshbach_shift_final - feshbach_shift_initial
    feshbach_extracted_density = science_functions.get_density_um_from_clock_shift_Hz(SAMPLE_FIELD_G, feshbach_mean_field_shift, 
                                                                                SAMPLE_INITIAL_STATE, SAMPLE_FINAL_STATE, SAMPLE_SPECTATOR_STATE, 
                                                                                length_source = "feshbach")
    assert np.isclose(feshbach_extracted_density, SAMPLE_SPECTATOR_DENSITY_UM)

    tabulated_scattering_length_initial_spectator = science_functions.get_scattering_length_tabulated_a0(initial_spectator_index_pair, SAMPLE_FIELD_G)
    tabulated_scattering_length_final_spectator = science_functions.get_scattering_length_tabulated_a0(final_spectator_index_pair, SAMPLE_FIELD_G)
    tabulated_shift_initial = science_functions.get_mean_field_shift_Hz(SAMPLE_SPECTATOR_DENSITY_UM, tabulated_scattering_length_initial_spectator)
    tabulated_shift_final = science_functions.get_mean_field_shift_Hz(SAMPLE_SPECTATOR_DENSITY_UM, tabulated_scattering_length_final_spectator)
    tabulated_mean_field_shift = tabulated_shift_final - tabulated_shift_initial
    tabulated_extracted_density = science_functions.get_density_um_from_clock_shift_Hz(SAMPLE_FIELD_G, tabulated_mean_field_shift, 
                                                                                       SAMPLE_INITIAL_STATE, SAMPLE_FINAL_STATE, SAMPLE_SPECTATOR_STATE, 
                                                                                       length_source = "tabulated")
    assert np.isclose(tabulated_extracted_density, SAMPLE_SPECTATOR_DENSITY_UM)