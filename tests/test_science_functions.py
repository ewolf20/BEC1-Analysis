import copy
import os
import shutil
import sys
import time

import matplotlib.pyplot as plt
import mpmath 
import numpy as np 


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

RESOURCES_DIRECTORY_PATH = "./resources"

from BEC1_Analysis.code import science_functions, loading_functions

def test_kardar_f_minus_function():
    LOWER_LOG_BOUND = -10
    UPPER_LOG_BOUND = 100
    S_VALUE_1 = 3/2
    S_VALUE_2 = 5/2
    log_z_values = np.linspace(LOWER_LOG_BOUND, UPPER_LOG_BOUND, 10000)
    vectorized_mpmath_polylog = np.vectorize(mpmath.fp.polylog, otypes = [complex])
    mp_math_values_1 = -vectorized_mpmath_polylog(S_VALUE_1, -np.exp(log_z_values))
    mp_math_values_2 = -vectorized_mpmath_polylog(S_VALUE_2, -np.exp(log_z_values))
    homebrew_values_1 = science_functions.kardar_f_minus_function(S_VALUE_1, log_z_values)
    homebrew_values_2 = science_functions.kardar_f_minus_function(S_VALUE_2, log_z_values)
    assert np.all(np.isclose(mp_math_values_1, homebrew_values_1, atol = 0.0, rtol = 1e-6))
    assert np.all(np.isclose(mp_math_values_2, homebrew_values_2, atol = 0.0, rtol = 1e-6))


def test_get_ideal_betamu_from_T_over_TF():
    T_over_TF_values = np.logspace(-3, 3, 1000)
    betamu_values_direct = science_functions.get_ideal_betamu_from_T_over_TF(T_over_TF_values)
    reconstituted_T_over_TF_values_direct = science_functions.ideal_T_over_TF(betamu_values_direct)
    assert np.all(np.isclose(T_over_TF_values, reconstituted_T_over_TF_values_direct, rtol = 1e-8, atol = 0.0))
    betamu_values_tabulated = science_functions.get_ideal_betamu_from_T_over_TF(T_over_TF_values, flag = "tabulated")
    reconstituted_T_over_TF_values_tabulated = science_functions.ideal_T_over_TF(betamu_values_tabulated) 
    assert np.all(np.isclose(T_over_TF_values, reconstituted_T_over_TF_values_tabulated, rtol = 1e-8, atol = 0.0))


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
    EXPECTED_AVERAGE_ENERGY_AUTOCUT = 5623.863607223448
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
    average_particle_energy_autocut = science_functions.get_hybrid_trap_average_energy(harmonic_positions, densities, 
                                                                        trap_cross_section_um, SAMPLE_TRAP_FREQ, autocut = True, 
                                                                        autocut_mode = "statistics")
    assert (np.isclose(average_particle_energy_autocut, EXPECTED_AVERAGE_ENERGY_AUTOCUT))


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



def test_get_hybrid_trap_compressibility():
    SAMPLE_TRAP_FREQ = 23.6
    sample_hybrid_trap_cut_data = np.load("resources/Sample_Box_Exp_Cut.npy")
    harmonic_positions, densities = sample_hybrid_trap_cut_data
    harmonic_energies = science_functions.get_li_energy_hz_in_1D_trap(harmonic_positions * 1e-6, SAMPLE_TRAP_FREQ)
    positions, compressibilities = science_functions.get_hybrid_trap_compressibilities(harmonic_positions, densities, SAMPLE_TRAP_FREQ)
    

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



def test_get_balanced_eos_functions():
    loaded_eos_data = loading_functions.load_unitary_EOS()
    independent_variable_options = ["betamu"]
    independent_variable_options.extend(science_functions.BALANCED_EOS_REVERSIBLE_VALUES_NAMES_LIST)
    lower_sensitivity_keys = ["kappa_over_kappa0", "Cv_over_NkB"]
    for independent_variable in independent_variable_options:
        independent_variable_experimental_data = loaded_eos_data[independent_variable]
        functions_dict = science_functions.get_balanced_eos_functions(independent_variable = independent_variable)
        for key in functions_dict:
            experimental_data = loaded_eos_data[key]
            function = functions_dict[key]
            function_calculated_data = function(independent_variable_experimental_data)
            if key in lower_sensitivity_keys:
                assert np.all(np.isclose(function_calculated_data, experimental_data, rtol = 1e-1))
            else:
                assert np.all(np.isclose(function_calculated_data, experimental_data, rtol = 2e-2))
    SAMPLE_VIRIAL_BETAMU_VALUE = -5.72
    for independent_variable in independent_variable_options:
        independent_variable_forward_function = science_functions._get_balanced_eos_virial_function(independent_variable)
        independent_variable_sample_value = independent_variable_forward_function(SAMPLE_VIRIAL_BETAMU_VALUE)
        functions_dict = science_functions.get_balanced_eos_functions(independent_variable = independent_variable)
        for key in functions_dict:
            function = functions_dict[key]
            function_calculated_value = function(independent_variable_sample_value) 
            direct_function = science_functions._get_balanced_eos_virial_function(key)
            direct_value = direct_function(SAMPLE_VIRIAL_BETAMU_VALUE)
            assert np.isclose(direct_value, function_calculated_value)


def _test_balanced_eos_virial_functions_helper(key, func, rtol = 2e-2):
    loaded_eos_data = loading_functions.load_unitary_EOS()
    experimental_betamu_values = loaded_eos_data["betamu"]
    experimental_values = loaded_eos_data[key]
    large_betamu_values = experimental_betamu_values[-4:] 
    large_betamu_experimental_values= experimental_values[-4:]
    virial_calculated_values = func(large_betamu_values)
    assert np.all(np.isclose(virial_calculated_values, large_betamu_experimental_values, rtol = rtol))


def test_balanced_kappa_over_kappa0_virial():
    _test_balanced_eos_virial_functions_helper("kappa_over_kappa0", science_functions.balanced_kappa_over_kappa0_virial, rtol = 1e-1)

def test_balanced_P_over_P0_virial():
    _test_balanced_eos_virial_functions_helper("P_over_P0", science_functions.balanced_P_over_P0_virial)

def test_balanced_Cv_over_NkB_virial():
    _test_balanced_eos_virial_functions_helper("Cv_over_NkB", science_functions.balanced_Cv_over_NkB_virial, rtol = 1e-1)

def test_balanced_T_over_TF_virial():
    _test_balanced_eos_virial_functions_helper("T_over_TF", science_functions.balanced_T_over_TF_virial)

def test_balanced_E_over_E0_virial():
    _test_balanced_eos_virial_functions_helper("E_over_E0", science_functions.balanced_E_over_E0_virial)

def test_balanced_mu_over_EF_virial():
    _test_balanced_eos_virial_functions_helper("mu_over_EF", science_functions.balanced_mu_over_EF_virial)

def test_balanced_S_over_NkB_virial():
    _test_balanced_eos_virial_functions_helper("S_over_NkB", science_functions.balanced_S_over_NkB_virial)

def test_balanced_F_over_E0_virial():
    _test_balanced_eos_virial_functions_helper("F_over_E0", science_functions.balanced_F_over_E0_virial)


def _test_ultralow_fugacity_function_helper(key, fun):
    betamu_value = -20.1
    full_virial_function = science_functions._get_balanced_eos_virial_function(key) 
    other_value = full_virial_function(betamu_value)
    assert np.isclose(betamu_value, fun(other_value))

def test_ultralow_fugacity_betamu_function_P_over_P0():
    _test_ultralow_fugacity_function_helper("P_over_P0", science_functions.ultralow_fugacity_betamu_function_P_over_P0)

def test_ultralow_fugacity_betamu_function_T_over_TF():
    _test_ultralow_fugacity_function_helper("T_over_TF", science_functions.ultralow_fugacity_betamu_function_T_over_TF)

def test_ultralow_fugacity_betamu_function_E_over_E0():
    _test_ultralow_fugacity_function_helper("E_over_E0", science_functions.ultralow_fugacity_betamu_function_E_over_E0)

def test_ultralow_fugacity_betamu_function_S_over_NkB():
    _test_ultralow_fugacity_function_helper("S_over_NkB", science_functions.ultralow_fugacity_betamu_function_S_over_NkB)


def test_get_balanced_eos_betamu_from_other_value_virial_function():
    name_to_function_dict = {
        "P_over_P0":science_functions.balanced_P_over_P0_virial, 
        "T_over_TF":science_functions.balanced_T_over_TF_virial, 
        "E_over_E0":science_functions.balanced_E_over_E0_virial, 
        "S_over_NkB":science_functions.balanced_S_over_NkB_virial
    }
    min_betamu = -30
    max_betamu = -1
    num_points = 10000
    betamu_values = np.linspace(min_betamu, max_betamu, num_points)
    for key in name_to_function_dict:
        forward_function = name_to_function_dict[key]
        reverse_function = science_functions._get_balanced_eos_betamu_from_other_value_virial_function(key) 
        other_values = forward_function(betamu_values) 
        restored_betamu_values = reverse_function(other_values) 
        assert np.all(np.isclose(restored_betamu_values, betamu_values))


def test_generate_and_save_balanced_eos_betamu_from_other_value_virial_data():
    min_betamu = -11
    max_betamu = -2
    num_points = 1000 
    name_to_function_dict = {
        "P_over_P0":science_functions.balanced_P_over_P0_virial, 
        "T_over_TF":science_functions.balanced_T_over_TF_virial, 
        "E_over_E0":science_functions.balanced_E_over_E0_virial, 
        "S_over_NkB":science_functions.balanced_S_over_NkB_virial
    }
    TEMP_WORKFOLDER_NAME = "temp"
    workfolder_pathname = os.path.join(RESOURCES_DIRECTORY_PATH, TEMP_WORKFOLDER_NAME)
    try:
        os.makedirs(workfolder_pathname)
        stored_data_filename = "Data_Temp.npy" 
        stored_metadata_filename = "Metadata_Temp.txt" 
        stored_data_pathname = os.path.join(workfolder_pathname, stored_data_filename)
        stored_metadata_pathname = os.path.join(workfolder_pathname, stored_metadata_filename) 
        science_functions.generate_and_save_balanced_eos_betamu_from_other_value_virial_data(
            stored_data_pathname, stored_metadata_pathname,
            min_betamu = min_betamu, 
            max_betamu = max_betamu, 
            num_points = num_points
        ) 
        assert os.path.exists(stored_data_pathname) 
        assert os.path.exists(stored_metadata_pathname) 
        loaded_stored_data = np.load(stored_data_pathname)
        betamu_values = loaded_stored_data[0] 
        for i, key in enumerate(name_to_function_dict):
            stored_other_values = loaded_stored_data[i + 1]
            other_values_virial_function = name_to_function_dict[key]
            assert np.all(np.isclose(stored_other_values, other_values_virial_function(betamu_values)))
    finally:
        shutil.rmtree(workfolder_pathname)

