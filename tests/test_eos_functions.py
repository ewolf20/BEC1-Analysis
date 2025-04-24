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

from BEC1_Analysis.code import eos_functions, loading_functions, science_functions


def test_kardar_f_minus_function():
    LOWER_LOG_BOUND = -10
    UPPER_LOG_BOUND = 100
    s_values = [1/2, 3/2, 5/2]
    log_z_values = np.linspace(LOWER_LOG_BOUND, UPPER_LOG_BOUND, 1000)
    vectorized_mpmath_polylog_fp = np.vectorize(mpmath.fp.polylog, otypes = [complex])
    vectorized_mpmath_polylog = np.vectorize(mpmath.polylog, otypes = [complex])
    for s_value in s_values:
        if s_value == 0.5:
            mp_math_values = -vectorized_mpmath_polylog(s_value, -np.exp(log_z_values))
        else:
            mp_math_values = -vectorized_mpmath_polylog_fp(s_value, -np.exp(log_z_values))
        homebrew_values = eos_functions.kardar_f_minus_function(s_value, log_z_values)
        assert np.all(np.isclose(mp_math_values, homebrew_values, atol = 0.0, rtol = 1e-6))


def test_thermal_de_broglie_mks():
    sample_kBT_J = 1000 * eos_functions.H_MKS
    sample_mass_kg = 1.67262e-27
    #Manually calculated value, independent of code:
    EXPECTED_WAVELENGTH_M = 7.94035e-6
    calculated_wavelength_m = eos_functions.thermal_de_broglie_mks(sample_kBT_J, sample_mass_kg)
    assert np.isclose(EXPECTED_WAVELENGTH_M, calculated_wavelength_m)

def test_fermi_energy_Hz_from_density_um():
    SAMPLE_DENSITY = 0.314 * 1e-18
    #Checked calculation for this value manually & cross-checked with another student
    EXPECTED_FERMI_ENERGY = 5.8969e-09
    energy = eos_functions.fermi_energy_Hz_from_density_um(SAMPLE_DENSITY) 
    assert (np.abs((energy - EXPECTED_FERMI_ENERGY) / EXPECTED_FERMI_ENERGY) < 1e-4)


def test_density_um_from_fermi_energy_Hz():
    test_fermi_energy = 1000
    extracted_density = eos_functions.density_um_from_fermi_energy_Hz(test_fermi_energy)
    extracted_fermi_energy = eos_functions.fermi_energy_Hz_from_density_um(extracted_density)
    assert np.isclose(extracted_fermi_energy, test_fermi_energy)


def test_fermi_pressure_Hz_um_from_density_um():
    SAMPLE_DENSITY = 2.71818e-18
    #Checked calculation for this value manually 
    EXPECTED_PRESSURE_HZ_UM = 2.703112e-26
    pressure_hz_um = eos_functions.fermi_pressure_Hz_um_from_density_um(SAMPLE_DENSITY)
    assert np.isclose(EXPECTED_PRESSURE_HZ_UM, pressure_hz_um, atol = 0.0, rtol = 1e-6)

def test_ideal_fermi_density_um():
    #First test for correct results in the ultra cold limit
    ultra_cold_betamu_values = np.linspace(10, 20, 100) 
    ultra_cold_kBT_Hz = 1337
    ultra_cold_mu_values = ultra_cold_betamu_values * ultra_cold_kBT_Hz 
    calculated_ultra_cold_densities_um = eos_functions.ideal_fermi_density_um(ultra_cold_betamu_values, ultra_cold_kBT_Hz)
    #This function is tested separately
    calculated_ultra_cold_density_fermi_energies = eos_functions.fermi_energy_Hz_from_density_um(calculated_ultra_cold_densities_um)
    assert np.all(np.isclose(ultra_cold_mu_values, calculated_ultra_cold_density_fermi_energies, rtol = 1e-2, atol = 0.0))
    #Then test for correct results in the ultra hot limit 
    ultra_hot_betamu_values = np.linspace(-15, -8, 100)
    ultra_hot_kBT_Hz = 420
    ultra_hot_kBT_J = eos_functions.H_MKS * ultra_hot_kBT_Hz
    calculated_ultra_hot_densities_um = eos_functions.ideal_fermi_density_um(ultra_hot_betamu_values, ultra_hot_kBT_Hz)
    #The thermal de Broglie wavelength is tested separately
    ultra_hot_thermal_de_Broglie_m = eos_functions.thermal_de_broglie_mks(ultra_hot_kBT_J, eos_functions.LI_6_MASS_KG)
    ultra_hot_thermal_de_Broglie_um = ultra_hot_thermal_de_Broglie_m * 1e6
    ultra_hot_z_values = np.exp(ultra_hot_betamu_values)
    predicted_ultra_hot_densities_um = ultra_hot_z_values * np.power(ultra_hot_thermal_de_Broglie_um, -3.0)
    assert np.all(np.isclose(predicted_ultra_hot_densities_um, calculated_ultra_hot_densities_um, atol = 0.0, rtol = 2e-4))


def test_ideal_fermi_pressure_Hz_um():
    kBT_Hz = 1234
    ultra_cold_betamu_values = np.linspace(20, 30, 100)
    #At ultra cold temperatures, the pressure should be equal to the fermi pressure 
    calculated_ultra_cold_pressures_Hz_um = eos_functions.ideal_fermi_pressure_Hz_um(ultra_cold_betamu_values, kBT_Hz)
    ultra_cold_densities_um = eos_functions.ideal_fermi_density_um(ultra_cold_betamu_values, kBT_Hz)
    expected_ultra_cold_pressures_Hz_um = eos_functions.fermi_pressure_Hz_um_from_density_um(ultra_cold_densities_um)
    assert np.allclose(calculated_ultra_cold_pressures_Hz_um, expected_ultra_cold_pressures_Hz_um, rtol = 2e-2)
    #At ultra high temperatures, use the direct expression from the virial 
    kBT_J = kBT_Hz * eos_functions.H_MKS
    mass_kg = eos_functions.LI_6_MASS_KG
    de_broglie_wavelength_m = eos_functions.thermal_de_broglie_mks(kBT_J, mass_kg)
    de_broglie_wavelength_um = de_broglie_wavelength_m * 1e6
    pressure_prefactor = kBT_Hz / np.power(de_broglie_wavelength_um, 3) 
    ultra_hot_betamu_values = np.linspace(-10, -15, 100) 
    ultra_hot_z_values = np.exp(ultra_hot_betamu_values)
    expected_ultra_hot_pressures_Hz_um = pressure_prefactor * ultra_hot_z_values
    calculated_ultra_hot_pressures_Hz_um = eos_functions.ideal_fermi_pressure_Hz_um(ultra_hot_betamu_values, kBT_Hz)
    assert np.allclose(calculated_ultra_hot_pressures_Hz_um, expected_ultra_hot_pressures_Hz_um, atol = 0.0)


#NOTE: Test is only of internal consistency. Need to figure out some way to compare to known ideal Fermi data
def test_get_ideal_eos_functions():
    testing_betamu_values = np.linspace(-20, 100, 13333)
    independent_variable_options = ["betamu"]
    independent_variable_options.extend(eos_functions.BALANCED_EOS_REVERSIBLE_VALUES_NAMES_LIST)
    betamu_functions = eos_functions.get_ideal_eos_functions(independent_variable = "betamu")
    for independent_variable in independent_variable_options:
        independent_variable_betamu_function = betamu_functions[independent_variable]
        independent_variable_test_values = independent_variable_betamu_function(testing_betamu_values)
        functions_dict = eos_functions.get_ideal_eos_functions(independent_variable = independent_variable)
        for key in functions_dict:
            dependent_function = functions_dict[key]
            function_calculated_dependent_data = dependent_function(independent_variable_test_values)
            dependent_betamu_function = betamu_functions[key] 
            betamu_calculated_dependent_data = dependent_betamu_function(testing_betamu_values)
            #Atol is necessary to avoid pathological behavior near mu = 0
            assert np.all(np.isclose(betamu_calculated_dependent_data, function_calculated_dependent_data, atol = 1e-6, rtol = 1e-4))


def test_get_balanced_eos_functions():
    loaded_eos_data = loading_functions.load_unitary_EOS()
    independent_variable_options = ["betamu"]
    independent_variable_options.extend(eos_functions.BALANCED_EOS_REVERSIBLE_VALUES_NAMES_LIST)
    lower_sensitivity_keys = ["kappa_over_kappa0", "Cv_over_NkB"]
    for independent_variable in independent_variable_options:
        independent_variable_experimental_data = loaded_eos_data[independent_variable]
        functions_dict = eos_functions.get_balanced_eos_functions(independent_variable = independent_variable)
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
        independent_variable_forward_function = eos_functions._get_balanced_eos_virial_function(independent_variable)
        independent_variable_sample_value = independent_variable_forward_function(SAMPLE_VIRIAL_BETAMU_VALUE)
        functions_dict = eos_functions.get_balanced_eos_functions(independent_variable = independent_variable)
        for key in functions_dict:
            function = functions_dict[key]
            function_calculated_value = function(independent_variable_sample_value) 
            direct_function = eos_functions._get_balanced_eos_virial_function(key)
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
    _test_balanced_eos_virial_functions_helper("kappa_over_kappa0", eos_functions.balanced_kappa_over_kappa0_virial, rtol = 1e-1)

def test_balanced_P_over_P0_virial():
    _test_balanced_eos_virial_functions_helper("P_over_P0", eos_functions.balanced_P_over_P0_virial)

def test_balanced_Cv_over_NkB_virial():
    _test_balanced_eos_virial_functions_helper("Cv_over_NkB", eos_functions.balanced_Cv_over_NkB_virial, rtol = 1e-1)

def test_balanced_T_over_TF_virial():
    _test_balanced_eos_virial_functions_helper("T_over_TF", eos_functions.balanced_T_over_TF_virial)

def test_balanced_E_over_E0_virial():
    _test_balanced_eos_virial_functions_helper("E_over_E0", eos_functions.balanced_E_over_E0_virial)

def test_balanced_mu_over_EF_virial():
    _test_balanced_eos_virial_functions_helper("mu_over_EF", eos_functions.balanced_mu_over_EF_virial)

def test_balanced_S_over_NkB_virial():
    _test_balanced_eos_virial_functions_helper("S_over_NkB", eos_functions.balanced_S_over_NkB_virial)

def test_balanced_F_over_E0_virial():
    _test_balanced_eos_virial_functions_helper("F_over_E0", eos_functions.balanced_F_over_E0_virial)


def test_balanced_density_um():
    #At low temperatures, the Fermi energy should be equal to the chemical potential divided by the Bertsch parameter
    BERTSCH_PARAMETER = 0.370
    cold_betamu_values = np.linspace(3.9, 3.2, 100)
    SAMPLE_KBT_HZ_VALUE = 1337.0 
    sample_large_betamu_mu_values_Hz = cold_betamu_values * SAMPLE_KBT_HZ_VALUE
    expected_large_betamu_fermi_energies = sample_large_betamu_mu_values_Hz / BERTSCH_PARAMETER
    calculated_large_betamu_densities = eos_functions.balanced_density_um(cold_betamu_values, SAMPLE_KBT_HZ_VALUE)
    calculated_large_betamu_fermi_energies = eos_functions.fermi_energy_Hz_from_density_um(calculated_large_betamu_densities)
    assert np.all(np.isclose(expected_large_betamu_fermi_energies, calculated_large_betamu_fermi_energies, rtol = 5e-2))
    #At high temperatures, the density is given by the inverse thermal de Broglie wavelength times the fugacity 
    hot_betamu_values = np.linspace(-15, -8, 100)
    hot_kBT_Hz = 420
    hot_kBT_J = eos_functions.H_MKS * hot_kBT_Hz
    calculated_hot_densities_um = eos_functions.ideal_fermi_density_um(hot_betamu_values, hot_kBT_Hz)
    #The thermal de Broglie wavelength is tested separately
    hot_thermal_de_Broglie_m = eos_functions.thermal_de_broglie_mks(hot_kBT_J, eos_functions.LI_6_MASS_KG)
    hot_thermal_de_Broglie_um = hot_thermal_de_Broglie_m * 1e6
    hot_z_values = np.exp(hot_betamu_values)
    predicted_hot_densities_um = hot_z_values * np.power(hot_thermal_de_Broglie_um, -3.0)
    assert np.all(np.isclose(predicted_hot_densities_um, calculated_hot_densities_um, atol = 0.0, rtol = 2e-4))






def _test_ultralow_fugacity_function_helper(key, fun):
    betamu_value = -20.1
    full_virial_function = eos_functions._get_balanced_eos_virial_function(key) 
    other_value = full_virial_function(betamu_value)
    assert np.isclose(betamu_value, fun(other_value))

def test_ultralow_fugacity_betamu_function_P_over_P0():
    _test_ultralow_fugacity_function_helper("P_over_P0", eos_functions.ultralow_fugacity_betamu_function_P_over_P0)

def test_ultralow_fugacity_betamu_function_T_over_TF():
    _test_ultralow_fugacity_function_helper("T_over_TF", eos_functions.ultralow_fugacity_betamu_function_T_over_TF)

def test_ultralow_fugacity_betamu_function_E_over_E0():
    _test_ultralow_fugacity_function_helper("E_over_E0", eos_functions.ultralow_fugacity_betamu_function_E_over_E0)

def test_ultralow_fugacity_betamu_function_S_over_NkB():
    _test_ultralow_fugacity_function_helper("S_over_NkB", eos_functions.ultralow_fugacity_betamu_function_S_over_NkB)


def test_get_balanced_eos_betamu_from_other_value_virial_function():
    name_to_function_dict = {
        "P_over_P0":eos_functions.balanced_P_over_P0_virial, 
        "T_over_TF":eos_functions.balanced_T_over_TF_virial, 
        "E_over_E0":eos_functions.balanced_E_over_E0_virial, 
        "S_over_NkB":eos_functions.balanced_S_over_NkB_virial
    }
    min_betamu = -30
    max_betamu = -1
    num_points = 10000
    betamu_values = np.linspace(min_betamu, max_betamu, num_points)
    for key in name_to_function_dict:
        forward_function = name_to_function_dict[key]
        reverse_function = eos_functions._get_balanced_eos_betamu_from_other_value_virial_function(key) 
        other_values = forward_function(betamu_values) 
        restored_betamu_values = reverse_function(other_values) 
        assert np.all(np.isclose(restored_betamu_values, betamu_values))


def test_generate_and_save_balanced_eos_betamu_from_other_value_virial_data():
    min_betamu = -11
    max_betamu = -2
    num_points = 1000 
    name_to_function_dict = {
        "P_over_P0":eos_functions.balanced_P_over_P0_virial, 
        "T_over_TF":eos_functions.balanced_T_over_TF_virial, 
        "E_over_E0":eos_functions.balanced_E_over_E0_virial, 
        "S_over_NkB":eos_functions.balanced_S_over_NkB_virial
    }
    TEMP_WORKFOLDER_NAME = "temp"
    workfolder_pathname = os.path.join(RESOURCES_DIRECTORY_PATH, TEMP_WORKFOLDER_NAME)
    try:
        os.makedirs(workfolder_pathname)
        stored_data_filename = "Data_Temp.npy" 
        stored_metadata_filename = "Metadata_Temp.txt" 
        stored_data_pathname = os.path.join(workfolder_pathname, stored_data_filename)
        stored_metadata_pathname = os.path.join(workfolder_pathname, stored_metadata_filename) 
        eos_functions.generate_and_save_balanced_eos_betamu_from_other_value_virial_data(
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


polaron_eos_A = eos_functions.POLARON_EOS_A
polaron_mstar_over_m = eos_functions.POLARON_MSTAR_OVER_M

def test_polaron_eos_minority_density_um():
    sample_T = 1
    sample_mu_up = 1000
    #First test for agreement at very high fugacity
    sample_mu_down_cold = 50
    adjusted_mu_down_cold = sample_mu_down_cold - polaron_eos_A * sample_mu_up
    #Then expected density is just given by the adjusted Fermi energy, adjusted by the mass ratio
    expected_fermi_energy_down = adjusted_mu_down_cold
    expected_bare_density_down_cold = eos_functions.density_um_from_fermi_energy_Hz(expected_fermi_energy_down) 
    expected_density_down_cold = np.power(polaron_mstar_over_m, 1.5) * expected_bare_density_down_cold
    extracted_density_down_cold = eos_functions.polaron_eos_minority_density_um(sample_mu_up, sample_mu_down_cold, sample_T)
    assert np.isclose(expected_density_down_cold, extracted_density_down_cold)
    #Also test for agreement at very low fugacity
    sample_mu_down_hot = -15 + polaron_eos_A * sample_mu_up 
    adjusted_mu_down_hot = sample_mu_down_hot - polaron_eos_A * sample_mu_up 
    adjusted_betamu_down = adjusted_mu_down_hot / sample_T
    sample_T_J = sample_T * eos_functions.H_MKS
    de_broglie_wavelength_m = eos_functions.thermal_de_broglie_mks(sample_T_J, eos_functions.LI_6_MASS_KG)
    sample_z_down_hot = np.exp(adjusted_betamu_down)
    expected_bare_density_down_m_hot = np.power(de_broglie_wavelength_m, -3) * sample_z_down_hot 
    expected_bare_density_down_um_hot = expected_bare_density_down_m_hot * 1e-18 
    expected_density_down_hot = np.power(polaron_mstar_over_m, 1.5) * expected_bare_density_down_um_hot 
    extracted_density_down_hot = eos_functions.polaron_eos_minority_density_um(sample_mu_up, sample_mu_down_hot, sample_T) 
    assert np.isclose(expected_density_down_hot, extracted_density_down_hot, atol = 0.0)


def test_polaron_eos_majority_density_um():
    sample_T = 1000 
    sample_mu_up = 2000 
    sample_mu_down = -500
    betamu_up = sample_mu_up / sample_T
    expected_bare_density_up = eos_functions.ideal_fermi_density_um(betamu_up, sample_T)
    density_down = eos_functions.polaron_eos_minority_density_um(sample_mu_up, sample_mu_down, sample_T)
    expected_density_up = expected_bare_density_up - polaron_eos_A * density_down
    extracted_density_up = eos_functions.polaron_eos_majority_density_um(sample_mu_up, sample_mu_down, sample_T)
    assert np.isclose(expected_density_up, extracted_density_up)


def test_polaron_eos_pressure_Hz_um():
    #Just test that we've implemented the equation correctly 
    sample_T = 1000 
    sample_mu_up = 2000 
    sample_mu_down = -500 
    betamu_up = sample_mu_up / sample_T 
    mu_down_adjusted = sample_mu_down - polaron_eos_A * sample_mu_up
    betamu_down_adjusted = mu_down_adjusted / sample_T
    pressure_contribution_up = eos_functions.ideal_fermi_pressure_Hz_um(betamu_up, sample_T)
    pressure_contribution_down = np.power(polaron_mstar_over_m, 1.5) * eos_functions.ideal_fermi_pressure_Hz_um(betamu_down_adjusted, sample_T)
    expected_pressure = pressure_contribution_up + pressure_contribution_down 
    extracted_pressure = eos_functions.polaron_eos_pressure_Hz_um(sample_mu_up, sample_mu_down, sample_T)
    assert np.isclose(expected_pressure, extracted_pressure)

def test_polaron_eos_entropy_density_um():
    #Utilize thermodynamic relation Phi_G / V = -P for our gas, and also Phi_G / V = E - Ts - mu N 
    sample_T_Hz = 1000
    sample_mu_up_Hz = 2000 
    sample_mu_down_Hz = -500 
    sample_pressure_Hz_um = eos_functions.polaron_eos_pressure_Hz_um(sample_mu_up_Hz, sample_mu_down_Hz, sample_T_Hz)
    expected_grand_potential_density_Hz_um = -sample_pressure_Hz_um
    sample_minority_density_um = eos_functions.polaron_eos_minority_density_um(sample_mu_up_Hz, sample_mu_down_Hz, sample_T_Hz)
    sample_majority_density_um = eos_functions.polaron_eos_majority_density_um(sample_mu_up_Hz, sample_mu_down_Hz, sample_T_Hz)
    sample_entropy_density_um = eos_functions.polaron_eos_entropy_density_um(sample_mu_up_Hz, sample_mu_down_Hz, sample_T_Hz) 
    sample_energy_density_Hz_um = sample_pressure_Hz_um * 3/2 
    #Use definition of grand potential
    extracted_grand_potential_density_Hz_um = (sample_energy_density_Hz_um 
                                               - sample_T_Hz * sample_entropy_density_um - sample_minority_density_um * sample_mu_down_Hz 
                                               - sample_majority_density_um * sample_mu_up_Hz)
    assert np.isclose(extracted_grand_potential_density_Hz_um, expected_grand_potential_density_Hz_um) 



def test_polaron_eos_entropy_per_particle():
    sample_T = 1000 
    sample_mu_up = 2000
    sample_mu_down = -500
    betamu_up = sample_mu_up / sample_T 
    betamu_down = sample_mu_down / sample_T 
    majority_density_um = eos_functions.polaron_eos_majority_density_um(sample_mu_up, sample_mu_down, sample_T)
    minority_density_um = eos_functions.polaron_eos_minority_density_um(sample_mu_up, sample_mu_down, sample_T) 
    entropy_density_um = eos_functions.polaron_eos_entropy_density_um(sample_mu_up, sample_mu_down, sample_T) 
    expected_entropy_per_particle = entropy_density_um / (majority_density_um + minority_density_um)
    extracted_entropy_per_particle = eos_functions.polaron_eos_entropy_per_particle(betamu_up, betamu_down)
    assert np.isclose(expected_entropy_per_particle, extracted_entropy_per_particle)


def test_polaron_eos_minimum_pressure_zero_T_Hz_um():
    sample_mu_up = 2000 
    sample_mu_down = -500 
    sample_T = 1
    sample_density_up_um = eos_functions.polaron_eos_majority_density_um(sample_mu_up, sample_mu_down, sample_T) 
    sample_density_down_um = eos_functions.polaron_eos_minority_density_um(sample_mu_up, sample_mu_down, sample_T) 
    sample_pressure_Hz_um = eos_functions.polaron_eos_pressure_Hz_um(sample_mu_up, sample_mu_down, sample_T)
    extracted_minimum_pressure = eos_functions.polaron_eos_minimum_pressure_zero_T_Hz_um(sample_density_up_um, sample_density_down_um)
    assert np.isclose(extracted_minimum_pressure, sample_pressure_Hz_um)

def test_polaron_eos_minority_to_majority_ratio():
    sample_T = 1000 
    sample_mu_up = 2000
    sample_mu_down = -500
    betamu_up = sample_mu_up / sample_T 
    betamu_down = sample_mu_down / sample_T
    majority_density = eos_functions.polaron_eos_majority_density_um(sample_mu_up, sample_mu_down, sample_T)
    minority_density = eos_functions.polaron_eos_minority_density_um(sample_mu_up, sample_mu_down, sample_T) 
    expected_minority_to_ideal_majority_ratio = minority_density / majority_density 
    calculated_minority_to_ideal_majority_ratio = eos_functions.polaron_eos_minority_to_majority_ratio(
                                                            betamu_up, betamu_down)
    assert np.isclose(expected_minority_to_ideal_majority_ratio, calculated_minority_to_ideal_majority_ratio)


def test_polaron_eos_pressure_to_ideal_majority_pressure_ratio():
    sample_T = 1000
    sample_mu_up = 2000 
    sample_mu_down = -500 
    betamu_up = sample_mu_up / sample_T 
    betamu_down = sample_mu_down / sample_T
    majority_density = eos_functions.polaron_eos_majority_density_um(sample_mu_up, sample_mu_down, sample_T) 
    ideal_pressure = eos_functions.fermi_pressure_Hz_um_from_density_um(majority_density) 
    total_pressure = eos_functions.polaron_eos_pressure_Hz_um(sample_mu_up, sample_mu_down, sample_T)
    expected_pressure_to_ideal_pressure_ratio = total_pressure / ideal_pressure 
    calculated_pressure_to_ideal_pressure_ratio = eos_functions.polaron_eos_pressure_to_ideal_majority_pressure_ratio(betamu_up, betamu_down)
    print(expected_pressure_to_ideal_pressure_ratio) 
    print(calculated_pressure_to_ideal_pressure_ratio)
    assert np.isclose(expected_pressure_to_ideal_pressure_ratio, calculated_pressure_to_ideal_pressure_ratio)