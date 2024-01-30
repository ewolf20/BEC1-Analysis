from re import L
import numpy as np 
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.signal import savgol_filter
from scipy.special import zeta, gamma

from . import statistics_functions, numerical_functions, loading_functions



#Taken from https://jet.physics.ncsu.edu/techdocs/pdf/PropertiesOfLi.pdf
LI_6_MASS_KG = 9.98834e-27
#Taken from https://physics.nist.gov/cgi-bin/cuu/Value?hbar
H_BAR_MKS = 1.054572e-34

#Data from https://jet.physics.ncsu.edu/techdocs/pdf/PropertiesOfLi.pdf
LI_HYPERFINE_CONSTANT_MHZ = 152.1368
LI_G_J = 2.0023010
LI_G_I = 0.0004476540
LI_I = 1
LI_F_PLUS = LI_I + 0.5

#Data from https://physics.nist.gov/cgi-bin/cuu/Value?mubshhz
BOHR_MAGNETON_IN_MHZ_PER_G = 1.3996245


#Data from https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.135301; Ku figure updated for Feshbach correction 
BERTSCH_PARAMETER = 0.370


#THERMODYNAMICS FUNCTIONS


#IDEAL FERMI GAS

#Homebrewed implementation of the f_minus function, defined in Kardar, "Statistical Physics of Particles", chapter 7.
#Observe that f_minus(s, z) = -polylog(s, -z). Observe also that this function takes the _log_ of z instead of the argument itself, 
#so as to be better behaved for large values of beta mu.

def kardar_f_minus_function(s, log_z):
    SOMMERFELD_EXPANSION_ORDER = 6
    SOMMERFELD_LOG_CUTOFF = 10
    POWER_SERIES_EXPANSION_ORDER = 70
    POWER_SERIES_LOG_CUTOFF = -0.1
    #Safety cast with minimal overhead for scalar log_z
    input_scalar = np.isscalar(log_z)
    if(input_scalar):
        log_z = np.atleast_1d(log_z)
    condition = np.zeros(log_z.shape, dtype = int)
    power_series_indices = (log_z < POWER_SERIES_LOG_CUTOFF)
    sommerfeld_series_indices = (log_z > SOMMERFELD_LOG_CUTOFF)
    other_indices = (np.logical_and(log_z >= POWER_SERIES_LOG_CUTOFF, log_z <= SOMMERFELD_LOG_CUTOFF))
    condition[power_series_indices] = 0 
    condition[other_indices] = 1
    condition[sommerfeld_series_indices] = 2
    kardar_f_minus_values = numerical_functions.smart_where(condition, log_z,
    lambda x: _kardar_highT_f_minus(s, POWER_SERIES_EXPANSION_ORDER, x),
    lambda x: _kardar_intermediateT_f_minus(s, x), 
    lambda x: _kardar_lowT_f_minus(s, SOMMERFELD_EXPANSION_ORDER, x)
    )
    #Undo safety cast
    if(input_scalar):
        return kardar_f_minus_values.item()
    else:
        return kardar_f_minus_values


#Implementation from large-z Sommerfeld expansion as given in Kardar.
def _kardar_lowT_f_minus(s, order, log_z):
    indices = np.arange(0, 2 * (order + 1), 2, dtype = float).reshape(1, order + 1)
    prefactor = np.power(log_z, s) / gamma(s + 1)
    reshaped_log_z = np.expand_dims(log_z, axis = 1) 
    summands = 2 * (1 - np.power(2.0, -indices + 1)) * zeta(indices) * gamma(s + 1) / gamma(s - indices + 1) * np.power(reshaped_log_z, -indices)
    return prefactor * np.sum(summands, axis = -1)


#Implementation from naive small-z expansion of polylog
def _kardar_highT_f_minus(s, order, logz):
    z = np.exp(logz) 
    indices = np.arange(1, order + 1, dtype = float).reshape(1, order)
    reshaped_z = np.expand_dims(z, axis = 1)
    summands = -np.power(-reshaped_z, indices) / np.power(indices, s)
    return np.sum(summands, axis = -1)


(polylog_analytic_continuation_centers, 
polylog_analytic_continuation_coeffs_3_2, 
polylog_analytic_continuation_coeffs_5_2) = loading_functions.load_polylog_analytic_continuation_parameters()

"""Implementation using Taylor expansions, with stored coefficients, about various relevant points for intermediate z."""
def _kardar_intermediateT_f_minus(s, logz):
    minus_z = -np.exp(logz)
    centers = polylog_analytic_continuation_centers
    if(s == 3/2):
        coeffs = polylog_analytic_continuation_coeffs_3_2
    elif(s == 5/2):
        coeffs = polylog_analytic_continuation_coeffs_5_2
    else:
        raise NotImplementedError("The fast analytic continuation implementation of the polylog is not supported for s != 3/2, 5/2.")
    return -numerical_functions.stored_coeffs_polylog_taylor_series(minus_z, centers, coeffs)



#FERMI GAS THERMODYNAMICS


def thermal_de_broglie_li_6_mks(kBT):
    return (2 * np.pi * H_BAR_MKS) / np.sqrt(2 * np.pi * LI_6_MASS_KG * kBT)

def ideal_fermi_P0(n, E_F):
    return 2 / 5 * n * E_F 

def ideal_fermi_kappa0(n, E_F):
    return 3 / (2 * n * E_F)


#Note: Only valid for box potentials
def ideal_fermi_E0_uniform(E_F):
    return 3/5 * E_F

#Derived from notes in Kardar, 'Statistical Physics of Particles', chapter 7
def ideal_fermi_P_over_p0(betamu):
    return 5.0 / 2.0 * 1.0 / (np.cbrt(9 * np.pi / 16)) * np.power(kardar_f_minus_function(3/2, betamu), -5/3) * (kardar_f_minus_function(5/2, betamu))


def ideal_T_over_TF(betamu):
    return 1.0 / (np.cbrt(9 * np.pi / 16) * np.power(kardar_f_minus_function(3/2, betamu), 2/3))

def _bruteforce_get_ideal_betamu_from_T_over_TF(T_over_TF):
    return fsolve(lambda x: ideal_T_over_TF(x) - T_over_TF, 0)

#Derived by slight alteration of Equation 64 of Cowan 2019: https://doi.org/10.1007/s10909-019-02228-0
def _low_T_get_ideal_betamu_from_T_over_TF(T_over_TF):
    COWAN_COEFFICIENTS = [1, -1/12, -1/80, 247/25920, -16291/777600] 
    indices = 2 * np.arange(len(COWAN_COEFFICIENTS))
    pi_T_over_TF = np.pi * T_over_TF 
    reshaped_pi_T_over_TF = np.expand_dims(pi_T_over_TF, axis = -1) 
    reshaped_coefficients = np.expand_dims(COWAN_COEFFICIENTS, tuple(np.arange(len(np.shape(pi_T_over_TF)))))
    reshaped_indices = np.expand_dims(indices, tuple(np.arange(len(np.shape(pi_T_over_TF)))))
    mu_over_EF = np.sum(reshaped_coefficients * np.power(reshaped_pi_T_over_TF, reshaped_indices), axis = -1)
    return mu_over_EF / T_over_TF

#From Equation 65 of Cowan 2019. Expansion of T/T_F fi
def _high_T_get_ideal_betamu_from_T_over_TF(T_over_TF):
    maxwell_term = T_over_TF * np.log(4.0 / (3.0 * np.sqrt(np.pi) * np.power(T_over_TF, 1.5)))
    COWAN_COEFFICIENTS = [1.0 / 3.0 * np.sqrt(2 / np.pi), 
                         -1.0 / (81 * np.pi) * (16 * np.sqrt(3) - 27), 
                        4 / (243 * np.power(np.pi, 1.5)) * (15 * np.sqrt(2) - 16 * np.sqrt(6) + 18)]
    T_powers = [-0.5, -2, -3.5]
    reshaped_coefficients = np.expand_dims(COWAN_COEFFICIENTS, tuple(np.arange(len(np.shape(T_over_TF)))))
    reshaped_T_over_TF = np.expand_dims(T_over_TF, axis = -1) 
    reshaped_T_powers = np.expand_dims(T_powers, tuple(np.arange(len(np.shape(T_over_TF)))))
    mu_over_EF = maxwell_term + np.sum(reshaped_coefficients * np.power(reshaped_T_over_TF, reshaped_T_powers), axis = -1)
    return mu_over_EF / T_over_TF


#Initialize globals to allow loading to be done once, and only if the relevant functions are used.
vectorized_bruteforce_get_ideal_betamu_from_T_over_TF = None
tabulated_ideal_betamu_interpolant = None

def get_ideal_betamu_from_T_over_TF(T_over_TF, flag = "direct"):
    LOW_T_CUTOFF = 0.01
    HIGH_T_CUTOFF = 5.0
    if(flag == "direct"):
        global vectorized_bruteforce_get_ideal_betamu_from_T_over_TF
        if(not vectorized_bruteforce_get_ideal_betamu_from_T_over_TF):
            vectorized_bruteforce_get_ideal_betamu_from_T_over_TF = np.vectorize(_bruteforce_get_ideal_betamu_from_T_over_TF, otypes = [float])
        input_scalar = np.isscalar(T_over_TF)
        if(input_scalar):
            T_over_TF = np.atleast_1d(T_over_TF)
        condition = np.zeros(T_over_TF.shape, dtype = int)
        low_T_indices = (T_over_TF < LOW_T_CUTOFF)
        high_T_indices = (T_over_TF > HIGH_T_CUTOFF)
        intermediate_T_indices = (np.logical_and(T_over_TF >= LOW_T_CUTOFF, T_over_TF <= HIGH_T_CUTOFF))
        condition[low_T_indices] = 0 
        condition[intermediate_T_indices] = 1
        condition[high_T_indices] = 2
        betamu_values = numerical_functions.smart_where(condition, T_over_TF,
        _low_T_get_ideal_betamu_from_T_over_TF,
        vectorized_bruteforce_get_ideal_betamu_from_T_over_TF, 
        _high_T_get_ideal_betamu_from_T_over_TF
        )
        #Undo safety cast
        if(input_scalar):
            return betamu_values.item()
        else:
            return betamu_values
    elif(flag == "tabulated"):
        global tabulated_ideal_betamu_interpolant
        if(not tabulated_ideal_betamu_interpolant):
            tabulated_T_over_TF, tabulated_ideal_betamu = loading_functions.load_tabulated_ideal_betamu_vs_T_over_TF()
            tabulated_ideal_betamu_interpolant = interp1d(tabulated_T_over_TF, tabulated_ideal_betamu, kind = "cubic")
        return tabulated_ideal_betamu_interpolant(T_over_TF)

def get_fermi_energy_hz_from_density(atom_density_m):
    fermi_k_m = np.cbrt(6 * np.square(np.pi) * atom_density_m)
    fermi_energy_J = np.square(H_BAR_MKS) * np.square(fermi_k_m) / (2 * LI_6_MASS_KG)
    fermi_energy_hz = fermi_energy_J / (2 * np.pi * H_BAR_MKS) 
    return fermi_energy_hz



#THERMODYNAMICS OF BALANCED UNITARY GAS

#Virial coefficients for high-temperature expansion of the grand potential of the unitary, balanced Fermi gas
#Taken from Liu et. al, PRL 102, 160401 (2009), and as cited in Ku 2012
#10.1103/PhysRevLett.102.160401
BALANCED_GAS_VIRIAL_COEFFICIENTS = np.array([1, 3 * np.sqrt(2) / 8, -0.29095295])


"""
Returns the experimentally determined and calculated EOS functions for a balanced unitary Fermi gas.

Returns a series of functions of thermodynamic quantities for the balanced Fermi gas, typically normalized by 
the corresponding values for a zero-temperature ideal Fermi gas. Optionally, a key may be passed in order to return only 
a single function; if no key is passed, all of the functions are returned as a dictionary.

To agree with internal conventions, all functions are returned as functions of the logarithm betamu of the fugacity. They may instead be 
expressed in terms of T/T_F by composing them with the separately provided inverse function for z in terms of T_F. 

Returned functions:

kappa_over_kappa0: The compressibility kappa, divided by the compressibility for an ideal Fermi gas
P_over_P0: The pressure P, normalized again to an ideal Fermi gas 
Cv_over_NkB: The dimensionless per-particle heat capacity at constant volume 
T_over_TF: The reduced temperature
E_over_E0: The reduced energy. Please note that E0 is the total energy of the homogeneous Fermi gas, equal to 3/5 E_F
mu_over_EF: The reduced _single-species_ chemical potential. Here E_F is used. 
F_over_E0: The reduced free energy. Here E_0 is used. 
S_over_NkB: The dimensionless entropy per particle. 


"""
def get_balanced_eos_functions(key = None):
    #Hard-coded value of beta mu below which to switch to high-temperature virial expansions of the equation of state
    VIRIAL_HANDOFF_BETAMU = -1.1
    virial_function_dict = {
        "kappa_over_kappa0":balanced_kappa_over_kappa0_virial,
        "P_over_P0":balanced_P_over_P0_virial,
        "Cv_over_NkB":balanced_Cv_over_NkB_virial,
        "T_over_TF":balanced_T_over_TF_virial,
        "E_over_E0":balanced_E_over_E0_virial,
        "mu_over_EF":balanced_mu_over_EF_virial,
        "F_over_E0":balanced_F_over_E0_virial,
        "S_over_NkB":balanced_S_over_NkB_virial
    }
    ku_experimental_values_dict = loading_functions.load_unitary_EOS()
    betamu_values = ku_experimental_values_dict["betamu"]
    returned_function_dict = {}
    for dict_key in ku_experimental_values_dict:
        if dict_key != "betamu":
            experimental_data = ku_experimental_values_dict[dict_key]
            interpolated_experimental_function = lambda x: np.interp(x, betamu_values, experimental_data)
            virial_function = virial_function_dict[dict_key]
            def virial_extended_function(betamu):
                return numerical_functions.smart_where(betamu > VIRIAL_HANDOFF_BETAMU, betamu, 
                                                       interpolated_experimental_function, virial_function)
            returned_function_dict[dict_key] = virial_extended_function
    if key is None:
        return returned_function_dict
    else:
        return returned_function_dict[key]
    

def balanced_eos_virial_f(z):
    number_coeffs = len(BALANCED_GAS_VIRIAL_COEFFICIENTS)
    power_indices = np.arange(number_coeffs) + 1
    reshaped_power_indices = np.expand_dims(power_indices, 0)
    reshaped_z = np.expand_dims(z, 1)
    reshaped_coefficients = np.expand_dims(BALANCED_GAS_VIRIAL_COEFFICIENTS, 0)
    return np.sum(reshaped_coefficients * np.power(reshaped_z, reshaped_power_indices), axis = 1)


def balanced_eos_virial_fprime(z):
    number_coeffs = len(BALANCED_GAS_VIRIAL_COEFFICIENTS)
    primed_coeffs = np.arange(1, number_coeffs + 1) * BALANCED_GAS_VIRIAL_COEFFICIENTS
    power_indices = np.arange(number_coeffs)
    reshaped_power_indices = np.expand_dims(power_indices, 0)
    reshaped_z = np.expand_dims(z, 1)
    reshaped_coefficients = np.expand_dims(primed_coeffs, 0)
    return np.sum(reshaped_coefficients * np.power(reshaped_z, reshaped_power_indices), axis = 1)


def balanced_kappa_over_kappa0_virial(betamu):
    return 0.0


def balanced_P_over_P0_virial(betamu):
    z = np.exp(betamu)
    return 10 / np.cbrt(36 * np.pi) * balanced_eos_virial_f(z) / np.power(z * balanced_eos_virial_fprime(z), 5/3)

def balanced_Cv_over_NkB_virial(betamu):
    return 0.0

def balanced_T_over_TF_virial(betamu):
    z = np.exp(betamu)
    return np.cbrt(16 / (9 * np.pi)) * np.power(z * balanced_eos_virial_fprime(z), -2/3)

def balanced_E_over_E0_virial(betamu):
    return balanced_P_over_P0_virial(betamu)

def balanced_mu_over_EF_virial(betamu):
    return 0.0 

def balanced_F_over_E0_virial(betamu):
    return 0.0 

def balanced_S_over_NkB_virial(betamu):
    return 0.0


#FUNCTIONS FOR CALCULATIONS IN BOX AND HYBRID TRAP

def get_box_fermi_energy_from_counts(atom_counts, box_cross_section_um, box_length_um):
    box_volume_m = box_cross_section_um * box_length_um * 1e-18
    atom_density_m = atom_counts / box_volume_m
    return get_fermi_energy_hz_from_density(atom_density_m)


def get_hybrid_trap_total_energy(harmonic_trap_positions_um, three_d_density_trap_profile_um, trap_cross_section_um, trap_freq, autocut = False, 
                                autocut_mode = "statistics"):
    if autocut:
        start_index, stop_index = hybrid_trap_autocut(three_d_density_trap_profile_um, mode = autocut_mode)
        harmonic_trap_positions_um = harmonic_trap_positions_um[start_index:stop_index]
        three_d_density_trap_profile_um = three_d_density_trap_profile_um[start_index:stop_index]
    harmonic_trap_energies = get_li_energy_hz_in_1D_trap(harmonic_trap_positions_um * 1e-6, trap_freq)
    total_potential_energy = trapezoid(harmonic_trap_energies * three_d_density_trap_profile_um * trap_cross_section_um, x = harmonic_trap_positions_um)
    #1D Virial theorem; see Zhenjie Yan's PhD thesis
    #Formula still holds for an imbalanced cloud
    total_energy = total_potential_energy * 4
    return total_energy


def get_hybrid_trap_average_energy(harmonic_trap_positions_um, three_d_density_trap_profile_um, trap_cross_section_um, trap_freq, autocut = False, 
                                    autocut_mode = "statistics"):
    total_energy = get_hybrid_trap_total_energy(harmonic_trap_positions_um, three_d_density_trap_profile_um, trap_cross_section_um, trap_freq, 
                                        autocut = autocut, autocut_mode = autocut_mode)
    total_counts = trapezoid(trap_cross_section_um * three_d_density_trap_profile_um, x = harmonic_trap_positions_um)
    return total_energy / total_counts

"""
Helper function for autocutting the hybrid trap data to avoid wings where it is zero."""
def hybrid_trap_autocut(three_d_density_trap_profile_um, mode = "statistics"):
    AUTOCUT_WINDOW = 101
    AUTOCUT_SAVGOL_ORDER = 2
    data_length = len(three_d_density_trap_profile_um)
    middle_index = int(np.floor(data_length / 2))
    if(mode == "statistics"):
        #Statistics-based autocut
        window_start = (-AUTOCUT_WINDOW + 1) // 2
        window_end = (AUTOCUT_WINDOW + 1) // 2
        window_range = np.arange(window_start, window_end).reshape(1, AUTOCUT_WINDOW)
        data_range = np.arange(data_length).reshape(data_length, 1) 
        #Hack that takes advantage of numpy broadcasting; final shape is (data_length, AUTOCUT_WINDOW)
        window_indices = window_range + data_range
        #Implement edge strategy analogous to "mirror" for savgol filter
        window_indices = np.where(window_indices < 0, np.abs(window_indices), window_indices)
        window_indices = np.where(window_indices > data_length - 1, 2 * (data_length - 1) - window_indices, window_indices)
        data_window_array = three_d_density_trap_profile_um[window_indices]
        data_window_is_nonzero_array = statistics_functions.mean_location_test(data_window_array, 0.0, axis = -1)
    elif(mode == "savgol"):
        filtered_data = savgol_filter(three_d_density_trap_profile_um, AUTOCUT_WINDOW, AUTOCUT_SAVGOL_ORDER)
        data_window_is_nonzero_array = filtered_data > 0.0
    else:
        raise RuntimeError("Unsupported mode for hybrid trap autocut")
    data_window_is_nonzero_first_half = data_window_is_nonzero_array[:middle_index] 
    data_window_is_nonzero_second_half = data_window_is_nonzero_array[middle_index:] 
    first_half_is_zero_indices, = (~data_window_is_nonzero_first_half).nonzero()
    if(len(first_half_is_zero_indices) > 0):
        last_first_half_zero_index = first_half_is_zero_indices[-1]
    else:
        last_first_half_zero_index = -1
    second_half_is_zero_indices = (~data_window_is_nonzero_second_half).nonzero()[0]
    if(len(second_half_is_zero_indices) > 0):
        first_second_half_zero_index = second_half_is_zero_indices[0] + middle_index
    else:
        first_second_half_zero_index = data_length
    start_index = last_first_half_zero_index + 1 
    stop_index_exclusive = first_second_half_zero_index
    return (start_index, stop_index_exclusive)

def get_hybrid_trap_compressibilities(harmonic_trap_positions_um, three_d_density_trap_profile_um, trap_freq, 
                                        energy_cutoff_hz = 10000):
    ENERGY_BIN_NUMBER = 100 
    SAVGOL_FILTER_WINDOW_LENGTH = 20
    SAVGOL_FILTER_POLYORDER = 2
    fermi_energies = get_fermi_energy_hz_from_density(three_d_density_trap_profile_um * 1e18)
    harmonic_energies = get_li_energy_hz_in_1D_trap(harmonic_trap_positions_um * 1e-6, trap_freq)
    energy_bins = np.linspace(0, energy_cutoff_hz, ENERGY_BIN_NUMBER)
    delta_E_bin = energy_bins[1] - energy_bins[0]
    #Minus 1 adopts the convention that index i is assigned to a value satisfying bins[i] <= val < bins[i + 1]
    bin_indices = np.digitize(harmonic_energies, energy_bins) - 1
    average_fermi_energies = np.zeros(ENERGY_BIN_NUMBER)
    fermi_energy_errors = np.zeros(ENERGY_BIN_NUMBER)
    for i in range(ENERGY_BIN_NUMBER):
        indices_for_current_bin = np.where((bin_indices) == i)
        current_bin_slice = fermi_energies[indices_for_current_bin]
        current_bin_average = np.sum(current_bin_slice) / current_bin_slice.size
        current_bin_standard_error_mean = np.sqrt(np.sum(np.square(current_bin_slice - current_bin_average))) / current_bin_slice.size
        fermi_energy_errors[i] = current_bin_standard_error_mean
        average_fermi_energies[i] = current_bin_average
    displacement_bins = get_li_displacement_um_from_1D_trap_energy(energy_bins, trap_freq)
    #TODO: Consider using the fermi energy errors in the savgol numerical differentiation.
    compressibilities = - savgol_filter(average_fermi_energies, SAVGOL_FILTER_WINDOW_LENGTH, SAVGOL_FILTER_POLYORDER, deriv = 1, delta = delta_E_bin)
    return (displacement_bins, compressibilities)

def get_li_energy_hz_in_1D_trap(displacement_m, trap_freq_hz):
    li_energy_mks = 0.5 * LI_6_MASS_KG * np.square(2 * np.pi * trap_freq_hz) * np.square(displacement_m)
    li_energy_hz = li_energy_mks / (2 * np.pi * H_BAR_MKS)
    return li_energy_hz

def get_li_energy_gradient_hz_um_in_1D_trap(displacement_m, trap_freq_hz):
    li_force_mks = LI_6_MASS_KG * np.square(2 * np.pi * trap_freq_hz) * displacement_m
    li_force_hz_per_um = li_force_mks / (2 * np.pi * H_BAR_MKS) * 1e-6
    return li_force_hz_per_um

def get_li_displacement_um_from_1D_trap_energy(li_energy_hz, trap_freq_hz):
    li_energy_mks = li_energy_hz * (2 * np.pi * H_BAR_MKS)
    displacement_m = np.sqrt(2 * li_energy_mks / (LI_6_MASS_KG * np.square(2 * np.pi * trap_freq_hz)))
    return displacement_m * 1e6


#FUNCTIONS FOR RF SPECTROSCOPY

#By convention, uses kHz as the base unit.
def two_level_system_population_rabi(t, omega_r, detuning):
    generalized_rabi = np.sqrt(np.square(omega_r) + np.square(detuning))
    population_excited = np.square(omega_r) / np.square(generalized_rabi) * np.square(np.sin(generalized_rabi / 2 * t))
    return np.array([1.0 - population_excited, population_excited])

"""
Function for getting the energy of the ground state hyperfine manifold of lithium.

Given a field in gauss and a state index (indices 1-6, labelling energies from least to greatest, 
return the energy of the state in MHz """
def get_li6_br_energy_MHz(field_in_gauss, state_index):
    m, plus_minus_bool = _convert_li_state_index_to_br_notation(state_index)
    dimensionless_field = _convert_gauss_to_li_x(field_in_gauss)
    dimensionless_energy = _breit_rabi_function(dimensionless_field, m, plus_minus_bool, LI_G_I, LI_G_J, LI_F_PLUS)
    energy_MHz = dimensionless_energy * LI_HYPERFINE_CONSTANT_MHZ
    return energy_MHz

"""
Given a resonance frequency between two Breit-Rabi states of 6Li, return the field in Gauss."""
def get_field_from_li6_resonance(resonance_MHz, states_tuple, initial_guess_gauss = 690):
    state_A_index, state_B_index = states_tuple
    m_A, plus_minus_bool_A = _convert_li_state_index_to_br_notation(state_A_index)
    m_B, plus_minus_bool_B = _convert_li_state_index_to_br_notation(state_B_index)
    def wrapped_splitting_function(x):
        dimless_state_A_energy = _breit_rabi_function(x, m_A, plus_minus_bool_A, LI_G_I, LI_G_J, LI_F_PLUS)
        dimless_state_B_energy = _breit_rabi_function(x, m_B, plus_minus_bool_B, LI_G_I, LI_G_J, LI_F_PLUS)
        return LI_HYPERFINE_CONSTANT_MHZ * np.abs(dimless_state_A_energy - dimless_state_B_energy) - resonance_MHz
    optimal_x = fsolve(wrapped_splitting_function, _convert_gauss_to_li_x(initial_guess_gauss))[0]
    return _convert_li_x_to_gauss(optimal_x)

def _breit_rabi_function(x, m, plus_minus_bool, g_I, g_J, F_plus):
    if(m == F_plus):
        return _breit_rabi_stretched_plus(x, g_I, g_J, F_plus)
    elif(m == -F_plus):
        return _breit_rabi_stretched_minus(x, g_I, g_J, F_plus)
    elif(plus_minus_bool):
        return -1.0 / 4.0 - g_I / (g_I + g_J) * F_plus * x * m + F_plus / 2.0 * np.sqrt(1 + 2 * m * x / F_plus + np.square(x)) 
    else:
        return -1.0 / 4.0 - g_I / (g_I + g_J) * F_plus * x * m - F_plus / 2.0 * np.sqrt(1 + 2 * m * x / F_plus + np.square(x)) 

def _breit_rabi_stretched_plus(x, g_I, g_J, F_plus):
    return -1.0 / 4.0 - g_I / (g_I + g_J) * F_plus * x * F_plus + F_plus / 2.0 * (1 + x) 

def _breit_rabi_stretched_minus(x, g_I, g_J, F_plus):
    return -1.0 / 4.0 + g_I / (g_I + g_J) * F_plus * x * F_plus + F_plus / 2.0 * (1 - x) 

def _convert_gauss_to_li_x(field_in_gauss):
    return field_in_gauss * (BOHR_MAGNETON_IN_MHZ_PER_G) * (LI_G_I + LI_G_J) / (LI_F_PLUS * LI_HYPERFINE_CONSTANT_MHZ)

def _convert_li_x_to_gauss(x):
    return x * (LI_F_PLUS * LI_HYPERFINE_CONSTANT_MHZ) / (BOHR_MAGNETON_IN_MHZ_PER_G * (LI_G_I + LI_G_J))


def _convert_li_state_index_to_br_notation(index):
    if(index == 1):
        return (0.5, False) 
    elif(index == 2):
        return (-0.5, False) 
    elif(index == 3):
        return (-1.5, False) 
    elif(index == 4):
        return (-0.5, True) 
    elif(index == 5):
        return (0.5, True) 
    elif(index == 6):
        return (1.5, True)