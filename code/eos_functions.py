import numpy as np 
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.special import zeta, gamma


from . import numerical_functions, loading_functions


#Taken from https://jet.physics.ncsu.edu/techdocs/pdf/PropertiesOfLi.pdf
LI_6_MASS_KG = 9.98834e-27
#Taken from https://physics.nist.gov/cgi-bin/cuu/Value?hbar
H_BAR_MKS = 1.054572e-34


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

To agree with internal conventions, all functions are returned by default as functions of the logarithm betamu of the fugacity. 
They may instead be expressed in terms of T/T_F by passing the independent_variable kwarg. 

Parameters:

key: (str) A key which specifies only a single function to return. If None, all functions are returned as a dict. 
independent_variable: (str) If "betamu" (default), functions are returned as a function of betamu, the log of the fugacity. 
    If "T_over_TF", functions are instead returned with this as the independent variable. 




Returned functions:

kappa_over_kappa0: The compressibility kappa, divided by the compressibility for an ideal Fermi gas
P_over_P0: The pressure P, normalized again to an ideal Fermi gas 
Cv_over_NkB: The dimensionless per-particle heat capacity at constant volume 
T_over_TF: The reduced temperature. (Nontrivial only for returns in terms of betamu)
E_over_E0: The reduced energy. Please note that E0 is the total energy of the homogeneous Fermi gas, equal to 3/5 E_F
mu_over_EF: The reduced _single-species_ chemical potential. Here E_F is used. 
F_over_E0: The reduced free energy. Here E_0 is used. 
S_over_NkB: The dimensionless entropy per particle. 
betamu: The dimensionless chemical potential; the log of the fugacity. (Nontrivial only for returns in terms of T_over_TF)


"""
def get_balanced_eos_functions(key = None, independent_variable = "betamu"):
    #Hard-coded value of betamu below which to switch to high-temperature virial expansions of the equation of state
    VIRIAL_HANDOFF_BETAMU = -1.20
    independent_var_virial_function = _get_balanced_eos_virial_function(independent_variable)
    independent_var_virial_handoff = independent_var_virial_function(VIRIAL_HANDOFF_BETAMU)
    independent_var_to_betamu_virial_function = _get_balanced_eos_betamu_from_other_value_virial_function(independent_variable)
    ku_experimental_values_dict = loading_functions.load_unitary_EOS()
    independent_variable_values = ku_experimental_values_dict[independent_variable]
    #np.interp requires ascending x axis
    independent_ascending_sort_indices = np.argsort(independent_variable_values)
    sorted_independent_variable_values = independent_variable_values[independent_ascending_sort_indices]
    returned_function_dict = {}
    for dict_key in ku_experimental_values_dict:
        experimental_data = ku_experimental_values_dict[dict_key]
        sorted_experimental_data = experimental_data[independent_ascending_sort_indices]
        #closure to avoid issues with changing values of sorted_experimental_data
        interpolated_experimental_function = _interp_wrapper(sorted_independent_variable_values, sorted_experimental_data)
        virial_function = _get_balanced_eos_virial_function(dict_key)
        converted_virial_function = _virial_wrapper(virial_function, independent_var_to_betamu_virial_function)
        if independent_variable == "betamu":
            returned_function_dict[dict_key] = _smart_where_wrapper(independent_var_virial_handoff, 
                                                                    interpolated_experimental_function, converted_virial_function)
        else:
            returned_function_dict[dict_key] = _smart_where_wrapper(independent_var_virial_handoff, 
                                                                    converted_virial_function, interpolated_experimental_function)
    if key is None:
        return returned_function_dict
    else:
        return returned_function_dict[key]


#Wrappers for smart_where and interp to avoid issues with scoping in above
def _smart_where_wrapper(cutoff, *funcs):
    def wrapped(x):
        return numerical_functions.smart_where(x > cutoff, x, *funcs)
    return wrapped

def _interp_wrapper(independent, experimental):
    def interp_wrapped(x):
        return np.interp(x, independent, experimental)
    return interp_wrapped

def _virial_wrapper(virial_function, independent_variable_to_betamu):
    def wrapped(indep):
        return virial_function(independent_variable_to_betamu(indep)) 
    return wrapped

def balanced_eos_virial_f(z):
    #handle scalar input
    if np.ndim(z) == 0:
        reshaped_scalar = True
        z = np.expand_dims(z, 0)
    else:
        reshaped_scalar = False
    number_coeffs = len(BALANCED_GAS_VIRIAL_COEFFICIENTS)
    power_indices = np.arange(number_coeffs) + 1
    reshaped_power_indices = np.expand_dims(power_indices, 0)
    reshaped_z = np.expand_dims(z, 1)
    reshaped_coefficients = np.expand_dims(BALANCED_GAS_VIRIAL_COEFFICIENTS, 0)
    return_value = np.sum(reshaped_coefficients * np.power(reshaped_z, reshaped_power_indices), axis = 1)
    if reshaped_scalar:
        return np.squeeze(return_value, axis = 0)
    else:
        return return_value


def balanced_eos_virial_fprime(z):
    #Handle scalar input
    if np.ndim(z) == 0:
        reshaped_scalar = True
        z = np.expand_dims(z, 0)
    else:
        reshaped_scalar = False
    number_coeffs = len(BALANCED_GAS_VIRIAL_COEFFICIENTS)
    primed_coeffs = np.arange(1, number_coeffs + 1) * BALANCED_GAS_VIRIAL_COEFFICIENTS
    power_indices = np.arange(number_coeffs)
    reshaped_power_indices = np.expand_dims(power_indices, 0)
    reshaped_z = np.expand_dims(z, 1)
    reshaped_coefficients = np.expand_dims(primed_coeffs, 0)
    return_value = np.sum(reshaped_coefficients * np.power(reshaped_z, reshaped_power_indices), axis = 1)
    if reshaped_scalar:
        return np.squeeze(return_value, axis = 0)
    else:
        return return_value
    

#Returns a quantity equal to zf'(z) + z^2 f''(z) which occurs in the EOS calculations of second derivatives
def balanced_eos_virial_f_twice_deriv(z):
    #Handle scalar input
    if np.ndim(z) == 0:
        reshaped_scalar = True
        z = np.expand_dims(z, 0)
    else:
        reshaped_scalar = False
    number_coeffs = len(BALANCED_GAS_VIRIAL_COEFFICIENTS)
    primed_coeffs = np.square(np.arange(1, number_coeffs + 1)) * BALANCED_GAS_VIRIAL_COEFFICIENTS
    power_indices = np.arange(number_coeffs) + 1
    reshaped_power_indices = np.expand_dims(power_indices, 0)
    reshaped_z = np.expand_dims(z, 1)
    reshaped_coefficients = np.expand_dims(primed_coeffs, 0)
    return_value = np.sum(reshaped_coefficients * np.power(reshaped_z, reshaped_power_indices), axis = 1)
    if reshaped_scalar:
        return np.squeeze(return_value, axis = 0)
    else:
        return return_value


def balanced_kappa_over_kappa0_virial(betamu):
    z = np.exp(betamu)
    return np.cbrt(np.pi / 6) * balanced_eos_virial_f_twice_deriv(z) / np.power(z * balanced_eos_virial_fprime(z), 1/3)

def balanced_P_over_P0_virial(betamu):
    z = np.exp(betamu)
    return 10 / np.cbrt(36 * np.pi) * balanced_eos_virial_f(z) / np.power(z * balanced_eos_virial_fprime(z), 5/3)

def balanced_Cv_over_NkB_virial(betamu):
    return 3.0 / 2.0 * (1.0 / balanced_T_over_TF_virial(betamu)) * (balanced_P_over_P0_virial(betamu) - 1.0 / balanced_kappa_over_kappa0_virial(betamu))

def balanced_T_over_TF_virial(betamu):
    z = np.exp(betamu)
    return np.cbrt(16 / (9 * np.pi)) * np.power(z * balanced_eos_virial_fprime(z), -2/3)

def balanced_E_over_E0_virial(betamu):
    return balanced_P_over_P0_virial(betamu)

def balanced_mu_over_EF_virial(betamu):
    return betamu * balanced_T_over_TF_virial(betamu)

def balanced_F_over_E0_virial(betamu):
    z = np.exp(betamu)
    return -np.cbrt(2000/(243 * np.pi)) * (balanced_eos_virial_f(z) - z * betamu * balanced_eos_virial_fprime(z)) / np.power(balanced_eos_virial_fprime(z) * z, 5/3)

def balanced_S_over_NkB_virial(betamu):
    z = np.exp(betamu)
    return 2.5 * balanced_eos_virial_f(z) /(z * balanced_eos_virial_fprime(z))  - betamu

#Included for compatibility with non_betamu returns
def balanced_betamu_virial(betamu):
    return betamu


# def balanced_betamu_from_T_over_TF_virial(T_over_TF):
#     z_star = 4.0 / (3 * np.sqrt(np.pi)) * np.power(T_over_TF, -3/2) 
#     virial_fprime_coeffs_ascending_power = (np.arange(len(BALANCED_GAS_VIRIAL_COEFFICIENTS)) + 1) * BALANCED_GAS_VIRIAL_COEFFICIENTS
#     virial_fprime_coeffs_descending_power = np.flip(virial_fprime_coeffs_ascending_power)
#     #By manual inspection, the third root (order = 2) is the correct one for z_star < Z_STAR_ROOT_SWITCH, then the second root is correct for 
#     #z_star > Z_STAR_ROOT_SWITCH 
#     #Note that there is no continuity between the roots when they switch, so it's vital to specify the crossover point as precisely as possible.
#     Z_STAR_ROOT_SWITCH = 0.521065275
#     order_to_use = np.where(z_star < Z_STAR_ROOT_SWITCH, 2, 1)
#     virial_inverted_z_values = numerical_functions.cubic_formula(*virial_fprime_coeffs_descending_power, -z_star, 
#                                                                  cube_root_order = order_to_use, cast_to_real = True)
#     virial_inverted_betamu_values = np.log(virial_inverted_z_values) 
#     return virial_inverted_betamu_values

def _get_balanced_eos_virial_function(key):
    virial_function_dict = {
        "kappa_over_kappa0":balanced_kappa_over_kappa0_virial,
        "P_over_P0":balanced_P_over_P0_virial,
        "Cv_over_NkB":balanced_Cv_over_NkB_virial,
        "T_over_TF":balanced_T_over_TF_virial,
        "E_over_E0":balanced_E_over_E0_virial,
        "mu_over_EF":balanced_mu_over_EF_virial,
        "F_over_E0":balanced_F_over_E0_virial,
        "S_over_NkB":balanced_S_over_NkB_virial,
        "betamu":balanced_betamu_virial
    }
    return virial_function_dict[key]


#List of balanced eos values which can be back-converted to betamu
BALANCED_EOS_REVERSIBLE_VALUES_NAMES_LIST = ["P_over_P0", "T_over_TF", "E_over_E0", "S_over_NkB"]


def _get_balanced_eos_betamu_from_other_value_virial_function(key):
    if key == "betamu":
        return balanced_betamu_virial
    ULTRALOW_FUGACITY_DICT = {
        "P_over_P0":ultralow_fugacity_betamu_function_P_over_P0,
        "T_over_TF":ultralow_fugacity_betamu_function_T_over_TF,
        "E_over_E0":ultralow_fugacity_betamu_function_E_over_E0,
        "S_over_NkB":ultralow_fugacity_betamu_function_S_over_NkB
    }
    key_to_index_dict = {key:i+1 for i, key in enumerate(BALANCED_EOS_REVERSIBLE_VALUES_NAMES_LIST)}
    if not key in BALANCED_EOS_REVERSIBLE_VALUES_NAMES_LIST:
        raise ValueError("Unsupported variable for conversion to betamu")
    ultralow_fugacity_function = ULTRALOW_FUGACITY_DICT[key] 
    tabulated_virial_betamu_data_array = loading_functions.load_tabulated_unitary_eos_virial_betamu_data()
    #Betamu is in decreasing order
    tabulated_betamu_values = tabulated_virial_betamu_data_array[0] 
    tabulated_experimental_values = tabulated_virial_betamu_data_array[key_to_index_dict[key]] 
    def tabulated_betamu_function(other_value):
        return np.interp(other_value, tabulated_experimental_values, tabulated_betamu_values)
    #All experimental values which are currently available are increasing with decreasing betamu
    maximum_experimental_value = tabulated_experimental_values[-1] 
    def betamu_from_other_value_virial_func(other_value):
        return numerical_functions.smart_where(other_value > maximum_experimental_value, other_value, 
                                               ultralow_fugacity_function, tabulated_betamu_function)
    return betamu_from_other_value_virial_func


def generate_and_save_balanced_eos_betamu_from_other_value_virial_data(data_path, metadata_path, 
                                                                max_betamu = -1, min_betamu = -10, 
                                                                num_points = 10000):
    betamu_values = np.linspace(max_betamu, min_betamu, num_points)
    stacked_array = betamu_values
    for key in BALANCED_EOS_REVERSIBLE_VALUES_NAMES_LIST:
        virial_function = _get_balanced_eos_virial_function(key)
        other_values = virial_function(betamu_values) 
        stacked_array = np.vstack((stacked_array, other_values))
    np.save(data_path, stacked_array)
    with open(metadata_path, 'w') as f:
        f.write("Axis 0: parameter (e.g. betamu, P/P_0)\n")
        f.write("Axis 1: Value\n")
        f.write("Axis 1 is in order of decreasing betamu.\n")
        f.write("--------\n") 
        f.write("Maximum betamu value: {0:.1f}\n".format(max_betamu))
        f.write("Minimum betamu value: {0:.1f}\n".format(min_betamu))
        f.write("Number points: {0:d}\n".format(num_points))
        f.write("--------\n")
        f.write("Parameter ordering:\n")
        f.write("Betamu\n")
        for key in BALANCED_EOS_REVERSIBLE_VALUES_NAMES_LIST:
            f.write("{0}\n".format(key))



def ultralow_fugacity_betamu_function_P_over_P0(P_over_P0):
    z_star = np.power(np.cbrt(36 * np.pi) / 10 * P_over_P0, -3/2)
    return np.log(z_star)

def ultralow_fugacity_betamu_function_T_over_TF(T_over_TF):
    z_star = 4.0 / (3 * np.sqrt(np.pi)) * np.power(T_over_TF, -3/2) 
    return np.log(z_star) 

def ultralow_fugacity_betamu_function_E_over_E0(E_over_E0):
    return ultralow_fugacity_betamu_function_P_over_P0(E_over_E0)

def ultralow_fugacity_betamu_function_S_over_NkB(S_over_NkB):
    return 5/2 - S_over_NkB
