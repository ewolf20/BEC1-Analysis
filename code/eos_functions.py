import numpy as np 
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.special import zeta, gamma


from . import numerical_functions, loading_functions


#Taken from https://jet.physics.ncsu.edu/techdocs/pdf/PropertiesOfLi.pdf
LI_6_MASS_KG = 9.98834e-27
#Taken from https://physics.nist.gov/cgi-bin/cuu/Value?hbar
H_BAR_MKS = 1.054572e-34
H_MKS = 2 * np.pi * H_BAR_MKS
KB_MKS = 1.380649e-23


#IDEAL FERMI GAS

#Homebrewed implementation of the f_minus function, defined in Kardar, "Statistical Physics of Particles", chapter 7.
#Observe that f_minus(s, z) = -polylog(s, -z). Observe also that this function takes the _log_ of z instead of the argument itself, 
#so as to be better behaved for large values of beta mu.

def kardar_f_minus_function(s, log_z):
    SOMMERFELD_EXPANSION_ORDER = 6
    SOMMERFELD_LOG_CUTOFF = 12
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
polylog_analytic_continuation_coeffs_1_2, 
polylog_analytic_continuation_coeffs_3_2, 
polylog_analytic_continuation_coeffs_5_2) = loading_functions.load_polylog_analytic_continuation_parameters()

"""Implementation using Taylor expansions, with stored coefficients, about various relevant points for intermediate z."""
def _kardar_intermediateT_f_minus(s, logz):
    minus_z = -np.exp(logz)
    centers = polylog_analytic_continuation_centers
    if s == 1/2:
        coeffs = polylog_analytic_continuation_coeffs_1_2
    elif s == 3/2:
        coeffs = polylog_analytic_continuation_coeffs_3_2
    elif s == 5/2:
        coeffs = polylog_analytic_continuation_coeffs_5_2
    else:
        raise NotImplementedError("The fast analytic continuation implementation of the polylog is not supported for s != 1/2, 3/2, 5/2.")
    return -numerical_functions.stored_coeffs_polylog_taylor_series(minus_z, centers, coeffs)



#IDEAL FERMI GAS THERMODYNAMICS

def ideal_fermi_f(betamu):
    return kardar_f_minus_function(5/2, betamu) 

def ideal_fermi_f_once_deriv(betamu):
    return kardar_f_minus_function(3/2, betamu) 

def ideal_fermi_f_twice_deriv(betamu):
    return kardar_f_minus_function(1/2, betamu)





def thermal_de_broglie_mks(kBT_J, mass_kg):
    return (2 * np.pi * H_BAR_MKS) / np.sqrt(2 * np.pi * mass_kg * kBT_J)

def ideal_fermi_P0(n, E_F):
    return 2 / 5 * n * E_F 

def ideal_fermi_kappa0(n, E_F):
    return 3 / (2 * n * E_F)

#Function for calculating the density of an ideal Fermi gas vs. betamu and T. 
#Because of a lack of normalization, this takes two parameters betamu and T, and requires the species to be 
#specified, so that the mass is known.
def ideal_fermi_density_um(betamu, kBT_Hz, species = "6Li"):
    if species == "6Li":
        mass_kg = LI_6_MASS_KG
    else:
        raise ValueError("Unsupported species")
    kBT_J = kBT_Hz * H_MKS
    thermal_de_broglie_m = thermal_de_broglie_mks(kBT_J, mass_kg)
    thermal_de_broglie_um = thermal_de_broglie_m * 1e6
    return np.power(thermal_de_broglie_um, -3.0) * ideal_fermi_f_once_deriv(betamu)


#Note: Only valid for box potentials
def ideal_fermi_E0_uniform(E_F):
    return 3/5 * E_F


def get_ideal_eos_functions(key = None, independent_variable = "betamu"):
    independent_variable_to_betamu_function = _get_ideal_eos_independent_to_betamu_function(independent_variable)
    returned_function_dict = {}
    for dict_key in IDEAL_EOS_FUNCTION_DICT:
        ideal_eos_function = IDEAL_EOS_FUNCTION_DICT[dict_key]
        converted_ideal_function = _variable_change_wrapper(ideal_eos_function, independent_variable_to_betamu_function)
        returned_function_dict[dict_key] = converted_ideal_function
    if key is None:
        return returned_function_dict
    else:
        return returned_function_dict[key]

#Derived from notes in Kardar, 'Statistical Physics of Particles', chapter 7
def ideal_fermi_kappa_over_kappa0(betamu):
    return _kappa_over_kappa0_f(betamu, ideal_fermi_f_once_deriv, ideal_fermi_f_twice_deriv)

def ideal_fermi_P_over_P0(betamu):
    return _P_over_P0_f(betamu, ideal_fermi_f, ideal_fermi_f_once_deriv)

def ideal_fermi_Cv_over_NkB(betamu):
    return _Cv_over_NkB_f(betamu, ideal_fermi_f, ideal_fermi_f_once_deriv, ideal_fermi_f_twice_deriv)

def ideal_fermi_T_over_TF(betamu):
    return _T_over_TF_f(betamu, ideal_fermi_f_once_deriv)

def ideal_fermi_E_over_E0(betamu):
    return _E_over_E0_f(betamu, ideal_fermi_f, ideal_fermi_f_once_deriv)

def ideal_fermi_mu_over_EF(betamu):
    return _mu_over_EF_f(betamu, ideal_fermi_f_once_deriv)

def ideal_fermi_F_over_E0(betamu):
    return _F_over_E0_f(betamu, ideal_fermi_f, ideal_fermi_f_once_deriv)

def ideal_fermi_S_over_NkB(betamu):
    return _S_over_NkB_f(betamu, ideal_fermi_f, ideal_fermi_f_once_deriv)

def ideal_fermi_betamu(betamu):
    return betamu

IDEAL_EOS_FUNCTION_DICT = {
    "kappa_over_kappa0":ideal_fermi_kappa_over_kappa0,
    "P_over_P0":ideal_fermi_P_over_P0,
    "Cv_over_NkB":ideal_fermi_Cv_over_NkB,
    "T_over_TF":ideal_fermi_T_over_TF, 
    "E_over_E0":ideal_fermi_E_over_E0,
    "mu_over_EF":ideal_fermi_mu_over_EF, 
    "F_over_E0":ideal_fermi_F_over_E0, 
    "S_over_NkB":ideal_fermi_S_over_NkB,
    "betamu":ideal_fermi_betamu
}


#Tabulate the values of various independent variables in terms of betamu, for numerical inversion 
#Inelegant, but sufficiently precise for our needs at a relatively trim 1 MB of data
def _get_ideal_eos_independent_to_betamu_function(key):
    if key == "betamu":
        return ideal_fermi_betamu
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
    #Betamu is in decreasing order
    tabulated_data = loading_functions.load_tabulated_ideal_eos_betamu_data()
    tabulated_betamu_values = tabulated_data[0] 
    tabulated_experimental_values = tabulated_data[key_to_index_dict[key]] 
    def tabulated_betamu_function(other_value):
        return np.interp(other_value, tabulated_experimental_values, tabulated_betamu_values)
    #All experimental values which are currently available are increasing with decreasing betamu
    maximum_experimental_value = tabulated_experimental_values[-1] 
    def betamu_from_other_value_func(other_value):
        return numerical_functions.smart_where(other_value > maximum_experimental_value, other_value, 
                                               ultralow_fugacity_function, tabulated_betamu_function)
    return betamu_from_other_value_func


def generate_and_save_ideal_eos_betamu_from_other_value_data(data_path, metadata_path, 
                                                                max_betamu = 20, min_betamu = -20, 
                                                                num_points = 10000):
    betamu_values = np.linspace(max_betamu, min_betamu, num_points)

    stacked_array = betamu_values
    for key in BALANCED_EOS_REVERSIBLE_VALUES_NAMES_LIST:
        ideal_eos_function = IDEAL_EOS_FUNCTION_DICT[key]
        eos_values = ideal_eos_function(betamu_values) 
        stacked_array = np.vstack((stacked_array, eos_values))
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
        converted_virial_function = _variable_change_wrapper(virial_function, independent_var_to_betamu_virial_function)
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

def _variable_change_wrapper(function_of_betamu, independent_variable_to_betamu):
    def wrapped(indep):
        return function_of_betamu(independent_variable_to_betamu(indep)) 
    return wrapped

#Function f(z) occurring in calculations of thermodynamic quantities. While f(z) is by definition a function of the fugacity 
#z, the input is betamu = log(z) for numerical stability
def balanced_eos_virial_f(betamu):
    z = np.exp(betamu)
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


#Returns a quantity equal to zf'(z) which occurs in the EOS calculations of first derivatives of the potential
def balanced_eos_virial_f_once_deriv(betamu):
    z = np.exp(betamu)
    #Handle scalar input
    if np.ndim(z) == 0:
        reshaped_scalar = True
        z = np.expand_dims(z, 0)
    else:
        reshaped_scalar = False
    number_coeffs = len(BALANCED_GAS_VIRIAL_COEFFICIENTS)
    primed_coeffs = np.arange(1, number_coeffs + 1) * BALANCED_GAS_VIRIAL_COEFFICIENTS
    power_indices = np.arange(number_coeffs) + 1
    reshaped_power_indices = np.expand_dims(power_indices, 0)
    reshaped_z = np.expand_dims(z, 1)
    reshaped_coefficients = np.expand_dims(primed_coeffs, 0)
    return_value = np.sum(reshaped_coefficients * np.power(reshaped_z, reshaped_power_indices), axis = 1)
    if reshaped_scalar:
        return np.squeeze(return_value, axis = 0)
    else:
        return return_value
    

#Returns a quantity equal to zf'(z) + z^2 f''(z) which occurs in the EOS calculations of second derivatives
def balanced_eos_virial_f_twice_deriv(betamu):
    z = np.exp(betamu)
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


_balanced_density_T_over_TF_vs_betamu_func = None
#Function for calculating the density of a balanced unitary Fermi gas vs. betamu and kBT. 
#As for the ideal gas, requires the species to be specified

#NOTE: Temp band-aid fix for balanced density fitting
ku_experimental_values_dict = loading_functions.load_unitary_EOS()
ku_maximum_betamu = ku_experimental_values_dict["betamu"][0] 
ku_maximum_betamu_mu = ku_experimental_values_dict["mu_over_EF"][0]

def balanced_density_um(betamu, kBT_Hz, species = "6Li"):
    if species == "6Li":
        mass_kg = LI_6_MASS_KG
    else:
        raise ValueError("Unsupported species")
    global _balanced_density_T_over_TF_vs_betamu_func
    if _balanced_density_T_over_TF_vs_betamu_func is None:
        _balanced_density_T_over_TF_vs_betamu_func =  get_balanced_eos_functions("T_over_TF")

    T_over_TF_values = np.where(betamu <= ku_maximum_betamu, _balanced_density_T_over_TF_vs_betamu_func(betamu), ku_maximum_betamu_mu / betamu)
    # T_over_TF_values = _balanced_density_T_over_TF_vs_betamu_func(betamu)
    kBT_J = kBT_Hz * H_MKS
    thermal_de_broglie_m = thermal_de_broglie_mks(kBT_J, mass_kg)
    thermal_de_broglie_um = thermal_de_broglie_m * 1e6
    return np.power(thermal_de_broglie_um, -3.0) * 4.0 / (3 * np.sqrt(np.pi)) * np.power(T_over_TF_values, -1.5)



def balanced_kappa_over_kappa0_virial(betamu):
    return _kappa_over_kappa0_f(betamu, balanced_eos_virial_f_once_deriv, balanced_eos_virial_f_twice_deriv)

def balanced_P_over_P0_virial(betamu):
    return _P_over_P0_f(betamu, balanced_eos_virial_f, balanced_eos_virial_f_once_deriv)

def balanced_Cv_over_NkB_virial(betamu):
    return _Cv_over_NkB_f(betamu, balanced_eos_virial_f, balanced_eos_virial_f_once_deriv, balanced_eos_virial_f_twice_deriv)

def balanced_T_over_TF_virial(betamu):
    return _T_over_TF_f(betamu, balanced_eos_virial_f_once_deriv)

def balanced_E_over_E0_virial(betamu):
    return _E_over_E0_f(betamu, balanced_eos_virial_f, balanced_eos_virial_f_once_deriv)

def balanced_mu_over_EF_virial(betamu):
    return _mu_over_EF_f(betamu, balanced_eos_virial_f_once_deriv)

def balanced_F_over_E0_virial(betamu):
    return _F_over_E0_f(betamu, balanced_eos_virial_f, balanced_eos_virial_f_once_deriv)

def balanced_S_over_NkB_virial(betamu):
    return _S_over_NkB_f(betamu, balanced_eos_virial_f, balanced_eos_virial_f_once_deriv)

#Included for compatibility with non_betamu returns
def balanced_betamu_virial(betamu):
    return _betamu_f(betamu)


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



#General functions for various normalized thermodynamic quantities in terms of a function f, defined such that 
#   Phi_G = -k_B T * V / (lambda_dB^3) f(z)
#   with z = e^betamu, Phi_G the grand potential

def _kappa_over_kappa0_f(betamu, f_once_deriv_func, f_twice_deriv_func):
    return np.cbrt(np.pi / 6) * f_twice_deriv_func(betamu) / np.power(f_once_deriv_func(betamu), 1/3)

def _P_over_P0_f(betamu, f_func, f_once_deriv_func):
    return 10 / np.cbrt(36 * np.pi) * f_func(betamu) / np.power(f_once_deriv_func(betamu), 5/3)

def _Cv_over_NkB_f(betamu, f_func, f_once_deriv_func, f_twice_deriv_func):
    prefactor = 3.0 / 2.0 * (1.0 / _T_over_TF_f(betamu, f_once_deriv_func))
    terms = (_P_over_P0_f(betamu, f_func, f_once_deriv_func) - 1.0 / _kappa_over_kappa0_f(betamu, f_once_deriv_func, f_twice_deriv_func))
    return prefactor * terms

def _T_over_TF_f(betamu, f_once_deriv_func):
    return np.cbrt(16 / (9 * np.pi)) * np.power(f_once_deriv_func(betamu), -2/3)

#ONLY VALID FOR THE SCALE-INVARIANT GAS!!!
def _E_over_E0_f(betamu, f_func, f_once_deriv_func):
    return _P_over_P0_f(betamu, f_func, f_once_deriv_func)

def _mu_over_EF_f(betamu, f_once_deriv_func):
    return betamu * _T_over_TF_f(betamu, f_once_deriv_func)

def _F_over_E0_f(betamu, f_func, f_once_deriv_func):
    return -np.cbrt(2000/(243 * np.pi)) * (f_func(betamu) -
                                    betamu * f_once_deriv_func(betamu)) / np.power(f_once_deriv_func(betamu), 5/3)

def _S_over_NkB_f(betamu, f_func, f_once_deriv_func):
    return 2.5 * f_func(betamu) /(f_once_deriv_func(betamu))  - betamu

#Included for compatibility with non_betamu returns
def _betamu_f(betamu):
    return betamu


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
