
import numpy as np 
from scipy.integrate import trapezoid
from scipy.optimize import fsolve
from scipy.signal import savgol_filter

from . import statistics_functions, eos_functions



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



def get_fermi_energy_hz_from_density(atom_density_m):
    fermi_k_m = np.cbrt(6 * np.square(np.pi) * atom_density_m)
    fermi_energy_J = np.square(H_BAR_MKS) * np.square(fermi_k_m) / (2 * LI_6_MASS_KG)
    fermi_energy_hz = fermi_energy_J / (2 * np.pi * H_BAR_MKS) 
    return fermi_energy_hz

def get_ideal_fermi_pressure_hz_um_from_density(atom_density_m):
    fermi_energy_hz = get_fermi_energy_hz_from_density(atom_density_m) 
    atom_density_um = atom_density_m * 1e-18
    ideal_pressure_hz_um = eos_functions.ideal_fermi_P0(atom_density_um, fermi_energy_hz) 
    return ideal_pressure_hz_um


#FUNCTIONS FOR CALCULATIONS IN BOX AND HYBRID TRAP

def get_box_fermi_energy_from_counts(atom_counts, box_cross_section_um, box_length_um):
    box_volume_m = box_cross_section_um * box_length_um * 1e-18
    atom_density_m = atom_counts / box_volume_m
    return get_fermi_energy_hz_from_density(atom_density_m)


def get_hybrid_trap_total_energy(harmonic_trap_positions_um, three_d_density_trap_profile_um, trap_cross_section_um, trap_freq):
    harmonic_trap_energies = get_li_energy_hz_in_1D_trap(harmonic_trap_positions_um * 1e-6, trap_freq)
    total_potential_energy = trapezoid(harmonic_trap_energies * three_d_density_trap_profile_um * trap_cross_section_um, x = harmonic_trap_positions_um)
    #1D Virial theorem; see Zhenjie Yan's PhD thesis
    #Formula still holds for an imbalanced cloud
    total_energy = total_potential_energy * 4
    return total_energy


def get_hybrid_trap_average_energy(harmonic_trap_positions_um, three_d_density_trap_profile_um, trap_cross_section_um, trap_freq):
    total_energy = get_hybrid_trap_total_energy(harmonic_trap_positions_um, three_d_density_trap_profile_um, trap_cross_section_um, trap_freq)
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

#TODO: Support the case where positions aren't in ascending order...
def get_hybrid_trap_compressibilities_savgol(harmonic_trap_positions_um, three_d_density_trap_profile_um, trap_freq, 
                                        savgol_window_length = 21, savgol_polyorder = 2):
    if not np.all(np.isclose(np.diff(harmonic_trap_positions_um), np.diff(harmonic_trap_positions_um)[0])):
        raise ValueError("Savitzky-Golay differentiation requires equal position spacings")
    fermi_energies = get_fermi_energy_hz_from_density(three_d_density_trap_profile_um * 1e18)
    potentials = get_li_energy_hz_in_1D_trap(harmonic_trap_positions_um * 1e-6, trap_freq)
    potential_index_deriv = np.diff(potentials)
    potential_index_deriv_extended = np.append(potential_index_deriv, potential_index_deriv[-1])
    fermi_energy_index_deriv = savgol_filter(fermi_energies, savgol_window_length, savgol_polyorder, deriv = 1)
    fermi_energy_potential_deriv = fermi_energy_index_deriv / potential_index_deriv_extended
    compressibilities = -fermi_energy_potential_deriv
    return compressibilities

#TODO: Speed up this unintelligent for loop implementation with numpy syntax
def get_hybrid_trap_compressibilities_window_fit(potentials_hz, three_d_density_trap_profile_um, breakpoint_indices, 
                                                 return_errors = False, polyorder = 2):
    fermi_energies = get_fermi_energy_hz_from_density(three_d_density_trap_profile_um * 1e18)
    compressibilities_list = []
    errors_list = []
    for lower_breakpoint_index, upper_breakpoint_index in zip(breakpoint_indices[:-1], breakpoint_indices[1:]):
        included_potentials = potentials_hz[lower_breakpoint_index:upper_breakpoint_index] 
        midpoint = (included_potentials[0] + included_potentials[-1]) / 2.0
        included_fermi_energies = fermi_energies[lower_breakpoint_index:upper_breakpoint_index]
        coeffs, pcov = np.polyfit(included_potentials - midpoint, included_fermi_energies, polyorder, cov = True)
        _, lin_sigma, _ = np.sqrt(np.diag(pcov))
        _, lin_coeff, _ = coeffs 
        fermi_energy_potential_deriv = lin_coeff 
        compressibility = -fermi_energy_potential_deriv
        compressibilities_list.append(compressibility) 
        errors_list.append(lin_sigma)
    compressibilities = np.array(compressibilities_list) 
    errors = np.array(errors_list)
    if return_errors:
        return (compressibilities, errors) 
    else:
        return compressibilities






    

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