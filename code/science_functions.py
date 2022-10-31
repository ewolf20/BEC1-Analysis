import numpy as np 
from scipy.integrate import trapezoid
from scipy.optimize import fsolve
from scipy.signal import savgol_filter

#Taken from https://jet.physics.ncsu.edu/techdocs/pdf/PropertiesOfLi.pdf
LI_6_MASS_KG = 9.98834e-27
#Taken from https://physics.nist.gov/cgi-bin/cuu/Value?hbar
H_BAR_MKS = 1.054572e-34


def get_box_fermi_energy_from_counts(atom_counts, box_radius_um, box_length_um):
    box_volume_m = np.pi * np.square(box_radius_um) * box_length_um * 1e-18
    atom_density_m = atom_counts / box_volume_m
    return get_fermi_energy_hz_from_density(atom_density_m)


def get_fermi_energy_hz_from_density(atom_density_m):
    fermi_k_m = np.cbrt(6 * np.square(np.pi) * atom_density_m) 
    fermi_energy_J = np.square(H_BAR_MKS) * np.square(fermi_k_m) / (2 * LI_6_MASS_KG)
    fermi_energy_hz = fermi_energy_J / (2 * np.pi * H_BAR_MKS) 
    return fermi_energy_hz


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

def get_hybrid_trap_compressibilities(harmonic_trap_positions_um, three_d_density_trap_profile_um, trap_freq):
    pixel_number = len(harmonic_trap_positions_um)
    position_delta = harmonic_trap_positions_um[1] - harmonic_trap_positions_um[0]
    window_width = pixel_number // 20
    deriv_polyorder = 2
    if not np.all(np.isclose(np.diff(harmonic_trap_positions_um), position_delta)):
        raise NotImplementedError("Compressibility for non-uniform harmonic trap pixel spacing is not implemented yet.")
    fermi_energies = get_fermi_energy_hz_from_density(three_d_density_trap_profile_um * 1e18)
    d_mu_d_y_um = savgol_filter(fermi_energies, window_width, deriv_polyorder, deriv = 1, delta = position_delta)
    d_V_d_y_um = get_li_energy_m_deriv_hz_in_1D_trap(harmonic_trap_positions_um * 1e-6, trap_freq) * 1e-6
    d_mu_d_V = d_mu_d_y_um / d_V_d_y_um
    compressibilities = - d_mu_d_V 
    return compressibilities

#By convention, uses kHz as the base unit.
def two_level_system_population_rabi(t, omega_r, detuning):
    generalized_rabi = np.sqrt(np.square(omega_r) + np.square(detuning))
    population_excited = np.square(omega_r) / np.square(generalized_rabi) * np.square(np.sin(generalized_rabi / 2 * t))
    return np.array([1.0 - population_excited, population_excited])

def get_li_energy_hz_in_1D_trap(displacement_m, trap_freq_hz):
    li_energy_mks = 0.5 * LI_6_MASS_KG * np.square(2 * np.pi * trap_freq_hz) * np.square(displacement_m)
    li_energy_hz = li_energy_mks / (2 * np.pi * H_BAR_MKS)
    return li_energy_hz

def get_li_energy_m_deriv_hz_in_1D_trap(displacement_m, trap_freq_hz):
    li_energy_deriv_m_mks = LI_6_MASS_KG * np.square(2 * np.pi * trap_freq_hz) * displacement_m
    li_energy_deriv_m_hz = li_energy_deriv_m_mks / (2 * np.pi * H_BAR_MKS)
    return li_energy_deriv_m_hz


#Data from https://jet.physics.ncsu.edu/techdocs/pdf/PropertiesOfLi.pdf
LI_HYPERFINE_CONSTANT_MHZ = 152.1368
LI_G_J = 2.0023010
LI_G_I = 0.0004476540
LI_I = 1
LI_F_PLUS = LI_I + 0.5

#Data from https://physics.nist.gov/cgi-bin/cuu/Value?mubshhz
BOHR_MAGNETON_IN_MHZ_PER_G = 1.3996245

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