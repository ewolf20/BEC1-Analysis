import numpy as np 

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


#By convention, uses kHz as the base unit.
def two_level_system_population_rabi(t, omega_r, detuning):
    generalized_rabi = np.sqrt(np.square(omega_r) + np.square(detuning))
    population_excited = np.square(omega_r) / np.square(generalized_rabi) * np.square(np.sin(generalized_rabi / 2 * t))
    return np.array([1.0 - population_excited, population_excited])

def get_li_energy_hz_in_1D_trap(displacement_m, trap_freq):
    li_energy_mks = 0.5 * LI_6_MASS_KG * np.square(trap_freq) * np.square(displacement_m)
    li_energy_hz = li_energy_mks / (2 * np.pi * H_BAR_MKS)
    return li_energy_hz
