import numpy as np 

LI_6_MASS_KG = 9.9883e-27
H_BAR_MKS = 1.054571e-34


def get_box_fermi_energy_from_counts(atom_counts, box_radius_um, box_length_um):
    box_volume_m = np.pi * np.square(box_radius_um) * box_length_um * 1e-18
    fermi_k_m = np.cbrt(6 * np.square(np.pi) * atom_counts / box_volume_m) 
    fermi_energy = np.square(H_BAR_MKS) * np.square(fermi_k_m) / (2 * LI_6_MASS_KG)  
    fermi_energy_hz = fermi_energy / (2 * np.pi * H_BAR_MKS)
    return fermi_energy_hz


#By convention, uses kHz as the base unit.
def two_level_system_population_rabi(t, omega_r, detuning):
    generalized_rabi = np.sqrt(np.square(omega_r) + np.square(detuning))
    population_excited = np.square(omega_r) / np.square(generalized_rabi) * np.square(np.sin(generalized_rabi / 2 * t))
    return np.array([1.0 - population_excited, population_excited])