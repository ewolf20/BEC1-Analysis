import numpy as np 

LI_6_MASS_KG = 9.9883e-27
H_BAR_MKS = 1.054571e-34


def get_box_fermi_energy_from_counts(atom_counts, box_radius_um, box_length_um):
    box_volume_m = np.pi * np.square(box_radius_um) * box_length_um * 1e-18
    fermi_k_m = np.cbrt(6 * np.square(np.pi) * atom_counts / box_volume_m) 
    fermi_energy = np.square(H_BAR_MKS) * np.square(fermi_k_m) / (2 * LI_6_MASS_KG)  
    fermi_energy_hz = fermi_energy / (2 * np.pi * H_BAR_MKS)
    return fermi_energy_hz


