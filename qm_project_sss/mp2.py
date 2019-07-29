"""
mp2.py
This packages contains class methods that returns the MP2 contribution to the total energy defined by the input Fock & interaction matrices
"""

import numpy as np

class MP2:
def __init__(self):
    self.partition_orbitals = partition_orbitals
    self.transform_interaction_tensor = transform_interaction_tensor
    self.calculate_mp2_energy = calculate_mp2_energy

def partition_orbitals(self):
    num_occ = (ionic_charge // 2) * np.size(self.fock_matrix,
                                            0) // orbitals_per_atom
    orbital_energy, orbital_matrix = np.linalg.eigh(self.fock_matrix)
    occupied_energy = orbital_energy[:num_occ]
    virtual_energy = orbital_energy[num_occ:]
    occupied_matrix = orbital_matrix[:, :num_occ]
    virtual_matrix = orbital_matrix[:, num_occ:]

    return occupied_energy, virtual_energy, occupied_matrix, virtual_matrix

def transform_interaction_tensor(self):
    chi2_tensor = np.einsum('qa,ri,qrp',
                            self.virtual_matrix,
                            self.occupied_matrix,
                            self.chi_tensor,
                            optimize=True)
    interaction_tensor = np.einsum('aip,pq,bjq->aibj',
                                   self.chi2_tensor,
                                   self.interaction_matrix,
                                   self.chi2_tensor,
                                   optimize=True)
    return interaction_tensor

def calculate_energy_mp2(self):
    E_occ, E_virt, occupied_matrix, virtual_matrix = partition_orbitals(
        self.fock_matrix)
    V_tilde = transform_interaction_tensor(occupied_matrix, virtual_matrix,
                                           self.interaction_matrix, self.chi_tensor)

    energy_mp2 = 0.0
    num_occ = len(E_occ)
    num_virt = len(E_virt)
    for a in range(num_virt):
        for b in range(num_virt):
            for i in range(num_occ):
                for j in range(num_occ):
                    energy_mp2 -= (
                        (2.0 * V_tilde[a, i, b, j]**2 -
                         V_tilde[a, i, b, j] * V_tilde[a, j, b, i]) /
                        (E_virt[a] + E_virt[b] - E_occ[i] - E_occ[j]))
    return energy_mp2
