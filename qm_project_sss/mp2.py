"""
mp2.py
The qm_project_sss package implements semi-empirical quantum mechanical (SCF+MP2) simulation parameterized to reproduce first-principles QM data using a minimal model.
This module contains a MP2 class implementing methods that returns the MP2 contribution to the total energy based on input Fock matrix and interaction matrix from Hartree-Fock calculations.
"""
from .hartree_fock import *
import numpy as np

class MP2(HartreeFock):
    def __init__(self, atomic_coordinates, gas_model):
        super().__init__(atomic_coordinates, gas_model)
        self.hf_energy = self.calculate_energy_scf()
        self.occupied_energy = self.partition_orbitals()[0]
        self.virtual_energy = self.partition_orbitals()[1]
        self.occupied_matrix = self.partition_orbitals()[2]
        self.virtual_matrix = self.partition_orbitals()[3]
        self.interaction_tensor = self.transform_interaction_tensor()
        self.mp2_energy = self.calculate_energy_mp2()
        

    def partition_orbitals(self):
        num_occ = (self.gas_model.ionic_charge // 2) * np.size(self.fock_matrix,
                                                0) // self.gas_model.orbitals_per_atom
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
                                    chi2_tensor,
                                    self.interaction_matrix,
                                    chi2_tensor,
                                    optimize=True)
        return interaction_tensor

    def calculate_energy_mp2(self):
        # E_occ, E_virt, occupied_matrix, virtual_matrix = self.partition_orbitals(self.fock_matrix)
        # V_tilde = self.transform_interaction_tensor(self.occupied_matrix, self.virtual_matrix,
        #                                     self.interaction_matrix, self.chi_tensor)
        V_tilde = self.interaction_tensor

        energy_mp2 = 0.0
        # num_occ = len(E_occ)
        num_occ = len(self.occupied_energy)
        # num_virt = len(E_virt)
        num_virt = len(self.virtual_energy)
        for a in range(num_virt):
            for b in range(num_virt):
                for i in range(num_occ):
                    for j in range(num_occ):
                        energy_mp2 -= (
                            (2.0 * V_tilde[a, i, b, j]**2 -
                             V_tilde[a, i, b, j] * V_tilde[a, j, b, i]) /
                            (self.virtual_energy[a] + self.virtual_energy[b] - self.occupied_energy[i] - self.occupied_energy[j]))
                        # energy_mp2 -= (
                        #     (2.0 * V_tilde[a, i, b, j]**2 -
                        #     V_tilde[a, i, b, j] * V_tilde[a, j, b, i]) /
                        #     (E_virt[a] + E_virt[b] - E_occ[i] - E_occ[j]))
                        
        return energy_mp2
