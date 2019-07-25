"""
hartree_fock.py
This package implements semi-empirical quantum mechanical (SCF+MP2) simulation parameterized to reproduce first-principles QM data using a minimal model.

Handles the primary functions
"""
import numpy as np

class Hartree_Fock:
    def __init__(self, atomic_coordinates, gas_model):
        self.atomic_coordiantes = atomic_coordinates

        self.gas_model = gas_model

        self.ndof = len(self.atomic_coordiantes) * self.gas_model.orbitals_per_atom

        self.interaction_matrix = self.calculate_interaction_matrix()

        self.density_matrix = self.calculate_density_matrix()

        self.chi_tensor = self.calculate_chi_tensor()

        self.hamiltonian_matrix = self.calculate_hamiltonian_matrix()

        self.fock_matrix = self.calculate_fock_matrix()


    def hopping_energy(self, o1, o2, r12):
        r12_rescaled = r12 / self.gas_model.model_parameters['r_hop']

        r12_length = np.linalg.norm(r12_rescaled)

        ans = np.exp( 1.0 - r12_length**2 )

        if o1 =='s' and o2 == 's':

            ans = ans * self.gas_model.model_parameters['t_ss']

        if o1 =='s' and o2 in self.gas_model.p_orbitals:

            ans = ans * np.dot(self.gas_model.vec[o2], r12_rescaled) * self.gas_model.model_parameters['t_sp']

        if o2 =='s' and o1 in self.gas_model.p_orbitals:

            ans = ans * -np.dot(self.gas_model.vec[o1], r12_rescaled) * self.gas_model.model_parameters['t_sp']

        if o1 in self.gas_model.p_orbitals and o2 in self.gas_model.p_orbitals:

            ans = ans * ( (r12_length**2) * np.dot(self.gas_model.vec[o1],self.gas_model.vec[o2]) * self.gas_model.model_parameters['t_pp2']
                     - np.dot(self.gas_model.vec[o1],r12_rescaled)* np.dot(self.gas_model.vec[o2],r12_rescaled)
                     * (self.gas_model.model_parameters['t_pp1'] + self.gas_model.model_parameters['t_pp2']) )

        return ans


    def coulomb_energy(self, o1, o2, r12):

        r12_length = np.linalg.norm(r12)

        ans = 1.0

        if o1 =='s' and o2 == 's':

            ans = ans * 1.0 / r12_length

        if o1 =='s' and o2 in self.gas_model.p_orbitals:

            ans = ans * np.dot(self.gas_model.vec[o2], r12) / r12_length**3

        if o2 =='s' and o1 in self.gas_model.p_orbitals:

            ans = ans * -np.dot(self.gas_model.vec[o1], r12) / r12_length**3

        if o1 in self.gas_model.p_orbitals and o2 in self.gas_model.p_orbitals:

            ans = ans * ( np.dot(self.gas_model.vec[o1],self.gas_model.vec[o2]) / r12_length**3
                     - 3.0 * np.dot(self.gas_model.vec[o1],r12)* np.dot(self.gas_model.vec[o2],r12) / r12_length**5 )

        return ans


    def pseudopotential_energy(self, o, r):

        r_rescaled = r / self.gas_model.model_parameters['r_pseudo']

        r_length = np.linalg.norm(r_rescaled)

        ans = self.gas_model.model_parameters['v_pseudo'] * np.exp( 1.0 - r_length**2 )

        if o in self.gas_model.p_orbitals:

            ans *= -2.0 * np.dot(self.gas_model.vec[o], r_rescaled)

        return ans


    def calculate_energy_ion(self):

        energy_ion = 0.0

        for i, r_i in enumerate(self.atomic_coordinates):

            for j, r_j in enumerate(self.atomic_coordinates):

                if i<j:

                    energy_ion += (self.gas_model.ionic_charge**2) * self.coulomb_energy('s', 's', r_i - r_j)

        return energy_ion


    def calculate_potential_vector(self):

        potential_vector = np.zeros(self.ndof)

        for p in range(self.ndof):

            for atom_i, r_i in enumerate(self.atomic_coordinates):

                r_pi = self.atomic_coordinates[self.gas_model.atom(p)] - r_i

                if atom_i != self.gas_model.atom(p):

                    potential_vector[p] += ( self.pseudopotential_energy(self.gas_model.orb(p), r_pi) -
                                        self.gas_model.ionic_charge * self.coulomb_energy(self.gas_model.orb(p), 's', r_pi) )

        return potential_vector
