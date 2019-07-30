"""
hartree_fock.py
The qm_project_sss package implements semi-empirical quantum mechanical (SCF+MP2) simulation parameterized to reproduce first-principles QM data using a minimal model.
This module contains a HartreeFock class which performs atomic SCF calculations.
"""
import numpy as np

class HartreeFock:
    def __init__(self, atomic_coordinates, gas_model):
        self.atomic_coordinates = atomic_coordinates
        # gas_model is a object of NobleGasModel
        self.gas_model = gas_model

        self.ndof = len(self.atomic_coordinates) * self.gas_model.orbitals_per_atom
        
        self.chi_tensor = self.calculate_chi_tensor()
        #
        self.potential_vector = self.calculate_potential_vector()

        self.interaction_matrix = self.calculate_interaction_matrix()

        self.density_matrix = self.calculate_atomic_density_matrix()

        self.hamiltonian_matrix = self.calculate_hamiltonian_matrix()

        self.fock_matrix = self.calculate_fock_matrix(self.density_matrix)

        # self.energy_scf = self.calculate_energy_scf()

    # 2. Slater-Koster tight-binding model
    def hopping_energy(self, o1, o2, r12):
        """ Returns hopping matrix element for a pair of orbitals of type o1 & o2 separated by a vector r12
        """
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

    # 3. Coulomb interaction
    def coulomb_energy(self, o1, o2, r12):
        """ Returns Coulomb matrix element for a pair of multipoles of type o1 & o2 separated by a vector r12.'''

        """
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
        """ Returns the energy of a pseudopotential between a multipole of type o and an atom separated by a vector r.'''
        """
        r_rescaled = r / self.gas_model.model_parameters['r_pseudo']

        r_length = np.linalg.norm(r_rescaled)

        ans = self.gas_model.model_parameters['v_pseudo'] * np.exp( 1.0 - r_length**2 )

        if o in self.gas_model.p_orbitals:

            ans *= -2.0 * np.dot(self.gas_model.vec[o], r_rescaled)

        return ans


    def calculate_energy_ion(self):
        """ Returns the ionic contribution to the total energy.'''
        
        Parameters
        ----------
        atomic_coordinates: list 
            list of atomic coordinates, e.g. [[0,0,0],[0,1,0]]
        
        Returns
        -------
        energy_ion: float
            ion-ion energy between atoms

        Examples
        --------
        >>> atomic_coordinates = np.array([ [0.0,0.0,0.0], [3.0,4.0,5.0] ])
        >>> calculate_energy_ion(atomic_coordinates)
        5.091168824543142
        """
        energy_ion = 0.0

        for i, r_i in enumerate(self.atomic_coordinates):

            for j, r_j in enumerate(self.atomic_coordinates):

                if i<j:

                    energy_ion += (self.gas_model.ionic_charge**2) * self.coulomb_energy('s', 's', r_i - r_j)

        return energy_ion


    def calculate_potential_vector(self):
        """ Returns the electron-ion potential energy vector.
        
        Returns
        -------
        potential_vector: list
            list of electron-ion interaction energies (float) omitting on-site Coulomb terms

        Examples
        --------
        >>> atomic_coordinates = np.array([ [0.0,0.0,0.0], [3.0,4.0,5.0] ])
        >>> calculate_potential_vector(atomic_coordinates, model_parameters)
        [-0.8 -0.1 -0.1 -0.1 -0.8  0.1  0.1  0.1]
        """
        potential_vector = np.zeros(self.ndof)

        for p in range(self.ndof):

            for atom_i, r_i in enumerate(self.atomic_coordinates):

                r_pi = self.atomic_coordinates[self.gas_model.atom(p)] - r_i

                if atom_i != self.gas_model.atom(p):

                    potential_vector[p] += ( self.pseudopotential_energy(self.gas_model.orb(p), r_pi) -
                                        self.gas_model.ionic_charge * self.coulomb_energy(self.gas_model.orb(p), 's', r_pi) )

        return potential_vector


    def calculate_interaction_matrix(self):
        """ Returns electron-electron interaction energy matrix.

        """
        interaction_matrix = np.zeros ((self.ndof,self.ndof))

        for p in range(self.ndof):

            for q in range(self.ndof):

                if self.gas_model.atom(p) != self.gas_model.atom(q):

                    r_pq = self.atomic_coordinates[self.gas_model.atom(p)] - self.atomic_coordinates[self.gas_model.atom(q)]

                    interaction_matrix[p,q] = self.coulomb_energy(self.gas_model.orb(p),self.gas_model.orb(q), r_pq)

                if p==q and self.gas_model.orb(p)=='s':

                    interaction_matrix[p,q] = self.gas_model.model_parameters['coulomb_s']

                if p==q and self.gas_model.orb(p) in self.gas_model.p_orbitals:

                    interaction_matrix[p,q] = self.gas_model.model_parameters['coulomb_p']

        return interaction_matrix

    # 4. Multipole decomposition
    def chi_on_atom(self, o1, o2, o3):
        """ Returns the dipole strength of intra-atomic s-p transition based on 3 input orbital types
        
        Parameters
        ----------
        o1: string
        o2: string
        o3: string

        Returns
        -------
        float
        """
        if o1 == o2 and o3 =='s':

            return 1.0

        if o1 == o3 and o3 in self.gas_model.p_orbitals and o2 =='s':

            return self.gas_model.model_parameters['dipole']

        if o2 == o3 and o3 in self.gas_model.p_orbitals and o1 =='s':

            return self.gas_model.model_parameters['dipole']

        return 0.0


    def calculate_chi_tensor(self):

        chi_tensor = np.zeros( (self.ndof, self.ndof, self.ndof) )

        for p in range(self.ndof):

            for orb_q in self.gas_model.orbital_types:

                q = self.gas_model.ao_index(self.gas_model.atom(p), orb_q)

                for orb_r in self.gas_model.orbital_types:

                    r = self.gas_model.ao_index(self.gas_model.atom(p), orb_r)

                    chi_tensor[p,q,r] = self.chi_on_atom(self.gas_model.orb(p), self.gas_model.orb(q), self.gas_model.orb(r))

        return chi_tensor

    # 5. 1-body Hamiltonian
    def calculate_hamiltonian_matrix(self):
        """ Returns 1-body Hamiltonian matrix, based on atomic coordinates and empirical parameters 'energy_s' and 'energy_p'
        """
        hamiltonian_matrix = np.zeros( (self.ndof, self.ndof) )

        # potential_vector = self.calculate_potential_vector()
        # potential_vector = self.potential_vector

        for p in range(self.ndof):

            for q in range(self.ndof):

                if self.gas_model.atom(p) != self.gas_model.atom(q):

                    r_pq = self.atomic_coordinates[self.gas_model.atom(p)] - self.atomic_coordinates[self.gas_model.atom(q)]

                    hamiltonian_matrix[p,q] = self.hopping_energy(self.gas_model.orb(p), self.gas_model.orb(q), r_pq)

                if self.gas_model.atom(p) == self.gas_model.atom(q):

                    if p == q and self.gas_model.orb(p) == 's':

                        hamiltonian_matrix[p,q] += self.gas_model.model_parameters['energy_s']

                    if p == q and self.gas_model.orb(p) in self.gas_model.p_orbitals:

                        hamiltonian_matrix[p,q] += self.gas_model.model_parameters['energy_p']

                    for orb_r in self.gas_model.orbital_types:

                        r = self.gas_model.ao_index(self.gas_model.atom(p), orb_r)

                        hamiltonian_matrix[p,q] += ( self.chi_on_atom(self.gas_model.orb(p), self.gas_model.orb(q), orb_r)
                                                 * self.potential_vector[r] )

        return hamiltonian_matrix


    def calculate_atomic_density_matrix(self):
        """ Returns a trial 1-electron density matrix which belongs to a system of isolated Argon atoms
        """

        density_matrix = np.zeros( (self.ndof, self.ndof) )

        for p in range(self.ndof):

            density_matrix[p,p] = self.gas_model.orbital_occupation[self.gas_model.orb(p)]

        return density_matrix


    def calculate_fock_matrix(self, old_density_matrix, construction_mode = 'slow', use_cpp_module = False):
        ''' Returns Fock matrix 
        '''
        
        if construction_mode == 'slow':
            fock_matrix = self.hamiltonian_matrix.copy()

            fock_matrix += 2.0*np.einsum('pqt,rsu,tu,rs->pq', self.chi_tensor, self.chi_tensor, self.interaction_matrix, old_density_matrix , optimize=True)

            fock_matrix -= np.einsum('rqt,psu,tu,rs->pq', self.chi_tensor, self.chi_tensor, self.interaction_matrix, old_density_matrix, optimize=True)
        
        if construction_mode == 'fast':
            if use_cpp_module:
                from qm_project_sss.hf_C import fock_matrix_rewrite
                dipole = self.gas_model.model_parameters['dipole']
                fock_matrix = fock_matrix_rewrite(self.hamiltonian_matrix, self.density_matrix, self.interaction_matrix, dipole)
            else:
                fock_matrix = self.hamiltonian_matrix.copy()
                # Hartree potential term
                for p in range(self.ndof):
                    for orb_q in self.gas_model.orbital_types:
                        q = self.gas_model.ao_index(self.gas_model.atom(p), orb_q)  # p & q on same atom
                        for orb_t in self.gas_model.orbital_types:
                            t = self.gas_model.ao_index(self.gas_model.atom(p), orb_t)  # p & t on same atom
                            chi_pqt = self.chi_on_atom(self.gas_model.orb(p), orb_q, orb_t,
                                                       self.gas_model.model_parameters)
                            for r in range(self.ndof):
                                for orb_s in self.gas_model.orbital_types:
                                    s = self.gas_model.ao_index(self.gas_model.atom(r), orb_s)  # r & s on same atom
                                    for orb_u in self.gas_model.orbital_types:
                                        u = self.gas_model.ao_index(self.gas_model.atom(r),
                                                                    orb_u)  # r & u on same atom
                                        chi_rsu = self.chi_on_atom(self.gas_model.orb(r), 
                                                                   orb_s, orb_u, self.gas_model.model_parameters)
                                        fock_matrix[p, q] += 2.0 * chi_pqt * chi_rsu * \
                                            self.interaction_matrix[t, u] * self.density_matrix[r, s]
                # Fock exchange term
                for p in range(self.ndof):
                    for orb_s in self.gas_model.orbital_types:
                        s = self.gas_model.ao_index(self.gas_model.atom(p), orb_s)  # p & s on same atom
                        for orb_u in self.gas_model.orbital_types:
                            u = self.gas_model.ao_index(self.gas_model.atom(p), orb_u)  # p & u on same atom
                            chi_psu = self.chi_on_atom(self.gas_model.orb(p), orb_s, orb_u,
                                                       self.gas_model.model_parameters)
                            for q in range(self.ndof):
                                for orb_r in self.gas_model.orbital_types:
                                    r = self.gas_model.ao_index(self.gas_model.atom(q), orb_r)  # q & r on same atom
                                    for orb_t in self.gas_model.orbital_types:
                                        t = self.gas_model.ao_index(self.gas_model.atom(q),
                                                                    orb_t)  # q & t on same atom
                                        chi_rqt = self.chi_on_atom(
                                            orb_r, self.gas_model.orb(q), orb_t, self.gas_model.model_parameters)
                                        fock_matrix[p, q] -= chi_rqt * chi_psu * self.interaction_matrix[t, u] * self.density_matrix[r, s]

        return fock_matrix


    def calculate_density_matrix(self):

        num_occ = (self.gas_model.ionic_charge//2) * np.size(self.fock_matrix,0) // self.gas_model.orbitals_per_atom

        orbital_energy, orbital_matrix = np.linalg.eigh(self.fock_matrix)

        occupied_matrix = orbital_matrix[:,:num_occ]

        density_matrix = occupied_matrix @ occupied_matrix.T

        return density_matrix

    def scf_cycle(self, max_scf_iterations=100, mixing_fraction=0.25, convergence_tolerance=1e-10, construction_mode='slow', use_cpp_module=False):

        self.density_matrix = self.calculate_density_matrix()

        old_density_matrix = self.density_matrix.copy()

        for iteration in range(max_scf_iterations):

            self.fock_matrix = self.calculate_fock_matrix(old_density_matrix, use_cpp_module)

            self.density_matrix = self.calculate_density_matrix()

            error_norm = np.linalg.norm( old_density_matrix - self.density_matrix)

            print(iteration, error_norm)

            if error_norm < convergence_tolerance:

                return self.density_matrix, self.fock_matrix

            old_density_matrix = (mixing_fraction * self.density_matrix + (1-mixing_fraction) * old_density_matrix )

        print("Warning: SCF Cycle did not converge")

        return self.density_matrix, self.fock_matrix


    def calculate_energy_scf(self):

        energy_scf = np.einsum('pq,pq', self.hamiltonian_matrix + self.fock_matrix, self.density_matrix)

        return energy_scf
