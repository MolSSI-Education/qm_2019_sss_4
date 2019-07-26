import numpy as np

# initialized with a gas type
class Noble_Gas_Model:

    def __init__(self, gas_type):

        """model_parameters : a dictionary of parameters"""

        if isinstance(gas_type, str):

            self.gas_type = gas_type
        else:

            raise TypeError("Name should be a string!")

        if (self.gas_type.lower() == "ar" or self.gas_type.lower() == "argon"):
            self.model_parameters = {
            'r_hop' : 3.1810226927827516,
            't_ss' : 0.03365982238611262,
            't_sp' : -0.029154833035109226,
            't_pp1' : -0.0804163845390335,
            't_pp2' : -0.01393611496959445,
            'r_pseudo' : 2.60342991362958,
            'v_pseudo' : 0.022972992186364977,
            'dipole' : 2.781629275106456,
            'energy_s' : 3.1659446174413004,
            'energy_p' : -2.3926873325346554,
            'coulomb_s' : 0.3603533286088998,
            'coulomb_p' : -0.003267991835806299
            }

        elif (self.gas_type.lower() == "ne" or self.gas_type.lower() == "neon"):
            self.model_parameters = {
            'coulomb_p': -0.010255409806855187,
            'coulomb_s': 0.4536486561938202,
            'dipole': 1.6692376991516769,
            'energy_p': -3.1186533988406335,
            'energy_s': 11.334912902362603,
            'r_hop': 2.739689713337267,
            'r_pseudo': 1.1800779720963734,
            't_pp1': -0.029546671673199854,
            't_pp2': -0.0041958662271044875,
            't_sp': 0.000450562836426027,
            't_ss': 0.0289251941290921,
            'v_pseudo': -0.015945813280635074
            }

        else:
            raise TypeError("Gas type cannot be recognized: Ar, Argon, Ne and Neon expected")

        self.ionic_charge = 0

        self.orbital_types = ['s', 'px', 'py', 'pz']

        self.orbitals_per_atom = len(self.orbital_types)

        self.p_orbitals = ['px', 'py', 'pz']

        self.vec = { 'px':[1,0,0], 'py':[0,1,0], 'pz':[0,0,1] }

        self.orbital_occupation = {'s':0, 'px':1, 'py':1, 'pz':1}


    def ao_index(self, atom_p, orb_p):

        p = atom_p * self.orbitals_per_atom

        p += self.orbital_types.index(orb_p)

        return p


    def atom(self, ao_index):

        return ao_index // self.orbitals_per_atom


    def orb(self, ao_index):

        orb_index = ao_index % self.orbitals_per_atom

        return self.orbital_types[orb_index]



argon = Noble_Gas_Model('Argon')
print(argon.model_parameters)
print(argon.ionic_charge)
print(argon.orbital_types)
print(argon.orbitals_per_atom)
print(argon.p_orbitals)
print(argon.vec)
print(argon.orbital_occupation)
print(argon.atom(5))
print(argon.orb(5))
print(argon.ao_index(argon.atom(5),argon.orb(5)))
