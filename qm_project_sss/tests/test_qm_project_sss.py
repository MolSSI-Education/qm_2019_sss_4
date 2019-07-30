"""
Unit and regression test for the qm_project_sss package.
"""

# Import package, test suite, and other packages as needed
import qm_project_sss
import pytest
import sys
import numpy as np
import numpy.testing as npt

"""Tests for NobelGasModel class
"""
@pytest.fixture()
def Ar_gas():
    n = 'Argon'
    argon = qm_project_sss.NobleGasModel(n)
    return argon

def test_qm_project_sss_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "qm_project_sss" in sys.modules

def test_create_failure():
    name = 20
    with pytest.raises(TypeError):
        helium = qm_project_sss.NobleGasModel(name)

def test_correct_dictionary(Ar_gas):
    test_dict_Ar = {
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

    expected_value = len(test_dict_Ar)

    matched_length = len(test_dict_Ar.items() & Ar_gas.model_parameters.items())

    assert expected_value == matched_length

def test_ionic_charge(Ar_gas):

    expected_value = 6.0

    assert expected_value == Ar_gas.ionic_charge


def test_orbital_types(Ar_gas):

    orb_type = ['s', 'px', 'py', 'pz']

    assert orb_type == Ar_gas.orbital_types


def test_orbitals_per_atom(Ar_gas):

    expected_value = 4.0

    assert expected_value == Ar_gas.orbitals_per_atom

@pytest.mark.parametrize("p1, expected_vec", [
    ('px', [1, 0, 0]),
    ('py', [0, 1, 0]),
    ('pz', [0, 0, 1])
])
def test_orbital_vec(p1, expected_vec, Ar_gas):

    orb_vec = Ar_gas.vec[p1]

    assert expected_vec == orb_vec


@pytest.mark.parametrize("orb, expected_value", [
    ('s', 0), ('px', 1), ('py', 1), ('pz', 1)
])
def test_orbital_occupation(orb, expected_value, Ar_gas):

    orb_occ = Ar_gas.orbital_occupation[orb]

    assert expected_value == orb_occ


@pytest.mark.parametrize("orbnum, expected_value", [
    (2, 0), (4, 1), (9,2), (15,3)
])
def test_atom_index(orbnum, expected_value, Ar_gas):

    atomnum = Ar_gas.atom(orbnum)

    assert expected_value == atomnum


@pytest.mark.parametrize("orbnum, expected_orb_type", [
    (0, 's'), (5, 'px'), (10, 'py'), (15, 'pz')
])
def test_orbital(orbnum, expected_orb_type, Ar_gas):

    orbtype =  Ar_gas.orb(orbnum)

    assert expected_orb_type == orbtype


@pytest.mark.parametrize("atomnum, orb_type, expected_ao_index", [
    (0, 'px', 1), (1, 's', 4), (2, 'pz', 11), (3, 'py', 14)
])
def test_ao_index(atomnum, orb_type, expected_ao_index, Ar_gas):

    atomorb_index = Ar_gas.ao_index(atomnum, orb_type)

    assert expected_ao_index == atomorb_index



"""Tests for HartreeFock class
"""
@pytest.fixture()
def hf_1(Ar_gas):
    atomic_coordinates = np.array([[0.0,0.0,0.0], [3.0,4.0,5.0]])
    hf1 = qm_project_sss.HartreeFock(atomic_coordinates, Ar_gas)
    return hf1

def test_scf_energy(hf_1):
    expected_hf_energy = -17.901180746673777
    hf_1.scf_cycle()
    calculated_hf_energy = hf_1.calculate_energy_ion() + hf_1.calculate_energy_scf()
    assert expected_hf_energy == calculated_hf_energy

def test_fock_implementation_fast(hf_1):
    expected_hf_energy = -17.901180746673777
    hf_1.scf_cycle(construction_mode='fast', use_cpp_module=False)
    calculated_hf_energy = hf_1.calculate_energy_ion() + hf_1.calculate_energy_scf()
    assert expected_hf_energy == calculated_hf_energy

# This test is to check the two fock matrix implementations
# def test_fock_diff

"""Tests for MP2 class
"""
@pytest.fixture()
def MP2_case1(Ar_gas):
    atomic_coordinates = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
    MP2_obj = qm_project_sss.MP2(atomic_coordinates, Ar_gas)
    return MP2_obj

def test_partition_orbitals(MP2_case1):
    expected_occupied_energy = [-0.6, -0.6, -0.6, -0.6, -0.6, -0.6]
    expected_virtual_energy = [5.4, 5.4]

    calculated_occupied_energy = MP2_case1.occupied_energy
    calculated_virtual_energy = MP2_case1.virtual_energy

    npt.assert_array_almost_equal(calculated_occupied_energy, expected_occupied_energy,decimal=1)
    npt.assert_array_almost_equal(calculated_virtual_energy, expected_virtual_energy,decimal=1)

# def test_transform_interaction_tensor(MP2_case1):

def test_calculate_energy_mp2(MP2_case1):
    expected_mp2_correction = -0.0012786819552120972
    calculated_mp2_correction = MP2_case1.mp2_energy
    npt.assert_almost_equal(calculated_mp2_correction, expected_mp2_correction)
