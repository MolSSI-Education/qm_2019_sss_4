"""
Unit and regression test for NobelGasModel class in the qm_project_sss package.
"""

import qm_project_sss
import pytest
import sys
# import numpy as np

# @pytest.fixture()
# def argon_model():
#     argon = qm_project_sss.NobleGasModel('Argon')
#     return argon

argon_model = qm_project_sss.NobleGasModel('Argon')
neon_model = qm_project_sss.NobleGasModel('Ne')

# Tests for ao_index function
@pytest.mark.parametrize("atom_idx, orb_type, expected_ao_idx, model_obj", [
    (0, 's', 0, argon_model), (0, 'px', 1, argon_model),
    (0, 'py', 2, argon_model), (0, 'pz', 3, argon_model),
    (1, 's', 4, neon_model), (1, 'px', 5, neon_model),
    (1, 'py', 6, neon_model), (1, 'pz', 7, neon_model),
])
def test_ao_index(atom_idx, orb_type, expected_ao_idx, model_obj):
    calculated_ao_idx = model_obj.ao_index(atom_idx, orb_type)
    assert calculated_ao_idx == expected_ao_idx

# Tests for atom function
@pytest.mark.parametrize("ao_idx, expected_atom_idx, model_obj", [
    (0, 0, argon_model), (1, 0, argon_model), (2, 0, argon_model), (3, 0, argon_model),
    (4, 1, neon_model), (5, 1, neon_model), (6, 1, neon_model), (7, 1, neon_model),
])
def test_atom(ao_idx, expected_atom_idx, model_obj):
    calculated_atom_idx = model_obj.atom(ao_idx)
    assert calculated_atom_idx == expected_atom_idx

# Tests for orb function
@pytest.mark.parametrize("ao_idx, expected_orb_type, model_obj", [
    (0, 's', argon_model), (1, 'px', argon_model), (2, 'py', argon_model), (3, 'pz', argon_model),
    (4, 's', neon_model), (5, 'px', neon_model), (6, 'py', neon_model), (7, 'pz', neon_model),
])
def test_orb(ao_idx, expected_orb_type, model_obj):
    calculated_orb_type = model_obj.orb(ao_idx)
    assert calculated_orb_type == expected_orb_type






