qm_project_sss
==============================
[//]: # (Badges)

[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/REPLACE_WITH_APPVEYOR_LINK/branch/master?svg=true)](https://ci.appveyor.com/project/REPLACE_WITH_OWNER_ACCOUNT/qm_project_sss/branch/master)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/qm_project_sss/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/qm_project_sss/branch/master)

The **qm_project_sss** package implements semi-empirical quantum mechanical (SCF+MP2) simulation parameterized to reproduce first-principles QM data using a minimal model. The package serves three main classes:

- NobleGasModel (in **Noble_Gas_Model** module)
- HartreeFock (in **hartree_fock** module)
- MP2 (in **mp2** module)

You can learn more about attributes and methods in three classes in the [documentation](https://qm-2019-sss-4.readthedocs.io/en/latest/). However, you can also keep reading below to get started.

Installation
------------

```bash
git clone https://github.com/MolSSI-Education/qm_2019_sss_4.git
cd qm_2019_sss_4
pip install -e .
```

You can run tests to make sure the package is installed successfully
```bash
pytest -v
```

Getting Started
---------------

In just a few lines, we can compute the Hartree-Fock energy and the MP2 energy correction of a system of two Argon atoms.

```python
from qm_project_sss import NobleGasModel, HartreeFock, MP2
import numpy as np

# Initilize a NobleGasModel object
Ar = NobleGasModel('ar')
atomic_coordinates = np.array([[0.0,0.0,0.0], [3.0,4.0,5.0]])

# Initilize a HartreeFock object
hf = qm_project_sss.HartreeFock(atomic_coordinates, Ar)

# Start SCF iterations
hf.scf_cycle()
hf_energy = hf.calculate_energy_ion() + hf.calculate_energy_scf()

# Compute MP2 correction to HF energy
mp2 = qm_project_sss.MP2(atomic_coordinates, Ar)
mp2_correction = mp2.mp2_energy
```


### Copyright

Copyright (c) 2019, sss_2019_qm_4


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
