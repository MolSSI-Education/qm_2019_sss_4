#include <pybind11/pybind11.h>
#include "hf.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(hf_C, m)
{
	m.doc() = "This is a trial for c++ version of calculate_fock_matrix";
	m.def("fock_matrix_rewrite", fock_matrix_rewrite, "Fock matrix rewrited in C++");
}
