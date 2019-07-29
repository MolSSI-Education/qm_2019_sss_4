#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

// Developed by Hanjie Jiang
// Used for rewriting the calculate_fock_matrix_fast function in C++ version

const int orbitals_per_atom = 4;
int ao_index(int atom, int orb_index);
int orb(int ao_index);
int atom(int ao_index);
double chi_tensor(int o1, int o2, int o3, double dipole);
int main();
Eigen::MatrixXd fock_matrix_rewrite(Eigen::MatrixXd hamiltonianMatrix, Eigen::MatrixXd densityMatrix, Eigen::MatrixXd interactionMatrix, double dipole);

