#include "hf.h"

Eigen::MatrixXd fock_matrix_rewrite(Eigen::MatrixXd hamiltonianMatrix, Eigen::MatrixXd densityMatrix, Eigen::MatrixXd interactionMatrix, double dipole)
{	
    int ndofRows = hamiltonianMatrix.rows();
	int ndofCols = hamiltonianMatrix.cols();
	//std::cout << "my matrix has " << hamiltonianMatrix.rows() << " number of rows" << std::endl;
	//std::cout << "my matrix has " << hamiltonianMatrix.cols() << " number of cols" << std::endl;
    Eigen::MatrixXd fockMatrix(ndofRows,ndofCols);
	std::vector<int> orbital_types;
    
    for (int i = 0; i < orbitals_per_atom; i++)
		orbital_types.push_back(i);
    
    for (size_t i = 0; i < orbital_types.size(); i++)     
        std::cout << orbital_types[i] << "\n";  

	for (int i = 0; i < ndofRows; i++)
		for (int j = 0; j < ndofCols; j++)
		{
	    	fockMatrix(i,j) = hamiltonianMatrix(i,j);
		}
	
	//std::cout << "printing Fock matrix" << std::endl;
    //std::cout << fockMatrix << std::endl;

    for (int p = 0; p < ndofRows; p++)
	{
		//Test passed
		//std::cout << "the program crashes before going into the second nested for loop" << std::endl;
		for (int orb_q = 0; orb_q < orbitals_per_atom; orb_q++)
		{
		    int atom_p = atom(p);
			int q = ao_index(atom_p, orb_q);
			//Test passed
		    //std::cout << "the program crashes before going into the second nested for loop" << std::endl;
			for (int orb_t = 0; orb_t < orbitals_per_atom; orb_t++)
			{	
				int t = ao_index(atom(p), orb_t);
				double chi_pqt = chi_tensor(orb(p), orb_q, orb_t, dipole);
				//Test passed
		        //std::cout << "the program crashes before going into the second nested for loop" << std::endl;
	            for (int r = 0; r < ndofCols; r++)
				{
		            for (int orb_s = 0; orb_s < orbitals_per_atom; orb_s++)
                    {
						//Test passed
			            //std::cout << "the program crashes before going into the second nested for loop" << std::endl;     
					    int s = ao_index(atom(r), orb_s);
			            for (int orb_u = 0; orb_u < orbitals_per_atom; orb_u++)
			            {
			                //std::cout << "the program crashes before going into the second nested for loop" << std::endl;     
                            //std::cout << "atom(r): " << atom(r) << std::endl;
							//std::cout << "orb_u: " << orb_u << std::endl;
				            int u = ao_index(atom(r), orb_u);
							//std::cout << "ao_index is causing trouble" << std::endl;
							double chi_rsu = chi_tensor(orb(r), orb_s, orb_t, dipole);
							//std::cout << "chi_pqt = " << chi_pqt << std::endl;
							//std::cout << "chi_rsu = " << chi_rsu << std::endl;
							//std::cout << "t = " << t << "u = " << u << "r = " << r << "s = " << s << std::endl;
							//std::cout << "interaction (t, u) = " << interactionMatrix(t,u) << std::endl;
							//std::cout << "density matrix (r, s) = " << densityMatrix(r,s) << std::endl;
							//std::cout << "fock matrix is causing trouble" << std::endl;
							//std::cout << "chi_rsu is causing trouble" << std::endl;
			                fockMatrix(p,q) += 2.0 * chi_pqt * chi_rsu * interactionMatrix(t,u) * densityMatrix(r,s); 
						}
		            }
				}
			}
	    }
	}
    std::cout << "The program reaches here" << std::endl;
	for (int p = 0; p < ndofRows; p++)
	{
		for (int orb_s = 0; orb_s < orbitals_per_atom; orb_s++)
		{
			int s = ao_index(atom(p), orb_s);
		    for (int orb_u = 0; orb_u < orbitals_per_atom; orb_u++)
			{
				int u = ao_index(atom(p), orb_u);
				double chi_psu = chi_tensor(orb(p), orb_s, orb_u, dipole);
				for (int q = 0; q < ndofCols; q++)
				{
					for (int orb_r = 0; orb_r < orbitals_per_atom; orb_r++)
					{
						int r = ao_index(atom(q), orb_r);
						for (int orb_t = 0; orb_t < orbitals_per_atom; orb_t++)
						{
							int t = ao_index(atom(q), orb_t);
							double chi_rqt = chi_tensor(orb_r, orb(q), orb_t, dipole);
							fockMatrix(p,q) -= chi_rqt * chi_psu * interactionMatrix(t,u) * densityMatrix(r,s);
						}
					}
				}
			}
		}	
	}
    return fockMatrix;
}


double chi_tensor(int o1, int o2, int o3, double dipole)
{
    if (o1 == o2 && o3 == 0) return 1.0;
	else if (o1 == o3 && o3 != 0 && o2 != 0) return dipole;
	else if (o2 == o3 && o3 != 0 && o1 != 0) return dipole;
	else return 0.0;
}	

int atom(int ao_index)
{
    int quotient = ao_index / orbitals_per_atom;
	return quotient;
}

int orb(int ao_index)
{
	int orb_index = ao_index % orbitals_per_atom;
	return orb_index;
}

int ao_index(int atom, int orb_index)
{
    int p = atom * orbitals_per_atom;
	p = p + orb_index;
	return p;
}	
