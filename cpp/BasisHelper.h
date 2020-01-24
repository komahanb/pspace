#ifndef BASIS_HELPER
#define BASIS_HELPER

#include "scalar.h"

class BasisHelper{
 public:
  // Constructor and destructor
  BasisHelper(int _basis_type = 0);
  ~BasisHelper();

  // Find tensor product of 1d rules
  void basisDegrees(const int nvars, const int *pmax, int *nindices, int **indx);
  void sparse(const int nvars, int *dmapi, int *dmapj, int *dmapk, 
              bool *sparse);

 private:
  void univariateBasisDegrees(const int nvars, const int *pmax, int *nindices, int **indx);

  // tensor product basis
  void bivariateBasisDegrees(const int nvars, const int *pmax, int *nindices, int **indx);
  void trivariateBasisDegrees(const int nvars, const int *pmax, int *nindices, int **indx);
  void quadvariateBasisDegrees(const int nvars, const int *pmax, int *nindices, int **indx);
  void pentavariateBasisDegrees(const int nvars, const int *pmax, int *nindices, int **indx);

  // Complete polynomial basis
  void bivariateBasisDegreesComplete(const int nvars, const int *pmax, int *nindices, int **indx);
  
  // basis types available: tensor, complete
  int basis_type;
};

#endif
