#ifndef BASIS_HELPER
#define BASIS_HELPER

#include "scalar.h"

/**
   Class for constructing multivariate basis from univariate basis
   functions
 */
class BasisHelper{
 public:
  // Constructor and destructor
  BasisHelper(int _basis_type = 0);
  ~BasisHelper();

  // Find tensor product of 1d rules
  void basisDegrees(const int nvars, const int *pmax, int *nbasis, int **basis_degrees);
  void sparse(const int nvars, int *dmapi, int *dmapj, int *dmapk, bool *sparse);

 private:
  void uniVariateBasisDegrees(const int nvars, const int *pmax, int *nbasis, int **basis_degrees);

  // Tensor product basis
  void biVariateBasisDegreesTensor(const int nvars, const int *pmax, int *nbasis, int **basis_degrees);
  void triVariateBasisDegreesTensor(const int nvars, const int *pmax, int *nbasis, int **basis_degrees);
  void quadVariateBasisDegreesTensor(const int nvars, const int *pmax, int *nbasis, int **basis_degrees);
  void pentaVariateBasisDegreesTensor(const int nvars, const int *pmax, int *nbasis, int **basis_degrees);

  // Complete polynomial basis
  void biVariateBasisDegreesComplete(const int nvars, const int *pmax, int *nbasis, int **basis_degrees);
  void triVariateBasisDegreesComplete(const int nvars, const int *pmax, int *nbasis, int **basis_degrees);
  void quadVariateBasisDegreesComplete(const int nvars, const int *pmax, int *nbasis, int **basis_degrees);
  void pentaVariateBasisDegreesComplete(const int nvars, const int *pmax, int *nbasis, int **basis_degrees);

  // basis types available: tensor=0, complete=1
  int basis_type;
};

#endif
