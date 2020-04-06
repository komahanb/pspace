#ifndef QUADRATURE_HELPER
#define QUADRATURE_HELPER

#include "scalar.h"

/**
   Class that performs multivariate quadrature from univariate
   quadrature.
 */
class QuadratureHelper {
 public:
  // Constructor and destructor
  QuadratureHelper(int quadrature_type=0);
  ~QuadratureHelper();

  // Find tensor product of 1d rules
  void tensorProduct(const int nvars, const int *nqpts,
                     scalar **zp, scalar **yp, scalar **wp,
                     scalar **zz, scalar **yy, scalar *ww);

 private:
  int quadrature_type;
};

#endif
