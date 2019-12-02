#ifndef QUADRATURE_HELPER
#define QUADRATURE_HELPER

#include "scalar.h"

class QuadratureHelper {
 public:
  // Constructor and destructor
  QuadratureHelper();
  ~QuadratureHelper();

  // Find tensor product of 1d rules
  void tensorProduct(const int nvars, const int *nqpts,
                     scalar **zp, scalar **yp, scalar **wp,
                     scalar **zz, scalar **yy, scalar *ww);
};

#endif
