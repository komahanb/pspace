#ifndef QUADRATURE_HELPER
#define QUADRATURE_HELPER

class QuadratureHelper {
 public:
  // Constructor and destructor
  QuadratureHelper();
  ~QuadratureHelper();

  // Find tensor product of 1d rules
  void tensorProduct(const int nvars, const int *nqpts,
                     double **zp, double **yp, double **wp,
                     double **zz, double **yy, double *ww);
};

#endif
