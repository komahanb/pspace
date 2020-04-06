#include "scalar.h"

/**
  Class to return points and weights for Gaussian Quadrature

  Author: Komahan Boopathy (komahanboopathy@gmail.com)
*/
class GaussianQuadrature {
 public:
  // Constructor and destructor
  GaussianQuadrature();
  ~GaussianQuadrature();

  // Quadrature implementations
  void hermiteQuadrature(int npoints, scalar mu, scalar sigma,
                         scalar *z, scalar *y, scalar *w);
  void legendreQuadrature(int npoints, scalar a, scalar b,
                          scalar *z, scalar *y, scalar *w);
  void laguerreQuadrature(int npoints, scalar mu, scalar beta,
                          scalar *z, scalar *y, scalar *w);
};
