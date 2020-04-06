#include "scalar.h"

/**
   Class to evaluate orthogonal polynomials

   @author Komahan Boopathy
*/
class OrthogonalPolynomials {
 public:
  // Constructor and destructor
  OrthogonalPolynomials();
  ~OrthogonalPolynomials();

  // Get Hermite polynomials  -- Normal distribution
  scalar hermite(scalar z, int d);
  scalar unit_hermite(scalar z, int d);

  // Get Legendre polynomials -- Uniform distribution
  scalar legendre(scalar z, int d);
  scalar unit_legendre(scalar z, int d);

  // Get Laguerre polynomials  -- Exponential distribution
  scalar laguerre(scalar z, int d);
  scalar unit_laguerre(scalar z, int d);

 private:
  // Useful functions
  scalar factorial(int n);
  scalar comb(int n, int r);

  // hermite algorithms
  scalar explicit_hermite(scalar z, int d);
  scalar recursive_hermite(scalar z, int d);

  // Laguerre algorithms
  scalar explicit_laguerre(scalar z, int d);
  scalar recursive_laguerre(scalar z, int d);

  // Legendre algorithms
  scalar explicit_legendre(scalar z, int d);
  scalar general_legendre(scalar z, int d);
};
