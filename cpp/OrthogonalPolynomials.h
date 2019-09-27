/*
  Class to evaluate orthogonal polynomials.

  Author: Komahan Boopathy (komahanboopathy@gmail.com)
 */
class OrthogonalPolynomials {
 public:

  // Constructor and destructor
  OrthogonalPolynomials();
  ~OrthogonalPolynomials();

  //  Get Hermite polynomials  -- Normal distribution
  double hermite(double z, int d);
  double unit_hermite(double z, int d);

  //  Get Legendre polynomials -- Uniform distribution  
  double legendre(double z, int d);
  double unit_legendre(double z, int d);

  //  Get Laguerre polynomials  -- Exponential distribution
  double laguerre(double z, int d);
  double unit_laguerre(double z, int d);

 private:
  // Useful functions
  double factorial(int n);
  double comb(int n, int r);

  // hermite algorithms
  double explicit_hermite(double z, int d);
  double recursive_hermite(double z, int d);

  // Laguerre algorithms
  double explicit_laguerre(double z, int d);
  double recursive_laguerre(double z, int d);

  // Legendre algorithms
  double explicit_legendre(double z, int d);
  double general_legendre(double z, int d);  
};
