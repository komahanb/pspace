/*
  Class to return points and weights for Gaussian Quadrature

  Author: Komahan Boopathy (komahanboopathy@gmail.com)
*/
class GaussianQuadrature {

 public:

  // Constructor and destructor
  GaussianQuadrature();
  ~GaussianQuadrature();

  // Quadrature implementations
  void hermiteQuadrature(int npoints, double mu, double sigma, 
                         double *z, double *y, double *w);
  void legendreQuadrature(int npoints, double a, double b, 
                          double *z, double *y, double *w);
  void laguerreQuadrature(int npoints, double mu, double beta, 
                          double *z, double *y, double *w);
};
