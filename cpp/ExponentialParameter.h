#include "AbstractParameter.h"

class ExponentialParameter : AbstractParameter{
 public:
  ExponentialParameter(int pid, double mu, double beta);
  ~ExponentialParameter();
  void quadrature(int npoints, double *z, double *y, double *w);
  void basis(double z, int d);
 private:
  double mu;
  double beta;
};
