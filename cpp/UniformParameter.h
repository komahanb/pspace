#include "AbstractParameter.h"

class UniformParameter : AbstractParameter{
 public:
  UniformParameter(int pid, double mu, double sigma);
  ~UniformParameter();
  void quadrature(int npoints, double *z, double *y, double *w);
  void basis(double z, int d);
 private:
  double a;
  double b;
};
