#include "AbstractParameter.h"

class NormalParameter : AbstractParameter{
 public:
  NormalParameter(int pid, double mu, double sigma);
  ~NormalParameter();
  void quadrature(int npoints, double *z, double *y, double *w);
  void basis(double z, int d);
 private:
  double mu;
  double sigma;
};
