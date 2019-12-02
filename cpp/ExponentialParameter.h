#include "scalar.h"
#include "AbstractParameter.h"

class ExponentialParameter : AbstractParameter{
 public:
  ExponentialParameter(int pid, scalar mu, scalar beta);
  ~ExponentialParameter();
  void quadrature(int npoints, scalar *z, scalar *y, scalar *w);
  scalar basis(scalar z, int d);
 private:
  scalar mu;
  scalar beta;
};
