#include "scalar.h"
#include "AbstractParameter.h"

class NormalParameter : AbstractParameter{
 public:
  NormalParameter(int pid, scalar mu, scalar sigma);
  ~NormalParameter();
  void quadrature(int npoints, scalar *z, scalar *y, scalar *w);
  scalar basis(scalar z, int d);
 private:
  scalar mu;
  scalar sigma;
};
