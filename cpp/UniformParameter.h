#include "scalar.h"
#include "AbstractParameter.h"

class UniformParameter : AbstractParameter{
 public:
  UniformParameter(int pid, scalar mu, scalar sigma);
  ~UniformParameter();
  void quadrature(int npoints, scalar *z, scalar *y, scalar *w);
  scalar basis(scalar z, int d);
 private:
  scalar a;
  scalar b;
};
