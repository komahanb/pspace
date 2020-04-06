#include "scalar.h"
#include "AbstractParameter.h"

/**
   A probabilistic parameter that follows uniform distribution

   @author Komahan Boopathy
*/
class UniformParameter : AbstractParameter{
 public:
  // Constructor and destructor
  UniformParameter(int pid, scalar mu, scalar sigma);
  ~UniformParameter();

  // Member functions
  void quadrature(int npoints, scalar *z, scalar *y, scalar *w);
  scalar basis(scalar z, int d);

 private:
  // Member variables
  scalar a;
  scalar b;
};
