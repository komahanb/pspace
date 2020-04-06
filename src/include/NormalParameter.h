#include "scalar.h"
#include "AbstractParameter.h"

/**
   A probabilistic parameter that follows exponential distribution

   @author Komahan Boopathy
*/
class NormalParameter : AbstractParameter{
 public:
  // Constructor and destructor
  NormalParameter(int pid, scalar mu, scalar sigma);
  ~NormalParameter();

  // Member functions
  void quadrature(int npoints, scalar *z, scalar *y, scalar *w);
  scalar basis(scalar z, int d);
 private:
  // Member variables
  scalar mu;
  scalar sigma;
};
