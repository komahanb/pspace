#include "scalar.h"
#include "AbstractParameter.h"

/**
   A probabilistic parameter that follows exponential distribution

   @author Komahan Boopathy
*/
class ExponentialParameter : AbstractParameter{
 public:
  // Constructor and destructor
  ExponentialParameter(int pid, scalar mu, scalar beta);
  ~ExponentialParameter();

  // Member functions
  void quadrature(int npoints, scalar *z, scalar *y, scalar *w);
  scalar basis(scalar z, int d);

 private:
  // Member variables
  scalar mu;
  scalar beta;
};
