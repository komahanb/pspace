#ifndef ABSTRACT_PARAMETER
#define ABSTRACT_PARAMETER

#include "scalar.h"
#include "GaussianQuadrature.h"
#include "OrthogonalPolynomials.h"
#include <stdio.h>
#include <list>
#include <map>

/**
   Abstract base class for probabilistically modeled parameters

   @author Komahan Boopathy
*/
class AbstractParameter {
 public:
  // Constructor and destructor
  //---------------------------
  AbstractParameter();
  ~AbstractParameter();

  // Deferred procedures
  //---------------------
  virtual void quadrature(int npoints, scalar *z, scalar *y, scalar *w) = 0;
  virtual scalar basis(scalar z, int d) = 0;

  // Accessors
  //--------------------
  int getParameterID();
  int getMaxDegree();

  // Mutators
  //--------------------
  void setParameterID(int pid);
  void setMaxDegree(int dmax);

 protected:
  GaussianQuadrature *gauss;
  OrthogonalPolynomials *polyn;

 private:
  int parameter_id;
  int dmax;
};

#endif
