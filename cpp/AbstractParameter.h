#ifndef ABSTRACT_PARAMETER
#define ABSTRACT_PARAMETER

#include "GaussianQuadrature.h"
#include "OrthogonalPolynomials.h"
#include <stdio.h>
#include <list>
#include <map>

class AbstractParameter {
 public:
  // Constructor and destructor
  //---------------------------
  AbstractParameter();
  ~AbstractParameter();

  // Deferred procedures
  //---------------------
  virtual void quadrature(int npoints, double *z, double *y, double *w) = 0;
  virtual double basis(double z, int d) = 0;

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
