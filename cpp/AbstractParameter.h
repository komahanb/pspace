#ifndef ABSTRACT_PARAMETER
#define ABSTRACT_PARAMETER

#include "GaussianQuadrature.h"
#include "OrthogonalPolynomials.h"

class AbstractParameter {
 public:
  // Constructor and destructor
  AbstractParameter();
  ~AbstractParameter();

  // Deferred procedures
  virtual void quadrature(int npoints, double *z, double *y, double *w) = 0;
  virtual double basis(double z, int d) = 0;

  // Implemented procedures
  int getParameterID();  
  void setParameterID(int pid);

  // Function pointer to set values
  void setClientFunction( void (*func)(void*, double) ){
    this->set = func;
  }

  // Function pointer to retrieve values
  void getClientFunction( double (*func)(void*) ){
    this->get = func;
  }

  void (*set)(void*, double); // handle to set things
  double (*get)(void*);   // handle to get things

 protected:
  GaussianQuadrature *gauss;
  OrthogonalPolynomials *polyn;
  
 private:
  int parameter_id;
};

#endif
