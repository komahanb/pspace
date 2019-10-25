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

  // Defer this to subclass??
  /* virtual update(double value) = 0; */
  /* virtual double fetch() = 0; */

  // Implemented procedures
  int getParameterID();  
  void setParameterID(int pid);

  // Make this a part of TACS maybe?
  // Fancy test stuff
  void setClient(void* obj){
    this->elem = obj;
  }
  void setClientFunction( void (*func)(void*, double) ){
    this->set = func;
  }

  void *elem; // points to element of consitituvive object to update 
  void (*set)(void*, double); // handle to set things
  double (*get)(void*);   // handle to get things
  void updateValue(double value){
    this->set(this->elem, value);
  };
  double getValue(){
    return this->get(this->elem);
  };

 protected:
  GaussianQuadrature *gauss;
  OrthogonalPolynomials *polyn;
  
 private:
  int parameter_id;
};

#endif
