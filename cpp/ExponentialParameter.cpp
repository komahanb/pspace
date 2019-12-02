#include "ExponentialParameter.h"

/*
  Construct exponential parameter with input parameters
*/
ExponentialParameter::ExponentialParameter(int pid, scalar mu, scalar beta)
  : AbstractParameter() {
  this->setParameterID(pid);
  this->mu = mu;
  this->beta = beta;
}

/*
  Destructor
*/
ExponentialParameter::~ExponentialParameter(){}

/*
  Evaluate the basis at the supplied point 
*/
void ExponentialParameter::quadrature(int npoints, scalar *z, scalar *y, scalar *w){
  this->gauss->laguerreQuadrature(npoints, 
                                 this->mu, this->beta, 
                                 z, y, w);
}

/*
  Evaluate the basis of order d at the point z
*/
scalar ExponentialParameter::basis(scalar z, int d){
  return this->polyn->unit_laguerre(z, d);
}
