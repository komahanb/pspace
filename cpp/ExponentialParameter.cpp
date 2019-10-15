#include "ExponentialParameter.h"

/*
  Construct exponential parameter with input parameters
*/
ExponentialParameter::ExponentialParameter(int pid, double mu, double beta)
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
void ExponentialParameter::quadrature(int npoints, double *z, double *y, double *w){
  this->gauss->laguerreQuadrature(npoints, 
                                 this->mu, this->beta, 
                                 z, y, w);
}

/*
  Evaluate the basis of order d at the point z
*/
double ExponentialParameter::basis(double z, int d){
  return this->polyn->unit_laguerre(z, d);
}
